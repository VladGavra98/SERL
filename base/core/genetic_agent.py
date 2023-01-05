import torch
import torch.nn as nn
from torch.optim import Adam
from parameters import Parameters
from core import replay_memory
from core.mod_utils import is_lnorm_key, LayerNorm, activations



class GeneticAgent:
    def __init__(self, args: Parameters):

        self.args = args

        self.actor = Actor(args)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-3)

        self.buffer = replay_memory.ReplayMemory(self.args.individual_bs, args.device)
        self.critical_buffer = replay_memory.ReplayMemory(self.args.individual_bs, args.device)
        self.loss = nn.MSELoss()

    def update_parameters(self, batch, p1, p2, critic) -> float:
        """ Crossover parameter update.

        Args:
            batch (tuple): Past experiences
            p1 (_type_): Parent actor 1
            p2 (_type_): Parent actor 1
            critic (_type_): Critic network  for filtering

        Returns:
            float: Policy clonning loss
        """
        state_batch, _, _, _, _ = batch

        #  Redeem parents' actions
        p1_action = p1(state_batch)
        p2_action = p2(state_batch)

        #  Redeem parents' Qs
        p1_q1,p1_q2 = critic(state_batch, p1_action)
        p1_q = torch.min(p1_q1,p1_q2).flatten()
        p2_q1,p2_q2 = critic(state_batch, p2_action)
        p2_q = torch.min(p2_q1,p2_q2).flatten()

        #  Select best behaving pparent based on Q-filtering:
        eps = 10**-5  # selection threshold -- how much better one action is wrt the other
        action_batch = torch.cat((p1_action[p1_q - p2_q > eps], p2_action[p2_q - p1_q >= eps])).detach()
        state_batch  = torch.cat((state_batch[p1_q - p2_q > eps], state_batch[p2_q - p1_q >= eps]))
        actor_action = self.actor(state_batch)

        #  Actor update
        self.actor_optim.zero_grad()
        sq = (actor_action - action_batch)**2  
        policy_loss = torch.sum(sq) + torch.mean(actor_action**2)   # clonning loss
        policy_mse = torch.mean(sq)
        policy_loss.backward()
        self.actor_optim.step()

        return policy_mse.item()

    def load_from_dict(self,actor_dict : dict):
        self.actor = actor_dict





class Actor(nn.Module):

    def __init__(self, args, init=False):
        super(Actor, self).__init__()
        self.args = args
        h = args.hidden_size
        L = args.num_layers
        activation = activations[args.activation_actor.lower()]

        layers = []

        # Input Layer 
        layers.extend([
            # nn.BatchNorm1d(args.state_dim, affine=True, track_running_stats=True),
            nn.Linear(args.state_dim, h),
            activation,
        ])
        
        # Hidden Layers
        for _ in range(L):
            layers.extend([
                nn.Linear(h, h),
                LayerNorm(h),
                activation
            ])

        # Output Layer 
        layers.extend([
            nn.Linear(h, args.action_dim),
            nn.Tanh(),
        ])
            
        self.net = nn.Sequential(*layers)
        self.to(args.device)

    def forward(self, state : torch.tensor) -> torch.tensor:
        return self.net(state)

    def select_action(self, state : torch.tensor):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.args.device)
        return self.forward(state).cpu().data.numpy().flatten()

    def get_novelty(self, batch):
        state_batch, action_batch, _, _, _ = batch
        novelty = torch.mean(torch.sum((action_batch - self.forward(state_batch))**2, dim=-1))
        self.novelty = novelty.item()
        return self.novelty

    # function to return current pytorch gradient in same order as genome's flattened parameter vector
    def extract_grad(self):
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count + sz] = param.grad.view(-1)
            count += sz
        return pvec.detach().clone()

    # function to grab current flattened neural network weights
    def extract_parameters(self):
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count + sz] = param.view(-1)
            count += sz
        return pvec.detach().clone()

    # function to inject a flat vector of ANN parameters into the model's current neural network weights
    def inject_parameters(self, pvec):
        count = 0
        for name, param in self.named_parameters():
            # only alter W -- skip norms and biases
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            raw = pvec[count:count + sz]
            reshaped = raw.view(param.size())
            param.data.copy_(reshaped.data)
            count += sz

    # count how many parameters are in the model
    def count_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            count += param.numel()
        return count


