import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from core import replay_memory

from core.genetic_agent import Actor
from core.mod_utils import hard_update, soft_update, LayerNorm


class Critic(nn.Module):

    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        l1 = 32; l2 = 64; l3 = int(l2/2)

        # Input layer
        self.bnorm = nn.BatchNorm1d(args.state_dim + args.action_dim)  # batch norm
        self.w_input = nn.Linear(args.state_dim + args.action_dim, l1)

        # Hidden Layer 1
        self.w_l1 = nn.Linear(l1, l2)
        self.lnorm1 = LayerNorm(l2)

        # Hidden Layer 1
        self.w_l2 = nn.Linear(l2, l3)
        self.lnorm2 = LayerNorm(l3)

        # Out
        self.w_out = nn.Linear(l3, 1)
        self.w_out.weight.data.mul_(0.1)
        self.w_out.bias.data.mul_(0.1)

        self.to(self.args.device)

    def forward(self, state, action):

        # Input Interface
        input = torch.cat((state, action), 1)
        input = self.bnorm(input)
        out = F.elu(self.w_input(input))
  
        # Hidden Layer 2
        out = self.w_l1(out)
        out = self.lnorm1(out)
        out = F.elu(out)

        # Hidden Layer 2
        out = self.w_l2(out)
        out = self.lnorm2(out)
        out = F.elu(out)

        # Output interface
        out = self.w_out(out)

        return out


class DDPG(object):
    def __init__(self, args):

        self.args = args
        self.buffer = replay_memory.ReplayMemory(args.individual_bs, args.device)
        self.critical_buffer = replay_memory.ReplayMemory(args.individual_bs, args.device)

        self.actor = Actor(args, init=True)
        self.actor_target = Actor(args, init=True)
        self.actor_optim = Adam(self.actor.parameters(), lr = self.args.lr)

        self.critic = Critic(args)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.args.lr)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def td_error(self, state, action, next_state, reward, done):
        next_action = self.actor_target.forward(next_state)
        next_q = self.critic_target(next_state, next_action)

        done = 1 if done else 0
        if self.args.use_done_mask: next_q = next_q * (1 - done)  # Done mask
        target_q = reward + (self.gamma * next_q)

        current_q = self.critic(state, action)
        dt = (current_q - target_q).abs()
        return dt.item()

    def update_parameters(self, batch, iteration : int):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch

        # Load everything to GPU if not already
        self.actor_target.to(self.args.device)
        self.critic_target.to(self.args.device)
        self.critic.to(self.args.device)
        state_batch = state_batch.to(self.args.device)
        next_state_batch = next_state_batch.to(self.args.device)
        action_batch = action_batch.to(self.args.device)
        reward_batch = reward_batch.to(self.args.device)
        if self.args.use_done_mask: done_batch = done_batch.to(self.args.device)

        # Critic Update
        next_action_batch = self.actor_target.forward(next_state_batch)
        next_q = self.critic_target.forward(next_state_batch, next_action_batch)
        if self.args.use_done_mask: next_q = next_q * (1 - done_batch) #Done mask
        target_q = reward_batch + (self.gamma * next_q).detach()

        self.critic_optim.zero_grad()
        current_q = self.critic.forward(state_batch, action_batch)
        delta = (current_q - target_q).abs()
        dt = torch.mean(delta**2)
        dt.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optim.step()

        # Actor Update
        self.actor_optim.zero_grad()

        policy_grad_loss = -(self.critic.forward(state_batch, self.actor.forward(state_batch))).mean()
        policy_loss = policy_grad_loss

        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return policy_grad_loss.data.cpu().numpy(), delta.data.cpu().numpy()



