import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam

from core import replay_memory
from core.genetic_agent import Actor
from core.mod_utils import hard_update, soft_update, LayerNorm, activations

from typing import Tuple
import logging

MAX_GRAD_NORM = 10



class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        # Layer sizes
        # l1 = 200; l2 = 300; l3 = l2    # original PDERL values (no tuning done)
        l1 = 64; l2 = 64;            # NOTE these worked for TD3-only control

        # Non-linearity:
        self.activation = activations[self.args.activation_actor.lower()]

        # Critic 1
        self.bnorm_1 = nn.BatchNorm1d(args.state_dim + args.action_dim)  # batch norm
        self.l1_1 = nn.Linear(args.state_dim + args.action_dim, l1)
        self.lnorm1_1 = LayerNorm(l1)
        self.l2_1 = nn.Linear(l1, l2)
        self.lnorm2_1 = LayerNorm(l2)
        self.lout_1 = nn.Linear(l2, 1)

        # Critic 2
        self.bnorm_2 = nn.BatchNorm1d(args.state_dim + args.action_dim)  # batch norm
        self.l1_2 = nn.Linear(args.state_dim + args.action_dim, l1)
        self.lnorm1_2 = LayerNorm(l1)
        self.l2_2 = nn.Linear(l1, l2)
        self.lnorm2_2 = LayerNorm(l2)
        self.lout_2 = nn.Linear(l2, 1)

        # Initlaise wights with smaller values
        self.lout_1.weight.data.mul_(0.1);self.lout_1.bias.data.mul_(0.1)
        self.lout_2.weight.data.mul_(0.1);self.lout_2.bias.data.mul_(0.1)

        self.to(self.args.device)

    def forward(self, state, action):
        # ------ Critic 1 ---------
        nn_input = torch.cat((state,action), 1)
        # nn_input = self.bnorm_1(nn_input)

        # hidden layer 1_1 (Input Interface)
        out = self.l1_1(nn_input)
        out = self.lnorm1_1(out)
        out = self.activation(out)

        # hidden layer 2_1
        out = self.l2_1(out)
        out = self.lnorm2_1(out)
        out = self.activation(out)

        # output interface
        out1 = self.lout_1(out)

        # ------ Critic 2 ---------
        # hidden layer 1_2 (Input Interface)
        nn_input = torch.cat((state,action), 1)

        out = self.l1_2(nn_input)
        out = self.lnorm1_2(out)
        out = self.activation(out)

        # hidden Layer 2_2
        out = self.l2_2(out)
        out = self.lnorm2_2(out)
        out = self.activation(out)

        # output interface
        out2 = self.lout_2(out)

        return out1, out2


class TD3(object):
    def __init__(self, args):

        self.args = args
        self.buffer = replay_memory.ReplayMemory(args.individual_bs, args.device)
        self.critical_buffer = replay_memory.ReplayMemory(args.individual_bs, args.device)

        # Initialise actor
        self.actor = Actor(args, init=True)
        self.actor_target = Actor(args, init=True)
        self.actor_optim = Adam(self.actor.parameters(), lr = self.args.lr)

        # Initialise critics
        self.critic = Critic(args)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.args.lr)

        # Initliase loss
        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss  = nn.MSELoss()

        # Make sure target starts with the same weights
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.caps_dict : dict = None

        if self.args.use_caps:
            self.caps_dict = {
                            'lambda_s' : 0.5,
                            'lambda_t' : 0.1,
                            'eps_sd'    : 0.05,
                             }


    def update_parameters(self, batch, iteration : int, champion_policy = False) -> Tuple[float,float]:
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch

        with torch.no_grad():
            # Load everything to GPU if not already
            self.actor_target.to(self.args.device)
            self.critic_target.to(self.args.device)
            self.critic.to(self.args.device)
            state_batch = state_batch.to(self.args.device)
            next_state_batch = next_state_batch.to(self.args.device)
            action_batch = action_batch.to(self.args.device)
            reward_batch = reward_batch.to(self.args.device)
            done_batch = done_batch.to(self.args.device)

            # Select target action
            noise = (torch.randn_like(action_batch) *\
                      self.args.noise_sd).clamp(-self.args.noise_clip, self.args.noise_clip)
            next_action_batch = torch.clamp(noise + self.actor_target.forward(next_state_batch), -1,1)

            # Compute the target Q values
            target_Q1, target_Q2 = self.critic_target.forward(next_state_batch, next_action_batch)
            next_Q = torch.min(target_Q1, target_Q2)
            next_Q = next_Q * (1 - done_batch)
            target_q = reward_batch + (self.gamma * next_Q).detach()

        # Get current Q estimates
        current_q1, current_q2 = self.critic.forward(state_batch, action_batch)

        # Compute critic losses
        loss_q1 = F.mse_loss(current_q1, target_q)
        loss_q2 = F.mse_loss(current_q2, target_q)
        TD =  loss_q1 + loss_q2

        # Optimize Critics
        self.critic_optim.zero_grad()
        TD.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
        self.critic_optim.step()
        TD_data = TD.data.cpu().numpy()

        # Soft Target Updates
        pgl = None
        if iteration % self.args.policy_update_freq == 0:
            policy_grad_loss = self.actor_update(state_batch, action_batch)

            if not champion_policy:
                # smooth target actor update
                soft_update(self.actor_target, self.actor, self.tau)

            soft_update(self.critic_target, self.critic, self.tau)
            pgl = policy_grad_loss.data.cpu().numpy()

        return pgl, TD_data

    def actor_update(self, state_batch, action_batch):
        self.actor_optim.zero_grad()

        # retrieve value of the critics
        est_q1,_ = self.critic.forward(state_batch, self.actor.forward(state_batch))  # objective reward
        policy_grad_loss = -torch.mean(est_q1)                                     # add minus to make it a loss

        if self.caps_dict is not None:
            next_action_batch = self.actor.forward(state_batch)
            state_bar  = state_batch + torch.rand_like(state_batch) * self.caps_dict['eps_sd']
            action_bar = self.actor.forward(state_bar)
            caps_loss  = self.caps_dict['lambda_t'] * F.mse_loss(action_batch, next_action_batch) + \
                         self.caps_dict['lambda_s'] * F.mse_loss(action_batch, action_bar)

            policy_grad_loss += caps_loss

        # backprop
        policy_grad_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
        self.actor_optim.step()

        return policy_grad_loss
