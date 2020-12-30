# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import numpy as np
import scipy.signal
import os

import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        N = len(obs)
        all_inputs = [o for o in obs] + [a for a in act]
        q = self.q(torch.cat(all_inputs, dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_spaces, action_spaces, common_actor, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        N = len(observation_spaces)
        obs_dim = observation_spaces[0].shape[0]
        act_dim = action_spaces[0].shape[0]
        act_limit = action_spaces[0].high[0]

        # build policy and value functions
        if common_actor:
            pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
            self.pis = [pi for i in range(N)]
            self.unique_pis = nn.ModuleList([pi])
        else:
            self.pis = []
            for i in range(N):
                pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
                self.pis.append(pi)
            self.pis = nn.ModuleList(self.pis)
            self.unique_pis = self.pis

        self.q = MLPQFunction(obs_dim*N, act_dim*N, hidden_sizes, activation)

    def act(self, obs):
        actions = []
        for i in range(len(obs)):
            with torch.no_grad():
                a = self.pis[i](obs[i])
                actions.append(a.numpy())
        return actions