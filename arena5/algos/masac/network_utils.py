# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import os


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


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


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

    def __init__(self, observation_spaces, action_spaces, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        N = len(observation_spaces)
        obs_dim = observation_spaces[0].shape[0]
        act_dim = action_spaces[0].shape[0]
        act_limit = action_spaces[0].high[0]

        # build policy and value functions
        self.pis = []
        for i in range(N):
            pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
            self.pis.append(pi)
        self.pis = nn.ModuleList(self.pis)

        self.q1 = MLPQFunction(obs_dim*N, act_dim*N, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim*N, act_dim*N, hidden_sizes, activation)


    def act(self, obs, deterministic=False):
        actions = []
        for i in range(len(obs)):

            with torch.no_grad():
                a, _ = self.pis[i](obs[i], deterministic, False)
                actions.append(a.numpy())

        return actions

    def current_pi(self, obs):
        actions = []
        logprobs = []
        for i in range(len(obs)):
            a, lp = self.pis[i](obs[i])
            actions.append(a)
            logprobs.append(lp)

        return actions, logprobs