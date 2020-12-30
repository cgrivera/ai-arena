# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import torch
from torch.distributions.normal import Normal


def generalized_advantage_estimation(reward, value, done, gamma=0.99, lam=0.95):
    last_part = 0
    advantages = torch.zeros_like(reward)
    for t in reversed(range(reward.size(0))):
        if done[t]:
            delta = reward[t] - value[t]
            last_part = 0
        else:
            delta = reward[t] + gamma * value[t + 1] - value[t]
        advantages[t] = delta + gamma * lam * last_part
        last_part = advantages[t]
    return advantages


def discounted_reward(reward, done, gamma=0.99):
    T = reward.size(0)
    discounted_rewards = torch.zeros_like(reward)
    for t in reversed(range(T)):
        if done[t]:
            discounted_rewards[t] = reward[t]
        else:
            discounted_rewards[t] = reward[t] + gamma * discounted_rewards[t + 1]
    return discounted_rewards


def ppo_loss(old_means, old_log_stds, new_means, new_log_stds, actions, advantages, clip_param=0.2):
    old_distribution = Normal(old_means, torch.exp(old_log_stds))
    old_log_prob = old_distribution.log_prob(actions)

    new_distribution = Normal(new_means, torch.exp(new_log_stds))
    new_log_prob = new_distribution.log_prob(actions)

    ratio = torch.exp(new_log_prob - old_log_prob)
    surrogate_loss = torch.min(advantages * ratio, advantages * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param))
    policy_loss = -surrogate_loss.sum(-1).mean()

    entropy = -(new_log_prob * torch.exp(new_log_prob)).sum(-1).mean()

    return policy_loss, entropy
