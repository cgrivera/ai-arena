# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import torch
from torch.distributions import Categorical
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import argparse
import gym
import sys
import torch.nn as nn
import torch.optim as optim



from mpi4py import MPI
import time


class GymEnvironment:
    def __init__(self, agent, rank, env,seed=None):
        self.done = False

        self.agent = agent
        self.rank = rank
        self.env = env
        if seed is not None:
            self.env.seed(seed)
        self.last_state = self.env.reset()

    def reset(self):
        self.done = False
        self.last_state = self.env.reset()
        print('finished reset')

    def step(self):
        print('stepping')
        action = self.agent.decide(torch.tensor(self.last_state).float())
        new_state, reward, done, _ = self.env.step(action)
        self.last_state = new_state
        self.agent.reward(reward)
        self.done = done
        print('done stepping')


class A2CAgent:
    def __init__(self):
        self.__iter = 0
        self.last_value = 0.0
        self.replay = Replay()
        self.total_reward = 0.0
        self.deterministic = False

    def decide(self, x):
        self.replay.states.append(x)
        probs, value = self.forward(x)

        distrib = Categorical(probs)
        action = distrib.sample().item() if not self.deterministic else torch.argmax(probs)
        self.replay.actions.append(action)
        self.last_value = value.item()
        self.__iter += 1
        self.replay.iter = self.__iter
        return action

    def reward(self, r):
        self.replay.rewards.append(r)
        self.total_reward += r

    def forward(self, x):
        raise NotImplementedError

    def on_reset(self, is_terminal):
        raise NotImplementedError

    def set_replay_hidden(self, hidden):
        self.replay.hidden0 = hidden

    def reset(self, new_episode=True):
        if new_episode:
            self.__iter = 0
            self.total_reward = 0.0
        self.replay = Replay()
        self.on_reset(new_episode)

    def end_replay(self, is_terminal):
        replay = self.replay
        replay.is_terminal = is_terminal
        replay.iter = self.__iter
        replay.total_reward = self.total_reward
        if not is_terminal:
            self.reset(new_episode=False)
        if replay.hidden0 is None:
            replay.hidden0 = torch.zeros(1)
        return replay

class Brain(nn.Module, A2CAgent):
    def __init__(self):
        nn.Module.__init__(self)
        A2CAgent.__init__(self)

        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        value = self.value_head(x)
        return F.softmax(action_scores, dim=-1), value

    def save(self, filename):
        f = open(filename, "wb")
        torch.save(self.state_dict(), f)
        f.close()

    def load(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)

    def on_reset(self, new_episode):
        pass



class Replay:
    def __init__(self):
        self.hidden0 = None
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminal = False
        self.iter = 0
        self.total_reward = 0.0

class MPIA2C:
    @staticmethod
    def init_mpi_rng(seed_base=None):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if seed_base is None:
            seed_base = int(time.time())
        torch.manual_seed(seed_base)

        seed = int(torch.randint(2 ** 32, (rank + 1,))[rank].item())
        seed = seed_base ^ seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        return seed

    #def __init__(self, comm, rank, optimizer, on_batch_done, on_episodes_done):
    def __init__(self,env,comm):
        self.use_gpu = False

        self.seed = MPIA2C.init_mpi_rng(0)
        self.comm = comm
        self.rank = MPI.COMM_WORLD.Get_rank()  #was passed by parameter previously
        self.replays = []
        self.episode = 0
        self.max_train_iters = 0
        self.agent = Brain()
        self.env = GymEnvironment(self.agent, self.rank, env, self.seed)

        self.gamma = 0.99
        self.steps_in_replay = 1

        #self.optimizer = optimizer
        self.optimizer = optim.Adam(self.agent.parameters(), lr=3e-3)
        self.ppo_clip = 0.2
        self.ppo_iters = 4

        self.on_batch_done = self.batch_done
        self.on_episodes_done = self.episodes_done
        print('finished agent init')


    def batch_done(self, loss, batch):
        pass

    def episodes_done(self, episode, batch_rewards, batch_iters):
        total_reward_avg = torch.tensor(batch_rewards).float().mean().item()
        self.episode_reward = total_reward_avg
        iter_avg = torch.tensor(batch_iters).float().mean().item()
        self.running_reward = self.running_reward * 0.99 + total_reward_avg * 0.01
        self.running_iter = self.running_iter * 0.99 + iter_avg * 0.01
        self.episode = episode

        if episode % self.log_interval == 0:
            dt = time.time() - self.lst_tick
            self.lst_tick = time.time()
            self.total_time += dt

            print('Episode {}\tLast length: {:5}\tAverage length: {:.2f}'.format(self.episode,  round(self.episode_reward), self.running_reward))
            sys.stdout.flush()
            if self.running_reward > self.threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(self.running_reward, round(self.episode_reward)))
                sys.exit(0)

        if episode % self.save_interval == 0:
            self.agent.save(self.dst)

    def ppo_loss(self, probs, old_probs, pred_vals, true_vals, mask):
        advantage = (true_vals - pred_vals).detach()
        prob_ratio = torch.exp(probs - old_probs)
        ploss_cpi = prob_ratio * advantage
        ploss_clip = torch.clamp(prob_ratio, 1-self.ppo_clip, 1+self.ppo_clip) * advantage
        ploss = -torch.min(ploss_cpi, ploss_clip)
        vloss = F.smooth_l1_loss(pred_vals, true_vals, reduce=False)
        loss = (ploss + vloss) * mask
        loss = loss.sum() / mask.sum()
        return loss

    def a2c_loss(self, logprobs, pred_vals, true_vals, mask):
        advantage = (true_vals - pred_vals).detach()
        ploss = -(logprobs * advantage)
        vloss = F.smooth_l1_loss(pred_vals, true_vals, reduce=False)
        loss = (ploss + vloss) * mask
        loss = loss.sum() / mask.sum()
        return loss

    def batch_replays(self, replays, max_pad):
        batch_size = len(replays)
        mask = torch.ones(batch_size, max_pad)

        init_done = False
        i = 0

        states = None
        actions = None
        true_values = None
        hiddens = None

        total_rewards = []
        iters = []

        for r in replays:
            if r.is_terminal:
                total_rewards.append(r.total_reward)
                iters.append(r.iter)

            seq_len = len(r.rewards)
            sys.stdout.flush()
            if seq_len < max_pad:
                mask[i, range(seq_len, max_pad)] = 0.0

            tv = torch.tensor(r.rewards)
            tv = F.pad(tv, (0, max_pad - seq_len))
            tv.unsqueeze_(0)

            st = r.states
            st = F.pad(st, (0, 0, 0, max_pad - seq_len))
            st.unsqueeze_(0)

            h0 = r.hidden0
            h0.unsqueeze_(0)

            ac = r.actions
            ac = F.pad(ac, (0, max_pad - seq_len))
            ac.unsqueeze_(0)

            actions = torch.cat((actions, ac)) if init_done else ac
            true_values = torch.cat((true_values, tv)) if init_done else tv
            states = torch.cat((states, st)) if init_done else st
            hiddens = torch.cat((hiddens, h0)) if init_done else h0

            init_done = True
            i += 1

        if self.use_gpu:
            actions = actions.cuda()
            true_values = true_values.cuda()
            states = states.cuda()
            hiddens = hiddens.cuda()
            mask = mask.cuda()

        if batch_size == 1:
            states = torch.split(states, 1, dim=1)
            actions = torch.split(actions, 1, dim=1)
        else:
            states = torch.split(states, 1, dim=1)
            actions = torch.split(actions, 1, dim=1)
        return states, actions, hiddens, true_values, mask, total_rewards, iters

    def master(self):
        agents = self.comm.size - 1
        total_rewards_in_replays = []
        iters_in_replays = []
        iter = 0
        episodes = 0

        if self.use_gpu:
            self.agent.cuda()

        self.comm.barrier()
        while iter < self.max_train_iters:
            replays = self.comm.gather(None, root=0)
            sys.stdout.flush()
            del replays[0]
            batch_size = len(replays)

            if batch_size == 0:
                break

            max_pad = len(max(replays, key=lambda r: len(r.rewards)).rewards)

            # self.replays.extend(replays)
            states, actions, hiddens, \
            true_values, mask, \
            batch_total_rewards, batch_iters = self.batch_replays(replays, max_pad)

            total_rewards_in_replays.extend(batch_total_rewards)
            iters_in_replays.extend(batch_iters)

            old_logprobs = None
            loss = 0.0
            for k in range(self.ppo_iters):
                self.agent.hidden = hiddens
                logprob_list = []
                pred_values_list = []

                for s, a in zip(states, actions):
                    if batch_size > 1:
                        s.squeeze_()
                        a.squeeze_()
                    else:
                        s = s.view(-1, s.size(2))

                    prob, value = self.agent.forward(s)
                    distrib = Categorical(prob)
                    logprob = distrib.log_prob(a)

                    logprob_list.append(logprob)
                    pred_values_list.append(value)

                logprobs = torch.stack(logprob_list).t() if batch_size > 1 else torch.stack(logprob_list).squeeze_()

                pred_vals = torch.stack(pred_values_list).squeeze().t() if batch_size > 1 else torch.stack(
                    pred_values_list).squeeze_()

                true_values.squeeze_()

                if self.ppo_iters > 1:
                    if old_logprobs is None:
                        old_logprobs = logprobs.detach()
                    loss = self.ppo_ßloss(logprobs, old_logprobs, pred_vals, true_values, mask)
                else:
                    loss = self.a2c_loss(logprobs, pred_vals, true_values, mask)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            iter += 1
            self.agent.reset()
            self.comm.bcast(self.agent.state_dict(), root=0)

            self.on_batch_done(loss, iter)

            # keep track of full episodes
            if len(total_rewards_in_replays) >= agents:
                self.on_episodes_done(episodes, total_rewards_in_replays, iters_in_replays)
                total_rewards_in_replays = []
                iters_in_replays = []
                episodes += 1
        print("Done!")

    def slave(self):
        env = self.env
        batch = 0

        self.comm.barrier()
        while batch < self.max_train_iters:
            self.agent.reset()
            env.reset()
            steps = 0
            while not env.done:
                env.step()
                steps += 1
                if (steps % self.steps_in_replay == 0 and steps != 0) or env.done:
                    replay = self.agent.end_replay(env.done)
                    replay.rewards = MPIA2C.discount_rewards(replay.rewards, self.gamma, env.done, self.agent.last_value)
                    replay.states = torch.stack(replay.states)

                    replay.hidden0 = replay.hidden0.squeeze()
                    replay.actions = torch.tensor(replay.actions)
                    self.comm.gather(replay, root=0)

                    batch += 1

                    new_weights = self.comm.bcast(None, root=0)
                    self.agent.load_state_dict(new_weights)

            self.episode += 1
        print("Agent %d terminated" % self.rank)


    def run(self, num_steps, data_dir, policy_record=None):
        print('starting to run the agent')
        env = self.env
        batch = 0
        self.max_train_iters = num_steps
        self.comm.barrier()
        while batch < self.max_train_iters:
            self.agent.reset()
            env.reset()
            steps = 0
            while not env.done:
                env.step()
                steps += 1
                if (steps % self.steps_in_replay == 0 and steps != 0) or env.done:
                    replay = self.agent.end_replay(env.done)
                    replay.rewards = MPIA2C.discount_rewards(replay.rewards, self.gamma, env.done, self.agent.last_value)
                    replay.states = torch.stack(replay.states)

                    replay.hidden0 = replay.hidden0.squeeze()
                    replay.actions = torch.tensor(replay.actions)
                    self.comm.gather(replay, root=0)

                    batch += 1

                    new_weights = self.comm.bcast(None, root=0)
                    self.agent.load_state_dict(new_weights)

            self.episode += 1
            #if self.rank == 0:
            replays = self.comm.gather(None, root=0)
            sys.stdout.flush()
            del replays[0]
            batch_size = len(replays)

            if batch_size == 0:
                break

            max_pad = len(max(replays, key=lambda r: len(r.rewards)).rewards)

            # self.replays.extend(replays)
            states, actions, hiddens, \
            true_values, mask, \
            batch_total_rewards, batch_iters = self.batch_replays(replays, max_pad)

            total_rewards_in_replays.extend(batch_total_rewards)
            iters_in_replays.extend(batch_iters)

            old_logprobs = None
            loss = 0.0
            for k in range(self.ppo_iters):
                self.agent.hidden = hiddens
                logprob_list = []
                pred_values_list = []

                for s, a in zip(states, actions):
                    if batch_size > 1:
                        s.squeeze_()
                        a.squeeze_()
                    else:
                        s = s.view(-1, s.size(2))

                    prob, value = self.agent.forward(s)
                    distrib = Categorical(prob)
                    logprob = distrib.log_prob(a)

                    logprob_list.append(logprob)
                    pred_values_list.append(value)

                logprobs = torch.stack(logprob_list).t() if batch_size > 1 else torch.stack(logprob_list).squeeze_()

                pred_vals = torch.stack(pred_values_list).squeeze().t() if batch_size > 1 else torch.stack(
                    pred_values_list).squeeze_()

                true_values.squeeze_()

                if self.ppo_iters > 1:
                    if old_logprobs is None:
                        old_logprobs = logprobs.detach()
                    loss = self.ppo_ßloss(logprobs, old_logprobs, pred_vals, true_values, mask)
                else:
                    loss = self.a2c_loss(logprobs, pred_vals, true_values, mask)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            iter += 1
            self.agent.reset()
            self.comm.bcast(self.agent.state_dict(), root=0)

            self.on_batch_done(loss, iter)

            # keep track of full episodes
            if len(total_rewards_in_replays) >= agents:
                self.on_episodes_done(episodes, total_rewards_in_replays, iters_in_replays)
                total_rewards_in_replays = []
                iters_in_replays = []
                episodes += 1

            #train the agent





    #def run(self):
        #if self.rank == 0:
        #    self.master()
        #else:
        #    self.slave()


    @staticmethod
    def discount_rewards(rewards, gamma, is_terminal, last_value):
        values = []
        v = last_value if not is_terminal else 0.0
        for r in rewards[::-1]:
            v = r + gamma * v
            values.insert(0, v)
        return values
