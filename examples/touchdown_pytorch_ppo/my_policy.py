import torch
from torch.distributions.normal import Normal
from torch.utils.data import SubsetRandomSampler, BatchSampler
import gym
from mpi4py import MPI
import os
from memory import Memory
from ppo import ppo_loss, generalized_advantage_estimation, discounted_reward


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Model(torch.nn.Module):
    def __init__(self, obs_space: gym.spaces.Box, ac_space: gym.spaces.Box, hidden_size=128, act=torch.nn.ReLU):
        super().__init__()

        def make_model(output_size):
            return torch.nn.Sequential(
                torch.nn.Conv2d(obs_space.shape[0], 32, 1, stride=4), act(),
                torch.nn.Conv2d(32, 16, 1, stride=3), act(),
                Flatten(),
                torch.nn.Linear(16 * 7 * 1, hidden_size), act(),
                torch.nn.Linear(hidden_size, hidden_size), act(),
                torch.nn.Linear(hidden_size, output_size)
            )

        self.policy = make_model(ac_space.shape[0])
        self.logstd = make_model(ac_space.shape[0])
        self.value = make_model(1)

    def forward(self, x):
        return self.policy(x), self.logstd(x), self.value(x)


class PPOCuriosity(object):
    TRAIN_BATCH_SIZE = 128
    NUM_TRAIN_EPOCHS = 8
    ENTROPY_COEF = 0.01

    ROLLOUT_BATCH_SIZE = 512

    def __init__(self, env, policy_comm):
        """
        env is an environment object that conforms to a gym interface

        policy_comm is an mpi4py comm object to communicate with the other processes that are working to train this policy.  See docs for details.
        """
        self.env = env
        self.comm = policy_comm
        self.rank = self.comm.Get_rank()
        self.world_rank = MPI.COMM_WORLD.Get_rank()
        self.root = 0
        self.is_root = self.rank == self.root

        print(f'{self.world_rank} {self.rank} {self.comm.Get_size()}')

        # Avoid slowdowns caused by each separate process's PyTorch using more than its fair share of CPU resources.
        torch.set_num_threads(max(int(torch.get_num_threads() / self.comm.Get_size()), 1))

        self.model: torch.nn.Module = Model(env.observation_space, env.action_space)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def run(self, num_steps, data_dir, policy_record=None):
        # since we do not synchronize step counts, get steps needed for individual worker:
        local_steps = int(num_steps / self.comm.Get_size())

        # print(f'{self.world_rank} {self.rank} loading...', flush=True)
        self.load_from_dir(data_dir)

        # print(f'{self.world_rank} {self.rank} waiting...', flush=True)
        # self.comm.barrier()

        rollout = self.rollouts(local_steps)

        while local_steps > 0:
            # print(f'{self.world_rank} {self.rank} Syncing weights', flush=True)
            self.mpi_sync_weights()

            # print(f'{self.world_rank} {self.rank} Rolling out', flush=True)
            memory, local_steps = next(rollout)

            # print(f'{self.world_rank} {self.rank} Gathering memory', flush=True)
            memory = self.mpi_gather_memory(memory)

            if self.is_root:
                # print(f'{self.world_rank} {self.rank} Training', flush=True)
                # assert (self.comm.Get_size() * steps_to_take == len(self.memory.data['obs']))
                self.train(memory)

            # print(f'{self.world_rank} {self.rank} waiting...', flush=True)
            # self.comm.barrier()

        self.save_to_dir(policy_record)

    def load_from_dir(self, data_dir):
        if os.path.exists(data_dir + "/model.pth"):
            self.model.load_state_dict(torch.load(data_dir + "/model.pth"))

    def save_to_dir(self, policy_record):
        if policy_record is not None:
            policy_record.save()
            torch.save(self.model.state_dict(), policy_record.data_dir + 'model.pth')

    def mpi_sync_weights(self):
        # get data to send
        if self.is_root:
            state_dict = self.model.state_dict()
        else:
            state_dict = None

        # broadcast data
        state_dict = self.comm.bcast(state_dict, root=self.root)

        # use the data
        self.model.load_state_dict(state_dict)

    def mpi_gather_memory(self, memory):
        data = memory.numpy()

        all_data = self.comm.gather(data, root=self.root)

        if self.is_root:
            assert (type(all_data) is list)
            assert (len(all_data) == self.comm.Get_size())
            memory.clear()
            for i, data in enumerate(all_data):
                memory.store_numpy(data)
            return memory
        else:
            assert all_data is None
            return None

    def rollouts(self, total_steps):
        memory = Memory()

        # do an initial reset of the environment
        obs = self.env.reset()

        # collect training data
        for i_step in range(total_steps):
            batch_done = (i_step > 0 and i_step % self.ROLLOUT_BATCH_SIZE == 0)
            rollouts_done = i_step == total_steps - 1

            torch_obs = torch.from_numpy(obs).float()
            with torch.no_grad():
                mean, logstd, value = self.model(torch_obs.unsqueeze(0))

            distribution = Normal(mean, torch.exp(logstd))
            action = distribution.sample()

            new_obs, reward, done, info = self.env.step(action.numpy())

            memory.store(obs=torch_obs, reward=reward, done=done or batch_done or rollouts_done, action=action,
                         mean=mean, logstd=logstd, value=value)

            if batch_done:
                yield memory, total_steps - i_step - 1
                del memory
                memory = Memory()

            if done:
                obs = self.env.reset()
            else:
                obs = new_obs

        yield memory, 0

    def train(self, memory: Memory):
        # train over training data
        obs, action, reward, done = memory.get('obs', 'action', 'reward', 'done')
        old_mean, old_logstd, old_value = memory.get('mean', 'logstd', 'value')

        # compute advantages
        advantage = generalized_advantage_estimation(reward, old_value, done)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # compute target values
        target_value = discounted_reward(reward, done)

        policy_losses = torch.zeros(self.NUM_TRAIN_EPOCHS)
        entropies = torch.zeros(self.NUM_TRAIN_EPOCHS)
        value_losses = torch.zeros(self.NUM_TRAIN_EPOCHS)

        for i_epoch in range(self.NUM_TRAIN_EPOCHS):
            for indices in BatchSampler(SubsetRandomSampler(range(obs.shape[0])), self.TRAIN_BATCH_SIZE,
                                        drop_last=True):
                new_mean, new_logstd, new_value = self.model(obs[indices])

                policy_loss, entropy = ppo_loss(old_mean[indices], old_logstd[indices],
                                                new_mean, new_logstd,
                                                action[indices], advantage[indices])

                value_loss = (target_value[indices] - new_value).pow(2).mean()

                loss = policy_loss - self.ENTROPY_COEF * entropy + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                policy_losses[i_epoch] += policy_loss
                entropies[i_epoch] += entropy
                value_losses[i_epoch] += value_loss

        print(f'{self.world_rank} {self.rank} Rewards {reward.sum().item()}', flush=True)
        print(f'{self.world_rank} {self.rank} Policy {policy_losses.detach().numpy()}', flush=True)
        print(f'{self.world_rank} {self.rank} Value {value_losses.detach().numpy()}', flush=True)
        print(f'{self.world_rank} {self.rank} Entropy {entropies.detach().numpy()}', flush=True)
