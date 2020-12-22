# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import random
from arena5.core.utils import mpi_print
from arena5.wrappers.mpi_logging_wrappers import MPISynchronizedPRUpdater
import arena5.algos.sac.sac_network_utils as core
from arena5.algos.sac.sac import sac

class SACPolicy():

    def __init__(self, env, policy_comm, eval_mode=False, worker_replay_size=int(1e5)):
        self.env = env
        self.comm = policy_comm
        self.eval_mode = eval_mode
        self.worker_replay_size = worker_replay_size


    def run(self, num_steps, data_dir, policy_record=None):

        self.env = MPISynchronizedPRUpdater(self.env, self.comm, policy_record)
        local_steps = int(num_steps / self.comm.Get_size())

        print(self.env.action_space, self.env.observation_space)
        policy = core.MLPActorCritic
        print('observation space shape', self.env.observation_space.shape)
        if len(self.env.observation_space.shape)>1:
            policy = core.ConvActorCritic
        sac(lambda: self.env, self.comm, data_dir, policy_record, self.eval_mode,
        	actor_critic=policy,
            ac_kwargs=dict(hidden_sizes=[256,256]), 
            gamma=0.99, seed=1337,steps_per_epoch=local_steps, epochs=1,
            replay_size=self.worker_replay_size, logger_kwargs=None)
