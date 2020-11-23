import random
from arena5.core.utils import mpi_print
from arena5.wrappers.mpi_logging_wrappers import MPISynchronizedPRUpdater
import arena5.algos.ddpg.ddpg_network_utils as core
from arena5.algos.ddpg.ddpg import ddpg

class DDPGPolicy():

    def __init__(self, env, policy_comm, eval_mode=False):
        self.env = env
        self.comm = policy_comm
        self.eval_mode = eval_mode
        # print(env.observation_spaces, env.action_spaces)


    def run(self, num_steps, data_dir, policy_record=None):

        self.env = MPISynchronizedPRUpdater(self.env, self.comm, policy_record)
        local_steps = int(num_steps / self.comm.Get_size())

        ddpg(lambda: self.env, self.comm, data_dir, policy_record, self.eval_mode,
        	actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[256,256]), 
            gamma=0.99, seed=1337, steps_per_epoch=local_steps, epochs=1,
            logger_kwargs=None)