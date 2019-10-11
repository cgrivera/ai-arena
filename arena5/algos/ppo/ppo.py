import random, os

from arena5.algos.ppo.ppo1_mod import PPO1
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLstmPolicy, CnnLstmPolicy

class PPOPolicy():

	def __init__(self, env, policy_comm, use_lstm=False):
		self.env = env
		self.comm = policy_comm

		if use_lstm:
			if len(self.env.observation_space.shape) > 2:
				pcy = CnnLstmPolicy
			else:
				pcy = MlpLstmPolicy

		else:
			if len(self.env.observation_space.shape) > 1:
				pcy = CnnPolicy
			else:
				pcy = MlpPolicy

		self.model = PPO1(pcy, env, policy_comm, timesteps_per_actorbatch=128, clip_param=0.2, entcoeff=0.01, 
			optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear', 
			verbose=1)
    

	def run(self, num_steps, data_dir, policy_record=None):

		if os.path.exists(data_dir+"/ppo_save.pkl"):
			self.model = PPO1.load(data_dir+"ppo_save", self.env, self.comm)

		self.model.learn(num_steps, policy_record)

		if policy_record is not None:
			policy_record.save()

			self.model.save(policy_record.data_dir+"ppo_save")