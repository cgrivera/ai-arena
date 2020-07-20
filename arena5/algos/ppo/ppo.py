import random, os

from arena5.algos.ppo.ppo1_mod import PPO1
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLstmPolicy, CnnLstmPolicy

def PPOLSTMPolicy(env, policy_comm):
	return PPOPolicy(env, policy_comm, True)

def PPOLSTMPolicyEval(env, policy_comm):
	return PPOPolicy(env, policy_comm, True, True)

def PPOPolicyEval(env, policy_comm):
	return PPOPolicy(env, policy_comm, False, True)

class PPOPolicy():

	def __init__(self, env, policy_comm, use_lstm=False, eval_mode=False, external_saved_file=None):
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

		self.eval_mode = eval_mode

		self.external_saved_file=external_saved_file


	def run(self, num_steps, data_dir, policy_record=None):

		if self.external_saved_file is not None:
			self.model = PPO1.load(self.external_saved_file, self.env, self.comm)
		elif os.path.exists(data_dir+"/ppo_save.pkl") or os.path.exists(data_dir+"/ppo_save.zip"):
			self.model = PPO1.load(data_dir+"ppo_save", self.env, self.comm)
			print("loaded model from saved file!")

		if self.eval_mode:
			self.model.evaluate(num_steps)
		else:	
			self.model.learn(num_steps, policy_record)

			if policy_record is not None:
				policy_record.save()

				self.model.save(policy_record.data_dir+"ppo_save")
