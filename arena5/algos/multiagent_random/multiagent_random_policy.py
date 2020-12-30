# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import random

class MARandomPolicy():

	# IMPORTANT!!!!!
	# env is a multiagent environment, not a standard gym environment
	# it exposes env.observation_spaces and env.action_spaces (NOTE PLURAL!)
	# reset returns a LIST of states, not one state
	# step takes a LIST of actions, and returns:
	#	 a LIST of states, LIST of rewards, global_done, LIST of infos

	def __init__(self, env, policy_comm):
		self.env = env
		self.comm = policy_comm

		self.num_entities = len(self.env.observation_spaces)

	def run(self, num_steps, data_dir, policy_record=None):

		# initial reset
		states = self.env.reset()

		cumr = 0.0
		ep_len = 0

		# since we do not synchronize step counts, get steps needed for individual worker:
		# NOTE: one step consists of an action from all entities, so this is independent from 
		# the number of entities we control
		local_steps = int(num_steps / self.comm.Get_size())

		for stp in range(local_steps):

			# collect random aciton for each entity
			actions = []
			for i in range(self.num_entities):
				a = self.env.action_spaces[i].sample()
				actions.append(a)

			# act
			states, rewards, done, infos = self.env.step(actions)

			# we will log the sum of all rewards but you could do whatever you want here
			for r in rewards:
				cumr += r
			ep_len += 1

			# handle done
			if done:
				states = self.env.reset()

				if policy_record is not None:
					policy_record.add_result(cumr, ep_len)

				cumr = 0.0
				ep_len = 0

		if policy_record is not None:
			policy_record.save()