# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import sys
import numpy as np
from arena5.core.utils import mpi_print

class EnvironmentProcess():

	def __init__(self, make_env_method, global_comm, local_comm, match_root_rank, call_render=False, env_kwargs={}):

		if isinstance(make_env_method, str):
			sys.path.append(make_env_method)
			from make_env import make_env
			self.env = make_env(**env_kwargs)
		else:
			self.env = make_env_method(**env_kwargs)

		mpi_print("made env")

		self.global_comm = global_comm
		self.local_comm = local_comm
		self.match_root_rank = match_root_rank
		self.call_render = call_render

	def proxy_sync(self):
		self.states = self.env.reset()
		self.local_comm.bcast(self.states, root=self.match_root_rank)

	def run(self, num_steps):

		#play steps
		for stp in range(num_steps):

			#get all actions
			actions = [[-1], [[0]]]
			actions = self.local_comm.gather(actions, root=self.match_root_rank)

			#some entries may represent multiple entities- convert all to single entity
			entity_actions = []
			for entry in actions:
				idxs = entry[0]
				actions_for_idxs = entry[1]

				for i in range(len(idxs)):
					entity_actions.append([idxs[i], actions_for_idxs[i]])

			#sort actions by entity id
			entity_actions = sorted(entity_actions, key=lambda x: x[0])

			#discard the first action, which is a dummy provided by this processes
			entity_actions = [x[1] for x in entity_actions[1:]]

			#step
			new_states, rewards, done, infos = self.env.step(entity_actions)

			# RENDER HERE
			if self.call_render:
				self.env.render()

			#send results back
			self.local_comm.bcast([new_states, rewards, done, infos], root=self.match_root_rank)

			if done:
				self.states = self.env.reset()

				#proxies will call reset -> respond with states
				self.local_comm.bcast(self.states, root=self.match_root_rank)

			else:
				self.states = new_states


			#mpi_print("env", stp+1, "/", num_steps)

		if hasattr(self.env, "close"):
			self.env.close()


