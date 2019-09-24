import sys
import numpy as np
from arena5.core.utils import mpi_print

class EnvironmentProcess():

	def __init__(self, make_env_location, global_comm, local_comm, match_root_rank):

		sys.path.append(make_env_location)
		from make_env import make_env
		self.env = make_env()

		mpi_print("made env")

		self.global_comm = global_comm
		self.local_comm = local_comm
		self.match_root_rank = match_root_rank

	def proxy_sync(self):
		self.states = self.env.reset()
		self.local_comm.bcast(self.states, root=self.match_root_rank)

	def run(self, num_steps):

		#play steps
		for stp in range(num_steps):

			#get all actions
			actions = [-1, [0]]
			actions = self.local_comm.gather(actions, root=self.match_root_rank)

			#sort actions by entity id
			actions = sorted(actions, key=lambda x: x[0])
			actions = [x[1] for x in actions[1:]]

			#step
			new_states, rewards, done, infos = self.env.step(actions)

			# RENDER HERE
			#self.env.render()

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


