
import numpy as np
from arena5.core.utils import mpi_print


# TODO: extend this to allow multiple entities (i.e. list of spaces)
# Or possibly create a separate proxy for multi-agent or vector envs
class proxy_env():

	def __init__(self, entity_idx, obs_space, act_space, match_comm, match_root_rank):
		self.observation_space = obs_space
		self.action_space = act_space
		self.comm = match_comm
		self.match_root_rank = match_root_rank
		self.entity_idx = entity_idx

	def seed(self, sd):
		pass

	def reset(self):
		states = self.comm.bcast(None, root=self.match_root_rank)
		state = states[self.entity_idx]
		return state

	def step(self, action):

		#make np array if it is not already
		action = np.asarray([action])

		#make sure action is a 1D array
		while len(action.shape) < 1:
			action = np.expand_dims(action, -1)
		while len(action.shape) > 1:
			action = np.squeeze(action)

		#convert to list
		action = action.tolist()

		#send actions to main env proc
		action_packet = [self.entity_idx, action]
		self.comm.gather(action_packet, root=self.match_root_rank)

		#get resulting info
		result = self.comm.bcast(None, root=self.match_root_rank)
		nss, rs, done, infos = result
		ns, r, info = nss[self.entity_idx], rs[self.entity_idx], infos[self.entity_idx]

		return ns, r, done, info