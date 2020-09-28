
import numpy as np
from arena5.core.utils import mpi_print

'''
This file contains magic arena sauce.

A environment proxy conforms to either the standard gym interface or the arena 
multiagent version depending on how many entity indexes it is passed.

Algorithms will interact with this proxy within their own process, and the proxy
will use MPI to shuttle the data back and forth to a different process running the environment.

For the environment side, see env_process.py
''' 

def make_proxy_env(entity_idxs, obs_spaces, act_spaces, match_comm, match_root_rank):
	if len(entity_idxs) == 1:
		return gym_proxy_env(entity_idxs[0], obs_spaces[0], act_spaces[0], match_comm, match_root_rank)
	else:
		return ma_proxy_env(entity_idxs, obs_spaces, act_spaces, match_comm, match_root_rank)


class gym_proxy_env():

	def __init__(self, entity_idx, obs_space, act_space, match_comm, match_root_rank):
		self.observation_space = obs_space
		self.action_space = act_space
		self.comm = match_comm
		self.match_root_rank = match_root_rank
		self.entity_idx = entity_idx
		self.is_multiagent = False

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
		action_packet = [[self.entity_idx], [action]]
		self.comm.gather(action_packet, root=self.match_root_rank)

		#get resulting info
		result = self.comm.bcast(None, root=self.match_root_rank)
		nss, rs, done, infos = result
		ns, r, info = nss[self.entity_idx], rs[self.entity_idx], infos[self.entity_idx]

		return ns, r, done, info


class ma_proxy_env():

	def __init__(self, entity_idxs, obs_spaces, act_spaces, match_comm, match_root_rank):
		self.observation_spaces = obs_spaces
		self.action_spaces = act_spaces
		self.comm = match_comm
		self.match_root_rank = match_root_rank
		self.entity_idxs = entity_idxs
		self.is_multiagent = True

	def seed(self, sd):
		pass

	def reset(self, **kwargs):
		states = self.comm.bcast(None, root=self.match_root_rank)
		ret_states = []
		for idx in self.entity_idxs:
			ret_states.append(states[idx])
		return ret_states

	def step(self, actions):

		#assume this contains properly formatted actions
		#and that it indeed contains more than one action
		#convert to a list in case it is a numpy array
		actions = np.asarray(actions).tolist()
		fmtactions = []
		for a in actions:
			#make np array if it is not already
			action = np.asarray([a])

			#make sure action is a 1D array
			while len(action.shape) < 1:
				action = np.expand_dims(action, -1)
			while len(action.shape) > 1:
				action = np.squeeze(action)

			#convert to list
			action = action.tolist()
			fmtactions.append(action)

		#send actions to main env proc
		action_packet = [self.entity_idxs, fmtactions]
		self.comm.gather(action_packet, root=self.match_root_rank)

		#get resulting info
		result = self.comm.bcast(None, root=self.match_root_rank)
		nss, rs, done, infs = result

		next_states = []
		rewards = []
		infos = []
		for idx in self.entity_idxs:
			next_states.append(nss[idx])
			rewards.append(rs[idx])
			infos.append(infs[idx])

		return next_states, rewards, done, infos