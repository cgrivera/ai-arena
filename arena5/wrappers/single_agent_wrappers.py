# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import numpy as np 
import cv2
from arena5.core.utils import *
from gym.spaces import Box, Discrete

# wrapper to make a standard gym env compatible with the arena,
# by re-framing it as a multi-agent problem with only one agent

class single_agent_wrapper():

	def __init__(self, env):

		self.env = env
		self.observation_spaces = [env.observation_space]
		self.action_spaces = [env.action_space]

	def reset(self):
		return [self.env.reset()]

	def step(self, action):
		action = self.prepare_action(action)
		s, r, d, info = self.env.step(action)

		# for i in range(4):
		# 	disp = s[:,:,i]
		# 	cv2.imshow("state"+str(i), disp)
		# 	cv2.waitKey(1)
		#mpi_print(r)
		return [s], [r], d, [info]

	def prepare_action(self, action):
		action = np.asarray(action)
		if isinstance(self.env.action_space, Discrete):
			action = int(np.squeeze(action))
		else:
			while len(action.shape)>1:
				action = action[0]
		return action

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()



# Some utility wrappers that mainly target atari games, may be useful elsewhere

class FrameStackWrapper():

	def __init__(self, env, stack):

		self.depth = stack
		self.state_buffer = []

		low = np.ndarray.flatten(env.observation_space.low)[0]
		high = np.ndarray.flatten(env.observation_space.high)[0]
		dtype = env.observation_space.dtype

		shape = list(env.observation_space.shape)
		shape[-1]*=self.depth
		self.observation_space = Box(low, high, shape=tuple(shape), dtype=dtype)
		self.action_space = env.action_space

		self.env = env

	def reset(self):
		s = self.env.reset()
		self.state_buffer = []
		while len(self.state_buffer) < self.depth:
			self.state_buffer.append(s)
		return self.compile_buffer()

	def step(self, a):
		ns, r, d, info = self.env.step(a)
		self.state_buffer.append(ns)
		self.state_buffer.pop(0)
		return self.compile_buffer(), r, d, info

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()

	def compile_buffer(self):
		if len(np.asarray(self.state_buffer).shape) == 3:
			final_state = np.asarray(self.state_buffer)
			final_state = np.transpose(final_state, (1,2,0))
			return final_state
		else:
			return np.concatenate([np.asarray(s) for s in self.state_buffer])


class GrayscaleWrapper():

	def __init__(self, env):
		self.env = env

		low = np.ndarray.flatten(env.observation_space.low)[0]
		high = np.ndarray.flatten(env.observation_space.high)[0]
		dtype = env.observation_space.dtype

		shape = list(env.observation_space.shape)
		shape[-1] = 1
		self.observation_space = Box(low, high, shape=tuple(shape), dtype=dtype)
		self.action_space = env.action_space

	def reset(self):
		return self.process_state(self.env.reset())

	def step(self, a):
		ns, r, d, info = self.env.step(a)
		return self.process_state(ns), r, d, info

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()

	def process_state(self, s):
		return cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


class ImageDownsizeWrapper():

	def __init__(self, env, square_size):
		self.env = env
		self.sz = square_size

		low = np.ndarray.flatten(env.observation_space.low)[0]
		high = np.ndarray.flatten(env.observation_space.high)[0]
		dtype = env.observation_space.dtype

		shape = list(env.observation_space.shape)
		for i in range(len(shape)-1):
			shape[i] = square_size
		shape[-1] = 1
		self.observation_space = Box(low, high, shape=tuple(shape), dtype=dtype)
		self.action_space = env.action_space

	def reset(self):
		return self.process_state(self.env.reset())

	def step(self, a):
		ns, r, d, info = self.env.step(a)
		return self.process_state(ns), r, d, info

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()

	def process_state(self, s):
		return cv2.resize(s, (self.sz, self.sz))