import gym
from arena5.wrappers.single_agent_wrappers import *

def make_env():
	env = gym.make("PongDeterministic-v4")
	env = GrayscaleWrapper(env) #also normalizes
	env = ImageDownsizeWrapper(env, 64)
	env = FrameStackWrapper(env, 4)
	env = single_agent_wrapper(env)
	return env
	
	