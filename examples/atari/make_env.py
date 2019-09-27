import gym
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from arena5.wrappers.single_agent_wrappers import *
from mpi4py import MPI

def make_env():
	# env = gym.make("PongNoFrameskip-v4")
	# env = GrayscaleWrapper(env) #also normalizes
	# env = ImageDownsizeWrapper(env, 64)
	# env = FrameStackWrapper(env, 4)

	env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
	workerseed = MPI.COMM_WORLD.Get_rank()*10000
	env.seed(workerseed)

	env = single_agent_wrapper(env)
	return env
	
	