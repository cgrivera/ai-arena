import gym
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from arena5.wrappers.single_agent_wrappers import *
from mpi4py import MPI

# Define a method to create a python object to house our environment
# This object needs to conform to the AI Arena modified gym interface
def make_env():

	# create pong environment and use wrappers from stable baselines
	env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
	workerseed = MPI.COMM_WORLD.Get_rank()*10000
	env.seed(workerseed)

	# convert standard gym interface to multiagent interface expected by ai arena
	env = single_agent_wrapper(env)
	return env
	
	