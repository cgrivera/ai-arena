# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import gym
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from arena5.wrappers.single_agent_wrappers import *
from mpi4py import MPI

def make_env():

	env = wrap_deepmind(make_atari("BreakoutNoFrameskip-v4"))
	workerseed = MPI.COMM_WORLD.Get_rank()*10000
	env.seed(workerseed)

	env = single_agent_wrapper(env)
	return env
	
	