# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import gym
from mpi4py import MPI
from arena5.wrappers.single_agent_wrappers import *
from CoverEnv import CoverEnv

def make_env():
	env = CoverEnv(N=2, headless=True)
	return env