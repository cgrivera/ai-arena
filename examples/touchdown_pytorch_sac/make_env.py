# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import gym
from mpi4py import MPI

from Touchdown import TouchdownEnv

def make_env():
	return TouchdownEnv(1, blue_obs="vector", blue_actions="continuous",
		red_obs="vector", red_actions="continuous")