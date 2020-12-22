# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
from Touchdown import TouchdownEnv

def make_env():
	return TouchdownEnv(1, blue_obs="image", blue_actions="discrete",
		red_obs="image", red_actions="discrete")