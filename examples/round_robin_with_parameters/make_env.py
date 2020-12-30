# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
from Touchdown import TouchdownEnv

# 2 v 2 touchdown environment
# this will therefore expose 4 entities: team 1 (1 and 2) and team 2 (3 and 4)
def make_env(clr1=[], clr2=[]):
	return TouchdownEnv(2, blue_obs="image", blue_actions="discrete",
		red_obs="image", red_actions="discrete", clr1=clr1, clr2=clr2)