from Touchdown import TouchdownEnv

# 2 v 2 touchdown environment
# this will therefore expose 6 entities: blue 1-3 and red 1-3 (indexed 0-5)
def make_env():
	return TouchdownEnv(3, blue_obs="image", blue_actions="discrete",
		red_obs="image", red_actions="discrete")