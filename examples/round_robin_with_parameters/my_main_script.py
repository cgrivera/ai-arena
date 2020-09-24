import math, time, random

from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *

import my_config as cfg

arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES)

# --- only the root process will get beyond this point ---

#define team colors, for fun
red = (1.0,0,0)
blue = (0,0,1.0)
green = (0,1.0,0)

# policy #1 and #2 will be using the PPO algorithm, policy #3 will be random
policy_types = {1:"ppo", 2:"ppo", 3:"random"}

# optionally set kwargs for the policies, we will adjust hyperparameters for ppo #2:
policy_kwargs = {2:{"optim_stepsize":0.0}}

# there are 3 nominal matches to be played:
match_list = [
	[1,1, 2,2],  # blue vs red
	[1,1, 3,3],  # blue vs green
	[2,2, 3,3]   # red  vs green
]

# define parameters for each environment
colors = [
	{"clr1":blue, "clr2":red}, 
	{"clr1":blue, "clr2":green}, 
	{"clr1":red, "clr2":green}
]

# define logging colors for the policies
plot_colors = {1:blue, 2:red, 3:green}

#train with this configuration
arena.kickoff(match_list, policy_types, 50000, render=True, scale=True, policy_kwargs=policy_kwargs,
	env_kwargs=colors, plot_colors=plot_colors)

