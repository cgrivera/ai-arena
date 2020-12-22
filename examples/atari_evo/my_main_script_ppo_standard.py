# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import math, time, random

from arena5.core.stems import *
from arena5.core.utils import *
from arena5.core.policy_record import *
from arena5.core.plot_utils import *

import my_config as cfg

arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES)

# --- only the root process will get beyond this point ---


for i in range(3):

	match_list = [[i+1]]*9
	policy_types = {i+1:"ppo"}

	# Specify number of steps to run across all matches:
	steps_per_match = total_steps_to_match_steps(match_list, 5000000)
	arena.kickoff(match_list, policy_types, steps_per_match, render=False, scale=False)