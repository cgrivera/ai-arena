import math, time, random

from arena5.core.stems import *
from arena5.core.utils import *
from arena5.core.policy_record import *
from arena5.core.plot_utils import *

import my_config as cfg


mpi_lag(wait=2.0)

arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES)

# --- only the root process will get beyond this point ---

match_list = [[[1, 1]]]
policy_types = {1:"maddpg"}

# Specify number of steps to run across all matches:
total_steps = 10000000
steps_per_match = total_steps_to_match_steps(match_list, total_steps)

arena.kickoff(match_list, policy_types, steps_per_match, render=False, scale=True)
