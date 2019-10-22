import math, time, random

from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *
from arena5.core.plot_utils import *

import my_config as cfg

arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES)

# --- only the root process will get beyond this point ---

# create 3 environments and assign to each a process executing policy number 1
match_list = [[1] for _ in range(3)]

# define policy number 1 to be "my_custom_random"
policy_types = {1:"my_custom_random"}

# run the training for 12,000 total steps, or 4,000 steps per worker
arena.kickoff(match_list, policy_types, 12000)
