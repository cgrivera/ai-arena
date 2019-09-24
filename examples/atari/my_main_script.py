import math, time, random

from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *
from arena5.core.plot_utils import *

import my_config as cfg

arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES)

# --- only the root process will get beyond this point ---

match_list = [[1] for _ in range(5)]
policy_types = {1:"ppo"}
arena.kickoff(match_list, policy_types, 2000000)
