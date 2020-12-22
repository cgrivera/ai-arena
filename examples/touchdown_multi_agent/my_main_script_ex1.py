# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import math, time, random

from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *

import my_config as cfg

arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES)

# --- only the root process will get beyond this point ---

# we define 3 sequential entities being fed to a worker of policy 1
# this can be dropped in to define any block of 3 entities being fed to the same worker
# see my_main_script_ex2.py for how to group non-sequential entities
policy_1_block = [1, 1, 1]

# policy 2 will use ungrouped entities to have one policy 2 worker per entity

# this is a list of assignments of entity <---> policy
# we can use the blocks defined above to create a match between groups of entities
match_list = [[policy_1_block, 2, 2, 2]]

#for each policy above, what type of policy is it
policy_types = {1:"multiagent_random", 2:"random"}

#train with this configuration
arena.kickoff(match_list, policy_types, 15000, render=True)

# NOTE: For the above configuration we will need to launch 6 processes:
# 1 root process
# 1 environment process
# 1 policy_1 worker, which will handle entities 0, 1, 2
# 3 policy_2 workers, which will separately handle 3, 4, 5

