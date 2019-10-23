import math, time, random

from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *

import my_config as cfg

arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES)

# --- only the root process will get beyond this point ---

# for this example we want to control non-sequentially indexed entities with the same policy workers
# we will define blocks [1,1,1] and [2,2,2]
# however, we will supply a entity index remapping of [0,2,4, 1,3,5] such that the match list
# now corresponds to entities in this order

# this is a list of assignments of entity <---> policy
match_list = [ [[1,1,1],[2,2,2]] ]
entity_maps = [ [0,2,4, 1,3,5] ]

#for each policy above, what type of policy is it
policy_types = {1:"multiagent_random", 2:"multiagent_random"}

#train with this configuration
arena.kickoff(match_list, policy_types, 15000, entity_remaps=entity_maps, render=True)

# NOTE: For the above configuration we will need to launch 6 processes:
# 1 root process
# 1 environment process
# 1 policy_1 worker, which will handle entities 0, 1, 2
# 3 policy_2 workers, which will separately handle 3, 4, 5

