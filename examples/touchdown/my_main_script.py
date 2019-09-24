

# To create a script for the arena, we need to interact with a UserStem object.
# This object is active only on the root process, and broadcasts commands to the
# other processes.

import math, time, random

from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *

import my_config as cfg

arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES)

# --- only the root process will get beyond this point ---

#this is a list of assignments of entity <---> policy
match_list = [[1,2], [1,2], [2,1]]

#for each policy above, what type of policy is it
#you can specify a string name or TODO: a path to a custom algo
policy_types = {1:"random", 2:"ppo"}

#train with this configuration
arena.kickoff(match_list, policy_types, 15000)

# get policy 2 record
# pr2 = PolicyRecord(2, cfg.LOG_COMMS_DIR)
# pr2.save()

# mpi_print(pr2.ep_results)

# # fork it and continue with 3 instead of 2
# pr3 = pr2.fork(3)
# match_list = [[1,3], [3,3], [3,1]]
# policy_types.append(2) #register policy 3 as using ppo
# arena.kickoff(match_list, policy_types, 15000)

# # get policy 3 record
# pr3 = PolicyRecord(3, cfg.LOG_COMMS_DIR)

# # fork it and continue with 4 instead of 3
# pr4 = pr3.fork(4)
# match_list = [[1,4], [4,4], [4,1]]
# policy_types.append(2) #register policy 4 as using ppo
# arena.kickoff(match_list, policy_types, 15000)