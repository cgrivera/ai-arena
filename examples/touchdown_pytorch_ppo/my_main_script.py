import os
from arena5.core.stems import make_stem
from arena5.core.policy_record import PolicyRecord, plot_policy_records

# prepare a method to build our custom policy
from my_policy import PPOCuriosity as Policy
from make_env import make_env

LOG_COMMS_DIR = os.getcwd() + "/log_comms/"
MAKE_ENV_LOCATION = os.getcwd()

temp = make_env()
OBS_SPACES = temp.observation_spaces
ACT_SPACES = temp.action_spaces
del temp

# pass the additional policies to the arena here:
arena = make_stem(MAKE_ENV_LOCATION, LOG_COMMS_DIR, OBS_SPACES, ACT_SPACES,
                  additional_policies={Policy.__name__: Policy})

# --- only the root process will get beyond this point ---

# this is a list of assignments of entity <---> policy
match_list = [[0, 1]]

# for each policy above, what type of policy is it
# you can specify a string name or a method to create a custom algorithm
policy_types = {0: Policy.__name__, 1: Policy.__name__}

# train with this configuration
NUM_ROUNDS = 10
for i in range(NUM_ROUNDS):
    arena.kickoff(match_list, policy_types, 15000, render=True, scale=True)

    records = [PolicyRecord(p, LOG_COMMS_DIR) for p in range(2)]
    # plot_policy_records(records, [100], [1.0], "plot_" + str(i) + ".png", colors=None, offsets=None)
