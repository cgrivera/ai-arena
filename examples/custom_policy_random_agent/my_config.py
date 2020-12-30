# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import os
from make_env import make_env

# CUSTOM POLICY ================================================================
# Tell the arena about non-built-in policies that it can access
# This is a dictionary with:
#	keys: 	string id for the policy, which the main script will use
#	values: [file_name, class_name] of the random policy

custom_policy_python_file = os.getcwd()+"/my_policy.py"
custom_policy_class_name = "MyCustomPolicyRandom"
ADDITIONAL_POLS = {"my_custom_random":[custom_policy_python_file, custom_policy_class_name]}
# ==============================================================================


# Tell the arena where it can put log files that describe the results of
# specific policies.  This is also used to pass results between root processes.

LOG_COMMS_DIR = os.getcwd()+"/log_comms/"


# Define where to find the environment
# if make_env is in this directory, os.getcwd() will suffice

MAKE_ENV_LOCATION = os.getcwd()


# Tell the arena what observation and action spaces to expect

temp = make_env()

OBS_SPACES = temp.observation_spaces
ACT_SPACES = temp.action_spaces

del temp
