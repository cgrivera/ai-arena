import math, time, random

from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *
from arena5.core.plot_utils import *

import my_config as cfg

arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES)

# --- only the root process will get beyond this point ---

rounds = 20
pop = 3
repeat_matches = 3
steps_per_round = 83333

# ---------------------------------------

pool = [i+1 for i in range(pop)]
latest = pop

all_policies = [i+1 for i in range(pop)]
curr_offset = 0

for i in range(rounds):

	match_list = []
	for rp in range(repeat_matches):
		match_list += [[p] for p in pool]

	policy_types = {}
	for p in pool:
		policy_types[p] = "ppo"
	arena.kickoff(match_list, policy_types, steps_per_round)
	curr_offset += steps_per_round

	#plot the results so far
	records = [PolicyRecord(p, cfg.LOG_COMMS_DIR) for p in all_policies]
	plot_policy_records(records, [100], [1.0], "plot_"+str(i)+".png", colors=None, offsets=None)

	#find the best policy to retain
	best_pol = 0
	best_score = -1000000
	for p in pool:
		pr = PolicyRecord(p, cfg.LOG_COMMS_DIR)
		length = min(len(pr.ep_results), 100)
		score = np.mean(pr.ep_results[-length:])
		if score > best_score:
			best_pol = p
			best_score = score

	#replace the bad policies if we have rounds to go
	if i < rounds-1:
		new_pool = [best_pol]
		while len(new_pool) < len(pool):
			latest += 1
			pr = PolicyRecord(best_pol, cfg.LOG_COMMS_DIR)
			pr.fork(latest)
			new_pool.append(latest)
			all_policies.append(latest)

		pool = new_pool