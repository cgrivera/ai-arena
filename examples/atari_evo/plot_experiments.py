# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import math, time, random, os

from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *
from arena5.core.plot_utils import *

import matplotlib.pyplot as plt


def winning_policy(pool, logsdir):
	best_pol = 0
	best_score = -1000000
	for p in pool:
		pr = PolicyRecord(p, logsdir)
		length = min(len(pr.channels["main"].ep_results), 100)
		score = np.mean(pr.channels["main"].ep_results[-length:])
		if score > best_score:
			best_pol = p
			best_score = score
	return PolicyRecord(best_pol, logsdir)

dead_color = "#ffae57"
live_color = "#de123e"
ppo_color = "#8215e8"

logs1 = os.getcwd()+"/run1/"
logs2 = os.getcwd()+"/run2/"
logs3 = os.getcwd()+"/run3/"
logsppo = os.getcwd()+"/ppo_runs/"

records = []
colors = []

# run 1
r1_records = [PolicyRecord(p, logs1) for p in list(range(1,30))]
best = winning_policy(list(range(1,30)), logs1)
records += r1_records
records.append(best)
colors += [dead_color for r in r1_records]
colors.append(live_color)

# run 2
r2_records = [PolicyRecord(p, logs2) for p in list(range(1,42))]
best = winning_policy(list(range(1,42)), logs2)
records += r2_records
records.append(best)
colors += [dead_color for r in r2_records]
colors.append(live_color)

# run 3
r3_records = [PolicyRecord(p, logs3) for p in list(range(1,32))]
best = winning_policy(list(range(1,32)), logs3)
records += r3_records
records.append(best)
colors += [dead_color for r in r3_records]
colors.append(live_color)

# ppo policies
records += [PolicyRecord(p, logsppo) for p in [1,2,3]]
colors += [ppo_color for p in [1,2,3]]

fig, ax = plot_policy_records(records, [100], [1.0], "plot_evos.png", colors=colors, return_figure=True)
ax.set_xlim(right=3500000)
plt.savefig("plot_evos.png")