# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import math, time, random, os

from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *
from arena5.core.plot_utils import *

import matplotlib.pyplot as plt


logs = os.getcwd()+"/log_comms/"

masac_record = PolicyRecord(2, logs)
maddpg_record = PolicyRecord(1, logs)

records = [masac_record, maddpg_record]
colors = ["#dd0000", "#0000dd"]


fig, ax = plot_policy_records(records, [20,100], [0.2,1.0], "plot_results.png", colors=colors, return_figure=True)
ax.set_xlim(right=4500000)
plt.savefig("plot_results.png")