import pickle, os
from shutil import copy

from arena5.core.plot_utils import *


def get_dir_for_policy(policy_id, log_comms_dir):
	return log_comms_dir + "policy_"+str(policy_id)+"/"

class PolicyRecord():

	def __init__(self, policy_id, log_comms_dir):

		self.ep_results = []
		self.ep_lengths = []
		self.ep_cumlens = []

		self.log_comms_dir = log_comms_dir
		self.data_dir = get_dir_for_policy(policy_id, log_comms_dir)

		if not os.path.exists(self.data_dir):
			os.makedirs(self.data_dir)
		else:
			self.load()

	def add_result(self, total_reward, ep_len):
		self.ep_results.append(total_reward)
		self.ep_lengths.append(ep_len)
		if len(self.ep_cumlens) == 0:
			self.ep_cumlens.append(ep_len)
		else:
			self.ep_cumlens.append(self.ep_cumlens[-1]+ep_len)

	def save(self):
		data = [self.ep_results, self.ep_lengths, self.ep_cumlens]
		pickle.dump(data, open(self.data_dir+"policy_record.p", "wb"))

		#save a plot also
		plot_policy_records([self], [20, 50, 100], [0.1, 0.3, 1.0], self.data_dir+"plot.png", colors=["#eb0033"])

	# WARNING: This will overwrite current recorded data
	def load(self):
		path = self.data_dir+"policy_record.p"
		if os.path.exists(path):
			data = pickle.load(open(path, "rb"))
			self.ep_results, self.ep_lengths, self.ep_cumlens = data


	def fork(self, new_id):
		new_pr = PolicyRecord(new_id, self.log_comms_dir)
		new_pr.ep_results = self.ep_results[:]
		new_pr.ep_lengths = self.ep_lengths[:]
		new_pr.ep_cumlens = self.ep_cumlens[:]
		
		# copy files over from existing log dir
		for f in os.listdir(self.data_dir):

			# HACK: check for f not having any extension
			if "." not in f:
				continue

			copy(self.data_dir+f, new_pr.data_dir)

		return new_pr
