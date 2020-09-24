import random

from arena5.core.utils import mpi_print
from arena5.wrappers.mpi_logging_wrappers import MPISynchronizedPRUpdater

class RandomPolicy():

	def __init__(self, env, policy_comm):
		self.env = env
		self.comm = policy_comm

	def run(self, num_steps, data_dir, policy_record=None):

		self.env = MPISynchronizedPRUpdater(self.env, self.comm, policy_record)
		
		self.env.reset()

		# cumr = 0.0
		# ep_len = 0

		#since we do not synchronize step counts, get steps needed for individual worker:
		local_steps = int(num_steps / self.comm.Get_size())

		for stp in range(local_steps):
			a = self.env.action_space.sample()
			_,r,done,_ = self.env.step(a)
			# cumr += r
			# ep_len += 1

			if done:
				self.env.reset()

				# if policy_record is not None:
				# 	policy_record.add_result(cumr, ep_len)

				# cumr = 0.0
				# ep_len = 0

		# if policy_record is not None:
		# 	policy_record.save()