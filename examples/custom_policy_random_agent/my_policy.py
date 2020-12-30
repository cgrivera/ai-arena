# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import random

'''
A custom policy must be a class with at least following methods:

def __init__(self, env, policy_comm):

	env is an environment object that conforms to a gym interface

	policy_comm is an mpi4py comm object to communicate with the other processes
		that are working to train this policy.  See docs for details.


def run(self, num_steps, data_dir, policy_record=none):

	num_steps is the number of steps that this policy should execute on self.env

	data_dir is a path to a directory for writing data that is specific to this policy
		All processes implementing this policy will be passed this location

	policy_record is a record keeping utility object that is passed to the root process
		for this policy.  Its use is completely optional.

'''

class MyCustomPolicyRandom():

	# create our policy, tracking the environment and comms objects
	def __init__(self, env, policy_comm):
		self.env = env
		self.comm = policy_comm

	# the most rudimentary random agent
	def run(self, num_steps, data_dir, policy_record=None):

		#do an initial reset of the environment
		self.env.reset()

		#since we do not synchronize step counts, get steps needed for individual worker:
		local_steps = int(num_steps / self.comm.Get_size())

		# run!
		for stp in range(local_steps):
			a = self.env.action_space.sample()
			_,_,done,_ = self.env.step(a)

			if done:
				self.env.reset()