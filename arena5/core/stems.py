
from mpi4py import MPI
from arena5.core.utils import mpi_print
from arena5.core.proxy_env import proxy_env
from arena5.core.env_process import EnvironmentProcess

from arena5.algos.random.random_policy import RandomPolicy
from arena5.algos.ppo.ppo import PPOPolicy

from arena5.core.policy_record import PolicyRecord, get_dir_for_policy

def make_stem(make_env_location, log_comms_dir, obs_spaces, act_spaces):
	rank = MPI.COMM_WORLD.Get_rank()
	if rank == 0:
		return UserStem(make_env_location, log_comms_dir, obs_spaces, act_spaces) #return an object to control arena
	else:
		stem = WorkerStem(make_env_location, log_comms_dir, obs_spaces, act_spaces)
		while True:
			stem.loop() #loop indefinitely


class UserStem(object):

	def __init__(self, make_env_location, log_comms_dir, obs_spaces, act_spaces):

		self.global_comm = MPI.COMM_WORLD
		self.global_rank = self.global_comm.Get_rank()
		mpi_print("I am the user stem, at rank", self.global_rank)

	def kickoff(self, match_list, policy_types, steps_per_match):

		#make sure there are enough procs to run the matches
		# TODO

		#broadcast the match list
		self.global_comm.bcast(match_list, root=0)

		#broadcast the policy types
		self.global_comm.bcast(policy_types, root=0)

		#broadcast the number of steps to run each match
		self.global_comm.bcast(steps_per_match, root=0)

		#create some unused groups - need to call this from root for it to work in other procs
		temp_match_group_comm = self.global_comm.Create(self.global_comm.group.Excl([]))
		temp_pol_group_comm = self.global_comm.Create(self.global_comm.group.Excl([]))

		# now wait for response from all the workers, gather dummy data as a syncing tool
		_ = self.global_comm.gather(None, root=0)

		mpi_print("ROUND COMPLETE!")

class WorkerStem(object):

	def __init__(self, make_env_location, log_comms_dir, obs_spaces, act_spaces):

		self.global_comm = MPI.COMM_WORLD
		self.global_rank = self.global_comm.Get_rank()
		self.make_env_location = make_env_location
		self.log_comms_dir = log_comms_dir
		self.obs_spaces = obs_spaces
		self.act_spaces = act_spaces

	def loop(self):
		
		#receive the match list
		match_list = self.global_comm.bcast(None, root=0)

		#receive the policy types
		policy_types = self.global_comm.bcast(None, root=0)

		#receive the number of steps to run each match
		steps_per_match = self.global_comm.bcast(None, root=0)

		#figure out mappings from ranks to policies, matches, entities
		policies_flat = [0] #root proc is taken
		match_num_flat = [0]
		entity_map = [-1]
		for midx, m in enumerate(match_list):
			policies_flat.append(-1) #one environment for each match
			policies_flat += m

			entity_map.append(-1)
			entity_map += list(range(len(m)))

			match_num_flat.append(midx+1) #env member
			for entry in m:
				match_num_flat.append(midx+1) #policy member

		# TODO: Need some sort of modulo here and should account for rank 0 being taken
		my_pol = policies_flat[self.global_rank]
		my_match = match_num_flat[self.global_rank]

		# create a local comm between members of a single match
		excluded = []
		for rank in range(len(match_num_flat)):
			if match_num_flat[rank] != my_match:
				excluded.append(rank)

		match_group_comm = self.global_comm.Create(self.global_comm.group.Excl(excluded))

		#everyone in the match group agrees on which rank is the environment
		my_packet = [match_group_comm.Get_rank(), my_pol==-1] #match rank and whether or not you are the environment

		#root 0 gathers and then bcasts to everyone in the match
		proc_info = match_group_comm.gather(my_packet, root=0)
		if match_group_comm.Get_rank()==0:
			for rank in range(len(proc_info)):
				if proc_info[rank][1]:
					root_proc = proc_info[rank][0]
					match_group_comm.bcast(root_proc, root=0)
					break
		else:
			root_proc = match_group_comm.bcast(None, root=0)

		# now all procs in the match should have the root process rank

		if my_pol == -1:
			mpi_print("I am an environment")
			self.process = EnvironmentProcess(self.make_env_location, self.global_comm, match_group_comm, root_proc)
			temp_pol_group_comm = self.global_comm.Create(self.global_comm.group.Excl([]))
			self.process.proxy_sync()
			self.process.run(steps_per_match)

			del self.process

		else:
			#determine which entity/ies this proc controls
			my_entity = entity_map[self.global_rank]

			mpi_print("I am a worker for policy", my_pol, "at entity", my_entity)

			#create a local comm among members of the same policy
			excluded = []
			for rank in range(len(policies_flat)):
				if policies_flat[rank] != my_pol:
					excluded.append(rank)

			mpi_print(excluded)
			policy_group_comm = self.global_comm.Create(self.global_comm.group.Excl(excluded))

			#calculate how many steps the policy needs to run for
			#for now this is (number pol workers) * (steps in match)
			steps_to_run = policy_group_comm.Get_size()*steps_per_match

			#make a proxy environment
			obs_space = self.obs_spaces[my_entity]
			act_space = self.act_spaces[my_entity]
			proxyenv = proxy_env(my_entity, obs_space, act_space, match_group_comm, root_proc)

			#make the policy
			pol_type = policy_types[my_pol]
			if pol_type == "random":
				policy = RandomPolicy(proxyenv, policy_group_comm)

			elif pol_type == "ppo":
				policy = PPOPolicy(proxyenv, policy_group_comm)

			elif pol_type == "ppo-lstm":
				policy = PPOPolicy(proxyenv, policy_group_comm, True)

			# TODO: other policy types here, including custom

			# compute full log comms directory for this policy
			data_dir =  get_dir_for_policy(my_pol, self.log_comms_dir)

			# create policy record for policy roots
			if policy_group_comm.Get_rank() == 0:
				pr = PolicyRecord(my_pol, self.log_comms_dir)
				pr.load()
				policy.run(steps_to_run, data_dir, pr)
			else:
				policy.run(steps_to_run, data_dir)

		# sync with main proc -> we are done working
		self.global_comm.gather(1, root=0)




