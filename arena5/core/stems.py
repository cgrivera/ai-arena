
from mpi4py import MPI
from arena5.core.utils import mpi_print, count_needed_procs
from arena5.core.proxy_env import make_proxy_env
from arena5.core.env_process import EnvironmentProcess

from arena5.algos.random.random_policy import RandomPolicy
from arena5.algos.multiagent_random.multiagent_random_policy import MARandomPolicy
from arena5.algos.ppo.ppo import PPOPolicy, PPOLSTMPolicy, PPOPolicyEval, PPOLSTMPolicyEval
#from arena5.algos.hppo.hppo import HPPOPolicy

from arena5.core.policy_record import PolicyRecord, get_dir_for_policy

def make_stem(make_env_method, log_comms_dir, obs_spaces, act_spaces, additional_policies={}):
	rank = MPI.COMM_WORLD.Get_rank()
	if rank == 0:
		# return an object to control arena
		return UserStem(make_env_method, log_comms_dir, obs_spaces, act_spaces, additional_policies)
	else:
		stem = WorkerStem(make_env_method, log_comms_dir, obs_spaces, act_spaces, additional_policies)
		while True:
			stem.loop() #loop indefinitely


class UserStem(object):

	def __init__(self, make_env_method, log_comms_dir, obs_spaces, act_spaces, additional_policies):

		self.global_comm = MPI.COMM_WORLD
		self.global_rank = self.global_comm.Get_rank()
		mpi_print("I am the user stem, at rank", self.global_rank)

	def kickoff(self, match_list, policy_types, steps_per_match, entity_remaps=[], render=False, scale=False, env_kwargs={}):

		#make sure there are enough procs to run the matches
		min_procs = count_needed_procs(match_list)
		avail_procs = MPI.COMM_WORLD.Get_size()
		if avail_procs < min_procs:
			raise RuntimeError("At least "+str(min_procs)+" processes are needed to run matches, but only "+str(avail_procs)+" exist.")

		# if scale is turned on, duplicate the match list until we run out of processes
		if scale:
			num_duplicates = (avail_procs-1) // (min_procs-1)
			new_match_list = []
			for m in match_list:
				for d in range(num_duplicates):
					new_match_list.append(m)
			match_list = new_match_list
			num_used_procs = count_needed_procs(match_list)
			num_unused_procs = avail_procs - num_used_procs

			if len(entity_remaps) > 0:
				new_emap_list = []
				for m in entity_remaps:
					for d in range(num_duplicates):
						new_emap_list.append(m)
				entity_remaps = new_emap_list

			mpi_print("\n================== SCALING REPORT ======================")
			mpi_print("AI Arena was able to duplicate the matches "+str(num_duplicates)+" times each.")
			mpi_print("There will be: "+str(num_unused_procs)+" unused processes.")
			mpi_print("You need to allocate: "+str(min_procs-num_unused_procs-1)+" more processes to duplicate again.")
			mpi_print("=========================================================\n")

		#broadcast the match list
		self.global_comm.bcast(match_list, root=0)

		#broadcast the policy types
		self.global_comm.bcast(policy_types, root=0)

		#broadcast the number of steps to run each match
		self.global_comm.bcast(steps_per_match, root=0)

		#broadcast the entity index maps
		self.global_comm.bcast(entity_remaps, root=0)

		#broadcast if we will be calling render() on environments
		self.global_comm.bcast(render, root=0)

		#broadcast any environment kwargs
		self.global_comm.bcast(env_kwargs, root=0)

		#create some unused groups - need to call this from root for it to work in other procs
		temp_match_group_comm = self.global_comm.Create(self.global_comm.group.Excl([]))
		temp_pol_group_comm = self.global_comm.Create(self.global_comm.group.Excl([]))

		# now wait for response from all the workers, gather dummy data as a syncing tool
		_ = self.global_comm.gather(None, root=0)

		mpi_print("ROUND COMPLETE!")

class WorkerStem(object):

	def __init__(self, make_env_method, log_comms_dir, obs_spaces, act_spaces, additional_policies):

		self.global_comm = MPI.COMM_WORLD
		self.global_rank = self.global_comm.Get_rank()
		self.make_env_method = make_env_method
		self.log_comms_dir = log_comms_dir
		self.obs_spaces = obs_spaces
		self.act_spaces = act_spaces
		self.additional_policies = additional_policies

	def loop(self):
		
		#receive the match list
		match_list = self.global_comm.bcast(None, root=0)

		#receive the policy types
		policy_types = self.global_comm.bcast(None, root=0)

		#receive the number of steps to run each match
		steps_per_match = self.global_comm.bcast(None, root=0)

		#receive the optional entity index maps for non-sequential entities 
		entity_remaps = self.global_comm.bcast(None, root=0)

		#receive if we will be calling render() on environments
		will_call_render = self.global_comm.bcast(None, root=0)
		mpi_print("will render:", will_call_render)

		#receive any environment kwargs
		env_kwargs = self.global_comm.bcast(None, root=0)

		#figure out mappings from ranks to policies, matches, entities
		policies_flat = [0] 	#index=rank, entry=policy number, root proc=0, envs=-1
		match_num_flat = [0] 	#index=rank, entry=match number, root proc=0
		entity_map = [[-1]]		#index=rank, entry=[entity numbers], root proc=[-1], envs=[-1]

		#populate the above mappings
		for midx, m in enumerate(match_list):

			# Create the rank --> policy mapping =======================================
			policies_flat.append(-1) #one environment for each match
			for entry in m:
				if isinstance(entry, int):
					#we have a single entity being controlled by a process
					policies_flat.append(entry)
				elif isinstance(entry, list):
					#we have a group of entities being controlled by a process
					policies_flat.append(entry[0])
				else:
					#we have an unknown specification
					raise ValueError("Match entry may contain only ints and lists of ints")

			# Create the rank --> entities mapping ======================================
			entity_map.append([-1])
			entity_nominal_num = 0
			for entry in m:
				if isinstance(entry, int):
					#we have a single entity being controlled by a process
					if len(entity_remaps) > 0:
						emap = entity_remaps[midx]
						entity_map.append([emap[entity_nominal_num]])
					else:
						entity_map.append([entity_nominal_num])
					entity_nominal_num += 1

				elif isinstance(entry, list):
					#we have a group of entities being controlled by a process
					ents = []
					for e in entry:
						if len(entity_remaps) > 0:
							emap = entity_remaps[midx]
							ents.append(emap[entity_nominal_num])
						else:
							ents.append(entity_nominal_num)
						entity_nominal_num += 1
					entity_map.append(ents)

				else:
					#we have an unknown specification
					raise ValueError("Match entry may contain only ints and lists of ints")

			match_num_flat.append(midx+1) #env member
			for entry in m:
				match_num_flat.append(midx+1) #policy member

		#look for any unused ranks
		unused_ranks = []
		for r in range(self.global_comm.Get_size()):
			if r >= len(policies_flat):
				unused_ranks.append(r)

		# lookup the policy and match for this process specifically
		if self.global_rank not in unused_ranks:
			my_pol = policies_flat[self.global_rank]
			my_match = match_num_flat[self.global_rank]
		else:
			#unused rank
			my_pol = -2
			my_match = -2

		# create a local comm between members of a single match
		excluded = []
		for rank in range(len(match_num_flat)):
			if match_num_flat[rank] != my_match:
				excluded.append(rank)
		for rank in unused_ranks:
			excluded.append(rank)

		if self.global_rank not in unused_ranks:
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
		else:
			unused_group_comm = self.global_comm.Create(self.global_comm.group.Excl(unused_ranks))
			unused_group_comm = self.global_comm.Create(self.global_comm.group.Excl(unused_ranks))

		# now all procs in the match should have the root process rank

		if my_pol == -1:
			mpi_print("I am an environment")
			self.process = EnvironmentProcess(self.make_env_method, self.global_comm, match_group_comm, root_proc, will_call_render, env_kwargs=env_kwargs)
			temp_pol_group_comm = self.global_comm.Create(self.global_comm.group.Excl([]))
			self.process.proxy_sync()
			self.process.run(steps_per_match)

			del self.process

		elif my_pol > -1:

			#determine which entity/ies this proc controls
			my_entities = entity_map[self.global_rank]

			mpi_print("I am a worker for policy", my_pol, "for entities", my_entities)

			#create a local comm among members of the same policy
			excluded = []
			for rank in range(len(policies_flat)):
				if policies_flat[rank] != my_pol:
					excluded.append(rank)
			for rank in unused_ranks:
				excluded.append(rank)

			mpi_print("excluded ranks:", excluded)
			policy_group_comm = self.global_comm.Create(self.global_comm.group.Excl(excluded))

			#calculate how many steps the policy needs to run for
			#for now this is (number pol workers) * (steps in match)
			steps_to_run = policy_group_comm.Get_size()*steps_per_match

			#make a proxy environment
			obs_spaces = [self.obs_spaces[e] for e in my_entities]
			act_spaces = [self.act_spaces[e] for e in my_entities]
			proxyenv = make_proxy_env(my_entities, obs_spaces, act_spaces, match_group_comm, root_proc)

			#make the collection of policy options
			available_policies = {
				"random":RandomPolicy,
				"ppo":PPOPolicy,
				"ppo-eval":PPOPolicyEval,
				"ppo-lstm":PPOLSTMPolicy,
				"ppo-lstm-eval":PPOLSTMPolicyEval,
				"multiagent_random":MARandomPolicy
			}

			#add custom policies here
			available_policies.update(self.additional_policies)

			#make the policy
			pol_type = policy_types[my_pol]
			policy_maker = available_policies[pol_type]
			policy = policy_maker(proxyenv, policy_group_comm)

			# compute full log comms directory for this policy
			data_dir =  get_dir_for_policy(my_pol, self.log_comms_dir)

			# create policy record for policy roots
			if policy_group_comm.Get_rank() == 0:
				pr = PolicyRecord(my_pol, self.log_comms_dir)
				pr.load()
				policy.run(steps_to_run, data_dir, pr)
			else:
				policy.run(steps_to_run, data_dir)


		else:
			mpi_print("Process with rank "+str(self.global_rank)+" is unused.")
			#this is an unused process :(

		# sync with main proc -> we are done working
		self.global_comm.gather(1, root=0)




