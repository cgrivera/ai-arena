
import sys

def mpi_print(*args):
    print(*args)
    sys.stdout.flush()

def count_needed_procs(match_list):
	#we need 1 process for each first-level entry in this list (environment)
	#for each second-level entry (match participant) we need another process
	#finally, we need 1 root process to orchestrate everything
	
	num_procs = 1 #root
	for m in match_list:
		num_procs += 1 #env
		for participant in m:
			num_procs += 1 #worker process

	return num_procs
