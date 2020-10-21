
import numpy as np
from mpi4py import MPI

### mpi utils for pytorch, mainly taken from spinning up
### NOTE: when changing pytorch.numpy(), this also changes the tensor


def sync_grads(comm, parameters):
    for p in parameters:
        if p.grad is None:
            print("WARNING: network parameter gradient does not exist.")
            continue

        p_grad_numpy = p.grad.numpy()
        avg_p_grad = mpi_avg(comm, p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def sync_weights(comm, parameters):
    for p in parameters:
        p_numpy = p.data.numpy()
        comm.Bcast(p_numpy, root=0)


def mpi_avg(comm, x):
    num_procs = comm.Get_size()
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    comm.Allreduce(x, buff, op=MPI.SUM)
    return buff / num_procs



# Additional tools employed by spinning up, with comm made explicit here

def mpi_op(comm, x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    comm.Allreduce(x, buff, op=op)
    return buff[0] if scalar else buff

def mpi_sum(comm, x):
    return mpi_op(comm, x, MPI.SUM)


def mpi_statistics_scalar(comm, x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum(comm, [np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(comm, np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(comm, np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(comm, np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std
