
import numpy as np
from mpi4py import MPI

### mpi utils for pytorch, mainly taken from spinning up
### NOTE: when changing pytorch.numpy(), this also changes the tensor


def sync_grads(comm, parameters):
    for p in parameters:
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


