from mpi4py import MPI
import time

start_time = time.time()
pid = 0
comm = MPI.COMM_WORLD
name = MPI.Get_processor_name()
rank = comm.Get_rank()
size = comm.Get_size()

if(size == 4):
    print("Processeur numéro  " + str(rank) + " sur la machine " + name + " parmi "+ str(size) + " processeurs")
else:
    if(rank == pid):
        print("Il faut avoir 4 processeurs")

comm.Barrier()
if rank == pid:
    print("--- %s seconds ---" % (time.time() - start_time))
