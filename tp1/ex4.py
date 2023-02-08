from mpi4py import MPI
import math
import time
import random
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
pid = 0
tag_direct = 0
tag_log = 1

parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true',
                    default=False, help="Enable the detialed training output.")
parser.add_argument('--n', type=int, default=10000,
                    help='Vectors size.')
args = parser.parse_args()


#Initialisation 
#On initialise les paramètres
size_data = math.floor(args.n / size)
if (rank < (args.n % size)):
    size_data += 1
v_vector = random.sample(range(0, 20 * args.n), size_data)
w_vector = random.sample(range(0, 20 * args.n), size_data)

#Begin dot product
start_time = time.time()
if rank == pid:
    print("Dot product")
comm.Barrier()
if (args.debug and rank == pid):
    print("Size :")
comm.Barrier()
for i in range(size):
    if(i == rank):
        print("Node ", i, ": ", size_data)
    comm.Barrier()
comm.Barrier()
if (args.debug and rank == pid):
    print("Direct")
comm.Barrier()
#Direct approach
start_time_direct = time.time()
sum_direct = 0
for i in range(len(v_vector)):
    sum_direct += v_vector[i] * w_vector[i]
#Communication
if(rank != pid):
    req = comm.isend(sum_direct, dest=pid, tag=tag_direct)
    req.wait()
else:
    for i in range(1, size):
        req = comm.irecv(source=i, tag=tag_direct)
        sum_direct += (req.wait())
comm.Barrier()
if rank == pid:
    print("result = " + str(sum_direct))
    print("Time direct %s seconds" % (time.time() - start_time_direct))


comm.Barrier()
if (args.debug and rank == pid):
    print("Log")
comm.Barrier()
#Log approach
start_time_log = time.time()
sum_log = 0
for i in range(len(v_vector)):
    sum_log += v_vector[i] * w_vector[i]
#Communication
i = 2
while(i <= size):
    if((rank%i) == i/2):
        req = comm.isend(sum_log, dest=rank - (i/2), tag=tag_direct)
        req.wait()
    if((rank%i) == 0):
        req = comm.irecv(source=rank + (i/2), tag=tag_direct)
        sum_log += (req.wait())
    i = i * 2
    comm.Barrier()
if rank == pid:
    print("result = " + str(sum_log))
    print("Time log %s seconds" % (time.time() - start_time))
#Barrier
comm.Barrier()
if rank == pid:
    print("--- %s seconds ---" % (time.time() - start_time))