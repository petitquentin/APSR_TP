from mpi4py import MPI
import math
import time
import random
import argparse

comm = MPI.COMM_WORLD
pid = comm.Get_rank()
size = comm.Get_size()
masterPid = 0
testpid = 1
tag_reduce = 0

parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true',
                    default=False, 
                    help="Enable the detailed training output.")
parser.add_argument('--n', type=int, default=1000,
                    help='Number of data to reduce.')
args = parser.parse_args()


#Initialisation Approche Naïve 
#On initialise les paramètres
size_data = math.floor(args.n / size)
if (pid < (args.n % size)):
    size_data += 1
randomlist = random.sample(range(0, 20 * args.n), size_data)

#Begin Reduction
if pid == masterPid:
    print("Reduce")
comm.Barrier()
if (args.debug and pid == masterPid):
    print("Size :")
comm.Barrier()
for i in range(size):
    if(i == pid):
        print("Node ", i, ": ", size_data)
    comm.Barrier()
comm.Barrier()
start_time = time.time()
sum = 0
for elem in randomlist:
    sum += elem
#Start communication
req = comm.isend(sum, dest=masterPid, tag=tag_reduce)
req.wait()
        
if pid == masterPid:
    result = 0
    for i in range(size):
        req = comm.irecv(source=i, tag=tag_reduce)
        result += req.wait()

#Barrier
comm.Barrier()
if pid == masterPid:
    print("--- %s seconds ---" % (time.time() - start_time))
 
comm.Barrier()   
if (args.debug and pid == masterPid):
    print("Result :")
    print(result)
comm.Barrier()

#On minimise les communications et le nombre de superstep