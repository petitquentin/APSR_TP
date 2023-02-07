from mpi4py import MPI
import math
import time
import random
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
masterRank = 0
testRank = 1
tag_reduce = 0

parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true',
                    default=False, help="Enable the detialed training output.")
parser.add_argument('--n', type=int, default=1000,
                    help='Number of data for the mean.')
args = parser.parse_args()


#Initialisation Gather scatter approach 
#On initialise les paramètres
data = None
if (rank == masterRank):
    data = []
    for i in range(size):
        size_data = math.floor(args.n / size)
        if (rank < (args.n % size)):
            size_data += 1
        data.append(random.sample(range(0, 20 * args.n), args.n))

#Begin Gather/sCatter
if rank == masterRank:
    print("Gather/Scatter")
comm.Barrier()
if (args.debug and rank == masterRank):
    print("Size : " + str(len(data)))
comm.Barrier()
start_time = time.time()
#Scatter
data_dist = comm.scatter(data, root = masterRank)
#Compute
sum = 0
for e in data_dist:
    sum += e
#Gather
final_sum = comm.gather(sum, root = masterRank)

if(rank == masterRank):
    means = 0
    for i in final_sum:
        means += i
    means = means/args.n
comm.Barrier()
if rank == masterRank:
    print("--- %s seconds ---" % (time.time() - start_time))
comm.Barrier()   
if (args.debug and rank == masterRank):
    print("Result :")
    print(means)
comm.Barrier()


#Begin Reduce approach
if rank == masterRank:
    print("Reduce")
comm.Barrier()
start_time = time.time()
#Compute
sum = 0
for e in data_dist:
    sum += e
#Gather
means = comm.reduce(sum, op=MPI.SUM, root = masterRank)
if(rank == masterRank):
    means = means/args.n
comm.Barrier()
if rank == masterRank:
    print("--- %s seconds ---" % (time.time() - start_time))
comm.Barrier()   
if (args.debug and rank == masterRank):
    print("Result :")
    print(means)
comm.Barrier()