from mpi4py import MPI
import time
import numpy as np
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
pid = 0
testRank = 1
tag_pram = 0
tag_bsp = 1

parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true',
                    default=False, help="Enable the detialed training output.")
parser.add_argument('--n', type=int, default=10000,
                    help='data to send.')
args = parser.parse_args()

#Initialisation PRAM
#On initialisation une liste de p données sur le noeud maître

data= np.zeros(args.n, dtype='d')
if rank == pid:
    for i in range(args.n):
        data[i] = size



#Begin PRAM
if rank == pid:
    print("PRAM")
if (args.debug and rank == pid):
    print("Broadcast :")
    print(data)
comm.Barrier()
if (args.debug and rank == testRank):
    print("Before :")
    print(data)
comm.Barrier()
start_time = time.time()
if rank == pid:
    for i in range(size):
        if(i != pid):
            req = comm.send(data, dest=i, tag=tag_pram)
else:
    data = comm.recv(source=pid, tag=tag_pram)
comm.Barrier()
if rank == pid:
    print("--- %s seconds ---" % (time.time() - start_time))
 
comm.Barrier()   
if (args.debug and rank == testRank):
    print("After :")
    print(data)
comm.Barrier()  


#Initialisation BSP
#On initialisation une liste de p données sur le noeud maître
data= np.zeros(args.n, dtype='d')
if rank == pid:
    for i in range(args.n):
        data[i] = size

#Begin BSP
if rank == pid:
    print("\nBSP")
if (args.debug and rank == pid):
    print("Broadcast :")
    print(data)
comm.Barrier()
if (args.debug and rank == testRank):
    print("Before :")
    print(data)
comm.Barrier()
#Première boucle
comm.Barrier()
start_time = time.time()
if rank == pid:
    for i in range(size):
        if(i != pid):
            req = comm.send(data[int(i * args.n /size): int((i+1)* args.n / size)], dest=i, tag=tag_bsp)
        else:
            tmp = data[int(i * args.n /size): int((i+1)* args.n / size)]
else:
    tmp = comm.recv(source=pid, tag=tag_bsp)
comm.Barrier()
time_ss1 = time.time() - start_time
#Deuxième boucle
start_time_ss2 = time.time()
comm.Allgather([tmp, MPI.INT], [data, MPI.INT])
comm.Barrier()
time_ss2 = time.time() - start_time_ss2
if rank == pid:
    print("--- Total %s seconds ---" % (time.time() - start_time))
    print("1st superstep: %s seconds" % time_ss1)
    print("2nd superstep: %s seconds" % time_ss2)
comm.Barrier()   
if (args.debug and rank == testRank):
    print("After :")
    print(data)
