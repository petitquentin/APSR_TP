from mpi4py import MPI
import math
import time
import argparse

comm = MPI.COMM_WORLD
pid = comm.Get_rank()
size = comm.Get_size()
masterPid = 0
testpid = 1
tag_naive = 0
tag_SdP = 1

parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true',
                    default=False, 
                    help="Enable the detailed training output.")
parser.add_argument('--n', type=int, default=1000,
                    help='Number of data.')
args = parser.parse_args()

#Initialisation Approche Naïve 
#On initialisation une liste de p données sur le noeud maître
data = []
for i in range(args.n):
    data.append(pid * args.n + 1 + i)



#Begin Naive approach
if pid == masterPid:
    print("Naive approach")
comm.Barrier()
if (args.debug and pid == masterPid):
    print("Before :")
comm.Barrier()
if (args.debug):
    for i in range(size):
        if(i == pid):
            print("Node ", i, ": ", data)
        comm.Barrier()
comm.Barrier()
start_time = time.time()
for i in range(1, len(data)):
    data[i] += data[i-1]
for i in range(pid, size):
    req = comm.isend(data[len(data)-1], dest=i, tag=tag_naive)
    req.wait()
sum = 0
for i in range(0, pid):
    req = comm.irecv(source=i, tag=tag_naive)
    sum += req.wait()
for j in range(len(data)):
    data[j] += sum


comm.Barrier()
if pid == masterPid:
    print("--- %s seconds ---" % (time.time() - start_time))
 
comm.Barrier()   
if (args.debug and pid == masterPid):
    print("After :")
comm.Barrier()
if (args.debug):
    for i in range(size):
        if(i == pid):
            print("Node ", i, ": ", data)
        comm.Barrier()
comm.Barrier()  


#Initialisation Saut de pointeur
#On initialisation une liste de p données sur le noeud maître
data = []
for i in range(args.n):
    data.append(pid * args.n + 1 + i)

#Begin SdP
if pid == masterPid:
    print("SdP approach")
comm.Barrier()
if (args.debug and pid == masterPid):
    print("Before :")
comm.Barrier()
if (args.debug):
    for i in range(size):
        if(i == pid):
            print("Node ", i, ": ", data)
        comm.Barrier()
comm.Barrier()
start_time = time.time()
for i in range(1, len(data)):
    data[i] += data[i-1]
tmp = 0
for i in range(math.ceil(math.log(size))):
    target = pid + math.pow(2,i)
    if (target < size):
        req = comm.isend(data[len(data)-1] + tmp, dest=target, tag=tag_SdP)
        req.wait()
    target = pid - math.pow(2,i)
    if (target >= 0):
        req = comm.irecv(source=target, tag=tag_SdP)
        tmp += req.wait()
    comm.Barrier()
for j in range(len(data)):
    data[j] += tmp

comm.Barrier()
if pid == masterPid:
    print("--- %s seconds ---" % (time.time() - start_time))
 
comm.Barrier()   
if (args.debug and pid == masterPid):
    print("After :")
comm.Barrier()

if (args.debug):
    for i in range(size):
        if(i == pid):
            print("Node ", i, ": ", data)
        comm.Barrier()