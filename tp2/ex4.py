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
tag_first_sample = 0
tag_second_step = 1

parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true',
                    default=False, 
                    help="Enable the detailed training output.")
parser.add_argument('--n', type=int, default=10000,
                    help='Number of data to sort.')
args = parser.parse_args()


#Initialisation Approche Naïve 
#On initialise les paramètres
size_data = math.floor(args.n / size)
if (pid < (args.n % size)):
    size_data += 1
random_list = random.sample(range(0, 20 * args.n), size_data)

#Begin Tiskin's algorithm
if pid == masterPid:
    print("Tiskin's sampling sort")
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
#Local sort
random_list.sort()
#Select for sampling
first_sample = []
for i in range(size + 1):
    first_sample.append(random_list[math.floor(i * (len(random_list)-1) / size)])
#Start communication
for i in range(size):
    req = comm.isend(first_sample, dest=i, tag=tag_first_sample)
    req.wait()
sample_list = []           
for i in range(size):
    req = comm.irecv(source=i, tag=tag_first_sample)
    sample_list += (req.wait())
comm.Barrier()
#Sort first samples
sample_list.sort()
second_sample = []
#Select second samples
for i in range(size + 1):
    second_sample.append(sample_list[math.floor(i * (len(sample_list)-1) / size)])
#Create messages
messages = []
for i in range(size):
    messages.append([])
current_node = 0;
for elem in random_list:
    if(elem > (second_sample[current_node + 1])):
        current_node += 1
        if(current_node >= size):
            print("Problem " + str(current_node))
    messages[current_node].append(elem)
#Communication part
for i in range(size):
    req = comm.isend(messages[i], dest=i, tag=tag_second_step)
    req.wait()
result = []           
for i in range(size):
    req = comm.irecv(source=i, tag=tag_second_step)
    result += req.wait()
result.sort()
#Barrier
comm.Barrier()
if pid == masterPid:
    print("--- %s seconds ---" % (time.time() - start_time))
 
comm.Barrier()   
if (args.debug and pid == masterPid):
    print("Result :")
    print(result)
comm.Barrier()