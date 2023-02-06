from mpi4py import MPI
import time
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print("I am process " + str(rank) + " of " + str(size) + " processors")