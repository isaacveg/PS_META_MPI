import os
import sys
import struct
import socket
import pickle
from time import sleep
import time

async def send_data(comm, data, client_rank, tag_epoch):
    data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)  
    comm.send(data, dest=client_rank, tag=tag_epoch)
    # print("after send")

async def get_data(comm, client_rank, tag_epoch):
    data = comm.recv(source=client_rank, tag=tag_epoch)
    data = pickle.loads(data)
    return data
