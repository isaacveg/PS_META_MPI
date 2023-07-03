from fileinput import filename
import os
import sys
import argparse
import socket
import pickle
import asyncio
import concurrent.futures
import json
from random import sample
import time
from tkinter.tix import Tree
import numpy as np
import threading
import torch
import copy
import math
from config import *
import torch.nn.functional as F
import datasets, models
from training_utils import test

from mpi4py import MPI

import logging


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['server_cuda']
device = torch.device("cuda" if cfg['server_use_cuda'] and torch.cuda.is_available() else "cpu")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
RESULT_PATH = os.getcwd() + '/server_log/'
MODEL_PATH = os.getcwd() + '/model_save/' + now + '/'
GLOBAL_MODEL_PATH = MODEL_PATH + now + "_global.model"
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH, exist_ok=True)

"""init logger"""
logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)

filename = RESULT_PATH + now + "_" +os.path.basename(__file__).split('.')[0] +'.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

comm_tags = np.ones(cfg['selected_num'] + 1)

def main():
    client_num = cfg['client_num']
    logger.info("Total number of clients: {}".format(client_num))
    logger.info("\nModel type: {}".format(cfg["model_type"]))
    logger.info("Dataset: {}".format(cfg["dataset_type"]))

    global_model = models.create_model_instance(cfg["dataset_type"], cfg["model_type"])
    init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    para_nums = init_para.nelement()
    model_size = init_para.nelement() * 4 / 1024 / 1024
    logger.info("para num: {}".format(para_nums))
    logger.info("Model Size: {} MB".format(model_size))

    # Create model instance
    train_data_partition, partition_sizes = partition_data(
        dataset_type=cfg['dataset_type'],
        partition_pattern=cfg['data_partition_pattern'],
        non_iid_ratio=cfg['non_iid_ratio'],
        client_num=client_num
    )
    logger.info('\nData partition: ')
    for i in range(len(partition_sizes)):
        s = ""
        for j in range(len(partition_sizes[i])):
            s += "{:.2f}".format(partition_sizes[i][j]) + " "
        logger.info(s)

    # print(init_para.device)
    init_para = init_para.to(device)
    # print(init_para.device)
    # create workers
    all_clients: List[ClientConfig] = list()
    for client_idx in range(client_num):
        client = ClientConfig(client_idx)
        client.lr = cfg['lr']
        client.params = init_para
        # print(client_idx, client.params.device)
        client.train_data_idxes = train_data_partition.use(client_idx)
        client.local_model_path = MODEL_PATH + now + "_local_" + str(client_idx) + ".model"
        client.global_model_path = GLOBAL_MODEL_PATH
        all_clients.append(client)

    # connect and send init config
    # communication_parallel(all_clients, 1, comm, action="init")

    # recoder: SummaryWriter = SummaryWriter()
    global_model.to(device)
    # print(global_model.device, "global_model_device")
    _, test_dataset = datasets.load_datasets(cfg['dataset_type'], cfg['dataset_path'])
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=cfg['test_batch_size'], shuffle=False)

    best_epoch = 1
    best_acc = 0.0
    best_loss = 0.0

    for epoch_idx in range(1, 1 + cfg['epoch_num']):
        logger.info("_____****_____\nEpoch: {:04d}".format(epoch_idx))
        print("_____****_____\nEpoch: {:04d}".format(epoch_idx))

        # The client selection algorithm can be implemented
        selected_num = cfg['selected_num']
        selected_client_idxes = sample(range(client_num), selected_num)
        logger.info("Selected client idxes: {}".format(selected_client_idxes))
        print("Selected client idxes: {}".format(selected_client_idxes))
        selected_clients = []
        for client_idx in selected_client_idxes:
            all_clients[client_idx].epoch_idx = epoch_idx
            all_clients[client_idx].params = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
            selected_clients.append(all_clients[client_idx])

        # send the configurations to the selected clients
        communication_parallel(selected_clients, action="send_config")
        print("send success")
        # when all selected clients have completed local training, receive their configurations
        communication_parallel(selected_clients, action="get_config")
        print("get success")
        # aggregate the clients' local model parameters
        aggregate_model_para(global_model, selected_clients)   
        # test and save the best global model
        test_loss, test_acc = test(global_model, test_loader, device, cfg['model_type'])

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch_idx
            model_save_path = MODEL_PATH + now + "_" + "best_global.model"
            torch.save(global_model.state_dict(), model_save_path)

        logger.info(
            "Test_Loss: {:.4f}\n".format(test_loss) +
            "Test_Acc: {:.4f}\n".format(test_acc) +
            "Best_Acc: {:.4f}\n".format(best_acc) +
            "Best_Epoch: {:04d}\n".format(best_epoch)
        )

        for m in range(len(selected_clients)):
            comm_tags[m + 1] += 1



def aggregate_model_para(global_model, worker_list):
    global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
    with torch.no_grad():
        # normal (or so called Reptile)
        if cfg['meta_method'] is None: 
            para_delta = torch.zeros_like(global_para)
            for worker in worker_list:
                # print(global_para.device, worker.params.device)
                model_delta = (worker.params - global_para)
                #gradient
                # model_delta = worker.config.neighbor_paras
                para_delta += worker.aggregate_weight * model_delta
            global_para += para_delta
        
        # First order MAML, use the last update gradients to update
        elif cfg['meta_method'] == 'fomaml':
            para_grads = torch.zeros_like(global_para)
            for worker in worker_list:
                # add up all grads
                para_grads += worker.params * worker.aggregate_weight
            global_para -= para_grads * cfg['meta_outer_lr']

        torch.nn.utils.vector_to_parameters(global_para, global_model.parameters())
    return global_para


async def send_config(client, client_rank, comm_tag):
    await send_data(comm, client, client_rank, comm_tag)


async def get_config(client, client_rank, comm_tag):
    config_received = await get_data(comm, client_rank, comm_tag)
    # for k, v in config_received.__dict__.items():
    for k, v in config_received.items():
        setattr(client, k, v)


def communication_parallel(client_list, action):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for m, client in enumerate(client_list): 
        if action == "send_config":
            print("sending to worker {}, tags {}".format(m+1, comm_tags[m+1]))
            task = asyncio.ensure_future(send_config(client, m + 1, comm_tags[m+1]))
        elif action == "get_config":
            print("get worker {}, tags {}".format(m+1, comm_tags[m+1]))
            task = asyncio.ensure_future(get_config(client, m + 1, comm_tags[m+1]))
        else:
            raise ValueError('Not valid action')
        tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


def non_iid_partition(ratio, train_class_num, worker_num):
    partition_sizes = np.ones((train_class_num, worker_num)) * ((1 - ratio) / (worker_num-1))

    for i in range(train_class_num):
        partition_sizes[i][i%worker_num]=ratio

    return partition_sizes


def partition_data(dataset_type, partition_pattern, non_iid_ratio, client_num=10):
    """
    partition_size should be in shape:
    classes_size * client_num, each number is ratio for each class on one client.
    """
    train_dataset, _ = datasets.load_datasets(dataset_type=dataset_type, data_path=cfg['dataset_path'])
    partition_sizes = np.ones((cfg['classes_size'], client_num))
    # iid
    # every client has same number of samples and corresponding classes
    if partition_pattern == 0:
        partition_sizes *= (1.0 / client_num)
    # non-iid
    # each client contains all classes of data, but the proportion of certain classes of data is very large
    elif partition_pattern == 1:
        if 0 < non_iid_ratio < 10:
            partition_sizes *= ((1 - non_iid_ratio * 0.1) / (client_num - 1))
            for i in range(cfg['classes_size']):
                partition_sizes[i][i % client_num] = non_iid_ratio * 0.1
        else:
            raise ValueError('Non-IID ratio is out of range')
    # non-iid
    # each client misses some classes of data, while the other classes of data are distributed uniformly
    elif partition_pattern == 2:
        if 0 < non_iid_ratio < 10:
            # calculate how many classes of data each worker is missing
            missing_class_num = int(round(cfg['classes_size'] * (non_iid_ratio * 0.1)))

            begin_idx = 0
            for worker_idx in range(client_num):
                for i in range(missing_class_num):
                    partition_sizes[(begin_idx + i) % cfg['classes_size']][worker_idx] = 0.
                begin_idx = (begin_idx + missing_class_num) % cfg['classes_size']

            for i in range(cfg['classes_size']):
                count = np.count_nonzero(partition_sizes[i])
                for j in range(client_num):
                    if partition_sizes[i][j] == 1.:
                        partition_sizes[i][j] = 1. / count
        else:
            raise ValueError('Non-IID ratio is too large')
    # non-iid
    # same as pattern 1 but every client has more than one major class
    elif partition_pattern == 3:
        if 0 < non_iid_ratio < 10:
            most_data_proportion = cfg['classes_size'] / client_num * non_iid_ratio * 0.1
            minor_data_proportion = cfg['classes_size'] / client_num * (1 - non_iid_ratio * 0.1) / (
                        cfg['classes_size'] - 1)
            partition_sizes *= minor_data_proportion
            for i in range(client_num):
                partition_sizes[i % cfg['classes_size']][i] = most_data_proportion
        else:
            raise ValueError('Non-IID ratio is out of range')
    else:
        raise ValueError('Not valid partition pattern')

    train_data_partition = datasets.LabelwisePartitioner(
        train_dataset, partition_sizes=partition_sizes, seed=cfg['data_partition_seed']
    )

    return train_data_partition, partition_sizes

if __name__ == "__main__":
    main()
