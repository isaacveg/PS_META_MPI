import os
import time
import socket
import pickle
import argparse
import asyncio
import concurrent.futures
import threading
import math
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from pulp import *
import random
from config import ClientConfig, cfg
from comm_utils import *
from training_utils import train, test
import datasets, models
from mpi4py import MPI
import logging

# parser = argparse.ArgumentParser(description='Distributed Client')
# parser.add_argument('--visible_cuda', type=str, default='-1')
# parser.add_argument('--use_cuda', action="store_false", default=True)

# args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

if cfg['client_cuda'] == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(int(rank)% 4 + 0)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['client_cuda']

device = torch.device("cuda" if cfg['client_use_cuda'] and torch.cuda.is_available() else "cpu")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# init logger
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
RESULT_PATH = os.getcwd() + '/clients_log/' + now + '/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

MASTER_RANK=0
best_acc=0.0
best_loss=0.0
best_epoch=1


def main():
    # logger.info("client_rank:{}".format(rank))
    client_config = ClientConfig(idx=0)
    
    train_dataset, test_dataset = datasets.load_datasets(cfg['dataset_type'], cfg['dataset_path'])
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=cfg['client_test_batch_size'], shuffle=False)

    comm_tag = 1

    while True:
        # receive the configuration from the server
        communicate_with_server(client_config, comm_tag, action='get_config')
        print("client {}, get params, epoch {}, comm_tag {}".format(client_config.idx, client_config.epoch_idx, comm_tag))

        logger = init_logger(comm_tag, client_config)
        logger.info("_____****_____\nEpoch: {:04d}".format(client_config.epoch_idx))

        # torch.random.seed()
        # load the test and train loader
        train_loader = datasets.create_dataloaders(
            train_dataset, batch_size=cfg['local_batch_size'], selected_idxs=client_config.train_data_idxes
        )

        # start local training
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = [
            asyncio.ensure_future(
                local_training(client_config, train_loader, test_loader, logger)
            )
        ]
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

        keys_to_send = ["params", "train_time", "send_time"]

        # 构造需要发送的字典
        config_to_send = {key: getattr(client_config, key) for key in keys_to_send}
        print("client {} with rank {}, send params, epoch {}, comm_tag {}".format(client_config.idx, rank, client_config.epoch_idx, comm_tag))
        communicate_with_server(config_to_send, comm_tag, 'send_config')

        # send the configuration to the server
        # if cfg['compress_method'] is not None:
        #     client_config.seed = compress.param_compress(model=None, action=cfg['compress_method'], extra_info=cfg['compress_ratio'])
        # communicate_with_server(client_config, comm_tag, action='send_config')
        comm_tag += 1

        if client_config.epoch_idx > cfg['epoch_num']:
            break



async def local_training(config, train_loader, test_loader, logger):
    local_model = models.create_model_instance(cfg['dataset_type'], cfg['model_type'], cfg['classes_size'])
    torch.nn.utils.vector_to_parameters(config.params, local_model.parameters())
    local_model.to(device)
    epoch_lr = config.lr
    local_steps = cfg['local_iters']

    if config.epoch_idx > 1:
        epoch_lr = max(cfg['decay_rate'] * epoch_lr, cfg['min_lr'])
        config.lr = epoch_lr
    logger.info("epoch-{} lr: {}".format(config.epoch_idx, epoch_lr))
    if cfg['momentum'] < 0:
        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.SGD(local_model.parameters(), momentum=cfg['momentum'], lr=epoch_lr, weight_decay=cfg['weight_decay'])
    train_loss, train_time = train(local_model, train_loader, optimizer, local_iters=local_steps, device=device, model_type=cfg['model_type'])
    logger.info(
        "Train_loss: {}\n".format(train_loss)+
        "Train_time: {}\n".format(train_time)
    )
    config.train_time = train_time

    test_loss, test_acc = test(local_model, test_loader, device, model_type=cfg['model_type'])
    logger.info(
        "Test_Loss: {}\n".format(test_loss) +
        "Test_ACC: {}\n".format(test_acc)
    )

    logger.info("send and save para")
    config.params = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()

    # logger.info("after send")
    config.epoch_idx += 1




def init_logger(comm_tag, client_config):
    logger = logging.getLogger(os.path.basename(__file__).split('.')[0] + str(comm_tag))
    logger.setLevel(logging.INFO)
    filename = RESULT_PATH + now + "_" + os.path.basename(__file__).split('.')[0] + '_' + str(
        client_config.idx) + '.log'
    file_handler = logging.FileHandler(filename=filename)
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


async def get_config(config, comm_tag):
    config_received = await get_data(comm, MASTER_RANK, comm_tag)
    for k, v in config_received.__dict__.items():
        setattr(config, k, v)


async def send_config(config, comm_tag):
    await send_data(comm, config, MASTER_RANK, comm_tag)


def communicate_with_server(config, comm_tag, action):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    if action == "send_config":
        task = asyncio.ensure_future(
            send_config(config, comm_tag)
        )
    elif action == "get_config":
        task = asyncio.ensure_future(
            get_config(config, comm_tag)
        )
    else:
        raise ValueError('Not valid action')
    tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


if __name__ == '__main__':
    main()
