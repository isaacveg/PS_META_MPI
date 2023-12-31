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
from training_utils import base, fomaml, reptile, mamlhf
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

VERBOSE = True

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
        if VERBOSE:
            print("client {}, get params, epoch {}, comm_tag {}".format(client_config.idx, client_config.epoch_idx, comm_tag))

        logger = init_logger(comm_tag, client_config)
        logger.info("_____****_____\nEpoch: {:04d}".format(client_config.epoch_idx))

        # torch.random.seed()
        # load the test and train loader
        meta_test_loader = datasets.create_dataloaders(
                test_dataset, batch_size=cfg['client_test_batch_size'], selected_idxs=client_config.test_data_idxes, shuffle=False)
        # print(f"{client_config.idx}: meta_test_loader_len", len(client_config.test_data_idxes))
        train_loader = datasets.create_dataloaders(
            train_dataset, batch_size=cfg['local_batch_size'], selected_idxs=client_config.train_data_idxes
        )

        # start local training
        local_training(client_config, train_loader, test_loader, meta_test_loader=meta_test_loader, logger=logger)

        keys_to_send = ["lr", "inner_lr", "outer_lr"]
        if client_config.is_eval:
            keys_to_send.extend(["adapt_time", "acc_bf_adpt", "loss_bf_adpt", "acc_af_adpt", "loss_af_adpt"])
        elif cfg['eval_while_training']:
            if (client_config.epoch_idx - 1)% cfg['eval_round'] == 0:
                keys_to_send.extend(["params", "adapt_time", "acc_bf_adpt", "loss_bf_adpt", "acc_af_adpt", "loss_af_adpt"])
            else:
                keys_to_send.extend(["params"])


        # 构造需要发送的字典
        config_to_send = {key: getattr(client_config, key) for key in keys_to_send}
        if VERBOSE:
            print("client {} with rank {}, send params, epoch {}, comm_tag {}".format(client_config.idx, rank, client_config.epoch_idx, comm_tag))
        communicate_with_server(config_to_send, comm_tag, 'send_config')

        comm_tag += 1

        if client_config.epoch_idx > cfg['epoch_num']:
            break



# async def local_training(config, train_loader, test_loader, meta_test_loader=None, logger=None):
def local_training(config, train_loader, test_loader, meta_test_loader=None, logger=None):
    local_model = models.create_model_instance(
        cfg['dataset_type'], cfg['model_type'], cfg['classes_size'])
    torch.nn.utils.vector_to_parameters(config.params, local_model.parameters())
    local_model.to(device)

    epoch_lr, inner_lr, outer_lr = config.lr, config.inner_lr, config.outer_lr
    local_steps, adapt_steps = cfg['local_iters'], cfg['adapt_steps']

    # update lr 
    if config.epoch_idx > 1:
        epoch_lr, inner_lr, outer_lr = [max(lr*cfg['decay_rate'], cfg['min_lr']) for lr in [epoch_lr, inner_lr, outer_lr]]
        config.lr, config.inner_lr, config.outer_lr = epoch_lr, inner_lr, outer_lr
    logger.info("epoch-{} lr: {} inner_lr {} outer_lr {}".format(
        config.epoch_idx, epoch_lr, inner_lr, outer_lr))

    # if eval or eval while training
    if config.is_eval or (cfg['eval_while_training'] and config.epoch_idx % cfg['eval_round'] == 0):
        # before adaptation
        test_loss, test_acc = base.test(local_model, meta_test_loader, device, model_type=cfg['model_type'])
        config.acc_bf_adpt, config.loss_bf_adpt = test_acc, test_loss
        logger.info("eval_acc_before_adaptation: {}, eval_loss_before_adaptation: {}".format(
            test_acc, test_loss))
    # Training clients
    if config.is_eval == False:
        if cfg['meta_method']=='fedavg':
            info_dic = base.train(
                local_model, train_loader, cfg['momentum'], cfg['weight_decay'], epoch_lr, local_steps, cfg['local_epochs'], device, cfg['model_type'])
        elif cfg['meta_method']=='fomaml':
            info_dic = fomaml.train(
                local_model, train_loader, inner_lr, outer_lr, local_steps, cfg['local_epochs'], device, cfg['model_type'])
        elif cfg['meta_method']=='mamlhf':
            info_dic = mamlhf.train(
                local_model, train_loader, inner_lr, outer_lr, local_steps, cfg['local_epochs'], device, cfg['model_type'])
        elif cfg['meta_method']=='reptile':
            info_dic = reptile.train(
                local_model, train_loader, cfg['momentum'], cfg['weight_decay'], inner_lr, outer_lr, local_steps, cfg['local_epochs'], device, cfg['model_type'])
        # save params to config for sending back
        config.params = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
        logger.info(
            "Train_loss: {}\n".format(info_dic["train_loss"])+
            "Train_time: {}\n".format(info_dic["train_time"])
        )
        config.train_time = info_dic["train_time"]
    
    # training and eval clients:
    if config.is_eval or (cfg['eval_while_training'] and config.epoch_idx % cfg['eval_round'] == 0):
        # after adaptation
        info_dic=base.train(local_model, train_loader, cfg['momentum'], cfg['weight_decay'], epoch_lr, adapt_steps, 1,  device, cfg['model_type'])
        test_loss, test_acc = base.test(local_model, meta_test_loader, device, model_type=cfg['model_type'])
        config.acc_af_adpt, config.loss_af_adpt = test_acc, test_loss
        logger.info("eval_acc_after_adaptation: {}, eval_loss_after_adaptation: {}".format(
            test_acc, test_loss))
        logger.info(
            "Adapt_loss: {}\n".format(info_dic["train_loss"])+
            "Adapt_time: {}\n".format(info_dic["train_time"])
        )
        config.adapt_time = info_dic["train_time"]
        
    test_loss, test_acc = base.test(local_model, test_loader, device, cfg['model_type'])
    logger.info(
        "Test_Loss: {}\n".format(test_loss) +
        "Test_ACC: {}\n".format(test_acc)
    )

    logger.info("Save para")
    config.epoch_idx += 1




def init_logger(comm_tag, client_config):
    logger = logging.getLogger(os.path.basename(__file__).split('.')[0] + str(comm_tag))
    logger.setLevel(logging.INFO)
    if client_config.is_eval:
        filename = RESULT_PATH + now + "_" + os.path.basename(__file__).split('.')[0] + '_eval_' + str(
            client_config.idx) + '.log'
    else:
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
