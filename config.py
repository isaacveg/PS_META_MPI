import os
from typing import List
import paramiko
from scp import SCPClient
from torch.utils.tensorboard import SummaryWriter
from comm_utils import *
import yaml

global cfg
if 'cfg' not in globals():
    with open('config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)


class ClientAction:
    LOCAL_TRAINING = "local_training"


class Worker:
    def __init__(self, config, rank):
        #这个config就是后面的client_config
        self.config = config
        self.rank = rank

    async def send_data(self, data, comm, epoch_idx):
        await send_data(comm, data, self.rank, epoch_idx)    

    async def send_init_config(self, comm, epoch_idx):
        print("before send", self.rank, "tag:", epoch_idx)
        await send_data(comm, self.config, self.rank, epoch_idx)    

    async def get_model(self, comm, epoch_idx):
        self.config.neighbor_paras = await get_data(comm, self.rank, epoch_idx)
    
    async def get_para(self, comm, epoch_idx):
        train_time,send_time = await get_data(comm, self.rank, epoch_idx)
        self.config.train_time=train_time
        self.config.send_time=send_time

class CommonConfig:
    def __init__(self):
        self.model_type = None
        self.dataset_type = None
        self.batch_size = None
        self.data_pattern = None
        self.lr = None
        self.decay_rate = None
        self.min_lr = None
        self.epoch = None
        self.momentum=None
        self.weight_decay=None
        self.para = None
        self.data_path = None
        #这里用来存worker的


class ClientConfig:
    def __init__(self,
                common_config,
                custom: dict = dict()
                ):
        self.para = None
        self.train_data_idxes = None
        self.common_config=common_config

        self.average_weight=0.1
        self.local_steps=20
        self.compre_ratio=1
        self.train_time=0
        self.send_time=0
        self.neighbor_paras=None
        self.neighbor_indices=None
