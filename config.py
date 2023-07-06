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


class ClientConfig:
    def __init__(self, idx):
        self.idx = idx
        self.epoch_idx = 0
        self.params = None
        self.local_model_path = None
        self.global_model_path = None
        self.train_data_idxes = None
        # meta training
        # self.support_data_idxes = None
        # self.query_data_idxes = None
        self.test_data_idxes = None
        self.lr = None
        self.aggregate_weight = 0.1
        self.train_time = 0
        self.send_time = 0
        self.seed = None
