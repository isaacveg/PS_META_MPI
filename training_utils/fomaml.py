import sys
import time
import math
import re
import gc
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fomaml(model, data_loader, optimizer, local_iters=None, device=torch.device("cpu"), model_type=None):
    t_start = time.time()
    model.train()
    if local_iters is None:
        local_iters = math.ceil(len(data_loader.dataset) / data_loader.batch_size)
    # print("local_iters: ", local_iters)

    train_loss = 0.0
    samples_num = 0

    
    for iter_idx in range(local_iters):
        data, target = next(iter(data_loader))

        if model_type == 'LR':
            data = data.squeeze(1).view(-1, 28 * 28)
            
        data, target = data.to(device), target.to(device)
        
        output = model(data)

        optimizer.zero_grad()
        
        loss_func = nn.CrossEntropyLoss() 
        loss =loss_func(output, target)
        # print("here")
        loss.backward()
        optimizer.step()

        # train_loss += (loss.item() * data.size(0))
        # samples_num += data.size(0)

        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)

    if samples_num != 0:
        train_loss /= samples_num
    
    return {'train_loss': train_loss, 'train_time': time.time()-t_start}

