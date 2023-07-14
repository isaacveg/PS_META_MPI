"""
Almost identical to FedAvg, I don't know whether it's gonna be good or ehh...
"""
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


def train(model, train_loader, momentum, weight_decay, alpha, beta, local_iters=None, device=torch.device("cpu"), model_type=None):
    t_start = time.time()
    model.train()

    # how many steps one iter
    steps = 3
    
    if local_iters is None:
        local_iters = math.ceil(len(train_loader.loader.dataset) / train_loader.loader.batch_size / steps)
    # print("local_iters: ", local_iters)

    if momentum < 0:
        optimizer = torch.optim.SGD(model.parameters(), lr=alpha, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=alpha, weight_decay=weight_decay)

    # print("here bf train")
    train_loss = 0.0
    samples_num = 0

    for iter_idx in range(local_iters):
        original_model = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

        for step in range(steps):
            data, target = next(train_loader)
            # print("here after data")

            if model_type == 'LR':
                data = data.squeeze(1).view(-1, 28 * 28)
                
            data, target = data.to(device), target.to(device)
            
            output = model(data)

            optimizer.zero_grad()
            
            loss_func = nn.CrossEntropyLoss().to(device) 
            loss = loss_func(output, target)
            # print("here")
            loss.backward()
            optimizer.step()

            train_loss += (loss.item() * data.size(0))
            samples_num += data.size(0)

        updated_para = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
        para_delta = updated_para - original_model
        original_model += para_delta * beta

    if samples_num != 0:
        train_loss /= samples_num

    # original_model = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

    # for iter_idx in range(local_iters):
    #     data, target = next(train_loader)
    #     # print("here after data")

    #     if model_type == 'LR':
    #         data = data.squeeze(1).view(-1, 28 * 28)
            
    #     data, target = data.to(device), target.to(device)
        
    #     output = model(data)

    #     optimizer.zero_grad()
        
    #     loss_func = nn.CrossEntropyLoss().to(device) 
    #     loss = loss_func(output, target)
    #     # print("here")
    #     loss.backward()
    #     optimizer.step()

    #     train_loss += (loss.item() * data.size(0))
    #     samples_num += data.size(0)

    # if samples_num != 0:
    #     train_loss /= samples_num

    # updated_para = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    # para_delta = updated_para - original_model
    # original_model += para_delta * beta
    
    return {'train_loss': train_loss, 'train_time': time.time()-t_start,
            'params': original_model}

