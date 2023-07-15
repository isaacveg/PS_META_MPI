"""
Use first order approximation of hessian
O(n) complexity, feasible. 
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


def train(model, train_loader, alpha, beta, local_iters=None, local_epochs=1,device=torch.device("cpu"), model_type=None):
    """
    Trains a given model using a First Order MAML approach.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        alpha (float): The learning rate for the one-step training.
        beta (float): The learning rate for the gradient update.
        local_iters (int, optional): The number of local iterations for the MAML training. 
            If not specified, it is calculated based on the size of the training dataset.
        local_epochs (int, optional): The number of local epochs for the MAML training. Default is 1.
        device (torch.device, optional): The device to run the training on. Default is 'cpu'.
        model_type (str, optional): The type of the model. Default is None.

    Returns:
        dict: A dictionary containing the following metrics:
            - 'train_loss' (float): The average training loss.
            - 'grad_loss' (float): The average gradient loss.
            - 'train_time' (float): The total training time.
            - 'params' (torch.Tensor): The model parameters.
    """
    t_start = time.time()
    model.train()
    
    # First Order MAML train for local_ites
    if local_iters is None:
        # 2 batches 1 update
        local_iters = math.ceil(len(train_loader.loader.dataset) / train_loader.loader.batch_size / 2)
    
    losses = [0.0, 0.0]
    sample_num = [0,0]

    for epoch_idx in range(local_epochs):
        for iter_idx in range(local_iters):
            temp_model = copy.deepcopy(model)
            # step 1: one step train
            batch_1 = next(train_loader)
            temp_model, one_step_loss = one_step(device, batch_1, temp_model, model_type, lr=alpha)

            # step 2: get grad
            batch_2 = next(train_loader)
            temp_model, grad_loss = get_grad(device, batch_2, model, model_type)

            # step 3: update model
            for param, grad_param in zip(model.parameters(), temp_model.parameters()):
                param.data.sub_(beta * grad_param.grad.data)
        
            losses = [losses[0]+one_step_loss, losses[1]+grad_loss]
            sample_num = [sample_num[0]+len(batch_1[0]), sample_num[1]+len(batch_2[0])]
        
 
    return {'train_loss': losses[0]/sample_num[0] if sample_num[0] != 0 else losses[0], 
            'grad_loss': losses[1]/sample_num[1] if sample_num[1] != 0 else losses[1], 
            'train_time': time.time()-t_start,
            'params': torch.nn.utils.parameters_to_vector(model.parameters()).detach()}


def one_step(device, data, model, model_type, lr):
    """
    Performs one step of training for a given device, data, model, and learning rate.

    Args:
        device (torch.device): The device (CPU or GPU) on which to perform the calculations.
        data (tuple): A tuple containing the input sequence (seq) and corresponding label (label).
        model (torch.nn.Module): The model to train.
        lr (float): The learning rate for the optimizer.

    Returns:
        tuple: A tuple containing the updated model and the loss value as a float.
    """
    seq, label = data

    if model_type == 'LR':
        seq = data.squeeze(1).view(-1, 28 * 28)

    seq = seq.to(device)
    label = label.to(device)
    y_pred = model(seq)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss().to(device)
    loss = loss_function(y_pred, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return model, loss.item()


def get_grad(device, data, model, model_type):
    """
        Calculate the gradient of the given model with respect to the input data.

        Parameters:
            device (torch.device): The device on which the computation will be performed.
            data (Tuple[torch.Tensor, torch.Tensor]): The input data and its corresponding labels.
            model (torch.nn.Module): The model for which the gradient will be calculated.

        Returns:
            Tuple[torch.nn.Module, float]: A tuple containing the updated model and the loss value as a float.
    """
    seq, label = data
    
    if model_type == 'LR':
        seq = data.squeeze(1).view(-1, 28 * 28)

    seq = seq.to(device)
    label = label.to(device)
    y_pred = model(seq)
    loss_function = nn.CrossEntropyLoss().to(device)
    loss = loss_function(y_pred, label)
    loss.backward()

    return model, loss.item()
