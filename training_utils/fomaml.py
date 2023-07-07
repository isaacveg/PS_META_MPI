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


def train(model, train_loader, alpha, beta, local_iters=None, device=torch.device("cpu"), model_type=None):
    """
    Train a model using the First Order MAML algorithm for a given number of iterations.
    
    Args:
        model: The model to be trained.
        train_loader: The data loader for training data.
        alpha: The learning rate for the one-step update.
        beta: The meta-learning rate for the gradient update.
        local_iters: The number of local iterations to perform. 
            If None, default to ceil(len(train_loader.loader.dataset) / train_loader.loader.batch_size / 2).
        device: The device to run the training on. Defaults to torch.device("cpu").
        model_type: The type of the model. Defaults to None.
    
    Returns:
        A dictionary containing the following:
        - train_loss: The average training loss per sample.
        - grad_loss: The average gradient loss per sample.
        - train_time: The total training time.
        - params: The model parameters.
    """
    t_start = time.time()
    model.train()
    
    # First Order MAML train for local_ites
    if local_iters is None:
        # 2 batches 1 update
        local_iters = math.ceil(len(train_loader.loader.dataset) / train_loader.loader.batch_size / 2)
    
    losses = [0.0, 0.0]
    sample_num = [0,0]

    for epoch in range(local_iters):
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
