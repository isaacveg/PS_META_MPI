"""
WARNING: 
You should NEVER use this method because it will take up n^2 memory! 
Even model as small as 3830000 params will take up
 3830000^2 * 4 / 1024/1024/1024 = 54646 GB memory!
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
    Trains a model using the MAML algorithm.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The data loader for training.
        alpha (float): The learning rate for the inner update.
        beta (float): The learning rate for the outer update.
        local_iters (int, optional): The number of local iterations. If not provided, it is calculated based on the size of the training data. 
        device (torch.device, optional): The device on which to perform the training. Defaults to torch.device("cpu").
        model_type (str, optional): The type of the model. Defaults to None.

    Returns:
        dict: A dictionary containing the losses during training, the training time, and the model parameters.
    """
    t_start = time.time()
    model.train()
    
    # MAML train for local_ites
    if local_iters is None:
        # 3 batches 1 update
        local_iters = math.ceil(len(train_loader.loader.dataset) / train_loader.loader.batch_size / 3)
    
    losses = [0.0, 0.0, 0.0]
    sample_num = [0,0,0]
    
    for epoch in range(local_iters):
        origin_model = copy.deepcopy(model)
        final_model = copy.deepcopy(model)

        # step1, one step on one batch, middle model
        batch_1 = next(train_loader)
        model, one_step_loss = one_step(device, batch_1, model, model_type, lr=alpha)
        # step2, get grad on next batch
        batch_2 = next(train_loader)
        model, grad_loss = get_grad(device, batch_2, model, model_type)
        # step3, get hessian on third batch using original model
        batch_3 = next(train_loader)
        hessian_params, hessian_loss = get_hessian(device, batch_3, origin_model, model_type)
        # step 4
        cnt = 0
        for param, param_grad in zip(final_model.parameters(), model.parameters()):
            hess = hessian_params[cnt]
            cnt += 1
            I = torch.ones_like(param.data)
            grad = (I - alpha * hess) * param_grad.grad.data
            param.data = param.data - beta * grad

        model = copy.deepcopy(final_model)

        sample_num = [sample_num[0] + len(batch_1[0]), sample_num[1] + len(batch_2[0]), sample_num[2] + len(batch_3[0])]
        losses = [losses[0] + one_step_loss, losses[1] + grad_loss, losses[2] + hessian_loss]
 
    return {'one_step_loss': losses[0]/sample_num[0] if sample_num[0] != 0 else losses[0], 
            'grad_loss': losses[1]/sample_num[1] if sample_num[1] != 0 else losses[1], 
            'hessian_loss': losses[2]/sample_num[2] if sample_num[2] != 0 else losses[2],
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


def get_hessian(device, data, model, model_type):
    """
    Computes the Hessian matrix of the model's parameters with respect to the given data.

    Args:
        device (torch.device): The device to use for computation.
        data (tuple): A tuple containing the input sequence and corresponding labels.
        model (nn.Module): The model to compute the Hessian matrix for.

    Returns:
        A tuple containing:
            - hessian_params (list of torch.Tensor): A list of tensors representing the Hessian matrix 
              for each parameter of the model.
            - loss (float): The loss value for the given input sequence and labels.
    """
    seq, label = data

    if model_type == 'LR':
        seq = data.squeeze(1).view(-1, 28 * 28)

    seq = seq.to(device)
    label = label.to(device)
    y_pred = model(seq)
    loss_function = nn.CrossEntropyLoss().to(device)
    loss = loss_function(y_pred, label)
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)

    hessian_params = []

    # 将一阶导数展平
    grads_flat = torch.cat([grad.view(-1) for grad in grads])

    # 计算二阶导数（海森矩阵）
    for i, grad in enumerate(grads_flat):
        grad2 = torch.autograd.grad(grad, model.parameters(), retain_graph=True)
        grad2_flat = torch.cat([g.contiguous().view(-1) for g in grad2])
        hessian_params.append(grad2_flat)
    hessian_params = torch.stack(hessian_params)

    # 还原成原来的形状
    new_hessian_params = []
    start = 0
    for param in model.parameters():
        end = start + param.numel()
        param_hessian = hessian_params[start:end].view(*param.shape, -1)
        new_hessian_params.append(param_hessian)
        start = end

    return new_hessian_params, loss.item()


