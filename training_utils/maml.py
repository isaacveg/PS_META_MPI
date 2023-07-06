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
    t_start = time.time()
    model.train()
    
    # MAML train for local_ites
    if local_iters is None:
        # 3 batches 1 update
        local_iters = math.ceil(len(train_loader.loader.dataset) / train_loader.loader.batch_size / 3)
    
    one_step_losses = []
    grad_losses = []
    hessian_losses = []
    
    for epoch in local_iters:
        origin_model = copy.deepcopy(model)
        final_model = copy.deepcopy(model)

        # step1, one step on one batch, middle model
        model, one_step_loss = one_step(device, next(train_loader), model, model_type, lr=alpha)
        # step2, get grad on next batch
        model, grad_loss = get_grad(device, next(train_loader), model, model_type)
        # step3, get hessian on third batch using original model
        hessian_params, hessian_loss = get_hessian(device, next(train_loader), origin_model, model_type)
        # step 4
        cnt = 0
        for param, param_grad in zip(final_model.parameters(), model.parameters()):
            hess = hessian_params[cnt]
            cnt += 1
            I = torch.ones_like(param.data)
            grad = (I - alpha * hess) * param_grad.grad.data
            param.data = param.data - beta * grad

        model = copy.deepcopy(final_model)
        one_step_losses.append(one_step_loss)
        grad_losses.append(grad_loss)
        hessian_losses.append(hessian_loss)
 
    return {'train_loss': one_step_losses, 
            'grad_loss': grad_losses, 
            'hessian_loss': hessian_losses,
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
    for k in range(len(grads)):
        hess_params = torch.zeros_like(grads[k])
        for i in range(grads[k].size(0)):
            # w or b?
            if len(grads[k].size()) == 2:
                for j in range(grads[k].size(1)):
                    hess_params[i, j] = torch.autograd.grad(grads[k][i][j], model.parameters(), retain_graph=True)[k][
                        i, j]
            else:
                hess_params[i] = torch.autograd.grad(grads[k][i], model.parameters(), retain_graph=True)[k][i]
        hessian_params.append(hess_params)

    #     # 将一阶导数展平
    # grads_flat = torch.cat([grad.view(-1) for grad in grads])

    # # 计算二阶导数（海森矩阵）
    # hessian = []
    # for i, grad in enumerate(grads_flat):
    #     grad2 = torch.autograd.grad(grad, model.parameters(), retain_graph=True)
    #     grad2_flat = torch.cat([g.contiguous().view(-1) for g in grad2])
    #     hessian.append(grad2_flat)
    # hessian = torch.stack(hessian)

    return hessian_params, loss.item()


