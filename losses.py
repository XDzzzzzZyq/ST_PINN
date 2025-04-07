"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


def get_optimizer(config, params, lr_mul=1.0):
    """Returns a flax optimizer object based on `config`."""

    lr = config.optim.lr
    decay = config.optim.weight_decay

    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, 
                               lr=lr*lr_mul, 
                               betas=(config.optim.beta1, 0.999), 
                               eps=config.optim.eps,
                               weight_decay=decay)
    else:
        raise NotImplementedError(f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def get_optimize_fn(config):
    """Returns an optimize_fn based on `config`."""

    lr = config.optim.lr

    def optimize_fn(optimizer, params, step, lr=lr, warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn


def get_step_fn(simulator, train, optimize_fn=None):
    """Create a one-step training/evaluation function.

    Args:
        simulator: simulator 
        train: train or eval
        optimize_fn: An optimization function.

    Returns:
        A one-step function for training or evaluation.
    """
    distance = torch.nn.MSELoss()

    # Simulation & Parameter space sampling
    def loss_fn(model, sample): # TODO: conditioning info
        if train:
            model.train()
        else:
            model.eval()        

        pred = model(sample.p, sample.t)
        return distance(pred, sample.dfdt)

    def step_fn(state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          state: A dictionary of training information, containing the score model, optimizer,
           EMA status, and number of optimization steps.
          batch: A mini-batch of training/evaluation data.

        Returns:
          loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())

        return loss

    return step_fn

def check_for_nans(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f'NaN detected in parameter: {name}')
            return True
    return False
