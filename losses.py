"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision


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
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, 
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
                g['lr'] = lr * min(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn

def get_step_fn(simulator, train, optimize_fn=None):
    distance = torch.nn.MSELoss()

    # Simulation & Parameter space sampling
    def loss_fn(model, sample, info):
        if train:
            model.train()
        else:
            model.eval()        

        if model.conditional:
            pred = model(sample.f, sample.t, info)
        else:
            pred = model(torch.cat([*info, sample.f], dim=1), sample.t)
        return distance(pred, sample.df_dt) + pred.mean(dim=(1,2,3)).square().mean()

    def step_fn(state, samples, info):
        model = state['model']
        loss = 0
        for sample in samples:
            if train:
                optimizer = state['optimizer']
                optimizer.zero_grad()
                loss_cur = loss_fn(model, sample, info) * (sample.t ** 2) * 1e2   # Loss normalization
                loss_cur.backward()
                optimize_fn(optimizer, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
            else:
                with torch.no_grad():
                    ema = state['ema']
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    loss_cur = loss_fn(model, sample, info)
                    ema.restore(model.parameters())
            loss += loss_cur.item()

        if train:
            state['step'] += 1
        return loss / (len(samples) - 1)

    return step_fn

def get_multi_shooting_step_fn(simulator, train, optimize_fn=None):
    distance = torch.nn.MSELoss()

    # Simulation & Parameter space sampling
    def loss_fn(model, sample1, sample2, info):
        if train:
            model.train()
        else:
            model.eval()        
          
        pred = simulator.reverse_aca_shooting(model, sample1, sample2, info)
        # TODO: Regularization
        # sigma = torch.sqrt(sample2.t).item() * 5
        # mask = torchvision.transforms.functional.gaussian_blur(info[0], (25, 25), (sigma, sigma))
        # mask = (mask > 0.05).float()
        # return distance(pred, sample2.f * mask)
        return distance(pred, sample2.f)

    def step_fn(state, samples, info):
        model = state['model']
        loss = 0
        for i in range(len(samples) - 1):
            if train:
                optimizer = state['optimizer']
                optimizer.zero_grad()
                loss_cur = loss_fn(model, samples[i+1], samples[i], info)
                loss_cur.backward()
                optimize_fn(optimizer, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
            else:
                with torch.no_grad():
                    ema = state['ema']
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    loss_cur = loss_fn(model, samples[i+1], samples[i], info)
                    ema.restore(model.parameters())
            loss += loss_cur.item()

        if train:
            state['step'] += 1
        return loss / (len(samples) - 1)

    return step_fn

def get_shooting_step_fn(simulator, train, optimize_fn=None):
    distance = torch.nn.MSELoss()

    # Simulation & Parameter space sampling
    def loss_fn(model, samples, info):
        if train:
            model.train()
        else:
            model.eval()        
        
        pred = simulator.reverse_adjoint_shooting(model, samples, info, rtol=1e-4, atol=1e-5)
        targ = torch.stack([sample.f for sample in samples[::-1]]).to(pred.device)
        return distance(pred, targ)

    def step_fn(state, samples, info):
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, samples, info)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['ema'].update(model.parameters())
            state['step'] += 1
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, samples, info)
                ema.restore(model.parameters())

        return loss

    return step_fn

def check_for_nans(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f'NaN detected in parameter: {name}')
            return True
    return False
