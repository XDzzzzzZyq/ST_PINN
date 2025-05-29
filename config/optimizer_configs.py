from config.default_configs import get_default_configs

def get_config():
    config = get_default_configs()
    
    config.optim.weight_decay = 1e-4
    config.optim.optimizer = 'AdamW'
    config.optim.lr = 1e-3
    config.optim.beta1 = 0.85
    config.optim.eps = 1e-8
    config.optim.warmup = 50
    config.optim.grad_clip = 5.

    config.model.ema_rate = 0.95

    return config