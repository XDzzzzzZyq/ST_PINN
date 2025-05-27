from config.default_configs import get_default_configs

def get_config():
    config = get_default_configs()
    
    config.model.type = 'unet'

    config.training.batch_size = 16
    config.training.sample_per_sol = 32

    config.optim.lr = 1e-3

    return config