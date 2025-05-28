from config.default_configs import get_default_configs

def get_config():
    config = get_default_configs()
    
    config.param.param.Re_min = config.param.param.Re_max = 1e4

    return config