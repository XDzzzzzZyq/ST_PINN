from config.default_configs import get_default_configs

def get_config():
    config = get_default_configs()
    config.data.dataset = 'SIMULATE'
    config.data.field = ['EEF1A1', 'S100A6']

    config.training.batch_size = 2
    config.training.sample_per_sol = 128

    return config