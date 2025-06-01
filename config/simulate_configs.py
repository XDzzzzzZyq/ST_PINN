from config.default_configs import get_default_configs

def get_config():
    config = get_default_configs()
    config.data.dataset = 'SIMULATE'
    config.data.field = ['EEF1A1', 'S100A6']
    config.data.pre_blur = 0.5
    config.data.image_size = 192

    config.data.factor = 3.0
    config.data.masked = True
    config.data.padding = 32

    assert config.data.image_size + 2 * config.data.padding == 256

    config.training.batch_size = 4
    config.training.sample_per_sol = 128

    return config