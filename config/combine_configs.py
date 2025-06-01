from config.default_configs import get_default_configs
from config.large_configs import get_config as get_large_config
from config.simulate_configs import get_config as get_simu_config

def get_config():
    config = get_default_configs()
    lrg_config = get_large_config()
    sim_config = get_simu_config()
    
    config.data = sim_config.data
    # config.training = sim_config.training
    config.training.batch_size = 16

    config.model = lrg_config.model
    config.model.rtol = 1e-3
    config.model.atol = 1e-3

    return config