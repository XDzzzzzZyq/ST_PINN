import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    config.seed = 1234
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # data
    config.data = data = ml_collections.ConfigDict()
    data.image_size = 256
    data.pre_blur = 0.8
    data.dataset = "MNIST" # MNIST / VISIUM / ...
    
    data.poisson_ratio = 0.5 # for MNIST data

    data.field = ['EPCAM'] # for Visium HD data
    data.path = '../STHD/analysis/exp1_full_patchify/patches/'
    
    # training 
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 64
    training.sample_per_sol = 32
    training.dt = 0.0005
    training.stride = 1
    
    training.n_iters = 50000
    training.snapshot_freq = 10000
    training.log_freq = 500
    training.eval_freq = 100
    training.snapshot_freq_save = 25000 # will not overwrite
    
    config.model = model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 32
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (8,)
    model.resamp_with_conv = True
    model.conditional = True
    model.dropout = 0.1
    model.embedding_type = 'fourier'
    
    model.level_feature_nums = [16, 32, 64, 96] # 4 levels of features for Unet_lite
    
    # parameters
    config.param = param = ml_collections.ConfigDict()
    param.t0 = 0
    param.t1 = 0.1
    param.t2 = 1
    param.dx = 1/200
    
    param.use_vel = True
    
    param.v_min = 0.0005
    param.v_max = 0.001
    
    param.p_min = 0.0
    param.p_max = 0.001
    
    param.Re_min = 10000.0
    param.Re_man = 1000000.0
    
    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    return config