from . import unet, unet_lite, transformer

def get_model(config):
    if config.model.type == 'unet':
        return unet.Unet(config)
    elif config.model.type == 'unet_lite':
        return unet_lite.Unet(config)
    elif config.model.type == 'transformer':
        return transformer.Unet(config)