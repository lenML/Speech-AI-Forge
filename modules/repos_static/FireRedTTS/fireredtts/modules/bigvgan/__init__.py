from fireredtts.modules.bigvgan.bigvgan import BigVGAN


def get_bigvgan_backend(bigvgan_config):
    generator = BigVGAN(**bigvgan_config)
    return generator
