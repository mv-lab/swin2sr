from configs.default_config import get_default_config
from core.data_type import DataType
from core.loss_type import LossType
from core.model_type import ModelType
from core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 256
    data.data_type = DataType.BioSR_MRC
    # data.channel_1 = 0
    # data.channel_2 = 1
    data.ch1_fname = 'ER/GT_all.mrc'
    data.ch2_fname = 'CCPs/GT_all.mrc'
    data.num_channels = 2   
    data.poisson_noise_factor = 1000
    data.gaussian_noise_factor = 2000
    
    data.enable_gaussian_noise = True
    data.trainig_datausage_fraction = 1.0
    config.data.synthetic_gaussian_scale = 5000
    # if True, then input has 'identical' noise as the target. Otherwise, noise of input is independently sampled.
    config.data.input_has_dependant_noise = True
    data.sampler_type = SamplerType.DefaultSampler
    
    data.normalized_input = True
    data.clip_percentile = 1
    data.use_one_mu_std = True
    data.train_aug_rotate = False
    data.randomized_channels = False
    data.multiscale_lowres_count = None
    data.padding_mode = 'reflect'
    data.padding_value = None

    loss = config.loss
    loss.loss_type = LossType.MSE

    model = config.model
    model.model_type = ModelType.Swin2SR
    model.img_shape = (1004,1004)
    model.upscale = 1
    model.in_chans = 1
    

    training = config.training
    training.lr = 0.001
    training.num_epochs = 400
    training.precision = 16
    
     
    return config