import ml_collections
from core.sampler_type import SamplerType


def get_default_config():
    config = ml_collections.ConfigDict()

    config.data = ml_collections.ConfigDict()
    config.data.sampler_type = SamplerType.DefaultSampler

    config.model = ml_collections.ConfigDict()
    config.model.use_vampprior = False
    config.model.encoder = ml_collections.ConfigDict()
    config.model.decoder = ml_collections.ConfigDict()
    config.model.decoder.conv2d_bias = True

    config.loss = ml_collections.ConfigDict()

    config.training = ml_collections.ConfigDict()
    config.training.batch_size = 4

    config.training.grad_clip_norm_value = 0.5  # Taken from https://github.com/openai/vdvae/blob/main/hps.py#L38
    config.training.gradient_clip_algorithm = 'value'
    config.training.earlystop_patience = 10
    config.training.precision = 32
    config.training.pre_trained_ckpt_fpath = ''

    config.git = ml_collections.ConfigDict()
    config.git.changedFiles = []
    config.git.branch = ''
    config.git.untracked_files = []
    config.git.latest_commit = ''

    config.workdir = '/FILL_IN_THE_WORKDIR'
    config.datadir = ''
    config.hostname = ''
    config.exptname = ''
    return config