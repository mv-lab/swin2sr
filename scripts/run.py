"""
run file for the disentangle work. 
"""
import json
import logging
import os
import pickle
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.data import DataLoader

import git
import ml_collections
# import tensorboard
from absl import app, flags
from denoisplit.config_utils import get_updated_config
from denoisplit.core.data_type import DataType
from denoisplit.core.loss_type import LossType
from denoisplit.core.model_type import ModelType
from denoisplit.core.sampler_type import SamplerType
from denoisplit.sampler.default_grid_sampler import DefaultGridSampler
from denoisplit.sampler.intensity_aug_sampler import IntensityAugSampler, IntensityAugValSampler
from denoisplit.sampler.nbr_sampler import NeighborSampler
from denoisplit.sampler.random_sampler import RandomSampler
from denoisplit.sampler.singleimg_sampler import SingleImgSampler
from denoisplit.training import create_dataset, train_network
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("logdir", '/group/jug/ashesh/wandb_backup/', "The folder name for storing logging")
flags.DEFINE_string("datadir", '/tmp2/ashesh/ashesh/VAE_based/data/MNIST/noisy/', "Data directory.")
flags.DEFINE_boolean("use_max_version", False, "Overwrite the max version of the model")
flags.DEFINE_string("load_ckptfpath", '', "The path to a previous ckpt from which the weights should be loaded")
flags.DEFINE_string("override_kwargs", '', 'There keys will be overwridden with the corresponding values')
flags.mark_flags_as_required(["workdir", "config", "mode"])


def add_git_info(config):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    repo = git.Repo(dir_path, search_parent_directories=True)
    config.git.changedFiles = [item.a_path for item in repo.index.diff(None)]
    config.git.branch = repo.active_branch.name
    config.git.untracked_files = repo.untracked_files
    config.git.latest_commit = repo.head.object.hexsha


def log_config(config, cur_workdir):
    # Saving config file.
    with open(os.path.join(cur_workdir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    print(f'Saved config to {cur_workdir}/config.pkl')


def set_logger():
    os.makedirs(FLAGS.workdir, exist_ok=True)
    fstream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(fstream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')


def get_new_model_version(model_dir: str) -> str:
    """
    A model will have multiple runs. Each run will have a different version.
    """
    versions = []
    for version_dir in os.listdir(model_dir):
        try:
            versions.append(int(version_dir))
        except:
            print(f'Invalid subdirectory:{model_dir}/{version_dir}. Only integer versions are allowed')
            exit()
    if len(versions) == 0:
        return '0'
    return f'{max(versions) + 1}'


def get_model_name(config):
    mtype = config.model.model_type
    dtype = config.data.data_type
    ltype = config.loss.loss_type
    stype = config.data.sampler_type

    return f'D{dtype}-M{mtype}-S{stype}-L{ltype}'


def get_month():
    return datetime.now().strftime("%y%m")


def get_workdir(config, root_dir, use_max_version, nested_call=0):
    rel_path = get_month()
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    rel_path = os.path.join(rel_path, get_model_name(config))
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    if use_max_version:
        # Used for debugging.
        version = int(get_new_model_version(cur_workdir))
        if version > 0:
            version = f'{version - 1}'

        rel_path = os.path.join(rel_path, str(version))
    else:
        rel_path = os.path.join(rel_path, get_new_model_version(cur_workdir))

    cur_workdir = os.path.join(root_dir, rel_path)
    try:
        Path(cur_workdir).mkdir(exist_ok=False)
    except FileExistsError:
        print(
            f'Workdir {cur_workdir} already exists. Probably because someother program also created the exact same directory. Trying to get a new version.'
        )
        time.sleep(2.5)
        if nested_call > 10:
            raise ValueError(f'Cannot create a new directory. {cur_workdir} already exists.')

        return get_workdir(config, root_dir, use_max_version, nested_call + 1)

    return cur_workdir, rel_path


def _update_config(config, key_levels, value):
    if len(key_levels) == 1:
        config[key_levels[0]] = value
    else:
        _update_config(config[key_levels[0]], key_levels[1:], value)


def overwride_with_cmd_params(config, params_dict):
    """
    It makes sure that config is updated correctly with the value typecasted to the same type as is already present in the config.
    """
    for key in params_dict:
        key_levels = key.split('.')
        _update_config(config, key_levels, params_dict[key])


def get_mean_std_dict_for_model(config, train_dset):
    """
    Computes the mean and std for the model. This will be subsequently passed to the model.
    """
    if config.data.data_type == DataType.TwoDset:
        mean_dict, std_dict = train_dset.compute_mean_std()
        for dset_key in mean_dict.keys():
            mean_dict[dset_key]['input'] = mean_dict[dset_key]['input'].reshape(1, 1, 1, 1)
    elif config.data.data_type == DataType.PredictedTiffData:
        mean_dict = {'input': None, 'target': None}
        std_dict = {'input': None, 'target': None}
        inp_mean, inp_std = train_dset.get_mean_std_for_input()
        mean_dict['input'] = inp_mean
        std_dict['input'] = inp_std
        if config.data.target_separate_normalization is True:
            data_mean, data_std = train_dset.compute_individual_mean_std()
        else:
            data_mean, data_std = train_dset.get_mean_std()
            # skip input channel
            data_mean = data_mean[1:].copy()
            data_std = data_std[1:].copy()

        mean_dict['target'] = data_mean
        std_dict['target'] = data_std

    else:
        mean_dict = {'input': None, 'target': None}
        std_dict = {'input': None, 'target': None}
        inp_mean, inp_std = train_dset.get_mean_std()
        mean_sq = inp_mean.squeeze()
        std_sq = inp_std.squeeze()
        for i in range(1, config.data.get('num_channels', 2)):
            assert mean_sq[0] == mean_sq[i]
            assert std_sq[0] == std_sq[i]
        mean_dict['input'] = np.mean(inp_mean, axis=1, keepdims=True)
        std_dict['input'] = np.mean(inp_std, axis=1, keepdims=True)

        if config.data.target_separate_normalization is True:
            data_mean, data_std = train_dset.compute_individual_mean_std()
        else:
            data_mean, data_std = train_dset.get_mean_std()

        mean_dict['target'] = data_mean
        std_dict['target'] = data_std

    return mean_dict, std_dict


def main(argv):
    config = FLAGS.config
    if FLAGS.override_kwargs:
        overwride_with_cmd_params(config, json.loads(FLAGS.override_kwargs))
    # making older configs compatible with current version.
    config = get_updated_config(config)

    assert os.path.exists(FLAGS.workdir)
    cur_workdir, relative_path = get_workdir(config, FLAGS.workdir, FLAGS.use_max_version)
    print(f'Saving training to {cur_workdir}')

    add_git_info(config)
    config.workdir = cur_workdir
    config.exptname = relative_path
    config.hostname = socket.gethostname()
    config.datadir = FLAGS.datadir
    config.training.pre_trained_ckpt_fpath = FLAGS.load_ckptfpath

    if FLAGS.mode == "train":
        set_logger()
        raw_data_dict = None

        # Now, config cannot be changed.
        config = ml_collections.FrozenConfigDict(config)
        log_config(config, cur_workdir)

        train_data, val_data = create_dataset(config, FLAGS.datadir, raw_data_dict=raw_data_dict)

        mean_dict, std_dict = get_mean_std_dict_for_model(config, train_data)

        # assert np.abs(config.data.mean_val - data_mean) < 1e-3, f'{config.data.mean_val - data_mean}'
        # assert np.abs(config.data.std_val - data_std) < 1e-3, f'{config.data.std_val - data_std}'

        if config.data.sampler_type == SamplerType.DefaultSampler:
            batch_size = config.training.batch_size
            shuffle = True

            train_dloader = DataLoader(train_data,
                                       pin_memory=False,
                                       num_workers=config.training.num_workers,
                                       shuffle=shuffle,
                                       batch_size=batch_size)
            val_dloader = DataLoader(val_data,
                                     pin_memory=False,
                                     num_workers=config.training.num_workers,
                                     shuffle=False,
                                     batch_size=batch_size)

        else:

            if config.data.sampler_type == SamplerType.RandomSampler:
                train_sampler = RandomSampler(train_data, config.training.batch_size)
                val_sampler = DefaultGridSampler(val_data, config.training.batch_size, grid_size=config.data.image_size)
            elif config.data.sampler_type == SamplerType.SingleImgSampler:
                train_sampler = SingleImgSampler(train_data, config.training.batch_size)
                val_sampler = SingleImgSampler(val_data, config.training.batch_size)
            elif config.data.sampler_type == SamplerType.NeighborSampler:
                assert 'gridsizes' in config.training, 'For this to work, gridsizes must be provided'
                nbr_set_count = config.data.nbr_set_count
                train_sampler = NeighborSampler(train_data,
                                                config.training.batch_size,
                                                valid_gridsizes=config.training.gridsizes,
                                                nbr_set_count=nbr_set_count)
                val_sampler = NeighborSampler(val_data, config.training.batch_size, nbr_set_count=0)
            elif config.data.sampler_type == SamplerType.DefaultGridSampler:
                train_sampler = DefaultGridSampler(train_data, config.training.batch_size)
                val_sampler = DefaultGridSampler(val_data, config.training.batch_size, grid_size=config.data.image_size)
            elif config.data.sampler_type == SamplerType.IntensityAugSampler:
                val_sampler = IntensityAugValSampler(val_data, config.data.image_size, config.training.batch_size)
                train_sampler = IntensityAugSampler(train_data,
                                                    len(train_data),
                                                    config.data.ch1_alpha_interval_count,
                                                    config.data.num_intensity_variations,
                                                    batch_size=config.training.batch_size)
            train_dloader = DataLoader(train_data,
                                       pin_memory=False,
                                       batch_sampler=train_sampler,
                                       num_workers=config.training.num_workers)
            val_dloader = DataLoader(val_data,
                                     pin_memory=False,
                                     batch_sampler=val_sampler,
                                     num_workers=config.training.num_workers)

        train_network(train_dloader, val_dloader, mean_dict, std_dict, config, 'BaselineVAECL', FLAGS.logdir)

    elif FLAGS.mode == "eval":
        pass
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == '__main__':
    print(socket.gethostname(), datetime.now().strftime("%y-%m-%d-%H:%M:%S"))
    print('Python version', sys.version)
    print('CUDA_HOME', CUDA_HOME)
    print('CudaToolKit Version', torch.version.cuda)
    print('torch Version', torch.__version__)
    print('torchvision Version', torchvision.__version__)
    app.run(main)
