"""
This is specific to the HDN => uSplit pipeline. 
"""
import os

from configs.config_utils import get_configdir_from_saved_predictionfile, load_config


def get_source_channel(pred_fname):
    den_config_dir1 = get_configdir_from_saved_predictionfile(pred_fname)
    config_temp = load_config(den_config_dir1)
    print(pred_fname, config_temp.model.denoise_channel, config_temp.data.ch1_fname, config_temp.data.ch2_fname)
    if config_temp.model.denoise_channel == 'Ch1':
        ch1 = config_temp.data.ch1_fname
    elif config_temp.model.denoise_channel == 'Ch2':
        ch1 = config_temp.data.ch2_fname
    else:
        raise ValueError('Unhandled channel', config_temp.model.denoise_channel)
    return ch1


def whether_to_flip(ch1_fname, ch2_fname, reference_config):
    """
    When one wants to get the highsnr data, then one does not know if the order of the channels is same as what uSplit predicts. 
    If not, then one needs to flip the channels.
    """
    ch1 = get_source_channel(ch1_fname)
    ch2 = get_source_channel(ch2_fname)
    channels = [reference_config.data.ch1_fname, reference_config.data.ch2_fname]
    assert ch1 in channels, f'{ch1} not in {channels}'
    assert ch2 in channels, f'{ch2} not in {channels}'
    assert ch1 != ch2, f'{ch1} and {ch2} are same'
    if ch1 == reference_config.data.ch2_fname:
        return True
    return False
