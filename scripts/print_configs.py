import argparse
import os
import torch

from utils.config_utils import load_config
from analysis.checkpoint_utils import get_best_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    assert os.path.exists(args.config)
    dir = args.config
    try:
        ckpt_fpath = get_best_checkpoint(dir)
        checkpoint = torch.load(ckpt_fpath)
        print(f'Model Trained till {checkpoint["epoch"]} epochs')
    except:
        print('No model was found in', dir)

    print(load_config(args.config))
