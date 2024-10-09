import argparse
import os
import pickle
from time import sleep

from analysis.results_handler import PaperResultsHandler


def rnd(obj):
    return f'{obj:.3f}'


def show(ckpt_dir, results_dir, only_test=True, skip_last_pixels=None):
    if ckpt_dir[-1] == '/':
        ckpt_dir = ckpt_dir[:-1]
    if results_dir[-1] == '/':
        results_dir = results_dir[:-1]

    fname = PaperResultsHandler.get_fname(ckpt_dir)
    print(ckpt_dir)
    for dir in sorted(os.listdir(results_dir)):
        if only_test and dir[:4] != 'Test':
            continue
        if skip_last_pixels is not None:
            sktoken = dir.split('_')[-1]
            assert sktoken[:2] == 'Sk'
            if int(sktoken[2:]) != skip_last_pixels:
                continue

        fpath = os.path.join(results_dir, dir, fname)
        # print(fpath)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                out = pickle.load(f)

            print(dir)
            if 'rmse' in out:
                print('RMSE', ' '.join([rnd(x) for x in out['rmse']]))
            if 'psnr' in out:
                print('PSNR', ' '.join([rnd(x) for x in out['psnr']]))
            if 'rangeinvpsnr' in out:
                print('RangeInvPSNR', ' '.join([rnd(x) for x in out['rangeinvpsnr']]))
            if 'ssim' in out:
                print('SSIM', ' '.join(rnd(x) for x in out['ssim']))
            if 'ms_ssim' in out:
                print('MS-SSIM', ' '.join(rnd(x) for x in out['ms_ssim']))
            print('')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('ckpt_dir', type=str)
    # parser.add_argument('results_dir', type=str)
    # parser.add_argument('--skip_last_pixels', type=int)
    # args = parser.parse_args()

    # ckpt_dir = '/home/ashesh.ashesh/training/disentangle/2210/D3-M3-S0-L0/117'
    # results_dir = '/home/ashesh.ashesh/data/paper_stats/'
    ckpt_dirs = [
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/93',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/88',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/109/',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/125',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/94',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/89',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/128',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/95',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/87',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/130',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/92',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/90',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/115',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/104',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/96',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/126',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/105',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/97',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/127',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/106',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/98',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/129',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/107',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/99',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/135',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/114',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/101',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/133',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/113',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/100',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/132',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/117',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/103',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/120',
        # '/home/ashesh.ashesh/training/disentangle/2402/D16-M23-S0-L0/102',
    ]

    for ckpt_dir in ckpt_dirs:
        show(ckpt_dir, '/group/jug/michele/data/paper_stats/', only_test=True, skip_last_pixels=44)
        sleep(1)

    # show(args.ckpt_dir, args.results_dir, only_test=True, skip_last_pixels=args.skip_last_pixels)
