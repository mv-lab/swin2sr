import os
import shutil

import pytorch_lightning as pl


class ValEveryNSteps(pl.Callback):
    """
    Run validation after every n step
    """
    def __init__(self, every_n_step):
        self.every_n_step = every_n_step

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.every_n_step == 0 and trainer.global_step != 0:
            trainer.run_evaluation()


def clean_up(dir):
    for yearmonth in os.listdir(dir):
        monthdir = os.path.join(dir, yearmonth)
        for modeltype in os.listdir(monthdir):
            modeltypedir = os.path.join(monthdir, modeltype)
            for modelid in os.listdir(modeltypedir):
                modeldir = os.path.join(modeltypedir, modelid)
                for fname in os.listdir(modeldir):
                    if fname[-10:] == '_last.ckpt':
                        fpath = os.path.join(modeldir, fname)
                        print('Removing', fpath)
                        os.remove(fpath)


def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def copy_config(src_dir, dst_dir):
    for yearmonth in os.listdir(src_dir):
        monthdir = os.path.join(src_dir, yearmonth)
        dst_monthdir = os.path.join(dst_dir, yearmonth)
        create_dir(dst_monthdir)
        for modeltype in os.listdir(monthdir):
            modeltypedir = os.path.join(monthdir, modeltype)
            dst_modeltypedir = os.path.join(dst_monthdir, modeltype)
            create_dir(dst_modeltypedir)
            for modelid in os.listdir(modeltypedir):
                modeldir = os.path.join(modeltypedir, modelid)
                dst_modeldir = os.path.join(dst_modeltypedir, modelid)
                create_dir(dst_modeldir)
                for fname in os.listdir(modeldir):
                    if fname[-5:] != '.ckpt' and fname[:7] != 'events.':
                        fpath = os.path.join(modeldir, fname)
                        dst_fpath = os.path.join(dst_modeldir, fname)
                        shutil.copyfile(fpath, dst_fpath)
