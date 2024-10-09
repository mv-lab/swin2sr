"""
Here, we compare a .py config file with a .pkl config file which gets generated from a training.
"""

import os.path
import ml_collections

from absl import app, flags
from ml_collections.config_flags import config_flags
from requests import delete
from scripts.compare_configs import (get_comparison_df, get_df_column_name, get_commit_key,
                                                 get_changed_files, display_changes)
from utils.config_utils import load_config

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("py_config", None, "Python config file", lock_config=True)
flags.DEFINE_string("pkl_config", None, "Work directory.")
flags.mark_flags_as_required(["py_config", "pkl_config"])


def main(argv):
    config1 = ml_collections.ConfigDict(FLAGS.py_config)
    config2 = ml_collections.ConfigDict(load_config(FLAGS.pkl_config))

    if 'encoder' not in config2.model:
        with config2.unlocked():
            config2.model.encoder = ml_collections.ConfigDict()
            for key in config1.model.encoder:
                if key in config2.model:
                    config2.model.encoder[key] = config2.model[key]

            assert 'decoder' not in config2.model
            config2.model.decoder = ml_collections.ConfigDict()
            for key in config1.model.decoder:
                if key in config2.model:
                    if key == 'multiscale_retain_spatial_dims':
                        config2.model.decoder[key] = False
                    else:
                        config2.model.decoder[key] = config2.model[key]

    df = get_comparison_df(config1, config2, 'python_config_file', get_df_column_name(FLAGS.pkl_config))

    changed_files = get_changed_files(*list(df.loc[get_commit_key()].values))
    display_changes(df, changed_files)


if __name__ == '__main__':
    app.run(main)
