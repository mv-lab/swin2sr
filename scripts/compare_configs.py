"""
Here, we compare two configs.
"""
import argparse
import os

import numpy as np
import pandas as pd

import git
import ml_collections
from denoisplit.config_utils import load_config


def _compare_config(config1, config2, prefix_key=''):
    keys = []
    val1 = []
    val2 = []
    for key in config1:
        if isinstance(config1[key], ml_collections.ConfigDict) or isinstance(config1[key],
                                                                             ml_collections.FrozenConfigDict):
            nested_key, nested_val1, nested_val2 = _compare_config(config1.get(key, {}),
                                                                   config2.get(key, {}),
                                                                   prefix_key=f'{key}.')
            keys += nested_key
            val1 += nested_val1
            val2 += nested_val2
        else:
            if key in config2:
                if isinstance(config1[key], list) or isinstance(config1[key], tuple) or isinstance(
                        config1[key], np.ndarray):
                    unequal = tuple(config1[key]) != tuple(config2[key])
                else:
                    unequal = config1[key] != config2[key]
                if unequal:
                    keys.append(prefix_key + key)
                    val1.append(config1[key])
                    val2.append(config2[key])
            else:
                keys.append(prefix_key + key)
                val1.append(config1[key])
                val2.append(None)

    return keys, val1, val2


def compare_raw_configs(config1, config2):
    keys, val1, val2 = _compare_config(config1, config2)
    keys_v2, val2_v2, val1_v2 = _compare_config(config2, config1)
    for idx, key_v2 in enumerate(keys_v2):
        if key_v2 in keys:
            continue
        assert val1_v2[
            idx] is None, 'Since this key is not present in keys, it means that it was not present in config1. So it must be none'
        keys.append(key_v2)
        val1.append(val1_v2[idx])
        val2.append(val2_v2[idx])

    val1_df = pd.Series(val1, index=keys).to_frame('Config1')
    val2_df = pd.Series(val2, index=keys).to_frame('Config2')
    df = pd.concat([val1_df, val2_df], axis=1)
    if 'workdir' in df.index:
        df = df.drop('workdir')
    if 'exptname' in df.index:
        df = df.drop('exptname')

    return df


def get_df_column_name(path):
    if path[-1] == '/':
        path = path[:-1]
    tokens = []
    depth = 3
    while depth > 0 and path != '':
        d0 = os.path.basename(path)
        path = os.path.dirname(path)
        tokens.append(d0)
        depth -= 1
    if depth > 0:
        return path

    return os.path.join(*reversed(tokens))


def get_changed_files(commit1, commit2):
    repo = git.Repo(search_parent_directories=True)
    fnames = repo.git.diff(f'{commit1}..{commit2}', name_only=True).split('\n')
    return fnames


def compare_config(config1_path, config2_path):
    """
    Compare two configs. This returns a dataframe with differing keys as index. It has two columns, one for each config.
    """
    config1 = load_config(config1_path)
    config2 = load_config(config2_path)
    if 'encoder' not in config1.model:
        config1 = ml_collections.config_dict.ConfigDict(config1)
        with config1.unlocked():
            config1.model.encoder = ml_collections.ConfigDict()
            assert 'decoder' not in config1.model
            config1.model.decoder = ml_collections.ConfigDict()

    if 'encoder' not in config2.model:
        config2.encoder = ml_collections.ConfigDict()
        assert 'decoder' not in config2.model
        config2.decoder = ml_collections.ConfigDict()

    c1_name, c2_name = get_df_column_name(config1_path), get_df_column_name(config2_path)
    df = get_comparison_df(config1, config2, c1_name, c2_name)
    return df, get_changed_files(*list(df.loc[get_commit_key()].values))


def get_commit_key():
    return 'git.latest_commit'


def get_comparison_df(config1, config2, config1_name, config2_name):
    df = compare_raw_configs(config1, config2)
    df.columns = [config1_name, config2_name]
    df = df.sort_index()

    commit_key = get_commit_key()
    if commit_key not in df.index:
        df.loc[commit_key] = [config1.git.latest_commit] * 2

    return df


def display_changes(df, changed_files):
    print('')
    print('************CHANGED FILES************')
    commit1, commit2 = list(df.loc['git.latest_commit'].values)
    print(commit1, '<==>', commit2)
    print()
    print('\n'.join(changed_files))
    print('')
    print('************CONFIG DIFFERENCE************')

    df = df.drop('git.latest_commit')
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config1', type=str)
    parser.add_argument('config2', type=str)
    args = parser.parse_args()
    assert os.path.exists(args.config1)
    assert os.path.exists(args.config2)
    df, changed_files = compare_config(args.config1, args.config2)
    display_changes(df, changed_files)
