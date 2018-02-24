# -*- coding: utf-8 -*-

import glob

import numpy as np
import pandas as pd


# Все завязано на конкретный формат файла. При необходимости - менять функцию
def read_file(filename):
    # system:index,blue,blue_1,current_slice,green,green_1,id,nir,nir_1,red,red_1,slice,swir1,swir1_1,swir2,swir2_1,tree_canopy_cover,uncertainty,.geo

    data = pd.read_csv(
        filename,
        delimiter=',',
    )

    # Delete old changes from sample:
    data = data.loc[
        ~(
            (data['current_slice'] > data['slice']) &
            (data['slice'] >= 0)
        )
    ]

    data['blue_1'] = data['blue_1'] / 10000.0
    data['green_1'] = data['green_1'] / 10000.0
    data['red_1'] = data['red_1'] / 10000.0
    data['nir_1'] = data['nir_1'] / 10000.0
    data['swir1_1'] = data['swir1_1'] / 10000.0
    data['swir2_1'] = data['swir2_1'] / 10000.0
    data['change'] = (data['current_slice'] == data['slice']).astype(int)

    names = [
        "id", "tree_canopy_cover", "current_slice", "slice",
        "blue", "blue_1",
        "green", "green_1",
        "red", "red_1",
        "nir", "nir_1",
        "swir1", "swir1_1",
        "swir2", "swir2_1",
    ]
    return data[list(names) + ['change']]


def get_data(pattern):
    fnames = glob.glob(pattern)
    df_lst = [read_file(f) for f in fnames]

    return pd.concat(df_lst)


def split_data(data, train_val_test=(0.66, 0.17, 0.17), seed=0):
    np.random.seed(seed)
    ids = set(data['id'])
    count = len(ids)

    train_count = int(count * train_val_test[0])
    val_count = int(count * train_val_test[1])
    # test_count = int(count*train_val_test[2])

    train = np.random.choice(list(ids), train_count, False)

    test_val_ids = ids.difference(set(train))
    val = np.random.choice(list(test_val_ids), val_count, False)

    test = np.array(list(test_val_ids.difference(set(val))))

    train = data[data['id'].isin(train)]
    val = data[data['id'].isin(val)]
    test = data[data['id'].isin(test)]

    names = [  # "current_slice",
        "blue", "blue_1",
        "green", "green_1",
        "red", "red_1",
        "nir", "nir_1",
        "swir1", "swir1_1",
        "swir2", "swir2_1", "change"]

    train = np.array(train[names])
    val = np.array(val[names])
    test = np.array(test[names])

    # np.random.shuffle(train)
    # np.random.shuffle(val)
    # np.random.shuffle(test)

    return train, val, test