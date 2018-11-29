# encoding: utf-8

import numpy as np



# Все завязано на конкретный формат файла. При необходимости - менять функцию
def prepare_data(data):
    # ['system:index', 'name', 'date', 'tree_canopy_cover', 'swir2', 'swir1', 'nir', 'red', 'green', 'blue', 'NDVI', 'NDSI', 'NDWI', 'realmed_swir2', 'realmed_swir1', 'realmed_nir', 'realmed_red', 'realmed_green', 'realmed_blue', 'realmed_NDVI', 'realmed_NDSI', 'realmed_NDWI']

    
    bands = [
        'swir2', 'swir1', 'nir', 'red', 'green', 'blue',
        'realmed_swir2', 'realmed_swir1', 'realmed_nir', 'realmed_red', 'realmed_green', 'realmed_blue'
    ]
    
    indices = [
        'NDVI', 'NDSI', 'NDWI', 
        'realmed_NDVI', 'realmed_NDSI', 'realmed_NDWI'
    ]
    
    inputs = bands + indices
    
        
    # В system:index содержится ID полигона (у нас много точек на полигон), чтобы избежать оптимистичной оценки из-за автокорреляции
    # выделим номер полигона и сохраним
    data['system:index'] = data['system:index'].str.extract(r'(.+)_\d+$')
    
    data['change'] = (data['name'] == 'Change').astype(int)

    
    addons = ['system:index', 'dayNumber', 'tree_canopy_cover', 'change']
    # addons = ['system:index', 'change']
        
    # names = set(inputs) - set(['name'])
    names = inputs + addons
    # print(names)
    return data[names]
    
def drop_N_groups(data, N, group='system:index', seed=42):  
    np.random.seed(seed)
    df = data.copy()
    grps = data[group].unique()
    drop = np.random.choice(grps, N, replace=False)

    count = len(grps) - N    
    
    for g in drop:
        df = df.drop(df[df[group] == g].index)
        
    return (df, count)
    
def split_x_y_group(data, group='system:index'):
    grp = np.array(data[group].factorize()[0] + 1)
    data = np.array(data.drop(labels=[group], axis=1)) 
    return data[:, :-1], data[:, -1], grp
    
    
def split_data(data, train_val_test=0.66, group='system:index', seed=42):
    np.random.seed(seed)
    ids = set(data[group])
    count = len(ids)
    print('Found %s unique areas' % (count))

    train_count = int(count * train_val_test)
    print('Used %s unique areas for train' % (train_count))
    
  
    train = np.random.choice(list(ids), train_count, False)

    val_ids = ids.difference(set(train))
    val = np.array(list(val_ids))
  
    train = data[data[group].isin(train)]
    val = data[data[group].isin(val)]
  
    train = np.array(train.drop(labels=[group], axis=1))
    val = np.array(val.drop(labels=[group], axis=1))

    return train, val
  


