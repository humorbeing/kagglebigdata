import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import time
import pickle


print()
print('This is [no drill] testing program.')
print()
since = time.time()
# print_time = str(int(time.time()))
data_dir = '../data/'
save_dir = '../saves/'

# !!!!!!!!!!!!!!!!!!!!!!!!!!!

model_name = '[]_0.6246_Light_gbdt_1512859793.model'

# !!!!!!!!!!!!!!!!!!!!!!!!!!!

print('loading model...')

model_name = model_name[:-6]
model = pickle.load(open(save_dir + 'model/' + model_name + '.model', "rb"))

print('loading complete.')

is_train = True
is_test = True


on = False

# on = [
#     'msno',
#     'song_id',
#     'target',
#     'source_system_tab',
#     'source_screen_name',
#     'source_type',
#     'language',
#     'artist_name',
#     'song_count',
#     'member_count',
#     'song_year',
# ]


if is_train:
    print('making new feature phase:')
    print('loading train set.')
    load_name = 'train_set'
    dt = pickle.load(open(save_dir + load_name + '_dict.save', "rb"))
    df = pd.read_csv(save_dir + load_name + ".csv", dtype=dt)
    del dt

    print('What we got:')
    print(df.dtypes)
    print('number of rows:', len(df))
    print('number of columns:', len(df.columns))

    if on:
        df = df[on]

    df.drop('target', axis=1, inplace=True)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype('category')

    print()
    print('on making feature:')
    print(df.dtypes)
    print('number of columns:', len(df.columns))
    print()

    X_test = df
    ids = df.index
    del df

    print('Making predictions...')

    p_test_1 = model.predict(X_test)

    if not is_test:
        del model

    print('prediction done.')
    print('creating new feature')
    subm = pd.DataFrame()
    subm['id'] = ids
    del ids
    subm['target'] = p_test_1
    del p_test_1
    subm.to_csv(save_dir + 'feature/' + model_name + '.csv',
                index=False, float_format='%.5f')
    print('[complete] featuring, name:', model_name + '.csv.gz')

if is_test:

    print('in test phase:')
    print('loading test set.')
    load_name = 'test_set'
    dt = pickle.load(open(save_dir + load_name + '_dict.save', "rb"))
    df = pd.read_csv(save_dir + load_name + ".csv", dtype=dt)
    del dt
    print('What we got:')
    print(df.dtypes)
    print('number of rows:', len(df))
    print('number of columns:', len(df.columns))

    if on:
        on.remove('target')
        on.append('id')
        df = df[on]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype('category')

    print()
    print('on test:')
    print(df.dtypes)
    print('number of columns:', len(df.columns))
    print()

    X_test = df.drop(['id'], axis=1)
    ids = df['id'].values
    del df

    print('Making predictions...')

    p_test_1 = model.predict(X_test)
    del model

    print('prediction done.')
    print('creating submission')
    subm = pd.DataFrame()
    subm['id'] = ids
    del ids
    subm['target'] = p_test_1
    del p_test_1
    subm.to_csv(save_dir+'submission/'+model_name+'.csv',
                index=False, float_format='%.5f')
    print('[complete] submission name:', model_name+'.csv.gz')

print()
print('test program complete.')

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


