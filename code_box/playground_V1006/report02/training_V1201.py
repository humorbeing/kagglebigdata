import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import time
import pickle
from sklearn.model_selection import train_test_split

since = time.time()

data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_set'
# dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
# df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
#
# del dt
# print('What we got:')
# print(df.dtypes)
# print('number of columns:', len(df.columns))
# print(type(df.head()))
# df = df.drop(['song_count', 'liked_song_count',
#               'disliked_song_count', 'artist_count',
#               'liked_artist_count', 'disliked_artist_count'], axis=1)
# df = df[['mn', 'sn', 'target']]
# df = df[['msno', 'song_id', 'language', 'target']]
# df['language'] = df['language'].astype('category')
working_on = ['source_system_tab',
              'source_screen_name',
              'source_type',
              'genre_ids',
              'composer',
              'lyricist',
              'rc',
              ]
for w in working_on:
    dt = pickle.load(open(save_dir + load_name + '_dict.save', "rb"))
    df = pd.read_csv(save_dir + load_name + ".csv", dtype=dt)

    del dt
    print('working on:', w)
    df = df[['msno', 'song_id', w, 'target']]
    # df['age_range'] = df['age_range'].astype('category')
    # df = df[['city', 'age', 'target']]
    print("Train test and validation sets")

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype('category')
            # test[col] = test[col].astype('category')

    print()
    print()
    print('After selection:')
    print(df.dtypes)
    print('number of columns:', len(df.columns))
    print()
    print()
    length = len(df)
    train_size = 0.76
    train_set = df.head(int(length*train_size))
    val_set = df.drop(train_set.index)
    # print(train_set.head(100))
    # print(len(train_set))
    # print(len(val_set))
    del df
    train_set = train_set.sample(frac=1)
    X_tr = train_set.drop(['target'], axis=1)
    Y_tr = train_set['target'].values

    X_val = val_set.drop(['target'], axis=1)
    Y_val = val_set['target'].values

    del train_set, val_set
    # X_test = test.drop(['id'], axis=1)
    # ids = test['id'].values
    # X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train,
    #                                             train_size=0.000001,
    #                                             shuffle=True,
    #                                             random_state=555,
    #                                             )
    t = len(Y_tr)
    t1 = sum(Y_tr)
    t0 = t - t1
    print('train size:', t, 'number of 1:', t1, 'number of 0:', t0)
    print('train: 1 in all:', t1/t, '0 in all:', t0/t, '1/0:', t1/t0)
    t = len(Y_val)
    t1 = sum(Y_val)
    t0 = t - t1
    print('val size:', t, 'number of 1:', t1, 'number of 0:', t0)
    print('val: 1 in all:', t1/t, '0 in all:', t0/t, '1/0:', t1/t0)
    print()
    print()
    # del X_train, Y_train

    train_set = lgb.Dataset(X_tr, Y_tr)
    val_set = lgb.Dataset(X_val, Y_val)
    del X_tr, Y_tr, X_val, Y_val

    # train_set = lgb.Dataset(X_train, Y_train,
    #                         categorical_feature=[0, 1],
    #                         )
    print('Training...')
    params = {'objective': 'binary',
              'metric': 'auc',
              # 'metric': 'binary_logloss',
              'boosting': 'gbdt',
              'learning_rate': 0.1,
              # 'verbosity': -1,
              'verbose': -1,
              'num_leaves': 100,

              'bagging_fraction': 0.8,
              'bagging_freq': 2,
              'bagging_seed': 1,
              'feature_fraction': 0.8,
              'feature_fraction_seed': 1,
              'max_bin': 63,
              'max_depth': -1,
              # 'min_data': 500,
              # 'min_hessian': 0.05,
              # 'num_rounds': 500,
              # "min_data_in_leaf": 1,
              # 'min_data': 1,
              # 'min_data_in_bin': 1,
              # 'lambda_l2': 0.5,
              # 'device': 'gpu',
              # 'gpu_platform_id': 0,
              # 'gpu_device_id': 0,
              # 'sparse_threshold': 1.0,
              # 'categorical_feature': (0,1,2,3),
              }
    model = lgb.train(params,
                      train_set,
                      num_boost_round=50000,
                      early_stopping_rounds=200,
                      valid_sets=val_set,
                      verbose_eval=10,
                      )
    # model_name = 'model_V1001'
    # pickle.dump(model, open(save_dir+model_name+'.save', "wb"))
    # print('model saved as: ', save_dir, model_name)
    del train_set, val_set
    print('complete on:', w)
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 "/media/ray/SSD/workspace/python/projects/big data kaggle/playground_V1006/one_round_train_V1001.py"
What we got:
msno                        object
song_id                     object
source_system_tab           object
source_screen_name          object
source_type                 object
target                       uint8
city                         uint8
registered_via               uint8
mn                           int64
age                           int8
age_range                     int8
membership_days              int64
membership_days_range         int8
registration_year            int64
registration_month           int64
registration_date            int64
expiration_year              int64
expiration_month             int64
expiration_date              int64
sex                           int8
sex_guess                     int8
song_length                  int64
genre_ids                   object
artist_name                 object
composer                    object
lyricist                    object
language                      int8
sn                           int64
lyricists_count               int8
composer_count                int8
genre_ids_count               int8
length_range                 int64
length_bin_range             int64
length_chunk_range           int64
song_year                    int64
song_year_bin_range          int64
song_year_chunk_range        int64
song_country                object
rc                          object
artist_composer               int8
artist_composer_lyricist      int8
song_count                   int64
liked_song_count             int64
disliked_song_count          int64
artist_count                 int64
liked_artist_count           int64
disliked_artist_count        int64
dtype: object
number of columns: 47
working on: source_system_tab
Train test and validation sets


After selection:
msno                 category
song_id              category
source_system_tab    category
target                  uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.640689
[20]	valid_0's auc: 0.645246
[30]	valid_0's auc: 0.648598
[40]	valid_0's auc: 0.651341
[50]	valid_0's auc: 0.652804
[60]	valid_0's auc: 0.654287
[70]	valid_0's auc: 0.655244
[80]	valid_0's auc: 0.655612
[90]	valid_0's auc: 0.655927
[100]	valid_0's auc: 0.656238
[110]	valid_0's auc: 0.656436
[120]	valid_0's auc: 0.65664
[130]	valid_0's auc: 0.656649
[140]	valid_0's auc: 0.656575
[150]	valid_0's auc: 0.656556
[160]	valid_0's auc: 0.656592
[170]	valid_0's auc: 0.656576
[180]	valid_0's auc: 0.656482
[190]	valid_0's auc: 0.656372
[200]	valid_0's auc: 0.65635
[210]	valid_0's auc: 0.656281
[220]	valid_0's auc: 0.656283
[230]	valid_0's auc: 0.656277
[240]	valid_0's auc: 0.656223
[250]	valid_0's auc: 0.65615
[260]	valid_0's auc: 0.656109
[270]	valid_0's auc: 0.656007
[280]	valid_0's auc: 0.65599
[290]	valid_0's auc: 0.655936
[300]	valid_0's auc: 0.655917
[310]	valid_0's auc: 0.655851
[320]	valid_0's auc: 0.655809
Early stopping, best iteration is:
[128]	valid_0's auc: 0.656662
complete on: source_system_tab
working on: source_screen_name
Traceback (most recent call last):
  File "/media/ray/SSD/workspace/python/projects/big data kaggle/playground_V1006/one_round_train_V1001.py", line 40, in <module>
    df = df[['msno', 'song_id', w, 'target']]
NameError: name 'df' is not defined

Process finished with exit code 1
'''