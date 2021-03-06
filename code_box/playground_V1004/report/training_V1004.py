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
load_name = 'train_fillna3'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)

del dt

df = df.drop(['song_count', 'liked_song_count',
              'disliked_song_count', 'artist_count',
              'liked_artist_count', 'disliked_artist_count'], axis=1)

print("Train test and validation sets")

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category')
        # test[col] = test[col].astype('category')

print(df.dtypes)
# train = df.sample(frac=0.6, random_state=5)
# val = df.drop(train.index)
# print('df len: ', len(df))
# del df
X_train = df.drop(['target'], axis=1)
y_train = df['target'].values
# X_val = val.drop(['target'], axis=1)
# Y_val = val['target'].values

# print('train len:', len(train))
# print('val len: ', len(val))
del df
# X_test = test.drop(['id'], axis=1)
# ids = test['id'].values
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train)

# del train, test; gc.collect();

train_set = lgb.Dataset(X_tr, y_tr)
val_set = lgb.Dataset(X_val, y_val)

print('Processed data...')
params = {'objective': 'binary',
          # 'metric': 'binary_logloss',
          'boosting': 'gbdt',
          'learning_rate': 0.02,
          'verbose': 0,
          'num_leaves': 200,
          # 'n_estimators': 10,
          'bagging_fraction': 0.8,
          'bagging_freq': 2,
          'bagging_seed': 1,
          'feature_fraction': 0.8,
          'feature_fraction_seed': 1,
          'max_bin': 256,
          'max_depth': -1,
          'num_rounds': 100000,
          'metric': 'auc',
          'lambda_l2': 0.5,
          # 'device': 'gpu',
          # 'gpu_platform_id': 0,
          # 'gpu_device_id': 0,
         }
model = lgb.train(params, train_set=train_set, early_stopping_rounds=10,
                  valid_sets=val_set, verbose_eval=5)
pickle.dump(model, open(save_dir+'model_V1002.save', "wb"))
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


# 66