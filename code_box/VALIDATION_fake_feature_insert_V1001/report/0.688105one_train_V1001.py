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
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt

# barebone = True
barebone = False
if barebone:
    ccc = [i for i in df.columns]
    ccc.remove('target')
    df.drop(ccc, axis=1, inplace=True)

# must be a fake feature
inner = [
    'FAKE_[]_0.6788_Light_gbdt_1512883008.csv'
]
inner = False


def insert_this(on):
    global df
    on = on[:-4]
    df1 = pd.read_csv('../saves/feature/'+on+'.csv')
    df1.drop('id', axis=1, inplace=True)
    on = on[-10:]
    df1.rename(columns={'target': 'FAKE_'+on}, inplace=True)
    df = df.join(df1)
    del df1


cc = df.drop('target', axis=1)
# print(cc.dtypes)
cols = cc.columns
del cc

counter = {}


def get_count(x):
    try:
        return counter[x]
    except KeyError:
        return 0


def add_this_counter_column(on_in):
    global counter, df
    read_from = '../fake/saves/'
    counter = pickle.load(open(read_from+'counter/'+'ITC_'+on_in+'_dict.save', "rb"))
    df['ITC_'+on_in] = df[on_in].apply(get_count).astype(np.int64)
    # counter = pickle.load(open(read_from + 'counter/' + 'CC11_' + on_in + '_dict.save', "rb"))
    # df['CC11_' + on_in] = df[on_in].apply(get_count).astype(np.int64)
    df.drop(on_in, axis=1, inplace=True)


for col in cols:
    add_this_counter_column(col)


def log10me(x):
    return np.log10(x)


def log10me1(x):
    return np.log10(x+1)


def xxx(x):
    d = x / (x + 1)
    return x


for col in cols:
    colc = 'ITC_'+col
    # df[colc + '_log10'] = df[colc].apply(log10me).astype(np.float64)
    df[colc + '_log10_1'] = df[colc].apply(log10me1).astype(np.float64)
    # df[colc + '_x_1'] = df[colc].apply(xxx).astype(np.float64)
    # col1 = 'CC11_'+col
    # df['OinC_'+col] = df[col1]/df[colc]
    df.drop(colc, axis=1, inplace=True)


load_name = 'train_set'
read_from = '../saves01/'
dt = pickle.load(open(read_from+load_name+'_dict.save', "rb"))
train = pd.read_csv(read_from+load_name+".csv", dtype=dt)
del dt

train.drop(
    [
        'target',
    ],
    axis=1,
    inplace=True
)

df = df.join(train)
del train
if inner:
    for i in inner:
        insert_this(i)

print('What we got:')
print(df.dtypes)
print('number of rows:', len(df))
print('number of columns:', len(df.columns))

num_boost_round = 2000
early_stopping_rounds = 50
verbose_eval = 10

boosting = 'gbdt'

learning_rate = 0.02
num_leaves = 511
max_depth = -1

max_bin = 255
lambda_l1 = 0.2
lambda_l2 = 0


bagging_fraction = 0.9
bagging_freq = 2
bagging_seed = 2
feature_fraction = 0.9
feature_fraction_seed = 2

params = {
    'boosting': boosting,

    'learning_rate': learning_rate,
    'num_leaves': num_leaves,
    'max_depth': max_depth,

    'lambda_l1': lambda_l1,
    'lambda_l2': lambda_l2,
    'max_bin': max_bin,

    'bagging_fraction': bagging_fraction,
    'bagging_freq': bagging_freq,
    'bagging_seed': bagging_seed,
    'feature_fraction': feature_fraction,
    'feature_fraction_seed': feature_fraction_seed,
}
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
# df = df[on]
fixed = [
    'target',
    'msno',
    'song_id',
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'artist_name',
    # 'composer',
    # 'lyricist',
    'song_year',
    'language',
    # 'rc',
    'ITC_song_id_log10_1',

    # 'ITC_source_system_tab_log10_1',
    # 'ITC_source_screen_name_log10_1',
    # 'ITC_source_type_log10_1',
    # 'ITC_artist_name_log10_1',
    # 'FAKE_1512883008',
]
result = {}
for w in df.columns:
    print("'{}',".format(w))

work_on = [

    # 'ITC_composer_log10_1',
    # 'ITC_lyricist_log10_1',
    # 'ITC_language_log10_1',
    'ITC_msno_log10_1',
    # 'ITC_song_year_log10_1',
    # 'ITC_song_country_log10_1',
    # 'ITC_rc_log10_1',
]
for w in work_on:
    if w in fixed:
        pass
    else:
        print('working on:', w)
        toto = [i for i in fixed]
        toto.append(w)
        df_on = df[toto]

        for col in df_on.columns:
            if df_on[col].dtype == object:
                df_on[col] = df_on[col].astype('category')


        print()
        print('Our guest selection:')
        print(df_on.dtypes)
        print('number of columns:', len(df_on.columns))
        print()

        save_me = True
        # save_me = False
        if save_me:
            print('creating train set.')
            save_name = 'train'
            vers = '_me2'
            d = df_on.dtypes.to_dict()
            # print(d)
            print('dtypes of df:')
            print('>' * 20)
            print(df_on.dtypes)
            print('number of columns:', len(df_on.columns))
            print('number of data:', len(df_on))
            print('<' * 20)
            df_on.to_csv(save_dir + save_name + vers + '.csv', index=False)
            pickle.dump(d, open(save_dir + save_name + vers + '_dict.save', "wb"))

            print('done.')


        length = len(df_on)
        train_size = 0.76
        train_set = df_on.head(int(length*train_size))
        val_set = df_on.drop(train_set.index)
        del df_on

        train_set = train_set.sample(frac=1)
        X_tr = train_set.drop(['target'], axis=1)
        Y_tr = train_set['target'].values

        X_val = val_set.drop(['target'], axis=1)
        Y_val = val_set['target'].values

        del train_set, val_set

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

        train_set = lgb.Dataset(
            X_tr, Y_tr,
            # weight=[0.1, 1]
        )
        # train_set.max_bin = max_bin
        val_set = lgb.Dataset(
            X_val, Y_val,
            # weight=[0.1, 1]
        )
        train_set.max_bin = max_bin
        val_set.max_bin = max_bin

        del X_tr, Y_tr, X_val, Y_val

        params['metric'] = 'auc'
        params['verbose'] = -1
        params['objective'] = 'binary'

        print('Training...')

        model = lgb.train(params,
                          train_set,
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          valid_sets=[train_set, val_set],
                          verbose_eval=verbose_eval,
                          )

        print('best score:', model.best_score['valid_1']['auc'])
        print('best iteration:', model.best_iteration)
        del train_set, val_set
        print('complete on:', w)
        result[w] = model.best_score['valid_1']['auc']
        print()


import operator
sorted_x = sorted(result.items(), key=operator.itemgetter(1))
# reversed(sorted_x)
# print(sorted_x)
for i in sorted_x:
    name = i[0] + ':  '
    name = name.rjust(40)
    name = name + str(i[1])
    print(name)

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION_fake_feature_insert_V1001/one_train_V1001.py
What we got:
target                               uint8
ITC_msno_log10_1                   float64
ITC_song_id_log10_1                float64
ITC_source_system_tab_log10_1      float64
ITC_source_screen_name_log10_1     float64
ITC_source_type_log10_1            float64
ITC_gender_log10_1                 float64
ITC_artist_name_log10_1            float64
ITC_composer_log10_1               float64
ITC_lyricist_log10_1               float64
ITC_language_log10_1               float64
ITC_name_log10_1                   float64
ITC_song_year_log10_1              float64
ITC_song_country_log10_1           float64
ITC_rc_log10_1                     float64
ITC_isrc_rest_log10_1              float64
ITC_top1_in_song_log10_1           float64
ITC_top2_in_song_log10_1           float64
ITC_top3_in_song_log10_1           float64
msno                                object
song_id                             object
source_system_tab                   object
source_screen_name                  object
source_type                         object
expiration_month                  category
genre_ids                           object
artist_name                         object
composer                            object
lyricist                            object
language                          category
name                                object
song_year                         category
song_country                      category
rc                                category
isrc_rest                         category
top1_in_song                      category
top2_in_song                      category
top3_in_song                      category
dtype: object
number of rows: 7377418
number of columns: 38
'target',
'ITC_msno_log10_1',
'ITC_song_id_log10_1',
'ITC_source_system_tab_log10_1',
'ITC_source_screen_name_log10_1',
'ITC_source_type_log10_1',
'ITC_gender_log10_1',
'ITC_artist_name_log10_1',
'ITC_composer_log10_1',
'ITC_lyricist_log10_1',
'ITC_language_log10_1',
'ITC_name_log10_1',
'ITC_song_year_log10_1',
'ITC_song_country_log10_1',
'ITC_rc_log10_1',
'ITC_isrc_rest_log10_1',
'ITC_top1_in_song_log10_1',
'ITC_top2_in_song_log10_1',
'ITC_top3_in_song_log10_1',
'msno',
'song_id',
'source_system_tab',
'source_screen_name',
'source_type',
'expiration_month',
'genre_ids',
'artist_name',
'composer',
'lyricist',
'language',
'name',
'song_year',
'song_country',
'rc',
'isrc_rest',
'top1_in_song',
'top2_in_song',
'top3_in_song',
working on: ITC_msno_log10_1
/media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION_fake_feature_insert_V1001/one_train_V1001.py:220: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  df_on[col] = df_on[col].astype('category')

Our guest selection:
target                    uint8
msno                   category
song_id                category
source_system_tab      category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
language               category
ITC_song_id_log10_1     float64
ITC_msno_log10_1        float64
dtype: object
number of columns: 11

creating train set.
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target                    uint8
msno                   category
song_id                category
source_system_tab      category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
language               category
ITC_song_id_log10_1     float64
ITC_msno_log10_1        float64
dtype: object
number of columns: 11
number of data: 7377418
<<<<<<<<<<<<<<<<<<<<
done.
train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:648: LGBMDeprecationWarning: The `max_bin` parameter is deprecated and will be removed in 2.0.12 version. Please use `params` to pass this parameter.
  'Please use `params` to pass this parameter.', LGBMDeprecationWarning)
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.795762	valid_1's auc: 0.66735
[20]	training's auc: 0.799679	valid_1's auc: 0.669626
[30]	training's auc: 0.80278	valid_1's auc: 0.670856
[40]	training's auc: 0.804122	valid_1's auc: 0.671351
[50]	training's auc: 0.808085	valid_1's auc: 0.673009
[60]	training's auc: 0.810563	valid_1's auc: 0.673764
[70]	training's auc: 0.813383	valid_1's auc: 0.674857
[80]	training's auc: 0.815013	valid_1's auc: 0.67545
[90]	training's auc: 0.818082	valid_1's auc: 0.676836
[100]	training's auc: 0.82107	valid_1's auc: 0.678073
[110]	training's auc: 0.82341	valid_1's auc: 0.678918
[120]	training's auc: 0.825949	valid_1's auc: 0.679953
[130]	training's auc: 0.828426	valid_1's auc: 0.681042
[140]	training's auc: 0.830694	valid_1's auc: 0.681911
[150]	training's auc: 0.832809	valid_1's auc: 0.682673
[160]	training's auc: 0.8347	valid_1's auc: 0.68324
[170]	training's auc: 0.836603	valid_1's auc: 0.683921
[180]	training's auc: 0.838343	valid_1's auc: 0.68446
[190]	training's auc: 0.839966	valid_1's auc: 0.684972
[200]	training's auc: 0.841622	valid_1's auc: 0.685492
[210]	training's auc: 0.84288	valid_1's auc: 0.685819
[220]	training's auc: 0.844232	valid_1's auc: 0.686249
[230]	training's auc: 0.845485	valid_1's auc: 0.686521
[240]	training's auc: 0.846585	valid_1's auc: 0.686788
[250]	training's auc: 0.847571	valid_1's auc: 0.686973
[260]	training's auc: 0.848599	valid_1's auc: 0.687162
[270]	training's auc: 0.849595	valid_1's auc: 0.687355
[280]	training's auc: 0.850467	valid_1's auc: 0.687523
[290]	training's auc: 0.851225	valid_1's auc: 0.687583
[300]	training's auc: 0.852089	valid_1's auc: 0.687706
[310]	training's auc: 0.852789	valid_1's auc: 0.687774
[320]	training's auc: 0.853587	valid_1's auc: 0.687848
[330]	training's auc: 0.854223	valid_1's auc: 0.687889
[340]	training's auc: 0.854841	valid_1's auc: 0.687966
[350]	training's auc: 0.855514	valid_1's auc: 0.688012
[360]	training's auc: 0.856135	valid_1's auc: 0.688053
[370]	training's auc: 0.856821	valid_1's auc: 0.688061
[380]	training's auc: 0.857419	valid_1's auc: 0.688064
[390]	training's auc: 0.857989	valid_1's auc: 0.688076
[400]	training's auc: 0.858586	valid_1's auc: 0.688037
[410]	training's auc: 0.859153	valid_1's auc: 0.688066
[420]	training's auc: 0.859716	valid_1's auc: 0.688099
[430]	training's auc: 0.860231	valid_1's auc: 0.688086
[440]	training's auc: 0.860764	valid_1's auc: 0.688087
[450]	training's auc: 0.861201	valid_1's auc: 0.688071
[460]	training's auc: 0.861656	valid_1's auc: 0.688074
Early stopping, best iteration is:
[413]	training's auc: 0.859358	valid_1's auc: 0.688105
best score: 0.68810488969
best iteration: 413
complete on: ITC_msno_log10_1

                     ITC_msno_log10_1:  0.68810488969

[timer]: complete in 44m 52s
'''