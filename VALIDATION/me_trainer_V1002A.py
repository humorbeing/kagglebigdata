import sys
sys.path.insert(0, '../')
from me import *
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import pickle

since = time.time()
result = {}
data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_me_play.csv'

df = read_df(load_name)
show_df(df)
on = [
    'msno',
    'song_id',
    # 'source_system_tab',
    'source_screen_name',
    'source_type',
    'target',
    # 'genre_ids',
    'artist_name',
    'composer',
    'lyricist',
    # 'language',
    'song_year',
    # 'song_country',
    'rc',
    # 'top1_in_song',
    # 'top2_in_song',
    # 'top3_in_song',
    'membership_days',
    'song_year_int',
    # 'ISC_top1_in_song',
    # 'ISC_top2_in_song',
    # 'ISC_top3_in_song',
    # 'ISC_language',
    'ISCZ_rc',
    'ISCZ_isrc_rest',
    'ISC_song_year',
    'song_length_log10',
    # 'ISCZ_genre_ids_log10',
    'ISC_artist_name_log10',
    'ISCZ_composer_log10',
    'ISC_lyricist_log10',
    # 'ISC_song_country_ln',
    'ITC_song_id_log10_1',
    'ITC_source_system_tab_log10_1',
    'ITC_source_screen_name_log10_1',
    'ITC_source_type_log10_1',
    'ITC_artist_name_log10_1',
    'ITC_composer_log10_1',
    'ITC_lyricist_log10_1',
    # 'ITC_song_year_log10_1',
    # 'ITC_top1_in_song_log10_1',
    # 'ITC_top2_in_song_log10_1',
    # 'ITC_top3_in_song_log10_1',
    'ITC_msno_log10_1',
    'OinC_msno',
    # 'ITC_language_log10_1',
    # 'OinC_language',
]
df = df[on]
show_df(df)

num_boost_round = 2000
early_stopping_rounds = 50
verbose_eval = 10

boosting = 'gbdt'

learning_rate = 0.022
num_leaves = 511
max_depth = 31

max_bin = 63
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


train, val = fake_df(df)
del df
model, cols = val_df(
    params, train, val,
    num_boost_round,
    early_stopping_rounds,
    verbose_eval,
    learning_rate=False
)
del train, val
show_mo(model)



print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))