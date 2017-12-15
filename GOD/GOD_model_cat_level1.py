import sys
sys.path.insert(0, '../')
from me import *
import pandas as pd
import lightgbm as lgb
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier
from sklearn import linear_model
# import h2o
#
# from h2o.estimators.random_forest import H2ORandomForestEstimator
# from h2o.estimators.gbm import H2OGradientBoostingEstimator
# from h2o.estimators.deeplearning import H2ODeepLearningEstimator
# from h2o.estimators.glm import H2OGeneralizedLinearEstimator

on_top2 = [
    'msno',
    'song_id',
    'source_screen_name',
    'source_type',
    'target',
    'artist_name',
    'song_year',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    # ------------------
    'top2_in_song',
    # 'language',
    # 'top3_in_song',

    # ------------------
    'source_system_tab',
    # 'ITC_source_system_tab_log10_1',
    # 'ISC_song_country_ln',

    # ------------------
    # 'membership_days',
    # 'ISC_song_year',
    # 'OinC_language',
]
on_language = [
    'msno',
    'song_id',
    'source_screen_name',
    'source_type',
    'target',
    'artist_name',
    'song_year',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    # ------------------
    # 'top2_in_song',
    'language',
    # 'top3_in_song',

    # ------------------
    'source_system_tab',
    # 'ITC_source_system_tab_log10_1',
    # 'ISC_song_country_ln',

    # ------------------
    # 'membership_days',
    # 'ISC_song_year',
    # 'OinC_language',
]
on_sst_c = [
    'msno',
    'song_id',
    'source_screen_name',
    'source_type',
    'target',
    'artist_name',
    'song_year',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    # ------------------
    # 'top2_in_song',
    # 'language',
    'top3_in_song',

    # ------------------
    # 'source_system_tab',
    'ITC_source_system_tab_log10_1',
    # 'ISC_song_country_ln',

    # ------------------
    # 'membership_days',
    # 'ISC_song_year',
    # 'OinC_language',
]
on_num = [
    # 'msno',
    # 'song_id',
    'source_screen_name',
    'source_type',
    'target',
    # 'artist_name',
    # 'song_year',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    # ------------------
    # 'top2_in_song',
    # 'language',
    # 'top3_in_song',

    # ------------------
    # 'source_system_tab',
    'ITC_source_system_tab_log10_1',
    'ISC_song_country_ln',

    # ------------------
    # 'membership_days',
    'ISC_song_year',
    'OinC_language',
]

on_hmm = [
    'msno',
    'song_id',
    'source_screen_name',
    'source_type',
    'target',
    'artist_name',
    'song_year',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    # ------------------
    # 'top2_in_song',
    # 'language',
    'top3_in_song',

    # ------------------
    # 'source_system_tab',
    'ITC_source_system_tab_log10_1',
    'ISC_song_country_ln',

    # ------------------
    'membership_days',
    # 'ISC_song_year',
    'OinC_language',
]

def cat_on_top2(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'cat_1'

    on = [

    ]
    iterations = 300
    learning_rate = 0.3
    depth = 6
    early_stop = 40
    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i + 1, '/', K)
        print()
        b = [i for i in range(K)]
        b.remove(i)
        c = [dfs[b[j]] for j in range(K - 1)]
        dt = pd.concat(c)
        model, cols = cat(
            dt[on_top2], test, iterations=iterations,
            learning_rate=learning_rate, depth=depth,
            early_stop=early_stop,
        )
        del dt
        print('- ' * 10)
        for c in cols:
            print("'{}',".format(c))
        print('- ' * 10)
        dfs_collector[i][r] = model.predict_proba(dfs[i][cols])[:, 1]
        print(dfs_collector[i].head())
        # dfs_collector[i][r+'a'] = model.predict_proba(dfs[i][cols])[:, 1]
        # print(dfs_collector[i].head())
        v += model.predict_proba(test[cols])[:, 1]

    test_collector[r] = v / K
    return dfs_collector, test_collector, r


