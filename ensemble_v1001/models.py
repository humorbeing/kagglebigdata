import sys
sys.path.insert(0, '../')
from me import *
import pandas as pd
import lightgbm as lgb
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier


def dart_1(
        K, dfs, dfs_collector, test,
        test_collector):

    r = 'dart_1'

    on = [

    ]
    params = {
        'boosting': 'dart',

        'learning_rate': 0.5,
        'num_leaves': 15,
        'max_depth': 5,

        'lambda_l1': 0,
        'lambda_l2': 0,
        'max_bin': 15,

        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'bagging_seed': 2,
        'feature_fraction': 0.8,
        'feature_fraction_seed': 2,
    }

    num_boost_round = 50
    early_stopping_rounds = 50
    verbose_eval = 1
    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i+1, '/', K)
        print()
        b = [i for i in range(K)]
        b.remove(i)
        c = [dfs[b[j]] for j in range(K - 1)]
        dt = pd.concat(c)
        model, cols = val_df(
            params, dt, test,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
        del dt
        dfs_collector[i][r] = model.predict(dfs[i])
        v += model.predict(test)

    test_collector[r] = v / K
    return dfs_collector, test
