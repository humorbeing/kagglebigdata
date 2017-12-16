import sys
sys.path.insert(0, '../')
from me import *
from fake_L_lvl2 import *
import pandas as pd
import lightgbm as lgb
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier



since = time.time()
K = 3
print()
print('This is [no drill] training.')
print()
data_dir = '../data/'
save_dir = '../saves/'
load_name = 'final_train_play.csv'
read_from = '../fake/saves/feature/level1/'

dfs, test = merge_fake()

show_df(test)

dfs_collector = []
for i in range(K):
    dc = pd.DataFrame()
    dc['target'] = dfs[i]['target']
    dfs_collector.append(dc)

test_collector = pd.DataFrame()
test_collector['target'] = test['target']


# !!!!!!!!!!!!!!!!!!!!!!!!!

# dfs_collector, test_collector, r = Ldrt_top2_1(
#     K, dfs, dfs_collector, test, test_collector
# )
#
from sklearn.metrics import roc_auc_score

dfs_collector, test_collector, r = Lgbt_top2_1(
    K, dfs, dfs_collector, test, test_collector
)
print(roc_auc_score(test['target'], test_collector[r]))
dfs_collector, test_collector, r = Lgos_top2_1(
    K, dfs, dfs_collector, test, test_collector
)

print(roc_auc_score(test['target'], test_collector[r]))


dfs_collector, test_collector, r = Lrf_top2_1(
    K, dfs, dfs_collector, test, test_collector
)
print(roc_auc_score(test['target'], test_collector[r]))
#
# dfs_collector, test_collector, r = Lgbt_top2_1(
#     K, dfs, dfs_collector, test, test_collector
# )
#
# #-----------------------------
#
dfs_collector, test_collector, r = Ldrt_top2_2(
    K, dfs, dfs_collector, test, test_collector
)
print(roc_auc_score(test['target'], test_collector[r]))


dfs_collector, test_collector, r = Lgos_top2_2(
    K, dfs, dfs_collector, test, test_collector
)
print(roc_auc_score(test['target'], test_collector[r]))



dfs_collector, test_collector, r = Lrf_top2_2(
    K, dfs, dfs_collector, test, test_collector
)
print(roc_auc_score(test['target'], test_collector[r]))



dfs_collector, test_collector, r = Lgbt_top2_2(
    K, dfs, dfs_collector, test, test_collector
)

print(roc_auc_score(test['target'], test_collector[r]))

# !!!!!!!!!!!!!!!!!!!!!!!!!

print(test_collector.head())
print(test_collector.tail())
save_name = 'L_lvl2'
save_here = '../fake/saves/feature/level2/'
for i in range(K):
    save_train = save_here + 'train' + str(i+1) + '/'
    save_df(dfs_collector[i], name=save_name,
            save_to=save_train)

save_df(test_collector, name=save_name,
            save_to=save_here+'test/')


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


