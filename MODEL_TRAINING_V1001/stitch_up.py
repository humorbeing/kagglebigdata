import sys
sys.path.insert(0, '../')
from me import *
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import pickle

since = time.time()

data_dir = '../data/'
save_dir = '../saves/'
read_from='../saves/submission/'
model_name = 'stitch_up_'
load_name = '0.68697.csv'
model_name += load_name[:6]+'_'
df1 = pd.read_csv(read_from + load_name)
load_name = '0.68624.csv'
model_name += load_name[:6]+'_'
df2 = pd.read_csv(read_from + load_name)
load_name = '0.68750stitch_up_0.6869_0.6862_0.6836_.csv'
model_name += load_name[:6]+'_'
df3 = pd.read_csv(read_from + load_name)
# load_name = ''
# df4 = read_df(load_name,read_from='../saves/submission/')
# load_name = ''
# df5 = read_df(load_name,read_from='../saves/submission/')

# show_df(df1)
show_df(df2)
show_df(df3)
# p = df1['target']
p = np.zeros(shape=[len(df2)])
p += df1['target']
p += df2['target']
p += df3['target']
p = p / 3

print(df1.head())
print(df2.head())
print(df3.head())
df = pd.DataFrame()
df['id'] = df2.id
df['target'] = p


print('-'*30)
print(df.head())

df.to_csv(save_dir+'submission/'+model_name+'.csv',
                index=False, float_format='%.5f')
print('[complete] submission name:', model_name+'.csv.')

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))