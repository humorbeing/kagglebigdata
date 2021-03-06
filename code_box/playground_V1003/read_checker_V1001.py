import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'
dt = pickle.load(open(save_dir+"custom_member_dict.save", "rb"))
df = pd.read_csv(save_dir+"custom_members_fixed.csv",
                 dtype=dt)
print(dt)
print(df.dtypes)
print(df.head())
print(len(df.columns))
print(len(df))
ddd = df.dtypes.to_dict()
for i in ddd:
    on = i
    print('inspecting:', on)
    print('>' * 20)
    print('any null:', df[on].isnull().values.any())
    print('null number:', df[on].isnull().values.sum())
    print('<'*20)
    print()

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

