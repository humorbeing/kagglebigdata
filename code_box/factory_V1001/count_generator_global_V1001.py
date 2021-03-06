import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
train = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt
train.drop(['target'], axis=1, inplace=True)
load_name = 'test_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
test = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt
test.drop(['id'], axis=1, inplace=True)

df = pd.concat([train, test])
del train, test

print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of rows:', len(df))
print('number of columns:', len(df.columns))
print('<'*20)


song_count = {k: v for k, v in df['song_id'].value_counts().iteritems()}
pickle.dump(song_count, open(save_dir+'total_song_count_dict.save', "wb"))
# pickle.dump(song_count, open(save_dir+'liked_song_count_dict.save', "wb"))
# pickle.dump(song_count, open(save_dir+'disliked_song_count_dict.save', "wb"))
del song_count
artist_count = {k: v for k, v in df['artist_name'].value_counts().iteritems()}
pickle.dump(artist_count, open(save_dir+'total_artist_count_dict.save', "wb"))
# pickle.dump(artist_count, open(save_dir+'liked_artist_count_dict.save', "wb"))
# pickle.dump(artist_count, open(save_dir+'disliked_artist_count_dict.save', "wb"))
del artist_count
member_count = {k: v for k, v in df['msno'].value_counts().iteritems()}
pickle.dump(member_count, open(save_dir+'total_member_count_dict.save', "wb"))
# pickle.dump(member_count, open(save_dir+'liked_member_count_dict.save', "wb"))
# pickle.dump(member_count, open(save_dir+'disliked_member_count_dict.save', "wb"))
del member_count

language_count = {k: v for k, v in df['language'].value_counts().iteritems()}
pickle.dump(language_count, open(save_dir+'total_language_count_dict.save', "wb"))
# pickle.dump(language_count, open(save_dir+'liked_language_count_dict.save', "wb"))
# pickle.dump(language_count, open(save_dir+'disliked_language_count_dict.save', "wb"))
del language_count


on = False
# on = 'target'
# on = 'source_system_tab'
# on = 'source_screen_name'
# on = 'source_type'
# on = 'count_song_played'
# on = 'count_artist_played'
# on = 'unliked_count_song_played'


print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('<'*20)
if on:
    print('inspecting:', on)
    print('>'*20)
    print('any null:', df[on].isnull().values.any())
    print('null number:', df[on].isnull().values.sum())
    print(on, 'dtype:', df[on].dtypes)
    print('describing', on, ':')
    print(df[on].describe())
    print('<'*20)
    l = df[on]
    s = set(l)
    print('list len:', len(l))
    print('set len:', len(s))
    # # print(s)
    print('<'*20)
# check_all = True
check_all = False
if check_all:
    ddd = df.dtypes.to_dict()
    for i in ddd:
        on = i
        print('inspecting:', on)
        print('>' * 20)
        print('any null:', df[on].isnull().values.any())
        print('null number:', df[on].isnull().values.sum())
        print('<'*20)
        print()

# plot = True
plot = False
if plot:
    plt.figure(figsize=(15, 12))
    # dff = pd.DataFrame()
    # dff[on] = df[on].dropna()
    # del df
    sns.distplot(df[on])
    # sns.countplot(df[on])
    plt.show()
# _dict_count_song_played_train = {k: v for k, v in df['song_id'].value_counts().iteritems()}
# for i in _dict_count_song_played_train:
#     print(i, ':', _dict_count_song_played_train[i])

plt.show()
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

