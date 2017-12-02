import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'


load_name = 'custom_song_fixed'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt


def fix_language(x):
    if x == -1.0:
        return 1
    elif x == 3.0:
        return 2
    elif x == 10.0:
        return 3
    elif x == 17.0:
        return 4
    elif x == 24.0:
        return 5
    elif x == 31.0:
        return 6
    elif x == 38.0:
        return 7
    elif x == 45.0:
        return 8
    elif x == 52.0:
        return 9
    elif x == 59.0:
        return 10
    else:
        return 6


def length_range(x):
    n = 5000
    split = np.linspace(185, 12174000, n)
    its = 1
    while True:
        if its == n:
            return 0
        else:
            if x < int(split[its]):
                return its
            else:
                its += 1


def length_bin_range(x):
    if x < 500000:
        return 1
    else:
        return 0


def length_chunk_range(x):

    n = 10
    split = np.linspace(185, 500000, n)
    its = 1
    while True:
        if its == n:
            return 0
        else:
            if x < int(split[its]):
                return its
            else:
                its += 1


# def isrc_to_year(isrc):
#     if type(isrc) == str:
#         if int(isrc[5:7]) > 17:
#             return 1900 + int(isrc[5:7])
#         else:
#             return 2000 + int(isrc[5:7])
#     else:
#         # print('here')
#         a = np.random.poisson(2016, 500)
#         a = int(np.mean(a))
#         if a > 2016:
#             a = int(np.random.uniform(1918, 2017))
#         return a


def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan


def song_year_bin_range(x):
    if x < 1980:
        return 0
    else:
        return 1


def song_year_chunk_range(x):
    if x < 1999:
        return 0
    else:
        return x


def isrc_to_c(isrc):
    if type(isrc) == str:
        return isrc[0:2]
    else:
        return 'US'


def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        a = sum(map(x.count, ['|', '/', '\\', ';'])) + 1
        # if a > 1:
        #     print(x)
        return a


def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1


def genre_id_count(x):
    # global kinds
    if x == 'no_genre_id':
        return 0
    else:
        a = x.count('|') + 1
        # if a == 1:
        #     kinds.add(x)
        # else:
        #     for i in x.split('|'):
        #         kinds.add(i)
        return a


def isrc_to_rc(isrc):
    if type(isrc) == str:
        return isrc[2:5]
    else:
        return 'no_rc'


count = {}
# count1 = {}
# count2 = {}


def get_count1(x):
    try:
        return count[x]
    except KeyError:
        return 1


def get_count(x):
    try:
        return count[x]
    except KeyError:
        return 0
#
#
# storage = 'storage/'
# count = pickle.load(open(save_dir + storage + 'song_count_dict.save', "rb"))
# df['song_count'] = df['song_id'].apply(get_count1).astype(np.int64)
# count = pickle.load(open(save_dir + storage + 'liked_song_count_dict.save', "rb"))
# df['liked_song_count'] = df['song_id'].apply(get_count).astype(np.int64)
# count = pickle.load(open(save_dir + storage + 'disliked_song_count_dict.save', "rb"))
# df['disliked_song_count'] = df['song_id'].apply(get_count).astype(np.int64)
#
#
# count = pickle.load(open(save_dir + storage + 'artist_count_dict.save', "rb"))
# df['artist_count'] = df['artist_name'].apply(get_count1).astype(np.int64)
# count = pickle.load(open(save_dir + storage + 'liked_artist_count_dict.save', "rb"))
# df['liked_artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)
# count = pickle.load(open(save_dir + storage + 'disliked_artist_count_dict.save', "rb"))
# df['disliked_artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)
# del count


# def c1_c2(x):
#     try:
#         c1 = count1[x]
#     except KeyError:
#         c1 = 0
#     try:
#         c2 = count2[x]
#     except KeyError:
#         c2 = 0
#
#     if c1 == 0 and c2 == 0:
#         return 0.5
#     else:
#         if c2 == 0:
#             return 100 * c1
#         else:
#             return c1 / c2
#
#
# count1 = pickle.load(open(save_dir + storage + 'liked_song_count_dict.save', "rb"))
# count2 = pickle.load(open(save_dir + storage + 'song_count_dict.save', "rb"))
# df['like_song_chance'] = df['song_id'].apply(c1_c2).astype(np.float16)
#
# count1 = pickle.load(open(save_dir + storage + 'disliked_song_count_dict.save', "rb"))
# count2 = pickle.load(open(save_dir + storage + 'song_count_dict.save', "rb"))
# df['dislike_song_chance'] = df['song_id'].apply(c1_c2).astype(np.float16)
#
# count1 = pickle.load(open(save_dir + storage + 'liked_song_count_dict.save', "rb"))
# count2 = pickle.load(open(save_dir + storage + 'disliked_song_count_dict.save', "rb"))
# df['song_like_dislike'] = df['song_id'].apply(c1_c2).astype(np.float16)
#
# ###
#
# count1 = pickle.load(open(save_dir + storage + 'liked_artist_count_dict.save', "rb"))
# count2 = pickle.load(open(save_dir + storage + 'artist_count_dict.save', "rb"))
# df['like_artist_chance'] = df['artist_name'].apply(c1_c2).astype(np.float16)
#
# count1 = pickle.load(open(save_dir + storage + 'disliked_artist_count_dict.save', "rb"))
# count2 = pickle.load(open(save_dir + storage + 'artist_count_dict.save', "rb"))
# df['dislike_artist_chance'] = df['artist_name'].apply(c1_c2).astype(np.float16)
#
# count1 = pickle.load(open(save_dir + storage + 'liked_artist_count_dict.save', "rb"))
# count2 = pickle.load(open(save_dir + storage + 'disliked_artist_count_dict.save', "rb"))
# df['artist_like_dislike'] = df['artist_name'].apply(c1_c2).astype(np.float16)


# df.drop(['isrc', 'name'], axis=1, inplace=True)
# df.drop(['lyricist',
#          'composer',
#          'genre_ids',
#          'song_length',
#          ],
#         axis=1, inplace=True)
# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
count = {}
# count1 = {}
# count2 = {}


def get_count1(x):
    try:
        return count[x]
    except KeyError:
        return 1


def get_count(x):
    try:
        return count[x]
    except KeyError:
        return 0

# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
storage = '../fake/'
count = pickle.load(open(storage + 'song_count_dict.save', "rb"))
df['fake_song_count'] = df['song_id'].apply(get_count1).astype(np.int64)
count = pickle.load(open(storage + 'liked_song_count_dict.save', "rb"))
df['fake_liked_song_count'] = df['song_id'].apply(get_count).astype(np.int64)
count = pickle.load(open(storage + 'disliked_song_count_dict.save', "rb"))
df['fake_disliked_song_count'] = df['song_id'].apply(get_count).astype(np.int64)
# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
count = pickle.load(open(storage + 'artist_count_dict.save', "rb"))
df['fake_artist_count'] = df['artist_name'].apply(get_count1).astype(np.int64)
count = pickle.load(open(storage + 'liked_artist_count_dict.save', "rb"))
df['fake_liked_artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)
count = pickle.load(open(storage + 'disliked_artist_count_dict.save', "rb"))
df['fake_disliked_artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)
del count


count = pickle.load(open(storage + 'member_count_dict.save', "rb"))
df['fake_member_count'] = df['song_id'].apply(get_count1).astype(np.int64)
count = pickle.load(open(storage + 'liked_member_count_dict.save', "rb"))
df['fake_member_song_count'] = df['song_id'].apply(get_count).astype(np.int64)
count = pickle.load(open(storage + 'disliked_member_count_dict.save', "rb"))
df['fake_disliked_member_count'] = df['song_id'].apply(get_count).astype(np.int64)
del count
# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

print('creating custom member.')
save_name = 'custom_song_'
vers = 'fixed'
d = df.dtypes.to_dict()
print(d)
print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('number of data:', len(df))
print('<'*20)
df.to_csv(save_dir+save_name+vers+'.csv', index=False)
pickle.dump(d, open(save_dir+save_name+vers+'_dict.save', "wb"))

print('done.')

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

