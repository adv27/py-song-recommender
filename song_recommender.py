import os
from math import sqrt

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split


def load_music_data(file_name):
    '''Get reviews data, from loccal csv'''
    if os.path.exists(file_name):
        print('--{} found locally'.format(file_name))
        df = pd.read_csv(file_name)

    return df


def values_to_map_index(values):
    map_index = {}
    index = 0
    for val in values:
        map_index[val] = index
        index += 1

    return map_index


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # You use np.newaxis so that mean_user_rating has the same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


# load music data with sampling fraction = 0.01 for reduce processing time
song_data = load_music_data(r'data/song_data.csv')
song_data = song_data.sample(frac=0.01, replace=False)

print('--Explore data')
# display(song_data.head())
print(song_data.head())

n_users = song_data.user_id.unique().shape[0]
n_items = song_data.song_id.unique().shape[0]
print('Number of users = {} | Number of songs = {}'.format(n_users, n_items))

print('--Showing the most popular songs in the dataset')
unique, counts = np.unique(song_data["song_id"], return_counts=True)
popular_songs = dict(zip(unique, counts))
# popular_songs = dict(zip(unique[:5], counts[:5]))
# print(list(popular_songs.items()))
df_popular_songs = pd.DataFrame(data=list(popular_songs.items()), columns=["Song", "Count"])
df_popular_songs = df_popular_songs.sort_values(by=["Count"], ascending=False)
print(df_popular_songs.head())

train_data, test_data = train_test_split(song_data, test_size=0.25)

user_idx = values_to_map_index(song_data.user_id.unique())
song_idx = values_to_map_index(song_data.song_id.unique())

train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[user_idx[line[1]], song_idx[line[2]]] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[user_idx[line[1]], song_idx[line[2]]] = line[3]

user_similarity = pairwise_distances(train_data_matrix, metric='cosine', n_jobs=1)
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine', n_jobs=-1)

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

print('User-based CF RMSE: {}'.format(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: {}'.format(rmse(item_prediction, test_data_matrix)))
