import pandas as pd
import numpy as np
import os, sys, pickle
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier

scope = 'playlist-modify-private playlist-read-private user-library-read'

saved_songs = pd.read_csv(os.getcwd() + '/song_info_saved2.csv') # load all saved songs
playlist_songs = pd.read_csv(os.getcwd() + '/playlist_info_batch_1.csv') # all playlist songs
test_playlist_songs = pd.read_csv(os.getcwd() + '/playlist_test_set.csv') # all playlist songs for testing

liked_songs = playlist_songs.loc[playlist_songs['Rating'] == 1] # filter only liked songs

liked_songs = playlist_songs.loc[playlist_songs['Rating'] == 1].append(saved_songs, sort=False) # combine saved liked songs with liked songs from playlist
disliked_songs = playlist_songs.loc[playlist_songs['Rating'] <= 0] # load disliked songs

training_features = liked_songs.append(disliked_songs, sort=False) # combine liked ad disliked songs
training_features = training_features.drop(columns=['Song Name', 'Artist', 'Album', 'Genre', 'Rating']).values # convert features to numpy array after dropping columns
training_labels = np.concatenate([np.ones((1, len(liked_songs))), np.zeros((1, len(disliked_songs)))], axis=1)[0] # set labels (1 for like, 0 for dislike)

scaler = preprocessing.StandardScaler().fit(training_features) # create scaler to standerdize dataset

training_features_scaled = scaler.transform(training_features) # scale training features

test_liked_songs = test_playlist_songs.loc[test_playlist_songs['Rating'] == 1] # load liked songs for testing features
test_disliked_songs = test_playlist_songs.loc[test_playlist_songs['Rating'] <= 0] # load disliked songs for testing feaures

testing_features = test_liked_songs.append(test_disliked_songs, sort=False)
testing_features = testing_features.drop(columns=['Song Name', 'Artist', 'Album', 'Rating']).values # convert to numpy array after dropping columns
testing_features_scaled = scaler.transform(testing_features) # scale testing features
testing_labels = np.concatenate([np.ones((1, len(test_liked_songs))), np.zeros((1, len(test_disliked_songs)))], axis=1)[0] # set labels

# model = TSNE(learning_rate=100, random_state=0)
# dbscan = DBSCAN()
#
# dbscan.fit(training_features)
#
# pca = PCA(n_components=2).fit(training_features)
# pca_2d = pca.transform(training_features)
#
# c1 = None
# c2 = None
# c3 = None
#
# for i in range(0, pca_2d.shape[0]):
#     if dbscan.labels_[i] == 0:
#         c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
#     elif dbscan.labels_[i] == 1:
#         c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
#     elif dbscan.labels_[i] == -1:
#         c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
#
# plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Noise'])
# plt.title('DBSCAN finds 2 clusters and Noise')
# plt.show()

# transformed = model.fit_transform(training_features)
#
# print(transformed)
#
# x_axis = transformed[:, 0]
# y_axis = transformed[:, 1]
#
# plt.scatter(x_axis, y_axis, c=training_labels)
#
# plt.show()

# clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0) # create Decision Tree Classifier
#
# clf.fit(training_features_scaled, training_labels) # train classifier on input data
#
# print(clf.score(testing_features_scaled, testing_labels)) # score it
# # print(clf.feature_importances_)
#
# joblib.dump(clf, 'dtclf.pkl') # pickle the classifier
