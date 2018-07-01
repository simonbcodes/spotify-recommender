import pandas as pd
import numpy as np
import os, sys, pickle
from sklearn.externals import joblib
from sklearn import tree
from sklearn import preprocessing

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

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0) # create Decision Tree Classifier

clf.fit(training_features_scaled, training_labels) # train classifier on input data

print(clf.score(testing_features_scaled, testing_labels)) # score it
# print(clf.feature_importances_)

joblib.dump(clf, 'dtclf.pkl') # pickle the classifier
