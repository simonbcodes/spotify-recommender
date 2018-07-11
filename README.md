# Spotify Recommender

This project is an attempt to create a model that can predict whether or not a user will like a specific song.

There are 3 files used to achieve this currently:

- download_song_info.py : used to download the characteristics of all the songs a user has saved.
- analyze_songs.py : generates a playlist for the user to listen to and rate (like or dislike).
- classify.py : creates the classifier that is used to predict whether a user will like a song or not.

The main problem currently is that there's no easy way to get a user's disliked tracks. Since all the saved songs a user has are likes, the dislikes can only be found if a user listens through all the songs in the generated playlist and rates them manually. This means that the proportions of like/dislike in the training datasets are currently incredibly off. In my case, it's something like 1750 likes to ~50 dislikes, meaning the the classifier is not as accurate as it could be.

## Keras approach

As a test, I used the Keras machine learning library to see if I could get different results than just a sklearn classifier. The files used are as follows:

- clean_data.py : this script takes in the playlist csvs and song_info csv to create a cleaned up, combined csv for use in training.
- nn.py : implements the Keras model and trains it.

This project uses a relatively simple Sequential model structure with 3 layers. The input layer takes in the 11 input features, corresponding to the elements of the songs we are reading in. The second layer is a 20 neuron hidden layer, which then feeds into a single output layer for classification. Though there is a clear imbalance in classes (much more likes than dislikes), the use of class weighting helps offset this. The model is fit on 70% of the input data, with around 30% being saved for testing. This led to a maximum of  85.25% accuracy with song classification accuracy, which is surprising given the small dataset.

## APIs/Dependencies Used:

- sklearn
- spotipy
- pandas
- numpy
- pygn
