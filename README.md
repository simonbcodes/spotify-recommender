# Spotify Recommender

This project is an attempt to create a model that can predict whether or not a user will like a specific song.

There are 3 files used to achieve this currently:

- download_song_info.py : used to download the characteristics of all the songs a user has saved.
- analyze_songs.py : generates a playlist for the user to listen to and rate (like or dislike)
- classify.py : creates the classifier that is used to predict whether a user will like a song or not.

The main problem currently is that there's no easy way to get a user's disliked tracks. Since all the saved songs a user has are likes, the dislikes can only be found if a user listens through all the songs in the generated playlist and rates them manually. This means that the proportions of like/dislike in the training datasets are currently incredibly off. In my case, it's something like 1750 likes to ~50 dislikes, meaning the the classifier is not as accurate as it could be.

## APIs/Dependencies Used:

- sklearn
- spotipy
- pandas
- numpy
- pygn
