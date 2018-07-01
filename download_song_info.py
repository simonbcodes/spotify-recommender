import pygn
import spotipy
import spotipy.util as util
import config
import pandas as pd
import sys
from time import time

#This program takes all of a user's saved songs and writes them to a CSV. It also catalogs genre from Gracenote and various features from Spotify.

scope = 'user-library-read'

clientID = config.pygn_client_id #pygn api (for Gracenote)
userID = pygn.register(clientID)

if len(sys.argv) > 1: #check for username input
    username = sys.argv[1]
else:
    print("Usage: {} username".format(sys.argv[1]))
    sys.exit()

client_id = config.spotipy_client_id #spotipy api
client_secret = config.spotipy_secret_id
redirect_uri = 'http://localhost/'

token = util.prompt_for_user_token(username, scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri) #get token
song_info = pd.DataFrame(columns=['Song Name', 'Artist', 'Album', 'Duration', 'Danceability', 'Instrumentalness', 'Speechiness', 'Energy', 'Mode', 'Tempo', 'Liveness', 'Loudness', 'Key', 'Acousticness'])
song_info = song_info.astype('object')

if token:
    sp = spotipy.Spotify(auth=token)
    offset = 0
    t0 = time()
    tracks = sp.current_user_saved_tracks(offset=offset, limit=20) # get 20 songs from user's saved songs

    while(len(tracks['items']) != 0): # haven't reached the end of saved tracks
        for song in tracks['items']:
            features = sp.audio_features(song['track']['id'])[0]
            metadata = pygn.search(clientID=clientID, userID=userID, artist=song['track']['artists'][0]['name'], album=song['track']['album']['name'], track=song['track']['name'])
            genres = []
            for key, genre in metadata['genre'].items():
                genres.append(metadata['genre'][str(key)]['TEXT']) # add all genres to list
                # print(metadata['genre'][str(key)]['TEXT'])
            if(len(genres) > 0):
                print('{} - {}'.format(song['track']['name'], metadata['genre']['1']['TEXT']))
            if features is not None: # if features exist (sometimes they don't), add row to df
                song_info = song_info.append({'Song Name': song['track']['name'], 'Artist': song['track']['artists'][0]['name'], 'Album': song['track']['album']['name'], \
                'Duration': features['duration_ms'], 'Danceability': features['danceability'], 'Instrumentalness': features['instrumentalness'], \
                'Speechiness': features['speechiness'], 'Energy': features['energy'], 'Mode': features['mode'], 'Tempo': features['tempo'], 'Liveness': features['liveness'], \
                'Loudness': features['loudness'], 'Key': features['key'], 'Acousticness': features['acousticness'], 'Genre': genres}, ignore_index=True)
        offset += 20 # next 20 songs
        tracks = sp.current_user_saved_tracks(offset=offset, limit=20) # get new set of tracks, then repeat
    print('time taken: {} seconds'.format(time() - t0)) # print total runtime
else:
    print("Can't get token for {}".format(username))

song_info.to_csv('song_info.csv', encoding='utf-8', index=False) #write to CSV
print('Song info written to CSV')
