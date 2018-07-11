import spotipy
import spotipy.util as util
from sklearn.externals import joblib
import pandas as pd
import os, sys
import config
from ast import literal_eval

scope = 'playlist-modify-private playlist-read-private user-library-read'

song_info = pd.read_csv(os.getcwd() + '/song_info_saved2.csv') # all saved songs
all_genres = pd.read_csv(os.getcwd() + '/all_genres.csv')['Genres'].tolist() # all Spotify genres
columns = ['Duration', 'Danceability', 'Instrumentalness', 'Speechiness', 'Energy', 'Mode', 'Tempo', 'Liveness', 'Loudness', 'Key', 'Acousticness']

def average_song(): # returns a dict of average characterstics from all saved songs
    average = {}
    average['Song Name'] = 'Average Song'
    average['Album'] = 'Average Album'
    average['Artist'] = 'Average Artist'
    for category in columns:
        average[category] = song_info[category].mean()
    return average

def song_to_dict(features):
    return {
    'Duration': features['duration_ms'], 'Danceability': features['danceability'], 'Instrumentalness': features['instrumentalness'], \
    'Speechiness': features['speechiness'], 'Energy': features['energy'], 'Mode': features['mode'], 'Tempo': features['tempo'], 'Liveness': features['liveness'], \
    'Loudness': features['loudness'], 'Key': features['key'], 'Acousticness': features['acousticness']}

def generate_playlist(sp):
    playlists = sp.user_playlists(user=sys.argv[1], limit=50) # get all playlists from user

    for play in playlists['items']: # check if playlist exists already
        if play['name'] == 'Generated_Playlist_Classifier':
            sp.user_playlist_unfollow(user=sys.argv[1], playlist_id=play['id']) # 'delete' it

    playlist = sp.user_playlist_create(user=sys.argv[1], name='Generated_Playlist_Classifier', public=False, description='Playlist for training the classifier') # new playlist

    keyword_parameters = {} # passed to spotipy api recommendation as a parameter
    keyword_parameters['seed_artists'] = None
    keyword_parameters['seed_tracks'] = None
    keyword_parameters['limit'] = 1 # one song at a time

    for option in columns: # set paramter for every characterestic in average song values
        keyword_parameters['target_{}'.format(option)] = average_song_values[option]

    saved_tracks = []
    classify_playlist = pd.DataFrame(columns=['Song Name', 'Artist', 'Album'] + columns)

    for genre in all_genres:
        keyword_parameters['seed_genres'] = [genre] # set seed genre to current genre
        recommended = sp.recommendations(**keyword_parameters) # recommend a single song
        if len(recommended['tracks']) != 0: # if generated
            saved_tracks.append(recommended['tracks'][0]['id'])
            print('{} -> {} - {}'.format(genre, recommended['tracks'][0]['name'], recommended['tracks'][0]['artists'][0]['name']))
            features = sp.audio_features(recommended['tracks'][0]['id'])[0] # get features of recommended song
            if features is not None: # if they exist
                classify_playlist = classify_playlist.append({'Song Name': recommended['tracks'][0]['name'], 'Artist': recommended['tracks'][0]['artists'][0]['name'], 'Album': recommended['tracks'][0]['album']['name'], \
                'Duration': features['duration_ms'], 'Danceability': features['danceability'], 'Instrumentalness': features['instrumentalness'], \
                'Speechiness': features['speechiness'], 'Energy': features['energy'], 'Mode': features['mode'], 'Tempo': features['tempo'], 'Liveness': features['liveness'], \
                'Loudness': features['loudness'], 'Key': features['key'], 'Acousticness': features['acousticness']}, ignore_index=True)

    added_tracks = 0
    for i in range(0, int(len(saved_tracks) / 100.0) + 1): # add songs to playlist
        sp.user_playlist_add_tracks(user=sys.argv[1], playlist_id=playlist['id'], tracks=saved_tracks[added_tracks:added_tracks + 100])
        added_tracks += 100

    classify_playlist.to_csv('playlist_info.csv', encoding='utf-8', index=False) # write added songs to CSV
    print('Playlist written to CSV')

song_info['Genre'] = song_info['Genre'].apply(literal_eval) #convert string array to list in df

if len(sys.argv) > 1: # get username
    username = sys.argv[1]
else:
    print("Usage: {} username".format(sys.argv[1]))
    sys.exit()

client_id = config.spotipy_client_id #spotipy api
client_secret = config.spotipy_secret_id
redirect_uri = 'http://localhost/'

token = util.prompt_for_user_token(username, scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)

clf = joblib.load('dtclf.pkl')

if token:
    average_song_values = average_song() # get average values
    sp = spotipy.Spotify(auth=token)
    # generate_playlist(sp)
    for genre in all_genres:
        print(genre)
        recommended = sp.recommendations(seed_genres=[genre], limit=50)
        tracks_to_analyze = []
        if len(recommended['tracks']) != 0:
            for track in recommended['tracks']:
                #print(track)
                tracks_to_analyze.append(track['id'])
            features = sp.audio_features(tracks_to_analyze)
            will_like_songs = 0
            total = 0
            for track in features:
                total += 1
                if clf.predict([list(song_to_dict(track).values())]) == 1:
                    will_like_songs += 1
            print('{}/{}'.format(will_like_songs, total))



else:
    print('no token')
