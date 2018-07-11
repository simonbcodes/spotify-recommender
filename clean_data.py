import pandas as pd

song_info = pd.read_csv('song_info_saved.csv') # load CSVs
playlist_test_set = pd.read_csv('playlist_test_set.csv')

# print(song_info)
# print(playlist_test_set)

song_info['Rating'] = 1 # set ratings of saved songs to 1

# print(song_info)

all_songs = pd.concat([song_info, playlist_test_set]) # combine 2 DataFrames

# print(all_songs)

all_songs = all_songs.drop(columns=['Song Name', 'Album', 'Artist']) # drop non-numerical data

all_songs.to_csv('all_songs.csv', encoding='utf-8', index=False) #write to CSV
