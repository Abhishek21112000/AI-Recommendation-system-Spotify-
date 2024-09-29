import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity #measure the similarity btw two Dot vectors 
import numpy as np
import plotly.express as px
import implicit  #temp chnage the data type for easy cal
from scipy.sparse import csr_matrix #store and perform fast when dealing with more zero's

# Spotify authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='add131255d68413b9b66f4832a41b64a',
                                               client_secret='c0c601f465084bfeb90f27253a4daf72',
                                               redirect_uri='http://localhost:8888/callback',
                                               scope="user-library-read"))

# UI
st.title("Spotify Music Recommendation System")
st.write("""
    This app fetches your saved tracks from Spotify and recommends similar songs 
    based on audio features like danceability, energy, and popularity.
""")

# Sidebar (ALS - predit user interaction alternating between optimizing user and item latent factor)
st.sidebar.header("Collaborative Filtering Hyperparameters")
factors = st.sidebar.slider('Number of factors', 10, 100, 50)
regularization = st.sidebar.slider('Regularization', 0.01, 1.0, 0.1)
iterations = st.sidebar.slider('Iterations', 10, 50, 30)

# Sidebar 
st.sidebar.header("Tune Feature Weights")
danceability_weight = st.sidebar.slider('Danceability Weight', 0.0, 1.0, 0.5)
energy_weight = st.sidebar.slider('Energy Weight', 0.0, 1.0, 0.5)
acousticness_weight = st.sidebar.slider('Acousticness Weight', 0.0, 1.0, 0.5)
valence_weight = st.sidebar.slider('Valence Weight', 0.0, 1.0, 0.5)

num_recommendations = st.sidebar.slider('Number of recommendations:', 1, 10, 3)

# Fetch user's track's
st.write("Fetching your saved tracks from Spotify...")
saved_tracks = sp.current_user_saved_tracks(limit=10)
track_data = []

# Store track info
for item in saved_tracks['items']:
    track = item['track']
    track_id = track['id']
    features = sp.audio_features(track_id)[0]  
    
    # Append data
    track_data.append({
        'name': track['name'],
        'artist': track['artists'][0]['name'],
        'popularity': track['popularity'], # based on the skipped, like and share etc
        'danceability': features['danceability'], # based on the rhythm its calucated 
        'energy': features['energy'], #Based on the sound it calucated like loud = 1 and low = 0.1
        'acousticness': features['acousticness'], #based on electronic sounds
        'valence': features['valence'], #based on +ve and -ve emotions
        'user_id': 1,  # Static user ID for now
        'song_id': track_id  # Unique song ID for interaction matrix
    })

# Convert pandas DataFrame
df = pd.DataFrame(track_data)

# fetched 
st.write("Your saved tracks:")
st.dataframe(df[['name', 'artist', 'popularity', 'danceability', 'energy', 'acousticness', 'valence']])

# Track selection
selected_track = st.selectbox('Select a track to get recommendations', df['name'].values)

# Index of selected track
track_index = df.index[df['name'] == selected_track][0]

# Adjust weights based on user input
df['danceability'] *= danceability_weight
df['energy'] *= energy_weight
df['acousticness'] *= acousticness_weight
df['valence'] *= valence_weight

# mapping song_id to integers
song_id_mapping = {song_id: idx for idx, song_id in enumerate(df['song_id'])}
df['song_id_mapped'] = df['song_id'].map(song_id_mapping)

interaction_matrix = csr_matrix((df['popularity'], (df['user_id'], df['song_id_mapped'])))

#Train the model with user-song interactions
als_model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)
st.write("Training ALS model with collaborative filtering...")
als_model.fit(interaction_matrix)

user_id = 1
recommended_song_ids, scores = als_model.recommend(user_id, interaction_matrix[user_id], N=num_recommendations)

recommended_song_ids = [list(song_id_mapping.keys())[list(song_id_mapping.values()).index(i)] for i in recommended_song_ids]

# Display ALS-based recommended tracks
st.write("Collaborative Filtering Recommendations:")
recommended_tracks = df[df['song_id'].isin(recommended_song_ids)][['name', 'artist']]
st.dataframe(recommended_tracks)

# Cosine similarity-based recommendations
features = df[['danceability', 'energy', 'acousticness', 'valence']]
similarity_matrix = cosine_similarity(features)

# Get indices of the most similar tracks
similar_tracks = similarity_matrix[track_index].argsort()[-(num_recommendations+1):][::-1]
similar_tracks = [idx for idx in similar_tracks if idx != track_index]  

# Display recommended tracks
st.write(f"Recommended tracks similar to **{selected_track}**:")
similar_recommended_tracks = df.iloc[similar_tracks][['name', 'artist']]
st.dataframe(similar_recommended_tracks)

# Audio preview
st.write("Play audio samples of recommended tracks:")
for idx in similar_tracks:
    track = saved_tracks['items'][idx]['track']
    st.write(f"{track['name']} by {track['artists'][0]['name']}")
    if 'preview_url' in track and track['preview_url']:
        st.audio(track['preview_url'], format='audio/mp3')
    else:
        st.write("No preview available")

# Precision@k and Recall@k
def precision_at_k(recommended_items, relevant_items, k):
    return len(set(recommended_items[:k]).intersection(set(relevant_items))) / k

def recall_at_k(recommended_items, relevant_items, k):
    return len(set(recommended_items[:k]).intersection(set(relevant_items))) / len(relevant_items)

relevant_items = df[df['user_id'] == user_id]['song_id'].values 
precision = precision_at_k(recommended_song_ids, relevant_items, num_recommendations)
recall = recall_at_k(recommended_song_ids, relevant_items, num_recommendations)

# Display Evaluation Metrics
st.write(f"Precision@{num_recommendations}: {precision:.2f}")
st.write(f"Recall@{num_recommendations}: {recall:.2f}")

# Visualization 
st.write("Visualize audio features of recommended tracks:")
radar_data = df.iloc[similar_tracks][['name', 'danceability', 'energy', 'acousticness', 'valence']]
fig = px.line_polar(radar_data, r='danceability', theta='name', line_close=True, title="Danceability of Recommended Tracks")
st.plotly_chart(fig)
