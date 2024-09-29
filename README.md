# AI-Recommendation-system-Spotify
Spotify Music Recommendation System
This project is a Spotify Music Recommendation System built using Python, Streamlit, Spotipy (Spotify Web API), and collaborative filtering techniques such as Alternating Least Squares (ALS). It also incorporates content-based recommendations using cosine similarity on audio features such as danceability, energy, acousticness, and valence.

The app fetches a user's saved tracks from Spotify, recommends songs based on a combination of collaborative filtering and audio features, and provides an interactive interface for users to explore their music and receive personalized recommendations.

Features
Spotify Authentication: Fetch saved tracks from the user's Spotify library using OAuth.
Collaborative Filtering: ALS model-based recommendations based on user-item interactions.
Content-based Filtering: Recommendations based on cosine similarity of audio features.
Feature Weight Tuning: Users can customize the importance of various audio features (danceability, energy, acousticness, valence).
Audio Preview: Play previews of recommended tracks.
Evaluation Metrics: Provides Precision@K and Recall@K metrics to evaluate the recommendation quality.
Visualizations: Radar chart visualization of audio features for recommended tracks.

Requirements
To run this project, ensure you have the following dependencies installed:

streamlit==1.x.x
spotipy==2.x.x
pandas==1.x.x
scikit-learn==0.x.x
numpy==1.x.x
plotly==5.x.x
implicit==0.x.x
scipy==1.x.x

How to Run
Clone the repository:

git clone https://github.com/Abhishek21112000/AI-Recommendation-system-Spotify-.git
cd spotify-music-recommendation
Install the required packages:

pip install -r requirements.txt
Set up your Spotify Developer credentials:

Create an application on the Spotify Developer Dashboard.
Copy your Client ID and Client Secret.
Replace the placeholders (client_id, client_secret, redirect_uri) in the code with your Spotify credentials.
Run the Streamlit app:

streamlit run app.py
Open the link provided in the terminal (usually http://localhost:8501) and authenticate your Spotify account.

File Structure
app.py: The main Streamlit app file that contains the logic for the music recommendation system.
requirements.txt: List of Python dependencies required to run the app.

How It Works
Spotify Authentication:

The app uses SpotifyOAuth to authenticate the user and fetch their saved tracks.
Collaborative Filtering:

Using the implicit library, we apply the Alternating Least Squares (ALS) algorithm to generate recommendations based on the user's listening history (track popularity).
Content-based Filtering:

Tracks are compared using cosine similarity based on features such as danceability, energy, acousticness, and valence. The similarity score helps in recommending similar tracks to the selected song.
Tune Feature Weights:

The user can adjust weights for the audio features via sliders to change the recommendation output.
Model Evaluation:

Precision@K and Recall@K are used to evaluate the recommendations provided by the ALS model.
Customization
You can further customize the system by:

Adjusting the ALS hyperparameters for better collaborative filtering results.
Extending the content-based filtering to include more audio features like tempo, loudness, and instrumentalness.
Implementing additional recommendation metrics.
Visualizations
The app provides a radar chart visualization that allows users to visually compare the audio features (danceability, energy, acousticness, and valence) of recommended tracks.
