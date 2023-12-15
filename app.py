import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
tennis_data = pd.read_csv("tennis_data.csv")

# Encoding categorical variables
label_encoder = LabelEncoder()
tennis_data['serving_player'] = label_encoder.fit_transform(tennis_data['serving_player'])

# Created a dictionary to map 'AD' to a numerical value
# Assuming that 'AD' is present in 'player_1_points' and 'player_2_points', and 'match_winning_player' is the target variable
points_mapping = {'AD':50}

# Apply the mapping to 'player_1_points' and 'player_2_points'
tennis_data['player_1_points'] = tennis_data['player_1_points'].replace(points_mapping)
tennis_data['player_2_points'] = tennis_data['player_2_points'].replace(points_mapping)

# Creating a binary target variable: 1 if player 1 wins, 0 otherwise
tennis_data['target'] = (tennis_data['player_1_points'].astype(int) > tennis_data['player_2_points'].astype(int)).astype(int)

# Selecting relevant features
features = ['serving_player', 'player_1_points', 'player_2_points', 'player_1_games', 'player_2_games', 'player_1_sets', 'player_2_sets']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tennis_data[features], tennis_data['target'], test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Streamlit App
st.title("Tennis Match Prediction")

# Add input widgets
serving_player = st.selectbox("Select serving player:", tennis_data['serving_player'].unique())
player_1_points = st.slider("Player 1 Points", min_value=0, max_value=50, value=0)
player_2_points = st.slider("Player 2 Points", min_value=0, max_value=50, value=0)
player_1_games = st.slider("Player 1 Games", min_value=0, max_value=6, value=0)
player_2_games = st.slider("Player 2 Games", min_value=0, max_value=6, value=0)
player_1_sets = st.slider("Player 1 Sets", min_value=0, max_value=2, value=0)
player_2_sets = st.slider("Player 2 Sets", min_value=0, max_value=2, value=0)

# Make prediction
input_data = [[label_encoder.fit_transform([serving_player])[0], player_1_points, player_2_points, player_1_games, player_2_games, player_1_sets, player_2_sets]]
prediction = model.predict(input_data)[0]

# Display prediction
st.subheader("Prediction:")
st.write(f"The predicted winner is: {'Player 1' if prediction == 1 else 'Player 2'}")
