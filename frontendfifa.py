# Importing necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Player Rating Prediction ⚽",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to scale user input
def scale_input(user_input):
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    user_input_df = pd.DataFrame([user_input], columns=feature_names)
    scaled_input = scaler.transform(user_input_df)
    return pd.DataFrame(scaled_input, columns=user_input_df.columns)

# Load trained model 
with open('ensemble_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Feature names
feature_names = [
    'movement_reactions', 
    'potential', 
    'wage_eur', 
    'passing', 
    'value_eur', 
    'dribbling', 
    'international_reputation', 
    'physic', 
    'shooting', 
    'skill_ball_control'
]

st.title('Player Rating Prediction ⚽')

# User input fields
st.sidebar.header('Player Features')

def user_input_features():
    movement_reactions = st.sidebar.slider('Movement Reactions', 0, 100, 50)
    potential = st.sidebar.slider('Potential', 0, 100, 50)
    wage_eur = st.sidebar.number_input('Wage (EUR)', min_value=0, max_value=int(1e6), value=50000)
    passing = st.sidebar.slider('Passing', 0, 100, 50)
    value_eur = st.sidebar.number_input('Value (EUR)', min_value=0, max_value=int(1e8), value=1000000)
    dribbling = st.sidebar.slider('Dribbling', 0, 100, 50)
    international_reputation = st.sidebar.slider('International Reputation', 1, 5, 1)
    physic = st.sidebar.slider('Physic', 0, 100, 50)
    shooting = st.sidebar.slider('Shooting', 0, 100, 50)
    skill_ball_control = st.sidebar.slider('Skill Ball Control', 0, 100, 50)

    data = {
        'movement_reactions': movement_reactions,
        'potential': potential,
        'wage_eur': wage_eur,
        'passing': passing,
        'value_eur': value_eur,
        'dribbling': dribbling,
        'international_reputation': international_reputation,
        'physic': physic,
        'shooting': shooting,
        'skill_ball_control': skill_ball_control
    }
    return data

input_data = user_input_features()

# Get predictions from model
st.subheader('Predict the Rating of Your Player! ⚽🔄🥁')
scaled_input = scale_input(input_data)
prediction = model.predict(scaled_input)

st.write(f"The predicted player rating is: {prediction[0]:.2f}")

# Explain model's prediction
if st.button('Prediction Explanation'):
    st.write("In this app, we are using an ensemble model that combines the predictions of three different models: Random Forest, Gradient Boosting, and XGBoost. The model predicts a player's rating based on various features such as movement reactions, potential, wage, passing, value, dribbling, international reputation, physic, shooting, and skill ball control.")
    st.write("The model was trained on a dataset of player ratings, using these features to predict the overall rating of a player. This is a demo project and doesn't use advanced model explanation techniques. Use with caution.")
