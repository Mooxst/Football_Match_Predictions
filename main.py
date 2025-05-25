# Main libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Streamlit
from streamlit import cache_data, cache_resource

st.set_page_config(page_title="Football Match Predictor", layout="wide") # Main page

st.title("Football Match Prediction System") # Title and team info
st.markdown("""
**Students:**  
- Bhavik Kumar (24007107)  
- Max Crooks (22009631)  
- Rehal Kumar (24016256)  
""")

# Load data
@st.cache_data
def load_data():
    classification_df = pd.read_csv("classification_df.csv")
    combined_team_data = pd.read_csv("combined_team_data.csv")
    return classification_df, combined_team_data

classification_df, combined_team_data = load_data()

def generate_match_features(classification_df, match_df): #  match-level dataset
    match_features = []
    outcomes = []

    for _, row in match_df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        result = row['FTR']  # H, D, A

        home_data = classification_df[classification_df['Team'] == home]
        away_data = classification_df[classification_df['Team'] == away]

        if home_data.empty or away_data.empty:
            continue

        home_data = home_data.iloc[0]
        away_data = away_data.iloc[0]

        feature = {
            'home_win%': home_data['Home Win%'],
            'home_loss%': home_data['Loss%'],
            'away_win%': away_data['Away Win%'],
            'away_loss%': away_data['Loss%'],
            'home_poss': home_data['Poss'],
            'away_poss': away_data['Poss'],
            'home_play_style': home_data['Play_Style_Label'],
            'away_play_style': away_data['Play_Style_Label'],
            'style_mismatch': int(away_data['Play_Style'] in home_data['Struggle_Against'])
        }

        match_features.append(feature)

        if result == 'H': # Encode result: 2 = Home Win, 1 = Draw, 0 = Away Win
            outcomes.append(2)
        elif result == 'D':
            outcomes.append(1)
        elif result == 'A':
            outcomes.append(0)
    return pd.DataFrame(match_features), outcomes


@st.cache_resource # Train model
def train_model(classification_df, combined_team_data):
    X, y = generate_match_features(classification_df, combined_team_data)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier() # Random forest
    clf.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(x_test))
    return clf, accuracy

model, accuracy = train_model(classification_df, combined_team_data)


st.sidebar.title("Navigation") # Navigation Sidebar
page = st.sidebar.radio("Go to", ["Prediction", "Data Exploration", "Model Info"])

if page == "Prediction": #Match prediction
    st.header("Match Outcome Prediction")
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Select Home Team", sorted(classification_df['Team'].unique()))
    with col2:
        away_team = st.selectbox("Select Away Team", sorted(classification_df['Team'].unique()))

    if st.button("Predict Outcome"):
        home_data = classification_df[classification_df['Team'] == home_team].iloc[0]
        away_data = classification_df[classification_df['Team'] == away_team].iloc[0]

        input_data = pd.DataFrame([{
            'home_win%': home_data['Home Win%'],
            'home_loss%': home_data['Loss%'],
            'away_win%': away_data['Away Win%'],
            'away_loss%': away_data['Loss%'],
            'home_poss': home_data['Poss'],
            'away_poss': away_data['Poss'],
            'style_mismatch': int(away_data['Play_Style'] in home_data['Struggle_Against'])
        }])

        probabilities = model.predict_proba(input_data)[0] # Predict probabilities
        outcome_map = {2: 'Home Win', 1: 'Draw', 0: 'Away Win'}


        st.subheader("Prediction Results") # Display results


        pred = model.predict(input_data)[0] # Create metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Home Win Probability", f"{probabilities[2] * 100:.1f}%")
        col2.metric("Draw Probability", f"{probabilities[1] * 100:.1f}%")
        col3.metric("Away Win Probability", f"{probabilities[0] * 100:.1f}%")


        st.success(f"**Most Likely Outcome:** {outcome_map[pred]}") # Show predicted outcome


        st.subheader("Team Styles") # Show team styles
        col1, col2 = st.columns(2)
        col1.info(f"**{home_team} Style:** {home_data['Play_Style']}")
        col2.info(f"**{away_team} Style:** {away_data['Play_Style']}")


        if input_data['style_mismatch'].iloc[0]: # Show style matchup
            st.warning(f" {home_team} tends to struggle against {away_data['Play_Style']} teams")
        else:
            st.success("No significant style mismatch detected")

elif page == "Data Exploration":
    st.header("Data Exploration")

    tab1, tab2, tab3 = st.tabs(["Team Statistics", "Visualizations", "Raw Data"])

    with tab1:
        st.dataframe(classification_df.sort_values('Win%', ascending=False))

    with tab2:
        st.subheader("Win Percentage vs Progressive Passes")
        fig, ax = plt.subplots()
        ax.scatter(classification_df['Win%'], classification_df['PrgP'])
        ax.set_xlabel('Win Percentage')
        ax.set_ylabel('Progressive Passes')
        st.pyplot(fig)

        st.subheader("Win Percentage vs Possession")
        fig, ax = plt.subplots()
        ax.scatter(classification_df['Win%'], classification_df['Poss'])
        ax.set_xlabel('Win Percentage')
        ax.set_ylabel('Possession Percentage')
        st.pyplot(fig)

        st.subheader("Loss Percentage vs Attacking Third Tackles")
        fig, ax = plt.subplots()
        ax.scatter(classification_df['Loss%'], classification_df['Att 3rd'])
        ax.set_xlabel('Loss Percentage')
        ax.set_ylabel('Tackles in Attacking Third')
        st.pyplot(fig)

    with tab3:
        st.dataframe(combined_team_data)

elif page == "Model Info":
    st.header("Model Information")
    st.write(f"Model Accuracy: {accuracy:.2%}")

    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    st.dataframe(feature_importance)

    fig, ax = plt.subplots()
    ax.barh(feature_importance['Feature'], feature_importance['Importance'])
    ax.set_xlabel('Importance')
    st.pyplot(fig)

    st.subheader("Model Details")
    st.write("Algorithm: Random Forest Classifier")
    st.write("This model predicts match outcomes (Home Win, Draw, Away Win) based on:")
    st.markdown("""
    - Team performance statistics (win%, loss%, possession)
    - Play style matchups
    - Historical performance data
    """)
