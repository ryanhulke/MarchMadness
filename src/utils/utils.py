import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def visual_test(model):
    df = pd.read_csv("data_2025/clean/new_data.csv")

    cols_to_drop = ["winner", "day", "team0_ID", "team1_ID", "team0_poss/gm", "team1_poss/gm", "year"]

    # the test set is all games after day 134
    X_train_df = df[df["day"] < 134]
    y_train_df = X_train_df["winner"]
    X_train_df = X_train_df.drop(columns=cols_to_drop)

    X_test_df_pre = df[df["day"] >= 134]
    y_test = X_test_df_pre["winner"]
    X_test_df = X_test_df_pre.drop(columns=cols_to_drop)

    X_train, X_val, y_train, y_val = train_test_split(X_train_df, y_train_df, test_size=.1, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test_df)
    # select games from 2024 tourney
    tourney_2023 = X_test_df_pre[-63:]

    # extract team ids and labels, delete days
    team0_ids = list(tourney_2023["team0_ID"])
    team1_ids = list(tourney_2023["team1_ID"])
    y_tourney_2023 = list(tourney_2023["winner"])
    tourney_2023 = tourney_2023.drop(columns=cols_to_drop + ["winner"])

    tourney_2023 = scaler.transform(tourney_2023)

    # map team0_teamID and team1_teamID to team names via data_2025/MTeams.csv
    teams = pd.read_csv("data_2025/MTeams.csv")
    team_id_to_name = dict(zip(teams["TeamID"], teams["TeamName"]))

    # Predict probabilities for the tournament games
    predicted_probs = model.predict_proba(tourney_2023)[:, 1]
    y_pred = model.predict(tourney_2023)
    # Iterate through each game
    for i in range(len(tourney_2023)):
        # Extract team IDs (assuming they are in columns 'team0_ID' and 'team1_ID')
        team0_id = int(team0_ids[i])
        team1_id = int(team1_ids[i])
        
        # Get team names
        team0_name = team_id_to_name.get(team0_id, "Unknown Team")
        team1_name = team_id_to_name.get(team1_id, "Unknown Team")
        
        # Get predicted probability (probability that team0 wins)
        predicted_prob = predicted_probs[i]
        prediction = y_pred[i]

        # Get actual winner (assuming 'label' column indicates the winner: 1 if team0 won, 0 if team1 won)
        actual_winner = y_tourney_2023[i] #int(y_tourney_2023.iloc[i])
        
        # Print the information
        print(f"Game {i+1}:")
        print(f"  {team0_name}")
        print(f"  {team1_name}")
        predicted_winner = team0_name if prediction == 1 else team1_name
        if predicted_prob < .5:
            predicted_prob = 1 - predicted_prob
        print(f"  Prediction: {predicted_winner}: {predicted_prob:.3f}")
        print(f"  Actual winner: {team0_name if actual_winner == 1 else team1_name}")
        print("-" * 30)
    print("accuracy:", accuracy_score(y_tourney_2023, y_pred))

def eval(y_pred, y_test, y_probs=None):
    # accuracy
    acc = accuracy_score(y_test, y_pred)
    print("accuracy:", acc)

    if y_probs is not None:
        plot_roc_auc(y_probs, y_test)
    
def plot_roc_auc(y_probs, y_test):
    # Compute ROC curve
    # Assuming y_test and y_probs are already defined
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    roc_auc = roc_auc_score(y_test, y_probs)
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()


def get_train_test_val(path, val_size=.1):
    df = pd.read_csv(path)
    cols_to_drop = ["winner", "day", "team0_ID", "team1_ID", "team0_poss/gm", "team1_poss/gm", "year"]
    # the test set is all games after day 134
    X_train_df = df[df["day"] < 134]
    y_train_df = X_train_df["winner"]
    X_train_df = X_train_df.drop(columns=cols_to_drop)

    X_test_df_pre = df[df["day"] >= 134]
    y_test = X_test_df_pre["winner"]
    X_test_df = X_test_df_pre.drop(columns=cols_to_drop)

    X_train, X_val, y_train, y_val = train_test_split(X_train_df, y_train_df, test_size=val_size, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test_df)
    return X_train, X_test, X_val, y_train, y_test, y_val