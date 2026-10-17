import pandas as pd
import numpy as np
import os

def build_and_save_dataset():
    data_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    
    players = pd.read_csv(os.path.join(data_dir, "players.csv"))
    appearances = pd.read_csv(os.path.join(data_dir, "appearances.csv"))
    clubs = pd.read_csv(os.path.join(data_dir, "clubs.csv"))
    competitions = pd.read_csv(os.path.join(data_dir, "competitions.csv"))
    
    appearances['date'] = pd.to_datetime(appearances['date'], errors='coerce')
    appearances = appearances.sort_values(by=['player_id', 'date'])
    
    player_career = appearances.groupby("player_id").agg({
        "minutes_played": "sum",
        "goals": "sum",
        "assists": "sum",
        "yellow_cards": "sum",
        "red_cards": "sum",
        "appearance_id": "count"
    }).rename(columns={"appearance_id": "total_matches"}).reset_index()
    
    recent_apps = appearances.groupby('player_id').tail(10)
    recent_stats = recent_apps.groupby('player_id').agg({
        'minutes_played': 'sum',
        'goals': 'sum',
        'assists': 'sum'
    }).reset_index()
    
    recent_stats['recent_minutes'] = recent_stats['minutes_played']
    recent_stats['recent_goal_contrib_per90'] = np.where(
        recent_stats['recent_minutes'] > 0,
        ((recent_stats['goals'] + recent_stats['assists']) / recent_stats['recent_minutes']) * 90,
        0
    )
    recent_stats = recent_stats[['player_id', 'recent_goal_contrib_per90', 'recent_minutes']]
    
    scout_df = players.merge(player_career, on="player_id", how="left")
    scout_df = scout_df.merge(recent_stats, on="player_id", how="left")
    
    clubs_subset = clubs[["club_id", "name"]].rename(columns={"name": "club_name"})
    scout_df = scout_df.merge(clubs_subset, left_on="current_club_id", right_on="club_id", how="left")
    
    comps_subset = competitions[["competition_id", "name", "country_name", "sub_type"]].rename(columns={"name": "competition_name"})
    scout_df = scout_df.merge(comps_subset, left_on="current_club_domestic_competition_id", right_on="competition_id", how="left")
    
    cols_to_drop = ["club_id", "competition_id"]
    scout_df = scout_df.drop(columns=[col for col in cols_to_drop if col in scout_df.columns])
    
    cols_to_fill = ["minutes_played", "goals", "assists", "yellow_cards", "red_cards", "total_matches", "recent_goal_contrib_per90", "recent_minutes"]
    scout_df[cols_to_fill] = scout_df[cols_to_fill].fillna(0)
    
    output_path = os.path.join(processed_dir, "scout_master.csv")
    scout_df.to_csv(output_path, index=False)
    
    print(f"\n--- DATASET SAVED TO {output_path} ---")
    print(f"Columns: {list(scout_df.columns)}")
    
    return scout_df

if __name__ == "__main__":
    df = build_and_save_dataset()