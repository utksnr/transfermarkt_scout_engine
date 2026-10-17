import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

def get_position_weights(sub_position):

    weights_dict = {
        'Centre-Forward': {'height_in_cm': 3.0, 'goals_per90': 5.0, 'assists_per90': 1.5, 'yellows_per90': 0.5, 'recent_goal_contrib_per90': 2.0},
        'Second Striker': {'height_in_cm': 1.5, 'goals_per90': 4.0, 'assists_per90': 3.0, 'yellows_per90': 0.5, 'recent_goal_contrib_per90': 2.0},
        'Right Winger': {'height_in_cm': 1.0, 'goals_per90': 3.0, 'assists_per90': 4.0, 'yellows_per90': 0.5, 'recent_goal_contrib_per90': 2.0},
        'Left Winger': {'height_in_cm': 1.0, 'goals_per90': 3.0, 'assists_per90': 4.0, 'yellows_per90': 0.5, 'recent_goal_contrib_per90': 2.0},
        'Attacking Midfield': {'height_in_cm': 1.0, 'goals_per90': 2.0, 'assists_per90': 5.0, 'yellows_per90': 0.5, 'recent_goal_contrib_per90': 2.0},
        'Central Midfield': {'height_in_cm': 1.5, 'goals_per90': 1.0, 'assists_per90': 3.5, 'yellows_per90': 2.0, 'recent_goal_contrib_per90': 1.0},
        'Defensive Midfield': {'height_in_cm': 2.5, 'goals_per90': 0.2, 'assists_per90': 1.0, 'yellows_per90': 4.0, 'recent_goal_contrib_per90': 0.5},
        'Centre-Back': {'height_in_cm': 5.0, 'goals_per90': 0.5, 'assists_per90': 0.2, 'yellows_per90': 3.5, 'recent_goal_contrib_per90': 0.2},
        'Left-Back': {'height_in_cm': 1.5, 'goals_per90': 0.5, 'assists_per90': 3.0, 'yellows_per90': 2.5, 'recent_goal_contrib_per90': 0.5},
        'Right-Back': {'height_in_cm': 1.5, 'goals_per90': 0.5, 'assists_per90': 3.0, 'yellows_per90': 2.5, 'recent_goal_contrib_per90': 0.5}
    }
    return weights_dict.get(sub_position, {'height_in_cm': 1.0, 'goals_per90': 1.0, 'assists_per90': 1.0, 'yellows_per90': 1.0, 'recent_goal_contrib_per90': 1.0})

def find_similar_players(df, player_name, min_total_minutes=600, min_recent_minutes=300, top_n=5, max_age=None, max_price=None):
    working_df = df.copy()
    
    if 'last_season' in working_df.columns:
        working_df = working_df[working_df['last_season'] >= (working_df['last_season'].max() - 1)]
        
    working_df['club_name'] = working_df['club_name'].replace(['0', 0, np.nan], 'Free Agent / Unknown')
    working_df['age'] = (pd.to_datetime('today') - pd.to_datetime(working_df['date_of_birth'], errors='coerce')).dt.days // 365
    working_df['height_in_cm'] = working_df['height_in_cm'].fillna(working_df['height_in_cm'].median())
    
    tier_1_leagues = ['GB1', 'ES1', 'IT1', 'L1', 'FR1', 'CL']
    working_df['league_multiplier'] = working_df.get('current_club_domestic_competition_id', pd.Series([''] * len(working_df))).apply(
        lambda x: 1.15 if str(x).upper() in tier_1_leagues else 1.0
    )
    
    working_df['goals_per90'] = ((working_df['goals'] / working_df['minutes_played']) * 90) * working_df['league_multiplier']
    working_df['assists_per90'] = ((working_df['assists'] / working_df['minutes_played']) * 90) * working_df['league_multiplier']
    working_df['yellows_per90'] = (working_df['yellow_cards'] / working_df['minutes_played']) * 90
    working_df = working_df.replace([np.inf, -np.inf], 0).fillna(0)
    
    working_df = working_df[(working_df['minutes_played'] >= min_total_minutes) & (working_df['recent_minutes'] >= min_recent_minutes)]
    
    features = ['height_in_cm', 'goals_per90', 'assists_per90', 'yellows_per90', 'recent_goal_contrib_per90']
    
    player_row = working_df[working_df['name'] == player_name]
    if player_row.empty:
        return None
    
    target_player = player_row.iloc[0]
    target_pos = target_player['sub_position']
    
    position_df = working_df[working_df['sub_position'] == target_pos].copy().reset_index(drop=True)
    target_idx = position_df[position_df['player_id'] == target_player['player_id']].index[0]
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(position_df[features])
    
    pos_weights_dict = get_position_weights(target_pos)
    weights_array = np.array([pos_weights_dict[feat] for feat in features])
    weighted_features = scaled_features * weights_array
    
    target_vector = weighted_features[target_idx].reshape(1, -1)
    distances = euclidean_distances(target_vector, weighted_features)[0]
    
    max_possible_dist = np.sqrt(np.sum(weights_array**2))
    raw_similarities = 1 - (distances / max_possible_dist)
    

    position_df['similarity_score'] = np.power(np.clip(raw_similarities, 0, 1), 3)
    
    similar_players = position_df[position_df['player_id'] != target_player['player_id']]
    
    if max_age:
        similar_players = similar_players[similar_players['age'] <= max_age]
    if max_price and 'market_value_in_eur' in similar_players.columns:
        similar_players = similar_players[similar_players['market_value_in_eur'] <= max_price]
        
    similar_players = similar_players.sort_values('similarity_score', ascending=False).head(top_n)
    
    cols = ['name', 'age', 'height_in_cm', 'club_name', 'goals_per90', 'assists_per90', 'similarity_score']
    if 'market_value_in_eur' in similar_players.columns:
        cols.insert(4, 'market_value_in_eur')
        
    return similar_players[cols]