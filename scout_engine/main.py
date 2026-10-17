import pandas as pd
import os
from similarity import find_similar_players

def run_scout_engine():
    processed_path = os.path.join("data", "processed", "scout_master.csv")
    df = pd.read_csv(processed_path)
    
    target = "Erling Haaland"
    print(f"\n--- PURE PERFORMANCE & FORM SEARCH: {target.upper()} ---")
    results = find_similar_players(df, player_name=target, min_total_minutes=800, min_recent_minutes=400, top_n=5)
    
    if isinstance(results, pd.DataFrame):
        print(results.to_string(index=False))
    else:
        print(results)

if __name__ == "__main__":
    run_scout_engine()