import fastf1
import pandas as pd

# Load the actual Race results for China 2026
session = fastf1.get_session(2026, 2, 'R')
session.load(telemetry=False, weather=False, laps=False)

results = session.results
if results is not None and not results.empty:
    print("\n--- ACTUAL CHINESE GP 2026 RESULTS ---")
    df = results[['Position', 'Abbreviation', 'Status', 'Points']]
    print(df.to_string(index=False))
else:
    print("Could not load race results.")
