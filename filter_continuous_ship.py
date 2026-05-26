import pandas as pd

# ====================================
# LOAD DATA
# ====================================

df = pd.read_csv(
    "ship_trajectory_data.csv"
)

# ====================================
# CONVERT TIMESTAMP
# ====================================

df['timestamp'] = pd.to_datetime(
    df['timestamp'],
    unit='s'
)

# ====================================
# SELECT ONE SHIP
# ====================================

ship_id = df['mmsi'].value_counts().idxmax()

print("Selected Ship:", ship_id)

df = df[
    df['mmsi'] == ship_id
]

# ====================================
# SORT BY TIME
# ====================================

df = df.sort_values(
    by='timestamp'
)

# ====================================
# CALCULATE TIME DIFFERENCE
# ====================================

df['time_diff'] = df[
    'timestamp'
].diff().dt.total_seconds()

# ====================================
# KEEP ONLY SMALL GAPS
# ====================================

df = df[
    (df['time_diff'] <= 60)
    | (df['time_diff'].isna())
]

# ====================================
# RESET INDEX
# ====================================

df = df.reset_index(drop=True)

# ====================================
# SAVE CLEAN TRACK
# ====================================

df.to_csv(
    "continuous_ship_trajectory.csv",
    index=False
)

print("\nContinuous trajectory saved!")

print(df.head())

print("\nTotal Points:", len(df))