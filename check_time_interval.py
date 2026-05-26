import pandas as pd

df = pd.read_csv(
    "ship_trajectory_data.csv"
)

# convert timestamp
df['timestamp'] = pd.to_datetime(
    df['timestamp'],
    unit='s'
)

# sort
df = df.sort_values(by='timestamp')

# calculate differences
time_diff = df['timestamp'].diff()

print(time_diff.describe())