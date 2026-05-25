import pandas as pd
import matplotlib.pyplot as plt

# ====================================
# LOAD DATA
# ====================================

df = pd.read_csv(
    "ship_trajectory_data.csv"
)

# ====================================
# SELECT ONE SHIP
# ====================================

ship_id = df['mmsi'].iloc[0]

df = df[
    df['mmsi'] == ship_id
]

# ====================================
# PLOT TRAJECTORY
# ====================================

plt.figure(figsize=(10,6))

plt.plot(
    df['lon'],
    df['lat']
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title("Ship Trajectory")

plt.grid(True)

plt.savefig(
    "ship_trajectory.png"
)

plt.show()


# import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load CSV
# df = pd.read_csv("trajectory_data.csv")

# print(df.head())

# # Create figure
# plt.figure(figsize=(10,6))

# # Plot trajectory
# plt.plot(df['lon'], df['lat'], color='blue')

# # Labels
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")

# # Title
# plt.title("Vehicle Trajectory")

# # Grid
# plt.grid(True)

# # Save image
# plt.savefig("trajectory.png")

# print("Trajectory image saved!")

# # Show plot
# plt.show()