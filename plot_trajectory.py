import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("trajectory_data.csv")

print(df.head())

# Create figure
plt.figure(figsize=(10,6))

# Plot trajectory
plt.plot(df['lon'], df['lat'], color='blue')

# Labels
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Title
plt.title("Vehicle Trajectory")

# Grid
plt.grid(True)

# Save image
plt.savefig("trajectory.png")

print("Trajectory image saved!")

# Show plot
plt.show()