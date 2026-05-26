import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("trajectory_data.csv")

print("Original Data:")
print(df.head())

# =========================
# NORMALIZE DATA
# =========================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

print("\nScaled Data Shape:")
print(scaled_data.shape)

# =========================
# CREATE SEQUENCES
# =========================

past_steps = 300
future_steps = 300

X = []
y = []

for i in range(len(scaled_data) - past_steps - future_steps):

    # Past sequence
    X.append(
        scaled_data[i:i+past_steps]
    )

    # Future sequence
    y.append(
        scaled_data[
            i+past_steps :
            i+past_steps+future_steps,
            0:2
        ]
    )

X = np.array(X)
y = np.array(y)

# =========================
# PRINT SHAPES
# =========================

print("\nInput Shape:")
print(X.shape)

print("\nOutput Shape:")
print(y.shape)

# =========================
# SAVE FILES
# =========================

np.save("X.npy", X)
np.save("y.npy", y)

print("\nSequences Saved Successfully!")