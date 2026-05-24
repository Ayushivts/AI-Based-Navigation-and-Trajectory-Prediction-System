import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
# =========================
# LOAD DATA
# =========================

X = np.load("X.npy")
y = np.load("y.npy")

# =========================
# CONVERT TO TENSOR
# =========================

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)

# =========================
# DEFINE MODEL
# =========================

class TrajectoryLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=7,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(64, 200)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        out = out.view(-1, 100, 2)

        return out

# =========================
# LOAD MODEL
# =========================

model = TrajectoryLSTM()

model.load_state_dict(
    torch.load("trajectory_model.pth")
)

model.eval()

# =========================
# SELECT SAMPLE
# =========================

sample_index = 5

input_sequence = X_tensor[
    sample_index:sample_index+1
]

actual_future = y[sample_index]

# =========================
# PREDICT
# =========================

with torch.no_grad():

    predicted_future = model(
        input_sequence
    )

predicted_future = predicted_future.numpy()[0]


# ============================================
# KALMAN FILTER SMOOTHING
# ============================================

kf = KalmanFilter(dim_x=2, dim_z=2)

kf.x = np.array([
    predicted_future[0,0],
    predicted_future[0,1]
])

kf.F = np.array([
    [1,0],
    [0,1]
])

kf.H = np.array([
    [1,0],
    [0,1]
])

kf.P *= 1000
kf.R *= 0.01
kf.Q *= 0.001

smoothed_predictions = []

for point in predicted_future:

    kf.predict()

    kf.update(point)

    smoothed_predictions.append(
        kf.x.copy()
    )

smoothed_predictions = np.array(
    smoothed_predictions
)
# =========================
# PLOT RESULTS
# =========================

plt.figure(figsize=(8,6))

# Actual trajectory
plt.plot(
    actual_future[:,1],
    actual_future[:,0],
    marker='o',
    label='Actual'
)

# Predicted trajectory
plt.plot(
    smoothed_predictions[:,1],
    smoothed_predictions[:,0],
    marker='x',
    label='Predicted'
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title("Trajectory Prediction")

plt.legend()

plt.grid()

plt.savefig("prediction_result.png")

plt.show()

print("\nPrediction Complete!")