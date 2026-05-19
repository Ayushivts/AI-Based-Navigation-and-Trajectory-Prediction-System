import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ============================================
# LOAD DATA
# ============================================

X = np.load("X.npy")
y = np.load("y.npy")

# ============================================
# DEFINE PARAMETERS
# ============================================

past_steps = 200
future_steps = 100

# ============================================
# DEFINE MODEL
# ============================================

class TrajectoryLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=7,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        # 100 future steps × 2 outputs
        self.fc = nn.Linear(64, 200)

    def forward(self, x):

        out, _ = self.lstm(x)

        # last timestep output
        out = out[:, -1, :]

        out = self.fc(out)

        # reshape to:
        # (batch, future_steps, features)
        out = out.view(-1, future_steps, 2)

        return out

# ============================================
# LOAD TRAINED MODEL
# ============================================

model = TrajectoryLSTM()

model.load_state_dict(
    torch.load("trajectory_model.pth")
)

model.eval()

print("Model Loaded Successfully!")

# ============================================
# SENSOR DENIAL SIMULATION
# ============================================

# choose one sample
sample_index = 5

# historical input sequence
input_sequence = X[sample_index]

# actual future trajectory
actual_future = y[sample_index]

# ============================================
# SENSOR DENIAL STARTS
# ============================================

print("\n--- SENSOR DENIED CONDITION ACTIVE ---")

# convert to tensor
input_tensor = torch.tensor(
    input_sequence,
    dtype=torch.float32
).unsqueeze(0)

# ============================================
# AI TRAJECTORY PREDICTION
# ============================================

with torch.no_grad():

    predicted_future = model(input_tensor)

predicted_future = predicted_future.numpy()[0]

print("Trajectory Prediction Completed!")

# ============================================
# CALCULATE ERROR
# ============================================

rmse = np.sqrt(
    mean_squared_error(
        actual_future.reshape(-1),
        predicted_future.reshape(-1)
    )
)

print(f"\nRMSE Error: {rmse:.6f}")

# ============================================
# PLOT RESULTS
# ============================================

plt.figure(figsize=(10,6))

# --------------------------------------------
# Actual Trajectory
# --------------------------------------------

plt.plot(
    actual_future[:,1],
    actual_future[:,0],
    marker='o',
    linewidth=2,
    label='Actual Trajectory'
)

# --------------------------------------------
# Predicted Trajectory
# --------------------------------------------

plt.plot(
    predicted_future[:,1],
    predicted_future[:,0],
    marker='x',
    linewidth=2,
    label='Predicted Trajectory'
)

# --------------------------------------------
# Sensor Denial Marker
# --------------------------------------------

plt.scatter(
    actual_future[0,1],
    actual_future[0,0],
    color='red',
    s=100,
    label='Sensor Denial Start'
)

# --------------------------------------------
# Labels
# --------------------------------------------

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "Trajectory Prediction During Sensor Denial"
)

plt.legend()

plt.grid(True)

# ============================================
# SAVE RESULT
# ============================================

plt.savefig(
    "sensor_denial_result.png"
)

plt.show()

print("\nResult Saved: sensor_denial_result.png")   