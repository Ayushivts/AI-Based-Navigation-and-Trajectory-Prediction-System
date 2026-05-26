import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================
# PARAMETERS
# ============================================

past_steps = 300
future_steps = 300

# ============================================
# LOAD DATA
# ============================================

X = np.load("X.npy")
y = np.load("y.npy")

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

        self.fc = nn.Linear(
            64,
            future_steps * 2
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        out = out.view(
            -1,
            future_steps,
            2
        )

        return out

# ============================================
# LOAD MODEL
# ============================================

model = TrajectoryLSTM()

model.load_state_dict(
    torch.load("trajectory_model.pth")
)

model.eval()

print("Model Loaded Successfully!")

# ============================================
# SELECT SAMPLE
# ============================================

sample_index = 10

current_sequence = X[
    sample_index
].copy()

# ============================================
# ACTUAL FUTURE
# ============================================

actual_5min = y[sample_index]

# next future block
actual_next_5min = y[
    sample_index + 1
]

actual_total = np.concatenate(
    [
        actual_5min,
        actual_next_5min
    ],
    axis=0
)

# ============================================
# RECURSIVE ROLLOUT
# ============================================

all_predictions = []

for rollout_step in range(2):

    input_tensor = torch.tensor(
        current_sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        prediction = model(
            input_tensor
        )

    prediction = prediction.numpy()[0]

    all_predictions.append(
        prediction
    )

    # ========================================
    # CREATE NEXT INPUT SEQUENCE
    # ========================================

    predicted_features = np.zeros(
        (future_steps, 7)
    )

    # lat lon
    predicted_features[:,0:2] = prediction

    # reuse previous motion values
    predicted_features[:,2:] = current_sequence[
        -future_steps:,
        2:
    ]

    # update rolling window
    current_sequence = predicted_features

# ============================================
# COMBINE PREDICTIONS
# ============================================

all_predictions = np.concatenate(
    all_predictions,
    axis=0
)

# ============================================
# PLOT RESULTS
# ============================================

plt.figure(figsize=(12,7))

# Actual trajectory
plt.plot(
    actual_total[:,1],
    actual_total[:,0],
    linewidth=2,
    label='Actual Future Path'
)

# Predicted trajectory
plt.plot(
    all_predictions[:,1],
    all_predictions[:,0],
    linewidth=2,
    label='Predicted 10-min Path'
)

# Sensor denial start
plt.scatter(
    actual_total[0,1],
    actual_total[0,0],
    color='red',
    s=100,
    label='Sensor Denial Start'
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "Recursive 10-Minute Maritime Trajectory Prediction"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "recursive_rollout_result.png"
)

plt.show()

print("\nRecursive Rollout Prediction Complete!")