import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

# =========================
# LOAD DATA
# =========================

X = np.load("X.npy")
y = np.load("y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# CONVERT TO TENSORS
# =========================

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# =========================
# CREATE DATALOADER
# =========================

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
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

        self.fc = nn.Linear(64, 20)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        out = out.view(-1, 10, 2)

        return out

# =========================
# CREATE MODEL
# =========================

model = TrajectoryLSTM()

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================
# TRAINING LOOP
# =========================

epochs = 50

for epoch in range(epochs):

    total_loss = 0

    for batch_X, batch_y in loader:

        predictions = model(batch_X)

        loss = criterion(
            predictions,
            batch_y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {total_loss:.6f}"
    )

# =========================
# SAVE MODEL
# =========================

torch.save(
    model.state_dict(),
    "trajectory_model.pth"
)

print("\nModel Saved Successfully!")