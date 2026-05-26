import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# =========================
# LOAD DATA
# =========================

X = np.load("X.npy")
y = np.load("y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])
# =========================
# CONVERT TO TENSORS
# =========================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
)
# =========================
# CREATE DATALOADER
# =========================

train_dataset = TensorDataset(
    X_train,
    y_train
)


loader = DataLoader(
    train_dataset,
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

        self.fc = nn.Linear(64, 600)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        out = out.view(-1, 300, 2)

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

epochs = 100

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
# TEST EVALUATION
# =========================

model.eval()

with torch.no_grad():

    test_predictions = model(X_test)

    test_loss = criterion(
        test_predictions,
        y_test
    )

print("\nTest Loss:", test_loss.item())
# =========================
# SAVE MODEL
# =========================

torch.save(
    model.state_dict(),
    "trajectory_model.pth"
)

print("\nModel Saved Successfully!")