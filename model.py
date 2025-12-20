import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from features import prepare_dataset
from result import plot_results, calculate_metrics

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

WINDOW = 40
MIN_SNR = 0.001
test_size = 0.1

x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled, x_scaler, y_scaler = prepare_dataset(WINDOW, MIN_SNR, test_size)

# Before training
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32).to(device).unsqueeze(1)
x_test_tensor  = torch.tensor(x_test_scaled, dtype=torch.float32).to(device).unsqueeze(1)

# Labels (already 2D)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
y_test_tensor  = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Fully connected output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.fc(out)
        return out


model = PredictionModel(input_dim=x_train_scaled.shape[1], hidden_dim=64, num_layers=3, output_dim=1).to(device) 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
num_epochs = 400

for epoch in range(num_epochs):
    y_train_pred = model(x_train_tensor)
    loss = criterion(y_train_pred, y_train_tensor) 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


model.eval()

y_test_pred = model(x_test_tensor)

y_train_pred = y_scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
y_train_real  = y_scaler.inverse_transform(y_train_tensor.detach().cpu().numpy())

y_test_pred = y_scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
y_test_real  = y_scaler.inverse_transform(y_test_tensor.detach().cpu().numpy())

plot_results(y_test_real, y_test_pred, title="Test Set: True vs Predicted Values", xlabel="Samples", ylabel="Values")
rmse, r2 = calculate_metrics(y_test_real, y_test_pred)
print(f"Test RMSE: {rmse}")
print(f"Test R-squared: {r2}")
