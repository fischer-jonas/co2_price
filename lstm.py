import explore

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

def create_sequences_multi(data, target, seq_length,step):
    xs = []
    ys = []
    for i in range(0,len(data) - seq_length,step):
        x = data[i:i+seq_length]      # seq_length x num_features
        y_val = target[i+seq_length]  # nur das nächste price
        xs.append(x)
        ys.append(y_val)
    return np.array(xs), np.array(ys)

df_merged = explore.explore()

X = df_merged.values.astype(float)
y = df_merged["price"].values.astype(float)

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

SEQ_LENGTH = 365  
step=1
X_seq, y_seq = create_sequences_multi(X_scaled, y_scaled, SEQ_LENGTH,step)

TRAIN_SIZE = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:TRAIN_SIZE], X_seq[TRAIN_SIZE:]
y_train, y_test = y_seq[:TRAIN_SIZE], y_seq[TRAIN_SIZE:]

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

print(X_train.shape)  # (num_sequences_train, seq_length, num_features)
print(y_train.shape)  # (num_sequences_train, 1)

#%%%%%%%%%%%
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=156, hidden_size=128, num_layers=3, output_size=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]     
        out = self.fc(out)
        return out

model = LSTMRegressor(input_size = X_train.shape[2] , hidden_size=128, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dataset = torch.utils.data.TensorDataset(X_train, y_train)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

epochs = 50

for epoch in range(epochs):
    for batch_x, batch_y in loader:
        out = model(batch_x)
        loss = criterion(out, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}  Loss: {loss.item():.4f}")

#%%%%%%%%%%
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test, y_test),
    batch_size=64,
    shuffle=False
)

model.eval()
preds = []
actual = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        out = model(batch_x)
        preds.append(out.numpy())
        actual.append(batch_y.numpy())

preds = np.concatenate(preds).flatten()
actual = np.concatenate(actual).flatten()

preds = scaler_y.inverse_transform(preds.reshape(-1, 1))
actual = scaler_y.inverse_transform(actual.reshape(-1, 1))

mse  = mean_squared_error(actual, preds)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(actual, preds)
r2   = r2_score(actual, preds)

print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)
print("R²  :", r2)

plt.plot(actual)
plt.plot(preds)