import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from yfin import get_option_chains_all
from transformer_1 import TransformerRegressor


# ===========================
# 1️⃣ Fetch and clean data
# ===========================
calls, puts = get_option_chains_all("AAPL", max_workers=4)
df = calls.copy()

# Drop NaN and infinite values
df = df.dropna(subset=["impliedVolatility"])
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Features and target
features = ["strike", "lastPrice", "bid",
            "ask", "volume", "openInterest", "TTM"]
target = "impliedVolatility"

# Remove extreme outliers (1st to 99th percentile)
df[features] = df[features].clip(lower=df[features].quantile(0.01),
                                 upper=df[features].quantile(0.99),
                                 axis=1)

X = df[features].values
y = df[target].values.reshape(-1, 1)

# ===========================
# 2️⃣ Scale features
# ===========================
scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Convert to tensors (batch_first=True → [batch, seq_len, input_dim])
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# ===========================
# 3️⃣ Model setup
# ===========================
model = TransformerRegressor(input_dim=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# ===========================
# 4️⃣ Training loop
# ===========================
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()

        # Prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for X_val, y_val in val_loader:
            preds = model(X_val)
            val_loss += loss_fn(preds, y_val).item()
        val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")

# ===========================
# 5️⃣ Save model
# ===========================
torch.save(model.state_dict(), "transformer_iv.pt")
print("✅ Training complete — model saved as transformer_iv.pt")
