import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader

from src.models.cnn import EdgeCNNModel
from src.models.dataset import BTCDataset

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


def load_data_set():
    df = pd.read_csv("data/btc_data.csv")
    features = ["Open", "High", "Low", "Close", "Volume", "obv", "bb", "ema_21", "rsi"]
    raw = df[features].values

    split_idx = int(len(raw) * 0.8)
    train_raw = raw[:split_idx]
    test_raw = raw[split_idx:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    train_set = BTCDataset(train_scaled, window=60)
    test_set = BTCDataset(test_scaled, window=60)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    return train_loader, test_loader


train_loader, test_loader = load_data_set()
model = EdgeCNNModel().to(device)


def train():
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()  # xóa gradient cũ
            loss.backward()  # tính gradient mới (backpropagation)
            optimizer.step()  # cập nhật weights (gradient descent)

            train_loss += loss.item()

        model.eval()
        test_loss = 0
        with torch.no_grad():  # không tính gradient khi eval
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_loss += criterion(pred, y).item()

        # 5. In kết quả mỗi 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss / len(train_loader):.6f} | "
                f"Test Loss: {test_loss / len(test_loader):.6f}"
            )

    # 6. Save model
    torch.save(model.state_dict(), "btc_model.pth")


train()
