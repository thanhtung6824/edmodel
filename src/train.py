from collections import Counter

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader

from src.models.cnn import EdgeCNNModel
from src.models.dataset import BTCDataset

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

features = ["Open", "High", "Low", "Close", "Volume", "obv", "bb", "ema_21", "rsi"]

def load_data_set():
    df = pd.read_csv("data/btc_4h.csv")
    raw = df[features].values

    split_idx = int(len(raw) * 0.8)
    train_raw = raw[:split_idx]
    test_raw = raw[split_idx:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    train_set = BTCDataset(train_scaled, train_raw, window=60)
    test_set = BTCDataset(test_scaled, train_raw, window=60)

    labels = [train_set[i][1].item() for i in range(len(train_set))]
    dist = Counter(labels)
    total = len(labels)
    print(f"\nLabel distribution:")
    print(f"  DOWN:    {dist[0]} ({dist[0] / total * 100:.1f}%)")
    print(f"  SIDEWAY: {dist[1]} ({dist[1] / total * 100:.1f}%)")
    print(f"  UP:      {dist[2]} ({dist[2] / total * 100:.1f}%)")
    print(f"  Total:   {total}\n")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


train_loader, test_loader = load_data_set()
model = EdgeCNNModel().to(device)

def train():
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    best_acc = 0
    counter = 0

    epochs = 100
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.squeeze().to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (pred.argmax(1) == y).sum().item()
            train_total += y.size(0)

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.squeeze().to(device)
                pred = model(x)
                test_correct += (pred.argmax(1) == y).sum().item()
                test_total += y.size(0)

        train_acc = train_correct / train_total * 100
        test_acc = test_correct / test_total * 100

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Loss: {train_loss / len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Test Acc: {test_acc:.1f}%")

        # Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  → Saved best model (Test Acc: {best_acc:.1f}%)")
        else:
            counter += 1
            if counter >= 15:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break


train()
