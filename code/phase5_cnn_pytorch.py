#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import time
import json

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class CNN1D(nn.Module):

    def __init__(self, n_channels=9, n_classes=6):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout3 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.global_pool(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)

        return x


class CNNClassifier:

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.device = device
        self.activities = {
            0: 'WALKING',
            1: 'WALKING_UPSTAIRS',
            2: 'WALKING_DOWNSTAIRS',
            3: 'SITTING',
            4: 'STANDING',
            5: 'LAYING'
        }

    def load_raw_signals(self):
        print("Loading raw signals...")
        data_path = Path("../UCI HAR Dataset")

        train_signals = []
        test_signals = []

        signal_names = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ]

        for signal_name in signal_names:
            train_file = data_path / "train" / "Inertial Signals" / f"{signal_name}_train.txt"
            test_file = data_path / "test" / "Inertial Signals" / f"{signal_name}_test.txt"
            train_signals.append(np.loadtxt(train_file))
            test_signals.append(np.loadtxt(test_file))

        X_train = np.stack(train_signals, axis=-1)
        X_test = np.stack(test_signals, axis=-1)

        y_train = np.loadtxt(data_path / "train" / "y_train.txt", dtype=int) - 1
        y_test = np.loadtxt(data_path / "test" / "y_test.txt", dtype=int) - 1

        print(f"✓ Loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

        return X_train, X_test, y_train, y_test

    def normalize_data(self, X_train, X_test):
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        n_timesteps = X_train.shape[1]
        n_channels = X_train.shape[2]

        X_train_flat = X_train.reshape(n_train, -1)
        X_test_flat = X_test.reshape(n_test, -1)

        X_train_normalized = self.scaler.fit_transform(X_train_flat)
        X_test_normalized = self.scaler.transform(X_test_flat)

        X_train_normalized = X_train_normalized.reshape(n_train, n_timesteps, n_channels)
        X_test_normalized = X_test_normalized.reshape(n_test, n_timesteps, n_channels)

        return X_train_normalized, X_test_normalized

    def prepare_data_loaders(self, X_train, y_train, X_test, y_test, batch_size=32):
        X_train_tensor = torch.FloatTensor(X_train).permute(0, 2, 1)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test).permute(0, 2, 1)
        y_test_tensor = torch.LongTensor(y_test)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        n_train = len(train_dataset)
        n_val = int(0.2 * n_train)
        n_train_split = n_train - n_val

        train_split, val_split = torch.utils.data.random_split(
            train_dataset, [n_train_split, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def train(self, train_loader, val_loader, n_epochs=100, learning_rate=0.001):
        self.model = CNN1D(n_channels=9, n_classes=6).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Device: {self.device}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        print("Training CNN...")
        start_time = time.time()

        best_val_acc = 0
        patience = 10
        patience_counter = 0

        train_losses = []
        val_accuracies = []

        for epoch in range(n_epochs):
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            val_acc = val_correct / val_total
            val_accuracies.append(val_acc)

            scheduler.step(1 - val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{n_epochs} - Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        self.model.load_state_dict(best_model_state)

        training_time = time.time() - start_time

        print(f"✓ Training completed in {training_time:.2f}s")
        print(f"  Best val accuracy: {best_val_acc:.4f}")

        return training_time, best_val_acc

    def evaluate(self, test_loader, y_test_original):
        self.model.eval()

        all_predictions = []
        all_labels = []

        start_time = time.time()

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        inference_time = time.time() - start_time

        y_pred = np.array(all_predictions)
        y_test = np.array(all_labels)

        accuracy = accuracy_score(y_test, y_pred)

        print(f"✓ Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        target_names = [self.activities[i] for i in range(6)]
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

        cm = confusion_matrix(y_test, y_pred)

        for i in range(6):
            mask = y_test == i
            if np.sum(mask) > 0:
                acc = accuracy_score(y_test[mask], y_pred[mask])

        return {
            'accuracy': accuracy,
            'inference_time': inference_time,
            'predictions': y_pred.tolist(),
            'confusion_matrix': cm.tolist()
        }


def main():
    print("PHASE 5: CNN IMPLEMENTATION (PyTorch)")
    print(f"Device: {device}")

    cnn = CNNClassifier()

    X_train, X_test, y_train, y_test = cnn.load_raw_signals()

    X_train_norm, X_test_norm = cnn.normalize_data(X_train, X_test)

    train_loader, val_loader, test_loader = cnn.prepare_data_loaders(
        X_train_norm, y_train, X_test_norm, y_test, batch_size=32
    )

    training_time, best_val_acc = cnn.train(train_loader, val_loader, n_epochs=100)

    results = cnn.evaluate(test_loader, y_test)
    results['training_time'] = training_time
    results['best_val_accuracy'] = best_val_acc

    with open('cnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(cnn.model.state_dict(), 'har_cnn_model.pth')
    print(f"✓ Model saved: har_cnn_model.pth")
    print(f"✓ Results saved: cnn_results.json")

    print("✓ PHASE 5 COMPLETED")

    return results


if __name__ == "__main__":
    results = main()
