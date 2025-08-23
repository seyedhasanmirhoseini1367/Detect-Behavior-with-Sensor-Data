import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from Kaggle.sensorydata import dataloader
from Kaggle.sensorydata import models

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DataConversionWarning)
# ======================
# Load Data
# ======================
train_path = r"D:\Kaggle\cmi-detect-behavior-with-sensor-data\train.csv"
test_path = r"D:\Kaggle\cmi-detect-behavior-with-sensor-data\test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# ======================
# Define columns
# ======================
acc_cols = ['acc_x', 'acc_y', 'acc_z']
rot_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
thm_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]

# ======================
# Train/Val split
# ======================
train_ids, val_ids = train_test_split(train['sequence_id'].unique(), test_size=0.2, random_state=42)

# ======================
# Collate fn for padding sequences
# ======================
from torch.nn.utils.rnn import pad_sequence
import torch


def collate_fn(batch, is_test=False):
    """
    batch: list of items from Dataset
    is_test: True if batch comes from test set (no target)
    """
    if is_test:
        # unpack features
        acc, rot, thm, tof = zip(*batch)
        acc_padded = pad_sequence(acc, batch_first=True)  # (B, T_max, 3)
        rot_padded = pad_sequence(rot, batch_first=True)  # (B, T_max, 4)
        thm_padded = pad_sequence(thm, batch_first=True)  # (B, T_max, 5)
        tof_padded = pad_sequence(tof, batch_first=True)  # (B, T_max, 320)
        return acc_padded, rot_padded, thm_padded, tof_padded
    else:
        acc, rot, thm, tof, target = zip(*batch)
        acc_padded = pad_sequence(acc, batch_first=True)
        rot_padded = pad_sequence(rot, batch_first=True)
        thm_padded = pad_sequence(thm, batch_first=True)
        tof_padded = pad_sequence(tof, batch_first=True)
        target_tensor = torch.stack(target)
        return acc_padded, rot_padded, thm_padded, tof_padded, target_tensor


# ======================
# DataLoaders
# ======================

# Create train dataset first to fit scalers and label encoder
train_dataset = dataloader.SensorSequenceDataset(train_ids[:500], train, is_test=False)

# Use the fitted scalers and label encoder for validation and test
val_dataset = dataloader.SensorSequenceDataset(val_ids[:200], train, is_test=False,
                                               label_encoder=train_dataset.le,
                                               scalers=train_dataset.scalers)

test_dataset = dataloader.SensorSequenceDataset(test['sequence_id'].unique(), test, is_test=True,
                                                label_encoder=train_dataset.le,
                                                scalers=train_dataset.scalers)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True,
                          collate_fn=lambda x: collate_fn(x, is_test=False))
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False,
                        collate_fn=lambda x: collate_fn(x, is_test=False))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                         collate_fn=lambda x: collate_fn(x, is_test=True))

# ======================
# Training Setup
# ======================
model = models.Fusion()

# Define loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# ======================
# Training Loop
# ======================
epochs = 50
best_val_loss = float('inf')

for epoch in range(epochs):
    # ==========================
    # TRAINING PHASE
    # ==========================
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        acc_tensor, rot_tensor, thm_tensor, tof_tensor, target_tensor = batch
        optimizer.zero_grad()

        # Forward pass
        outputs = model(acc_tensor, rot_tensor, thm_tensor, tof_tensor)
        target_tensor = target_tensor.long()

        # Compute loss
        loss = criterion(outputs, target_tensor)

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == target_tensor).sum().item()
        total += target_tensor.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    # ==========================
    # VALIDATION PHASE
    # ==========================
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            acc_tensor, rot_tensor, thm_tensor, tof_tensor, target_tensor = batch
            outputs = model(acc_tensor, rot_tensor, thm_tensor, tof_tensor)
            target_tensor = target_tensor.long()

            loss = criterion(outputs, target_tensor)
            val_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == target_tensor).sum().item()
            val_total += target_tensor.size(0)

    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total

    # Update learning rate
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Saved best model with val loss: {val_loss:.4f}")

    print(f"Epoch [{epoch + 1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

# ======================
# Load best model for testing
# ======================
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

print("Training completed. Best model saved as 'best_model.pth'")