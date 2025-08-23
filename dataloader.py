import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ======================
# Load Data
# ======================


class SensorSequenceDataset(Dataset):
    def __init__(self, sequence_ids, dataframe):
        super().__init__()
        self.sequence_ids = sequence_ids
        self.df = dataframe

        self.l = LabelEncoder()
        self.df['gesture_encoded'] = self.l.fit_transform(self.df['gesture'])

    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, idx):
        # Get the sequence_id
        seq_id = self.sequence_ids.iloc[idx] if hasattr(self.sequence_ids, 'iloc') else self.sequence_ids[idx]

        # Get all rows for this sequence_id
        sequence_data = self.df[self.df['sequence_id'] == seq_id]

        # Define sensor columns
        tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
        acc_cols = ['acc_x', 'acc_y', 'acc_z']
        rot_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
        thm_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']

        # Initialize lists for the sequence
        acc_sequence, rot_sequence, thm_sequence, tof_sequence = [], [], [], []

        for _, row in sequence_data.iterrows():
            # Handle NaN values
            acc = np.nan_to_num(row[acc_cols].values.astype(np.float32), nan=0.0)
            rot = np.nan_to_num(row[rot_cols].values.astype(np.float32), nan=0.0)
            thm = np.nan_to_num(row[thm_cols].values.astype(np.float32), nan=0.0)
            tof = np.nan_to_num(row[tof_cols].values.astype(np.float32), nan=0.0)

            acc_sequence.append(acc)
            rot_sequence.append(rot)
            thm_sequence.append(thm)
            tof_sequence.append(tof)

            target = row['gesture_encoded']  # Same for all rows in sequence

        # Convert to tensors
        # Shape: [sequence_length, num_features]
        acc_tensor = torch.tensor(np.array(acc_sequence), dtype=torch.float32)
        rot_tensor = torch.tensor(np.array(rot_sequence), dtype=torch.float32)
        thm_tensor = torch.tensor(np.array(thm_sequence), dtype=torch.float32)
        tof_tensor = torch.tensor(np.array(tof_sequence), dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.long)

        return acc_tensor, rot_tensor, thm_tensor, tof_tensor, target_tensor


