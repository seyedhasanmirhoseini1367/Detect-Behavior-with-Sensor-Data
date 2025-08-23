import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class SensorSequenceDataset(Dataset):
    def __init__(self, sequence_ids, dataframe, is_test=False, label_encoder=None, scalers=None):
        """
        sequence_ids: list of unique sequence IDs
        dataframe: full dataframe with all sequences
        is_test: True for test set (no targets)
        label_encoder: fitted LabelEncoder for gestures (required if not training)
        scalers: dict of fitted StandardScalers for each modality
        """
        self.sequence_ids = sequence_ids
        self.df = dataframe
        self.is_test = is_test

        # Define columns
        self.acc_cols = ['acc_x', 'acc_y', 'acc_z']
        self.rot_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
        self.thm_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
        self.tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]

        # Label encoding
        if not is_test:
            if label_encoder is None:
                self.le = LabelEncoder()
                self.df['gesture_encoded'] = self.le.fit_transform(self.df['gesture'])
            else:
                self.le = label_encoder
                self.df['gesture_encoded'] = self.le.transform(self.df['gesture'])

        # Feature scaling - only use data from provided sequence_ids for fitting
        if scalers is None:
            # Fit scalers only on the current dataset sequences (usually train)
            train_data = self.df[self.df['sequence_id'].isin(self.sequence_ids)]
            self.scalers = {
                'acc': StandardScaler().fit(train_data[self.acc_cols]),
                'rot': StandardScaler().fit(train_data[self.rot_cols]),
                'thm': StandardScaler().fit(train_data[self.thm_cols]),
                'tof': StandardScaler().fit(train_data[self.tof_cols])
            }
        else:
            self.scalers = scalers

    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, idx):
        seq_id = self.sequence_ids[idx]
        sequence_data = self.df[self.df['sequence_id'] == seq_id]

        # Get target from first row (should be same for all rows in sequence)
        if not self.is_test:
            target = sequence_data.iloc[0]['gesture_encoded']

        # Extract all data at once and fill NaNs
        acc_data = sequence_data[self.acc_cols].fillna(0).values.astype(np.float32)
        rot_data = sequence_data[self.rot_cols].fillna(0).values.astype(np.float32)
        thm_data = sequence_data[self.thm_cols].fillna(0).values.astype(np.float32)
        tof_data = sequence_data[self.tof_cols].fillna(0).values.astype(np.float32)

        # Normalize using fitted scalers (batch transform for efficiency)
        acc_normalized = self.scalers['acc'].transform(acc_data)
        rot_normalized = self.scalers['rot'].transform(rot_data)
        thm_normalized = self.scalers['thm'].transform(thm_data)
        tof_normalized = self.scalers['tof'].transform(tof_data)

        # Convert to tensors
        acc_tensor = torch.tensor(acc_normalized, dtype=torch.float32)
        rot_tensor = torch.tensor(rot_normalized, dtype=torch.float32)
        thm_tensor = torch.tensor(thm_normalized, dtype=torch.float32)
        tof_tensor = torch.tensor(tof_normalized, dtype=torch.float32)

        if self.is_test:
            return acc_tensor, rot_tensor, thm_tensor, tof_tensor
        else:
            target_tensor = torch.tensor(target, dtype=torch.long)
            return acc_tensor, rot_tensor, thm_tensor, tof_tensor, target_tensor

