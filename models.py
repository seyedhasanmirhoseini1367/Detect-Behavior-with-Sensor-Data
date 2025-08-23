import torch
import torch.nn as nn


# ===============================================================

class GRUEncoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, bidirectional=False):
        """
        General GRU encoder for any time series modality
        Args:
            input_size: number of features per timestep
            hidden_size: GRU hidden size
            bidirectional: use bidirectional GRU or not
        """
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        Returns:
            pooled_embedding: mean over time
            gru_out: GRU outputs for each timestep
        """
        gru_out, _ = self.gru(x)
        pooled_embedding = gru_out.mean(dim=1)
        return pooled_embedding


# def model(name, data, input_size, hidden_size):
#     model = GRUEncoder(input_size=input_size, hidden_size=hidden_size)
#     pooled_emb_imu, gru_out_imu = model(data)
#     print(f"{name} pooled embedding shape:, {pooled_emb_imu.shape}")
#     print(f"{name} GRU shape:, {gru_out_imu.shape}")
#
# imu_data = torch.randn(1, 57, 3) # batch=1, seq_len=57, feature_dim=3
# thm_data = torch.randn(1, 57, 5) # batch=1, seq_len=57, feature_dim=5
# rot_data = torch.randn(1, 57, 4) # batch=1, seq_len=57, feature_dim=4
# model('IMU', imu_data, input_size=3, hidden_size=32)

# ===============================================================

class TOFEncoder(nn.Module):
    def __init__(self, in_channels=5, conv_channels=[16, 32], kernel_size=(3, 3), pool_size=(2, 2), gru_hidden_size=64):
        super(TOFEncoder, self).__init__()

        # CNN for spatial features
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=conv_channels[0], kernel_size=kernel_size[0], padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=kernel_size[1],
                      padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((pool_size[0], pool_size[1]))
        )

        self.gru_input_size = conv_channels[-1] * pool_size[0] * pool_size[1]
        self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=gru_hidden_size, batch_first=True)

        # Define output_dim for Fusion
        self.output_dim = self.gru_input_size + gru_hidden_size  # CNN + GRU concatenated

    def forward(self, x):
        B, T, F = x.shape  # (batch, seq_len, feature_dim=320)
        x = x.view(B, T, 5, 8, 8)  # reshape to (B, T, 5, 8, 8)

        # merge batch and time for CNN
        x_reshaped = x.view(B * T, 5, 8, 8)

        # CNN forward
        cnn_out = self.cnn(x_reshaped)  # (B*T, 32, 2, 2)
        cnn_out_flat = cnn_out.view(B, T, -1)  # (B, T, 128)

        # GRU forward
        gru_out, _ = self.gru(cnn_out_flat)  # (B, T, 64)

        # Concatenate CNN + GRU features
        combined_features = torch.cat([cnn_out_flat, gru_out], dim=-1)  # (B, T, 192)
        pooled_embedding = combined_features.mean(dim=1)  # (B, 192)

        return pooled_embedding


#
# # Example test
# data = torch.randn(1, 57, 320)  # batch=1, seq_len=57, feature_dim=320
# model = TOFEncoder()
# pooled_embedding, combined_features, cnn_features, gru_features = model(data)
#
# print("Pooled Embedding shape:", pooled_embedding.shape)
# print("Combined features shape:", combined_features.shape)
# print("CNN features shape:", cnn_features.shape)
# print("GRU features shape:", gru_features.shape)


# ======================================================================


class Fusion(nn.Module):
    def __init__(self, acc_dim=3, acc_hidden_size=32, rot_dim=4, rot_hidden_size=64,
                 thm_dim=5, thm_hidden_size=32, num_classes=18):
        super(Fusion, self).__init__()

        # GRU encoders
        self.acc_gru = GRUEncoder(input_size=acc_dim, hidden_size=acc_hidden_size)
        self.rot_gru = GRUEncoder(input_size=rot_dim, hidden_size=rot_hidden_size)
        self.thm_gru = GRUEncoder(input_size=thm_dim, hidden_size=thm_hidden_size)

        # TOF encoder
        self.tof_encoder = TOFEncoder()  # TOFEncoder returns pooled_embedding of size output_dim

        # Compute in_features dynamically
        self.in_features = acc_hidden_size + rot_hidden_size + thm_hidden_size + self.tof_encoder.output_dim

        # Classification MLP
        self.classification = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, acc_tensor, rot_tensor, thm_tensor, tof_tensor):
        pooled_acc = self.acc_gru(acc_tensor)  # (batch, acc_hidden_size)
        pooled_rot = self.rot_gru(rot_tensor)  # (batch, rot_hidden_size)
        pooled_thm = self.thm_gru(thm_tensor)  # (batch, thm_hidden_size)
        pooled_tof = self.tof_encoder(tof_tensor)  # (batch, tof_feature_dim)

        # Concatenate all features
        fused = torch.cat([pooled_acc, pooled_rot, pooled_thm, pooled_tof], dim=1)

        # Classification
        out = self.classification(fused)
        return out
