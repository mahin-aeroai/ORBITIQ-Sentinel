"""
ORBITIQ Sentinel — LSTM Autoencoder Anomaly Detection Model
"""
import torch
import torch.nn as nn
import numpy as np

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim=64, latent_dim=32, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim  = hidden_dim
        self.latent_dim  = latent_dim

        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.output_fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        enc_out, (h, _) = self.encoder(x)
        # Use last hidden state
        latent = self.enc_fc(h[-1])               # (batch, latent_dim)

        # Decode: repeat latent across seq_len
        dec_inp = self.dec_fc(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_inp)
        recon = self.output_fc(dec_out)           # (batch, seq_len, n_features)
        return recon, latent

    def reconstruction_error(self, x):
        """Per-sample mean reconstruction error"""
        self.eval()
        with torch.no_grad():
            recon, _ = self.forward(x)
            err = torch.mean((x - recon) ** 2, dim=(1, 2))
        return err.cpu().numpy()

    def feature_errors(self, x):
        """Per-feature reconstruction error for explainability"""
        self.eval()
        with torch.no_grad():
            recon, _ = self.forward(x)
            err = torch.mean((x - recon) ** 2, dim=1)  # (batch, n_features)
        return err.cpu().numpy()


def train_model(X_train, n_features, epochs=40, batch_size=64, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = LSTMAutoencoder(n_features=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = nn.MSELoss()

    dataset = torch.FloatTensor(X_train).to(device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dataset, dataset),
        batch_size=batch_size, shuffle=True
    )

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for xb, _ in loader:
            optimizer.zero_grad()
            recon, _ = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_train)
        scheduler.step()
        history.append(epoch_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {epoch_loss:.6f}")

    return model, history, device


def compute_threshold(model, X_train, device, percentile=95):
    """Set anomaly threshold from training reconstruction errors"""
    model.eval()
    tensor = torch.FloatTensor(X_train).to(device)
    errors = model.reconstruction_error(tensor)
    threshold = np.percentile(errors, percentile)
    print(f"  Threshold (p{percentile}): {threshold:.6f}")
    return threshold, errors
