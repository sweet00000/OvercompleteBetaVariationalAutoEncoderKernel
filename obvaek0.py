import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# We import PyTorch locally inside the fit method if possible, 
# or globally, but we don't rely on it for inference.
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. THE INTERNAL PYTORCH ENGINE
# ==========================================
class _InternalOBVAE(nn.Module):
    """ The temporary PyTorch engine used only during training. """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder
        self.enc_hidden = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.enc_hidden(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decoder(z), mu, logvar


# ==========================================
# 2. THE SCIKIT-LEARN TRANSFORMER (OBVAEK)
# ==========================================
class OBVAEK(BaseEstimator, TransformerMixin):
    """
    Overcomplete Beta-VAE Kernel (OBVAEK)
    Auto-senses the required non-linear orthogonal dimension,
    trains via PyTorch, and executes in pure NumPy.
    """
    def __init__(self, latent_dim='auto', expansion_factor=10, hidden_dim=64, beta=2.0, epochs=300, lr=0.005):
        self.latent_dim = latent_dim
        self.expansion_factor = expansion_factor
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.epochs = epochs
        self.lr = lr
        
        # NumPy Weight Matrices
        self.W_h = None
        self.b_h = None
        self.W_mu = None
        self.b_mu = None
        
        self.effective_dim_ = None 
        self.is_fitted_ = False

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float32)
        input_dim = X_arr.shape[1]
        
        # === AUTO-SENSE LOGIC ===
        if self.latent_dim == 'auto':
            training_dim = min(input_dim * self.expansion_factor, 500) 
        else:
            training_dim = self.latent_dim
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _InternalOBVAE(input_dim, self.hidden_dim, training_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        X_tensor = torch.FloatTensor(X_arr).to(device)
        
        # --- PyTorch Training Loop ---
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            recon, mu, logvar = model(X_tensor)
            mse_loss = nn.functional.mse_loss(recon, X_tensor, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse_loss + (self.beta * kld_loss)
            loss.backward()
            optimizer.step()
            
        # --- The NumPy Extraction & Pruning ---
        model.eval()
        with torch.no_grad():
            _, final_mu, _ = model(X_tensor)
            node_variances = torch.var(final_mu, dim=0).cpu().numpy()
            
            # Keep only nodes with meaningful variance
            active_indices = np.where(node_variances > 1e-4)[0]
            self.effective_dim_ = len(active_indices)
            
            print(f"OBVAEK: Inflated to {training_dim}D, Pruned down to {self.effective_dim_} active features.")

            self.W_h = model.enc_hidden[0].weight.cpu().numpy()
            self.b_h = model.enc_hidden[0].bias.cpu().numpy()
            
            # Slice matrices to drop the mathematical dead weight
            self.W_mu = model.fc_mu.weight.cpu().numpy()[active_indices, :]
            self.b_mu = model.fc_mu.bias.cpu().numpy()[active_indices]
            
        self.is_fitted_ = True
        
        # Cleanup memory
        del model
        del X_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return self

    def transform(self, X):
        """ Executes the coordinate warp in 100% pure NumPy. """
        if not self.is_fitted_:
            raise RuntimeError("You must fit the OBVAEK before calling transform.")
            
        X_arr = np.asarray(X, dtype=np.float32)
        
        # H = ReLU(X * W_h^T + b_h)
        Z_hidden = np.dot(X_arr, self.W_h.T) + self.b_h
        H = np.maximum(0, Z_hidden) 
        
        # Mu = H * W_mu^T + b_mu
        Mu = np.dot(H, self.W_mu.T) + self.b_mu
        
        return Mu