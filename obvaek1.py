import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class _InternalOBVAEK:
    """
    Pure NumPy VAE engine with manual backpropagation.
    Replicates: Linear -> ReLU -> Linear (mu/logvar) -> sampling -> Linear -> ReLU -> Linear
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Xavier/Glorot initialization
        def init_weights(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, (fan_out, fan_in))
        
        # Encoder weights
        self.W_enc = init_weights(input_dim, hidden_dim)  # (hidden_dim, input_dim)
        self.b_enc = np.zeros(hidden_dim)
        
        self.W_mu = init_weights(hidden_dim, latent_dim)   # (latent_dim, hidden_dim)
        self.b_mu = np.zeros(latent_dim)
        
        self.W_logvar = init_weights(hidden_dim, latent_dim)  # (latent_dim, hidden_dim)
        self.b_logvar = np.zeros(latent_dim)
        
        # Decoder weights
        self.W_dec1 = init_weights(latent_dim, hidden_dim)  # (hidden_dim, latent_dim)
        self.b_dec1 = np.zeros(hidden_dim)
        
        self.W_dec2 = init_weights(hidden_dim, input_dim)   # (input_dim, hidden_dim)
        self.b_dec2 = np.zeros(input_dim)
        
        # Cache for backprop
        self.cache = {}
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_deriv(self, x):
        return (x > 0).astype(float)
    
    def encode(self, x):
        """x: (batch_size, input_dim)"""
        # h = relu(x @ W_enc.T + b_enc)
        z_enc = x @ self.W_enc.T + self.b_enc  # (batch, hidden_dim)
        h = self.relu(z_enc)
        
        mu = h @ self.W_mu.T + self.b_mu       # (batch, latent_dim)
        logvar = h @ self.W_logvar.T + self.b_logvar  # (batch, latent_dim)
        
        self.cache['encode'] = (x, z_enc, h, mu, logvar)
        return mu, logvar, h
    
    def reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        z = mu + std * eps
        self.cache['z'] = (mu, logvar, std, eps, z)
        return z
    
    def decode(self, z):
        """z: (batch_size, latent_dim)"""
        # h_dec = relu(z @ W_dec1.T + b_dec1)
        z_dec1 = z @ self.W_dec1.T + self.b_dec1  # (batch, hidden_dim)
        h_dec = self.relu(z_dec1)
        
        x_recon = h_dec @ self.W_dec2.T + self.b_dec2  # (batch, input_dim)
        
        self.cache['decode'] = (z, z_dec1, h_dec, x_recon)
        return x_recon
    
    def forward(self, x):
        mu, logvar, h_enc = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def compute_loss(self, x_recon, x, mu, logvar, beta):
        """
        MSE reconstruction loss + Beta * KL divergence
        """
        # MSE loss (sum over batch and features)
        mse_loss = np.sum((x_recon - x) ** 2)
        
        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kld_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
        
        loss = mse_loss + beta * kld_loss
        return loss, mse_loss, kld_loss
    
    def backward(self, x, x_recon, mu, logvar, beta):
        """
        Manual backpropagation through the entire VAE.
        Returns gradients for all parameters.
        """
        batch_size = x.shape[0]
        
        # --- Decoder gradients ---
        z, z_dec1, h_dec, _ = self.cache['decode']
        mu_cache, logvar_cache, std, eps, z_sample = self.cache['z']
        
        # dL/dx_recon (MSE derivative: 2*(x_recon - x))
        dx_recon = 2 * (x_recon - x)  # (batch, input_dim)
        
        # dL/dW_dec2, dL/db_dec2
        dW_dec2 = dx_recon.T @ h_dec  # (input_dim, hidden_dim)
        db_dec2 = np.sum(dx_recon, axis=0)  # (input_dim,)
        
        # dL/dh_dec
        dh_dec = dx_recon @ self.W_dec2  # (batch, hidden_dim)
        dz_dec1 = dh_dec * self.relu_deriv(z_dec1)  # (batch, hidden_dim)
        
        # dL/dW_dec1, dL/db_dec1
        dW_dec1 = dz_dec1.T @ z  # (hidden_dim, latent_dim)
        db_dec1 = np.sum(dz_dec1, axis=0)  # (hidden_dim,)
        
        # dL/dz (from decoder + from KL if z is mu + std*eps)
        dz = dz_dec1 @ self.W_dec1  # (batch, latent_dim)
        
        # --- KL divergence gradients w.r.t mu and logvar ---
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # dKL/dmu = -0.5 * (-2*mu) = mu
        # dKL/dlogvar = -0.5 * (1 - exp(logvar))
        dmu_kl = mu
        dlogvar_kl = -0.5 * (1 - np.exp(logvar))
        
        # Reparameterization: z = mu + std * eps, std = exp(0.5*logvar)
        # dz/dmu = 1, dz/dlogvar = 0.5 * std * eps
        dmu = dz + beta * dmu_kl  # (batch, latent_dim)
        dlogvar = dz * (0.5 * std * eps) + beta * dlogvar_kl  # (batch, latent_dim)
        
        # --- Encoder gradients ---
        x_orig, z_enc, h_enc, _, _ = self.cache['encode']
        
        # Gradients through mu and logvar branches
        dW_mu = dmu.T @ h_enc  # (latent_dim, hidden_dim)
        db_mu = np.sum(dmu, axis=0)  # (latent_dim,)
        
        dW_logvar = dlogvar.T @ h_enc  # (latent_dim, hidden_dim)
        db_logvar = np.sum(dlogvar, axis=0)  # (latent_dim,)
        
        # Backprop to hidden layer
        dh_mu = dmu @ self.W_mu  # (batch, hidden_dim)
        dh_logvar = dlogvar @ self.W_logvar  # (batch, hidden_dim)
        dh_enc = dh_mu + dh_logvar
        
        dz_enc = dh_enc * self.relu_deriv(z_enc)  # (batch, hidden_dim)
        
        # dL/dW_enc, dL/db_enc
        dW_enc = dz_enc.T @ x_orig  # (hidden_dim, input_dim)
        db_enc = np.sum(dz_enc, axis=0)  # (hidden_dim,)
        
        grads = {
            'W_enc': dW_enc, 'b_enc': db_enc,
            'W_mu': dW_mu, 'b_mu': db_mu,
            'W_logvar': dW_logvar, 'b_logvar': db_logvar,
            'W_dec1': dW_dec1, 'b_dec1': db_dec1,
            'W_dec2': dW_dec2, 'b_dec2': db_dec2
        }
        
        return grads


class AdamOptimizer:
    """Pure NumPy Adam optimizer"""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        
        # Initialize moments
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.params = params
        
    def step(self, grads):
        self.t += 1
        
        for key in self.params.keys():
            g = grads[key]
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)
            
            # Compute bias-corrected estimates
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        pass  # Gradients are computed fresh each time in our implementation


class OBVAEK(BaseEstimator, TransformerMixin):
    """
    Overcomplete Beta-VAE Kernel (OBVAEK) - Pure NumPy Implementation
    Auto-senses the required non-linear orthogonal dimension,
    trains via pure NumPy with manual backpropagation, and executes in pure NumPy.
    
    No PyTorch/TensorFlow/JAX dependencies for training or inference.
    """
    def __init__(self, latent_dim='auto', expansion_factor=10, hidden_dim=64, 
                 beta=2.0, epochs=300, lr=0.005, batch_size=None, 
                 verbose=True, random_state=None):
        self.latent_dim = latent_dim
        self.expansion_factor = expansion_factor
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size  # None = full batch
        self.verbose = verbose
        self.random_state = random_state
        
        # NumPy Weight Matrices (extracted after training)
        self.W_h = None
        self.b_h = None
        self.W_mu = None
        self.b_mu = None
        
        self.effective_dim_ = None 
        self.is_fitted_ = False

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float32)
        n_samples, input_dim = X_arr.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # === AUTO-SENSE LOGIC ===
        if self.latent_dim == 'auto':
            training_dim = min(input_dim * self.expansion_factor, 500) 
        else:
            training_dim = self.latent_dim
            
        # Initialize model
        model = _InternalOBVAENumPy(input_dim, self.hidden_dim, training_dim, 
                                    seed=self.random_state)
        
        # Flatten parameters for optimizer
        params = {
            'W_enc': model.W_enc, 'b_enc': model.b_enc,
            'W_mu': model.W_mu, 'b_mu': model.b_mu,
            'W_logvar': model.W_logvar, 'b_logvar': model.b_logvar,
            'W_dec1': model.W_dec1, 'b_dec1': model.b_dec1,
            'W_dec2': model.W_dec2, 'b_dec2': model.b_dec2
        }
        
        optimizer = AdamOptimizer(params, lr=self.lr)
        
        # --- Training Loop ---
        batch_size = self.batch_size if self.batch_size is not None else n_samples
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_arr[indices]
            
            epoch_loss = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                
                # Forward pass
                x_recon, mu, logvar = model.forward(X_batch)
                
                # Compute loss
                loss, mse, kld = model.compute_loss(x_recon, X_batch, mu, logvar, self.beta)
                epoch_loss += loss
                
                # Backward pass
                grads = model.backward(X_batch, x_recon, mu, logvar, self.beta)
                
                # Update weights
                optimizer.step(grads)
            
            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/n_samples:.4f}")
            
        # --- NumPy Extraction & Pruning ---
        # Final forward pass on all data to get variances
        _, final_mu, _ = model.forward(X_arr)
        node_variances = np.var(final_mu, axis=0)
        
        # Keep only nodes with meaningful variance
        active_indices = np.where(node_variances > 1e-4)[0]
        self.effective_dim_ = len(active_indices)
        
        if self.verbose:
            print(f"OBVAEK: Inflated to {training_dim}D, Pruned down to {self.effective_dim_} active features.")

        # Extract encoder weights for inference (only need encoder for transform)
        self.W_h = model.W_enc  # (hidden_dim, input_dim)
        self.b_h = model.b_enc  # (hidden_dim,)
        
        # Slice matrices to drop dead neurons
        self.W_mu = model.W_mu[active_indices, :]  # (effective_dim, hidden_dim)
        self.b_mu = model.b_mu[active_indices]     # (effective_dim,)
        
        # Store full weights if user wants to implement inverse_transform later
        self._full_W_mu = model.W_mu
        self._full_b_mu = model.b_mu
        self._active_indices = active_indices
        
        # Decoder weights (for potential inverse_transform)
        self._W_dec1 = model.W_dec1
        self._b_dec1 = model.b_dec1
        self._W_dec2 = model.W_dec2
        self._b_dec2 = model.b_dec2
        
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Executes the coordinate warp in 100% pure NumPy."""
        if not self.is_fitted_:
            raise RuntimeError("You must fit the OBVAEK before calling transform.")
            
        X_arr = np.asarray(X, dtype=np.float32)
        
        # H = ReLU(X @ W_h.T + b_h)
        Z_hidden = X_arr @ self.W_h.T + self.b_h
        H = np.maximum(0, Z_hidden) 
        
        # Mu = H @ W_mu.T + b_mu
        Mu = H @ self.W_mu.T + self.b_mu
        
        return Mu
    
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, Z):
        """
        Decode latent representation back to original space.
        Note: This uses the pruned latent space, not the full overcomplete space.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before inverse_transform.")
        
        Z_arr = np.asarray(Z, dtype=np.float32)
        
        # Pad with zeros for inactive latent dimensions if needed
        if Z_arr.shape[1] != self._full_W_mu.shape[0]:
            Z_full = np.zeros((Z_arr.shape[0], self._full_W_mu.shape[0]))
            Z_full[:, self._active_indices] = Z_arr
        else:
            Z_full = Z_arr
        
        # Decode: z -> h_dec -> x_recon
        z_dec1 = Z_full @ self._W_dec1.T + self._b_dec1
        h_dec = np.maximum(0, z_dec1)
        x_recon = h_dec @ self._W_dec2.T + self._b_dec2
        
        return x_recon
    
    def get_feature_variances(self):
        """Return variances of latent features (useful for analysis)."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first.")
        return self._feature_variances if hasattr(self, '_feature_variances') else None


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                               n_redundant=5, random_state=42)
    X = StandardScaler().fit_transform(X)
    
    print("Original shape:", X.shape)
    
    # Fit OBVAEK
    obvaek = OBVAEK(latent_dim='auto', expansion_factor=5, hidden_dim=64, 
                    beta=2.0, epochs=200, lr=0.01, verbose=True)
    
    X_transformed = obvaek.fit_transform(X)
    print("Transformed shape:", X_transformed.shape)
    
    # Test inverse transform
    X_recon = obvaek.inverse_transform(X_transformed)
    print("Reconstruction shape:", X_recon.shape)
    print("Reconstruction MSE:", np.mean((X - X_recon) ** 2))