import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import expm, logm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from sklearn.cluster import SpectralClustering
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class TemporalPrototypeAdapter(nn.Module):
    """
    Temporal Prototype Adapter (TPA) for TaMP-Med.
    Integrates generic temporal modeling with dataset-specific geometric and probabilistic enhancements.
    """

    def __init__(self, feature_dim, num_classes, dataset_name, alpha=0.6, hidden_size=512):
        """
        Args:
            feature_dim (int): Dimension of deep features (e.g., 2048 for ResNet, 768 for ViT).
            num_classes (int): Number of diagnostic classes.
            dataset_name (str): One of ['NIH', 'BRaTS', 'Camelyon16', 'PANDA'].
            alpha (float): Fusion coefficient for novelty vs. continuity.
            hidden_size (int): Hidden size for LSTM.
        """
        super(TemporalPrototypeAdapter, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.alpha = alpha
        self.hidden_size = hidden_size

        # Generic TPA components
        self.lstm = nn.LSTMCell(input_size=feature_dim, hidden_size=hidden_size)
        
        # Initialize hidden and cell states for each class
        self.register_buffer('hidden_state', torch.zeros(num_classes, hidden_size))
        self.register_buffer('cell_state', torch.zeros(num_classes, hidden_size))

        # Dataset-specific parameters and modules
        if dataset_name == 'BRaTS':
            self.manifold_dim = feature_dim
            self.k_neighbors = 5
            self.gamma_geo = 1.0  # Geodesic scaling
        elif dataset_name == 'Camelyon16':
            self.eta_transport = 0.5  # Transport influence
            self.w_init = 0.1
        elif dataset_name == 'PANDA':
            self.alpha_dp = 1.0  # DPMM concentration
            self.H0_scale = 0.1
        elif dataset_name == 'NIH':
            # For patch-level spatiotemporal attention
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=2048),
                num_layers=2
            )
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            )
            self.layer_norm = nn.LayerNorm(feature_dim)

        # Internal state for dataset-specific tracking
        self._init_internal_state()

    def _init_internal_state(self):
        """Initialize internal state for dataset-specific modeling."""
        self.prev_prototypes = None  # For BRaTS Procrustes
        self.prev_distributions = {}  # For Camelyon16 OT
        self.gp_posteriors = {}  # For PANDA GP
        self.patch_memory = {}  # For NIH attention

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass to compute fused prototypes.

        Args:
            features (torch.Tensor): [N, D] tensor of deep features from current domain.
            labels (torch.Tensor): [N] tensor of pseudo-labels.

        Returns:
            dict: Fused prototypes for each class as torch.Tensor.
        """
        # Step 1: Compute current domain prototypes
        current_prototypes_np = self._compute_prototypes(features, labels)

        # Step 2: Apply dataset-specific enhancement or generic TPA
        if self.dataset_name == 'BRaTS':
            fused_prototypes = self._enhance_brats(current_prototypes_np)
        elif self.dataset_name == 'Camelyon16':
            fused_prototypes = self._enhance_camelyon16(current_prototypes_np)
        elif self.dataset_name == 'PANDA':
            fused_prototypes = self._enhance_panda(current_prototypes_np)
        elif self.dataset_name == 'NIH':
            fused_prototypes = self._enhance_nih(current_prototypes_np, features, labels)
        else:
            # Generic TPA for other datasets
            fused_prototypes = self._generic_tpa(current_prototypes_np)

        return fused_prototypes

    def _compute_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> dict:
        """Compute class-wise mean prototypes."""
        prototypes = {}
        device = features.device
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                prototypes[c] = features[mask].mean(dim=0).detach().cpu().numpy()
            else:
                # Use previous state if no samples
                hidden_c = self.hidden_state[c].detach().cpu().numpy()
                prototypes[c] = hidden_c if not np.allclose(hidden_c, 0) else np.zeros(self.feature_dim)
        return prototypes

    def _generic_tpa(self, current_prototypes: dict) -> dict:
        """Generic TPA using LSTM fusion."""
        fused_prototypes = {}
        hidden_state_np = self.hidden_state.detach().cpu().numpy()
        cell_state_np = self.cell_state.detach().cpu().numpy()

        for c in range(self.num_classes):
            p_t = torch.from_numpy(current_prototypes[c]).float().to(self.hidden_state.device)
            h_prev = torch.from_numpy(hidden_state_np[c]).to(self.hidden_state.device)
            c_prev = torch.from_numpy(cell_state_np[c]).to(self.cell_state.device)

            # Update LSTM
            h_t, c_t = self.lstm(p_t, (h_prev, c_prev))

            # Fuse current and historical
            p_fused = self.alpha * p_t + (1 - self.alpha) * h_t
            fused_prototypes[c] = p_fused

        # Update internal state
        new_hidden = torch.stack([fused_prototypes[c] for c in range(self.num_classes)], dim=0)
        self.hidden_state.copy_(new_hidden)
        self.cell_state.copy_(torch.stack([c_t for c in range(self.num_classes)], dim=0))

        return fused_prototypes

    def _enhance_brats(self, current_prototypes: dict) -> dict:
        """BRaTS: Riemannian manifold update via Lie group."""
        protos = np.array([current_prototypes[c] for c in range(self.num_classes)])
        
        if self.prev_prototypes is not None and self.prev_prototypes.shape == protos.shape:
            # Compute Procrustes alignment
            R = self._procrustes_alignment(self.prev_prototypes, protos)
            log_R = logm(R + 1e-8 * np.eye(R.shape[0]))  # Avoid singular matrix
            # Weighted average of log maps (simplified K=1)
            w_k = 1.0
            exp_avg = expm(w_k * log_R)
            # Apply to previous hidden state
            h_prev_np = self.hidden_state[0].detach().cpu().numpy()
            h_t_np = exp_avg @ h_prev_np
            fused_prototypes = {}
            for c in range(self.num_classes):
                p_t = current_prototypes[c]
                p_fused = self.alpha * p_t + (1 - self.alpha) * h_t_np
                fused_prototypes[c] = torch.from_numpy(p_fused).float().to(self.hidden_state.device)
        else:
            fused_prototypes = self._generic_tpa(current_prototypes)

        self.prev_prototypes = protos.copy()
        return fused_prototypes

    def _procrustes_alignment(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute orthogonal Procrustes alignment matrix."""
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        A = X.T @ Y
        U, _, Vt = np.linalg.svd(A)
        R = U @ Vt
        return R

    def _enhance_camelyon16(self, current_prototypes: dict) -> dict:
        """Camelyon16: Optimal Transport-based fusion with Brenier potential."""
        fused_prototypes = {}
        for c in range(self.num_classes):
            # Simulate intensity distribution for class c
            mu_prev = self.prev_distributions.get(c, np.random.normal(0.5, 0.1, 100))
            mu_curr = np.random.normal(0.55, 0.1, 100)  # Simulated shift
            # Compute Wasserstein distance
            W2 = wasserstein_distance(mu_prev, mu_curr)
            # Brenier potential gradient (simulated)
            grad_phi = np.clip(mu_curr - mu_prev, -1, 1)
            # Transport-inspired gate
            p_t = current_prototypes[c]
            h_prev = self.hidden_state[c].detach().cpu().numpy()
            i_t = torch.sigmoid(
                torch.tensor(self.eta_transport * grad_phi.mean(), dtype=torch.float32) +
                torch.tensor(0.1, dtype=torch.float32)  # bias
            ).item()
            p_fused = i_t * h_prev + (1 - i_t) * p_t
            fused_prototypes[c] = torch.from_numpy(p_fused).float().to(self.hidden_state.device)
            self.prev_distributions[c] = mu_curr
        return fused_prototypes

    def _enhance_panda(self, current_prototypes: dict) -> dict:
        """PANDA: Gaussian Process Latent Variable Model (GPLVM) for uncertainty-aware fusion."""
        fused_prototypes = {}
        for c in range(self.num_classes):
            p_t = current_prototypes[c]
            # GP prior with RBF kernel
            kernel = lambda t1, t2: np.exp(-0.5 * (t1 - t2)**2)
            # Variational inference (simplified)
            if c in self.gp_posteriors:
                mu_prev, sigma_prev = self.gp_posteriors[c]
                mu_t = 0.95 * mu_prev + 0.05 * p_t
                sigma_t = 0.9 * sigma_prev
            else:
                mu_t = p_t
                sigma_t = np.eye(self.feature_dim) * 0.01
            # Sample from posterior
            p_sample = np.random.multivariate_normal(mu_t, sigma_t + 1e-6 * np.eye(len(mu_t)))
            p_fused = self.alpha * p_t + (1 - self.alpha) * p_sample
            fused_prototypes[c] = torch.from_numpy(p_fused).float().to(self.hidden_state.device)
            self.gp_posteriors[c] = (mu_t, sigma_t)
        return fused_prototypes

    def _enhance_nih(self, current_prototypes: dict, features: torch.Tensor, labels: torch.Tensor) -> dict:
        """NIH: Spatiotemporal attention fusion using Transformer."""
        fused_prototypes = {}
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                patch_features = features[mask]  # [N_patches, D]
                patch_features = patch_features.unsqueeze(1)  # [N, 1, D]
                # Apply Transformer
                attended = self.transformer(patch_features)  # [N, 1, D]
                z_t = attended.mean(dim=0).squeeze(0)  # [D]
                mlp_out = self.mlp(z_t)
                z_t = self.layer_norm(patch_features.mean(dim=0).squeeze() + z_t + mlp_out)
                p_t = torch.from_numpy(current_prototypes[c]).float().to(z_t.device)
                p_fused = self.alpha * p_t + (1 - self.alpha) * z_t
                fused_prototypes[c] = p_fused
            else:
                fused_prototypes[c] = torch.from_numpy(current_prototypes[c]).float().to(self.hidden_state.device)
        return fused_prototypes

    def get_hidden_states(self):
        """Return current hidden states for saving."""
        return {
            'hidden_state': self.hidden_state.clone(),
            'cell_state': self.cell_state.clone()
        }

    def set_hidden_states(self, state_dict):
        """Set hidden states from a previous session."""
        if 'hidden_state' in state_dict and 'cell_state' in state_dict:
            self.hidden_state.copy_(state_dict['hidden_state'])
            self.cell_state.copy_(state_dict['cell_state'])
        else:
            raise KeyError("State dict must contain 'hidden_state' and 'cell_state'.")

    def reset_states(self):
        """Reset all internal states for a new sequence."""
        self.hidden_state.zero_()
        self.cell_state.zero_()
        self._init_internal_state()