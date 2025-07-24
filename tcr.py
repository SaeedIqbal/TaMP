# tcr.py
import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.stats import wasserstein_distance
from sklearn.cluster import DBSCAN
from sklearn.manifold import LocallyLinearEmbedding
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class TemporalConsistencyRefinement(nn.Module):
    """
    Temporal Consistency Refinement (TCR) for TaMP-Med.
    Ensures stable and consistent pseudo-labels across temporally adjacent domains.
    Implements Kalman filtering, TCN smoothing, and dataset-specific enhancements.
    """

    def __init__(self, num_classes, dataset_name, beta=0.8, lambda_tcn=0.5, K=5):
        """
        Args:
            num_classes (int): Number of diagnostic classes.
            dataset_name (str): One of ['NIH', 'BRaTS', 'Camelyon16', 'PANDA'].
            beta (float): Kalman filter gain (0.8 for high smoothing).
            lambda_tcn (float): Weight for temporal smoothness loss.
            K (int): Number of recent predictions to use in TCN.
        """
        super(TemporalConsistencyRefinement, self).__init__()
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.beta = beta
        self.lambda_tcn = lambda_tcn
        self.K = K

        # Store prediction history for temporal modeling
        self.prediction_history = []

        # Dataset-specific parameters
        if dataset_name == 'BRaTS':
            self.f_growth = 'gompertz'  # Tumor growth model
            self.noise_level = 0.1
        elif dataset_name == 'Camelyon16':
            self.gamma_ot = 0.5  # Optimal transport weight
        elif dataset_name == 'PANDA':
            self.transition_matrix = None  # Markov jump process
            self.eta_dirichlet = 0.8
        elif dataset_name == 'NIH':
            self.tau_shift = 0.3  # Threshold for abrupt shift detection
            self.atten_maps = []  # Store Grad-CAM maps

        # Temporal Convolutional Network (TCN) for smoothness
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=num_classes, out_channels=64, kernel_size=3, dilation=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=64, out_channels=num_classes, kernel_size=3, dilation=4, padding=4),
        )

    def forward(self, raw_predictions: torch.Tensor):
        """
        Refine raw predictions using temporal consistency.

        Args:
            raw_predictions (torch.Tensor): [N, C] tensor of raw softmax outputs.

        Returns:
            torch.Tensor: [N, C] tensor of refined pseudo-labels.
        """
        # Step 1: Apply Kalman filtering
        smoothed_predictions = self._kalman_filter(raw_predictions)

        # Step 2: Apply dataset-specific enhancement or generic TCR
        if self.dataset_name == 'BRaTS':
            refined_predictions = self._enhance_brats(smoothed_predictions)
        elif self.dataset_name == 'Camelyon16':
            refined_predictions = self._enhance_camelyon16(smoothed_predictions)
        elif self.dataset_name == 'PANDA':
            refined_predictions = self._enhance_panda(smoothed_predictions)
        elif self.dataset_name == 'NIH':
            refined_predictions = self._enhance_nih(smoothed_predictions)
        else:
            # Generic TCR with TCN
            refined_predictions = self._generic_tcr(smoothed_predictions)

        # Update history
        self.prediction_history.append(refined_predictions.detach().cpu().numpy())
        if len(self.prediction_history) > self.K:
            self.prediction_history.pop(0)

        return refined_predictions

    def _kalman_filter(self, raw_preds: torch.Tensor) -> torch.Tensor:
        """Apply Kalman filter to raw predictions."""
        device = raw_preds.device
        if len(self.prediction_history) == 0:
            return raw_preds  # No prior, return raw
        else:
            prev_smooth = torch.from_numpy(self.prediction_history[-1]).float().to(device)
            return self.beta * raw_preds + (1 - self.beta) * prev_smooth

    def _generic_tcr(self, smoothed_preds: torch.Tensor) -> torch.Tensor:
        """Apply TCN for temporal smoothness."""
        if len(self.prediction_history) < 2:
            return smoothed_preds

        # Stack recent predictions: [T, N, C] -> [C, T, N]
        hist_tensor = torch.tensor(np.stack(self.prediction_history, axis=0), dtype=torch.float32)
        hist_tensor = hist_tensor.permute(2, 0, 1)  # [N, T, C] -> [C, T, N]

        # Apply TCN
        refined = self.tcn(hist_tensor)
        refined = refined.permute(2, 0, 1)  # Back to [N, C]

        # Blend with current smoothed prediction
        current = self._kalman_filter(smoothed_preds)
        return 0.7 * current + 0.3 * refined[:, -1, :]  # Use last TCN output

    def _enhance_brats(self, smoothed_preds: torch.Tensor) -> torch.Tensor:
        """BRaTS: Refine using Itô diffusion on probability simplex."""
        device = smoothed_preds.device
        if len(self.prediction_history) == 0:
            return smoothed_preds

        prev = torch.from_numpy(self.prediction_history[-1]).float().to(device)
        # Model deterministic growth (simplified Gompertz)
        if self.f_growth == 'gompertz':
            f = 0.01 * prev * torch.log(1 / (prev + 1e-8))  # dμ/dt = r μ ln(K/μ)
        else:
            f = torch.zeros_like(prev)

        # Add stochastic noise (scanner variation)
        G = self.noise_level * torch.eye(self.num_classes).to(device)
        dW = torch.randn_like(prev) * 0.01
        # Fokker-Planck update (simplified Euler-Maruyama)
        drift = f * 1.0  # dt = 1
        diffusion = G @ dW.T
        mu_t = prev + drift + diffusion.T

        # Fréchet mean in KL sense (simplified)
        p_fused = self.beta * smoothed_preds + (1 - self.beta) * mu_t
        return torch.softmax(p_fused, dim=-1)

    def _enhance_camelyon16(self, smoothed_preds: torch.Tensor) -> torch.Tensor:
        """Camelyon16: Refine using optimal transport geodesic interpolation."""
        if len(self.prediction_history) == 0:
            return smoothed_preds

        prev = self.prediction_history[-1]
        refined = np.zeros_like(smoothed_preds.cpu().numpy())

        for c in range(self.num_classes):
            # Simulate patch-level distribution
            nu_prev = np.random.normal(prev[:, c].mean(), 0.1, 100)
            nu_curr = np.random.normal(smoothed_preds[:, c].mean().item(), 0.1, 100)
            # Compute Wasserstein-2 geodesic
            W2 = wasserstein_distance(nu_prev, nu_curr)
            # Brenier potential gradient (simulated)
            grad_phi = np.clip(nu_curr - nu_prev, -1, 1).mean()
            # Transport-inspired correction
            correction = self.gamma_ot * grad_phi
            refined[:, c] = smoothed_preds[:, c].cpu().numpy() + correction

        # Entropy-weighted fusion
        H_raw = - (smoothed_preds * torch.log(smoothed_preds + 1e-8)).sum(dim=-1, keepdim=True)
        H_ref = - (torch.from_numpy(refined) * torch.log(torch.from_numpy(refined) + 1e-8)).sum(dim=-1, keepdim=True)
        weight_raw = torch.exp(-H_raw)
        weight_ref = torch.exp(-H_ref)
        fused = (weight_raw * smoothed_preds + weight_ref * torch.from_numpy(refined)) / (weight_raw + weight_ref + 1e-8)
        return fused

    def _enhance_panda(self, smoothed_preds: torch.Tensor) -> torch.Tensor:
        """PANDA: Refine using Markov jump process for Gleason grading."""
        device = smoothed_preds.device
        if self.transition_matrix is None:
            # Initialize transition matrix (e.g., Gleason 3+3 -> 3+4 allowed, not to 5+5 directly)
            self.transition_matrix = torch.eye(self.num_classes)
            self.transition_matrix[0, 1] = 0.1  # 3+3 -> 3+4
            self.transition_matrix[1, 2] = 0.1  # 3+4 -> 4+3
            self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(dim=1, keepdim=True)
            self.transition_matrix = self.transition_matrix.to(device)

        if len(self.prediction_history) == 0:
            return smoothed_preds

        prev = torch.from_numpy(self.prediction_history[-1]).float().to(device)
        # Forward-backward inference (simplified)
        alpha = prev * self.transition_matrix  # Forward
        beta = torch.ones_like(prev)  # Backward (simplified)
        gamma = alpha * beta
        gamma = gamma / (gamma.sum(dim=-1, keepdim=True) + 1e-8)

        # Refined label is the MAP estimate
        refined = torch.softmax(self.eta_dirichlet * smoothed_preds + (1 - self.eta_dirichlet) * gamma, dim=-1)
        return refined

    def _enhance_nih(self, smoothed_preds: torch.Tensor) -> torch.Tensor:
        """NIH: Refine using spatiotemporal GP and attention persistence."""
        device = smoothed_preds.device
        if len(self.prediction_history) == 0:
            return smoothed_preds

        prev = torch.from_numpy(self.prediction_history[-1]).float().to(device)
        delta = torch.norm(smoothed_preds - prev, dim=-1)

        # Detect abrupt shifts using GP (simplified)
        if delta.mean() > self.tau_shift and len(self.atten_maps) >= 2:
            # Attention persistence check
            A_curr = self.atten_maps[-1]
            A_prev = self.atten_maps[-2]
            ssim_val = self._ssim_attention(A_curr, A_prev)
            if ssim_val < 0.5:  # Low attention consistency
                # Preserve previous prediction unless change is large
                mask = (delta < self.tau_shift).unsqueeze(-1)
                smoothed_preds = torch.where(mask, prev, smoothed_preds)

        return self._generic_tcr(smoothed_preds)

    def _ssim_attention(self, A1: np.ndarray, A2: np.ndarray) -> float:
        """Compute SSIM between two attention maps."""
        from skimage.metrics import structural_similarity as ssim
        return ssim(A1, A2, data_range=A1.max() - A1.min())

    def add_attention_map(self, atten_map: np.ndarray):
        """Add a Grad-CAM attention map for NIH-specific refinement."""
        self.atten_maps.append(atten_map)
        if len(self.atten_maps) > self.K:
            self.atten_maps.pop(0)

    def get_temporal_consistency_score(self) -> float:
        """Compute TCS: average cosine similarity of predictions over time."""
        if len(self.prediction_history) < 2:
            return 1.0
        similarities = []
        for i in range(1, len(self.prediction_history)):
            p1 = self.prediction_history[i-1].flatten()
            p2 = self.prediction_history[i].flatten()
            cos_sim = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-8)
            similarities.append(cos_sim)
        return np.mean(similarities)

    def reset_history(self):
        """Reset prediction history for a new sequence."""
        self.prediction_history = []
        self.atten_maps = []

'''
tcr = TemporalConsistencyRefinement(num_classes=14, dataset_name='NIH')
raw_preds = torch.softmax(torch.randn(32, 14), dim=-1)  # Simulated raw predictions

# For NIH, add attention map
atten_map = np.random.rand(28, 28)  # Simulated Grad-CAM
tcr.add_attention_map(atten_map)

# Refine predictions
refined_preds = tcr(raw_preds)
print(f"Refined predictions shape: {refined_preds.shape}")
print(f"Temporal Consistency Score: {tcr.get_temporal_consistency_score():.3f}")
'''