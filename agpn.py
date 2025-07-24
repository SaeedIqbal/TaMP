import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import eigh, solve
from scipy.sparse.linalg import cg, LinearOperator
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import AgglomerativeClustering
import torch.nn.functional as F

class AdaptiveGraphPropagationNetwork(nn.Module):
    """
    Adaptive Graph Propagation Network (AGPN) for TaMP-Med.
    Enables efficient, real-time label propagation with adaptive steps and dataset-specific dynamics.
    """

    def __init__(self, feature_dim, num_classes, dataset_name, 
                 lambda_prop=0.7, gamma=1.0, K_cheby=3, n_max=25, tau_step=0.05):
        """
        Args:
            feature_dim (int): Dimension of prototype vectors.
            num_classes (int): Number of diagnostic classes.
            dataset_name (str): One of ['NIH', 'BRaTS', 'Camelyon16', 'PANDA'].
            lambda_prop (float): Propagation coefficient (0 < lambda < 1).
            gamma (float): Sparsity control for adjacency matrix.
            K_cheby (int): Order of Chebyshev polynomial approximation.
            n_max (int): Maximum number of propagation steps.
            tau_step (float): Threshold for domain shift to determine steps.
        """
        super(AdaptiveGraphPropagationNetwork, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.lambda_prop = lambda_prop
        self.gamma = gamma
        self.K_cheby = K_cheby
        self.n_max = n_max
        self.tau_step = tau_step

        # Store previous domain's prototypes for shift calculation
        self.prev_prototypes = None

        # Dataset-specific parameters
        if dataset_name == 'BRaTS':
            self.sigma_adaptive = True
            self.precond_method = 'amg'  # Algebraic Multigrid
        elif dataset_name == 'Camelyon16':
            self.hierarchical_levels = 3
            self.coarsening_factor = 4
        elif dataset_name == 'PANDA':
            self.alpha_beta = (1.0, 1.0)  # Beta prior parameters
        elif dataset_name == 'NIH':
            self.rank_krylov = 50  # For Lanczos approximation
            self.nu_temporal = 0.1  # Temporal persistence decay

    def forward(self, prototypes: torch.Tensor, soft_labels: torch.Tensor, domain_shift: float = None):
        """
        Perform adaptive label propagation.

        Args:
            prototypes (torch.Tensor): [C, D] tensor of class prototypes.
            soft_labels (torch.Tensor): [N, C] initial soft labels (e.g., from classifier).
            domain_shift (float): Optional KL divergence between current and previous domains.

        Returns:
            torch.Tensor: [N, C] refined soft labels.
        """
        # Step 1: Construct time-varying adjacency matrix
        W_t = self._construct_adjacency(prototypes)

        # Step 2: Dynamically adjust propagation steps
        n_step = self._adaptive_step_control(domain_shift)

        # Step 3: Apply dataset-specific propagation or generic AGPN
        if self.dataset_name == 'BRaTS':
            refined_labels = self._propagate_brats(W_t, soft_labels)
        elif self.dataset_name == 'Camelyon16':
            refined_labels = self._propagate_camelyon16(prototypes, W_t, soft_labels)
        elif self.dataset_name == 'PANDA':
            refined_labels = self._propagate_panda(W_t, soft_labels)
        elif self.dataset_name == 'NIH':
            refined_labels = self._propagate_nih(W_t, soft_labels)
        else:
            # Generic AGPN with Chebyshev approximation
            refined_labels = self._generic_agpn(W_t, soft_labels, n_step)

        # Update for next domain
        if domain_shift is not None:
            self.prev_prototypes = prototypes.detach().cpu()

        return refined_labels

    def _construct_adjacency(self, prototypes: torch.Tensor) -> torch.Tensor:
        """Construct time-varying adjacency matrix W_ij = Softmax(-γ ||p_i - p_j||^2)."""
        # Compute pairwise squared Euclidean distances
        dist_sq = torch.cdist(prototypes, prototypes, p=2).pow(2)  # [C, C]
        # Apply Gaussian kernel
        W = torch.exp(-self.gamma * dist_sq)  # [C, C]
        # Apply Softmax per row for stochasticity
        W = F.softmax(W, dim=1)
        # Zero the diagonal
        W.fill_diagonal_(0)
        return W

    def _adaptive_step_control(self, domain_shift: float) -> int:
        """Dynamically adjust the number of propagation steps based on domain shift magnitude."""
        if domain_shift is None:
            return self.n_max  # Default to max if shift unknown
        # n_step = min(n_max, ceil(D_KL / tau_step))
        n_step = min(self.n_max, int(np.ceil(domain_shift / self.tau_step)))
        return max(1, n_step)  # Ensure at least one step

    def _generic_agpn(self, W: torch.Tensor, S: torch.Tensor, n_step: int) -> torch.Tensor:
        """Generic AGPN using Chebyshev polynomial approximation."""
        # Normalize adjacency to get Laplacian
        D = torch.diag(W.sum(dim=1))  # Degree matrix
        L = D - W  # Unnormalized Laplacian
        # Symmetrically normalized Laplacian: L_sym = D^{-1/2} L D^{-1/2}
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(D.diagonal()) + 1e-8))
        L_sym = D_inv_sqrt @ L @ D_inv_sqrt
        # Scale L_sym to [-1, 1] for Chebyshev stability
        lambda_max = torch.symeig(L_sym, eigenvectors=False).eigenvalues.max()
        L_tilde = (2.0 / lambda_max) * L_sym - torch.eye(L_sym.size(0))

        # Chebyshev polynomial approximation of (I - lambda*L)^{-1}
        # Y = (1-lambda)(I - lambda*L)^{-1} S ≈ sum_{k=0}^K theta_k T_k(L_tilde) S
        I = torch.eye(L_tilde.size(0))
        Y = S.clone()
        T_0 = I
        T_1 = L_tilde

        for k in range(n_step):
            if k == 0:
                Y_k = T_0 @ S
            elif k == 1:
                Y_k = T_1 @ S
            else:
                T_k = 2 * L_tilde @ T_1 - T_0  # T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)
                Y_k = T_k @ S
                T_0, T_1 = T_1, T_k
            # Use fixed Chebyshev coefficients (can be learned)
            theta_k = 1.0 if k == 0 else 2.0
            Y = Y + theta_k * Y_k

        # Final scaling
        Y = (1 - self.lambda_prop) * Y
        return Y

    def _propagate_brats(self, W: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """BRaTS: Manifold-regularized harmonic function with PCG."""
        # Use geodesic distance on Riemannian manifold (simplified via k-NN)
        # Here, we assume W is already computed from geodesic distances
        L = torch.diag(W.sum(dim=1)) - W  # Graph Laplacian
        # Solve: (L + mu*I) Y = mu * S  (Manifold Regularization)
        mu = 0.5
        A = L + mu * torch.eye(L.size(0))
        b = mu * S.T  # [C, N]

        # Use Conjugate Gradient (PCG) for efficiency
        A_np = A.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        Y_np = np.zeros_like(b_np)

        def pcg_matvec(v):
            return A_np @ v

        A_op = LinearOperator((A_np.shape[0], A_np.shape[0]), matvec=pcg_matvec)
        for i in range(b_np.shape[1]):
            Y_np[:, i], _ = cg(A_op, b_np[:, i], tol=1e-3, maxiter=50)

        Y = torch.from_numpy(Y_np.T).float().to(S.device)
        return Y

    def _propagate_camelyon16(self, prototypes: torch.Tensor, W: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Camelyon16: Hierarchical spectral propagation on a graph pyramid."""
        # Build multi-scale graph pyramid
        W_levels = [W]
        for l in range(1, self.hierarchical_levels):
            # Coarsen graph
            n_coarse = max(1, W_levels[-1].shape[0] // self.coarsening_factor)
            clustering = AgglomerativeClustering(n_clusters=n_coarse).fit(prototypes.detach().cpu().numpy())
            labels = clustering.labels_
            # Prolongation operator P: maps coarse to fine
            P = torch.zeros(prototypes.shape[0], n_coarse)
            for i, lbl in enumerate(labels):
                P[i, lbl] = 1.0
            P = P / (P.sum(dim=0, keepdim=True) + 1e-8)  # Normalize
            # Coarsen adjacency
            W_coarse = P.T @ W_levels[-1] @ P
            W_levels.append(W_coarse)

        # Bottom-up propagation
        Y_levels = [S]
        for l in range(self.hierarchical_levels - 1, -1, -1):
            W_l = W_levels[l]
            S_l = Y_levels[-1]
            # Apply one step of generic AGPN
            Y_l = self._generic_agpn(W_l, S_l, n_step=1)
            Y_levels.append(Y_l)

        return Y_levels[-1]

    def _propagate_panda(self, W: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """PANDA: Bayesian graph sharpening with edge reliability."""
        # Model edge reliability with Beta-Bernoulli
        alpha_0, beta_0 = self.alpha_beta
        # Assume we have a way to compute KL divergence between prototypes
        # For simplicity, use W as a proxy for similarity
        delta_ij = 1 - W  # High W means low divergence
        # Update Beta posterior
        alpha_ij = alpha_0 + (delta_ij < 0.1).float()  # If divergence low, increase alpha
        beta_ij = beta_0 + (delta_ij >= 0.1).float()  # If divergence high, increase beta
        # Expected edge weight
        E_w_ij = alpha_ij / (alpha_ij + beta_ij)
        # Sharpen adjacency
        W_sharpened = E_w_ij * W
        # Propagate
        Y = self._generic_agpn(W_sharpened, S, n_step=self.n_max)
        return Y

    def _propagate_nih(self, W: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """NIH: Attention-gated and temporally persistent propagation."""
        # Simulate attention-guided masking (in practice, use Grad-CAM)
        # Assume we have attention maps A_q, A_s for query and support
        # For simulation, use random attention
        n_nodes = W.shape[0]
        A_q = torch.rand(n_nodes, 1)
        A_s = torch.rand(n_nodes, 1)
        # Compute attention consistency mask
        M = torch.sigmoid(5.0 * torch.abs(A_q - A_s).mean())  # [n_nodes, 1]
        # Apply to adjacency
        W_atten = W * M @ M.T  # Outer product for pairwise mask

        # Add temporal persistence (if previous prototypes exist)
        if self.prev_prototypes is not None:
            # Compute temporal edge persistence P_temp(e_ij) = exp(-nu |t_i - t_j|) * I(y_i = y_j)
            # Simplified: assume all previous edges have high persistence
            P_temp = torch.ones_like(W_atten) * torch.exp(-self.nu_temporal)
            W_final = W_atten * P_temp
        else:
            W_final = W_atten

        # Use Krylov subspace approximation (Lanczos) for efficiency
        # Here, we approximate with a low-rank Chebyshev (K=3)
        Y = self._generic_agpn(W_final, S, n_step=min(3, self.n_max))
        return Y

    def get_propagation_residual(self, Y_prev: torch.Tensor, Y_curr: torch.Tensor) -> float:
        """Compute the label propagation residual ||Y^{(k)} - Y^{(k-1)}||_F."""
        return torch.norm(Y_curr - Y_prev, p='fro').item()

    def reset(self):
        """Reset the internal state (e.g., for a new patient sequence)."""
        self.prev_prototypes = None
'''
agpn = AdaptiveGraphPropagationNetwork(feature_dim=2048, num_classes=14, dataset_name='NIH')
prototypes = torch.randn(14, 2048)  # Class prototypes
soft_labels = torch.softmax(torch.randn(100, 14), dim=-1)  # Initial soft labels
domain_shift = 0.15  # Estimated KL divergence

# Perform adaptive propagation
refined_labels = agpn(prototypes, soft_labels, domain_shift)
print(f"Refined labels shape: {refined_labels.shape}")
'''