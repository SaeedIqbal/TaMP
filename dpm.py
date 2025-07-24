import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.mixture import BayesianGaussianMixture
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class DynamicPrototypeManager(nn.Module):
    """
    Dynamic Prototype Manager (DPM) for TaMP-Med.
    Manages a prototype bank by pruning obsolete prototypes and merging similar ones
    to prevent memory explosion while preserving semantic discriminability.
    """

    def __init__(self, feature_dim, dataset_name, 
                 tau_prune=0.1, tau_merge=0.2, max_prototypes=50, K=5):
        """
        Args:
            feature_dim (int): Dimension of the prototype vectors.
            dataset_name (str): One of ['NIH', 'BRaTS', 'Camelyon16', 'PANDA'].
            tau_prune (float): Threshold for KL divergence to prune prototypes.
            tau_merge (float): Threshold for affinity to merge prototypes.
            max_prototypes (int): Maximum number of prototypes (circular buffer).
            K (int): Target number of clusters for spectral clustering (adaptive).
        """
        super(DynamicPrototypeManager, self).__init__()
        self.feature_dim = feature_dim
        self.dataset_name = dataset_name
        self.tau_prune = tau_prune
        self.tau_merge = tau_merge
        self.max_prototypes = max_prototypes
        self.K = K

        # Prototype bank: list of dicts {prototype: torch.Tensor, meta dict}
        self.prototype_bank = []

        # Dataset-specific parameters
        if dataset_name == 'BRaTS':
            self.gamma_geo = 1.0  # Geodesic decay rate
            self.epsilon_trust = 1e-3  # Trust region threshold
        elif dataset_name == 'Camelyon16':
            self.lambda_weights = None  # For Wasserstein barycenter
        elif dataset_name == 'PANDA':
            self.alpha_dpmm = 1.0  # DPMM concentration parameter
        elif dataset_name == 'NIH':
            self.tau_atten = 0.7  # Saliency consistency threshold
            self.atten_maps = {}  # Store Grad-CAM maps for prototypes

    def add_prototypes(self, prototypes_dict, domain_id=None, timestamp=None):
        """
        Add new prototypes to the bank.

        Args:
            prototypes_dict (dict): {class_id: prototype_tensor}
            domain_id (str): Identifier for the source domain.
            timestamp (int): Time step of addition.
        """
        for class_id, proto in prototypes_dict.items():
            metadata = {
                'class_id': class_id,
                'domain_id': domain_id,
                'timestamp': timestamp,
                'modality': self.dataset_name
            }
            self.prototype_bank.append({
                'prototype': proto.detach().cpu(),
                'metadata': metadata
            })
        # Enforce circular buffer
        while len(self.prototype_bank) > self.max_prototypes:
            self.prototype_bank.pop(0)

    def _compute_kl_divergence_matrix(self):
        """Compute pairwise KL divergence between all prototypes in the bank."""
        n = len(self.prototype_bank)
        if n < 2:
            return np.zeros((n, n))

        # Assume each prototype p_i has a covariance Σ_i from its domain
        # For simplicity, we use a fixed covariance or estimate from feature variance
        covs = [0.1 * np.eye(self.feature_dim) for _ in range(n)]  # Simplified
        means = [p['prototype'].numpy() for p in self.prototype_bank]

        kl_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    kl_matrix[i, j] = self._kl_divergence_gaussians(
                        means[i], covs[i], means[j], covs[j]
                    )
        return kl_matrix

    def _kl_divergence_gaussians(self, mu1, Sigma1, mu2, Sigma2):
        """Compute KL divergence D_KL(N(mu1,Sigma1) || N(mu2,Sigma2))."""
        d = len(mu1)
        inv_Sigma2 = np.linalg.inv(Sigma2 + 1e-8 * np.eye(d))
        term1 = np.trace(inv_Sigma2 @ Sigma1)
        term2 = (mu2 - mu1).T @ inv_Sigma2 @ (mu2 - mu1)
        term3 = np.log(np.linalg.det(Sigma2) / (np.linalg.det(Sigma1) + 1e-8))
        return 0.5 * (term1 + term2 - d + term3)

    def _prune_obsolete_prototypes(self, kl_matrix):
        """Remove prototypes that are too similar to a more recent one (low KL)."""
        to_remove = set()
        n = kl_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i < j:  # j is more recent than i
                    if kl_matrix[i, j] < self.tau_prune:
                        to_remove.add(i)
        # Sort in reverse to avoid index shifting
        for idx in sorted(to_remove, reverse=True):
            del self.prototype_bank[idx]

    def _merge_similar_prototypes(self, kl_matrix):
        """Merge similar prototypes using spectral clustering."""
        n = kl_matrix.shape[0]
        if n < 2:
            return

        # Create affinity matrix from KL divergence
        affinity = np.exp(-kl_matrix / np.clip(kl_matrix.std(), 1e-8, None))
        # Estimate number of clusters via eigen-gap
        k = self._estimate_optimal_clusters(affinity)
        k = min(k, n, self.K)

        if k < n:
            clustering = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
            labels = clustering.fit_predict(affinity)
            # Merge prototypes in the same cluster
            new_bank = []
            for cluster_id in range(k):
                mask = (labels == cluster_id)
                if mask.sum() > 0:
                    cluster_protos = [self.prototype_bank[i]['prototype'].numpy() for i in range(n) if mask[i]]
                    merged_proto = np.mean(cluster_protos, axis=0)
                    # Take metadata from the most recent prototype in the cluster
                    recent_idx = max([i for i in range(n) if mask[i]])
                    metadata = self.prototype_bank[recent_idx]['metadata'].copy()
                    new_bank.append({
                        'prototype': torch.from_numpy(merged_proto),
                        'metadata': metadata
                    })
            self.prototype_bank = new_bank

    def _estimate_optimal_clusters(self, affinity_matrix):
        """Estimate optimal number of clusters using eigen-gap heuristic."""
        # Eigenvalues of the graph Laplacian
        laplacian = np.diag(affinity_matrix.sum(axis=1)) - affinity_matrix
        eigenvals = np.linalg.eigvalsh(laplacian)
        eigenvals = np.sort(eigenvals)
        # Find the largest gap
        gaps = np.diff(eigenvals)
        k = np.argmax(gaps) + 1
        return max(k, 1)

    def _enhance_brats(self):
        """BRaTS: Prune based on Riemannian trust region."""
        if len(self.prototype_bank) < 2:
            return
        # Model on statistical manifold with Fisher-Rao metric
        # Use KL as Bregman divergence approximation
        for i in range(len(self.prototype_bank) - 1, -1, -1):  # Iterate backwards
            p_i = self.prototype_bank[i]['prototype'].numpy()
            # Compute Riemannian weight w_t^{(i)} = exp(-gamma * d_M^2(p_t, p_i))
            weights = []
            for j in range(len(self.prototype_bank)):
                if i != j:
                    d_sq = self._kl_divergence_gaussians(p_i, 0.1*np.eye(self.feature_dim), 
                                                         self.prototype_bank[j]['prototype'].numpy(), 0.1*np.eye(self.feature_dim))
                    weight = np.exp(-self.gamma_geo * d_sq)
                    weights.append(weight)
            if weights and min(weights) < self.epsilon_trust:
                # Prune if outside trust region
                del self.prototype_bank[i]

    def _enhance_camelyon16(self):
        """Camelyon16: Merge using Wasserstein barycenter fusion."""
        if len(self.prototype_bank) < 2:
            return
        # Group prototypes by class
        class_groups = {}
        for p in self.prototype_bank:
            c = p['metadata']['class_id']
            if c not in class_groups:
                class_groups[c] = []
            class_groups[c].append(p)

        new_bank = []
        for c, group in class_groups.items():
            if len(group) > 1:
                # Compute Wasserstein barycenter for Gaussians
                mus = [p['prototype'].numpy() for p in group]
                Sigmas = [0.1 * np.eye(self.feature_dim) for _ in group]  # Simplified
                weights = self.lambda_weights or [1.0 / len(mus)] * len(mus)
                mu_bary, Sigma_bary = self._wasserstein_barycenter_gaussian(mus, Sigmas, weights)
                # Create merged prototype
                merged_proto = torch.from_numpy(mu_bary)
                # Take metadata from the most recent
                recent_idx = np.argmax([p['metadata']['timestamp'] for p in group])
                metadata = group[recent_idx]['metadata'].copy()
                new_bank.append({
                    'prototype': merged_proto,
                    'metadata': metadata
                })
            else:
                new_bank.append(group[0])
        self.prototype_bank = new_bank

    def _wasserstein_barycenter_gaussian(self, mus, Sigmas, weights):
        """Compute the Wasserstein barycenter of Gaussian distributions."""
        # Closed-form solution for Gaussians
        mu_bary = np.sum([w * mu for w, mu in zip(weights, mus)], axis=0)
        # Iterative solution for Sigma_bary (simplified)
        Sigma_bary = np.sum([w * sigma for w, sigma in zip(weights, Sigmas)], axis=0)
        return mu_bary, Sigma_bary

    def _enhance_panda(self):
        """PANDA: Merge using Dirichlet Process Mixture Model (DPMM)."""
        if len(self.prototype_bank) < 2:
            return
        # Group by class
        class_groups = {}
        for p in self.prototype_bank:
            c = p['metadata']['class_id']
            if c not in class_groups:
                class_groups[c] = []
            class_groups[c].append(p['prototype'].numpy())

        new_bank = []
        for c, group in class_groups.items():
            if len(group) > 1:
                # Apply DPMM for automatic clustering
                dpmm = BayesianGaussianMixture(n_components=len(group), 
                                              weight_concentration_prior=self.alpha_dpmm,
                                              random_state=42, 
                                              covariance_type='full')
                labels = dpmm.fit_predict(np.array(group))
                # Merge clusters
                for cluster_id in np.unique(labels):
                    mask = (labels == cluster_id)
                    cluster_protos = np.array(group)[mask]
                    merged_proto = np.mean(cluster_protos, axis=0)
                    # Find the most recent prototype in the cluster
                    recent_idx = np.argmax([p['metadata']['timestamp'] for p in class_groups[c] if p['metadata']['class_id'] == c])
                    metadata = self.prototype_bank[recent_idx]['metadata'].copy()
                    new_bank.append({
                        'prototype': torch.from_numpy(merged_proto),
                        'metadata': metadata
                    })
            else:
                # Re-add single prototype
                for p in self.prototype_bank:
                    if p['metadata']['class_id'] == c:
                        new_bank.append(p)
                        break
        self.prototype_bank = new_bank

    def _enhance_nih(self, saliency_maps):
        """NIH: Merge only if high KL similarity AND high saliency alignment."""
        if len(self.prototype_bank) < 2:
            return
        kl_matrix = self._compute_kl_divergence_matrix()
        # Update saliency map memory
        for i, p in enumerate(self.prototype_bank):
            c = p['metadata']['class_id']
            if i < len(saliency_maps) and c in saliency_maps[i]:
                self.atten_maps[p['metadata']['timestamp']] = saliency_maps[i][c]

        # Compute saliency consistency matrix
        n = len(self.prototype_bank)
        saliency_matrix = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    t_i = self.prototype_bank[i]['metadata']['timestamp']
                    t_j = self.prototype_bank[j]['metadata']['timestamp']
                    if t_i in self.atten_maps and t_j in self.atten_maps:
                        A_i = self.atten_maps[t_i]
                        A_j = self.atten_maps[t_j]
                        S_ij = self._saliency_consistency(A_i, A_j)
                        saliency_matrix[i, j] = S_ij

        # Prune based on combined criterion
        to_merge = []
        for i in range(n):
            for j in range(i+1, n):
                if kl_matrix[i, j] < self.tau_merge and saliency_matrix[i, j] > self.tau_atten:
                    to_merge.append((i, j))
        
        # Merge prototypes
        merged_indices = set()
        new_bank = []
        for i, p in enumerate(self.prototype_bank):
            if i in merged_indices:
                continue
            # Find all prototypes to merge with i
            partners = [j for (x, y) in to_merge if x == i or y == i for j in [x, y] if j != i]
            partners = [p for p in partners if p not in merged_indices]
            if partners:
                # Merge i and all partners
                all_protos = [self.prototype_bank[i]['prototype'].numpy()] + [self.prototype_bank[j]['prototype'].numpy() for j in partners]
                merged_proto = np.mean(all_protos, axis=0)
                # Use the most recent metadata
                all_timestamps = [self.prototype_bank[i]['metadata']['timestamp']] + [self.prototype_bank[j]['metadata']['timestamp'] for j in partners]
                recent_idx = np.argmax(all_timestamps)
                metadata = self.prototype_bank[[i] + partners][recent_idx]['metadata'].copy()
                new_bank.append({
                    'prototype': torch.from_numpy(merged_proto),
                    'metadata': metadata
                })
                merged_indices.add(i)
                merged_indices.update(partners)
            else:
                new_bank.append(p)
        self.prototype_bank = new_bank

    def _saliency_consistency(self, A_i, A_j):
        """Compute saliency consistency between two attention maps."""
        # Flatten top-K masks
        K = 10
        flat_A_i = A_i.flatten()
        flat_A_j = A_j.flatten()
        top_k_i = np.argsort(flat_A_i)[-K:]
        top_k_j = np.argsort(flat_A_j)[-K:]
        mask_i = np.zeros_like(flat_A_i)
        mask_j = np.zeros_like(flat_A_j)
        mask_i[top_k_i] = 1
        mask_j[top_k_j] = 1
        # Cosine similarity
        return np.dot(mask_i, mask_j) / (np.linalg.norm(mask_i) * np.linalg.norm(mask_j) + 1e-8)

    def forward(self, prototypes_dict, saliency_maps=None):
        """
        Apply DPM: prune and merge prototypes.

        Args:
            prototypes_dict (dict): New prototypes to add.
            saliency_maps (list of dict): Optional, for NIH-specific merging.

        Returns:
            list: The current prototype bank after pruning and merging.
        """
        # Step 1: Add new prototypes
        self.add_prototypes(prototypes_dict, domain_id='current', timestamp=len(self.prototype_bank))

        # Step 2: Compute KL divergence matrix
        kl_matrix = self._compute_kl_divergence_matrix()

        # Step 3: Apply dataset-specific enhancement or generic DPM
        if self.dataset_name == 'BRaTS':
            self._enhance_brats()
        elif self.dataset_name == 'Camelyon16':
            self._enhance_camelyon16()
        elif self.dataset_name == 'PANDA':
            self._enhance_panda()
        elif self.dataset_name == 'NIH' and saliency_maps is not None:
            self._enhance_nih(saliency_maps)
        else:
            # Generic DPM
            self._prune_obsolete_prototypes(kl_matrix)
            self._merge_similar_prototypes(kl_matrix)

        return self.prototype_bank

    def get_prototype_bank(self):
        """Return the current prototype bank."""
        return self.prototype_bank

    def get_memory_efficiency(self, current_domain):
        """Compute Memory Efficiency (ME)."""
        return len(self.prototype_bank) / (current_domain + 1)

    def reset_bank(self):
        """Clear the prototype bank."""
        self.prototype_bank = []
'''
# Example usage
dpm = DynamicPrototypeManager(feature_dim=2048, dataset_name='PANDA')
new_prototypes = {0: torch.randn(2048), 1: torch.randn(2048)}  # Simulated new prototypes

# Update the bank
updated_bank = dpm(new_prototypes)
print(f"Current prototype bank size: {len(updated_bank)}")
print(f"Memory Efficiency: {dpm.get_memory_efficiency(current_domain=5):.3f}")
'''