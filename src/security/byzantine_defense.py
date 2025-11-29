"""
Byzantine Defense Mechanisms
Protects against malicious clients in federated learning
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ByzantineDefense:
    """
    Defense mechanisms against Byzantine (malicious) clients
    
    Implements various methods to detect and filter out malicious updates.
    
    Args:
        method: Defense method name
        tolerance: Fraction of Byzantine clients to tolerate
    """
    
    def __init__(self, method: str = "krum", tolerance: float = 0.2):
        self.method = method
        self.tolerance = tolerance
        self.detection_history = []
        
        logger.info(f"Initialized Byzantine defense: {method}, tolerance={tolerance}")
    
    def filter_updates(
        self,
        client_updates: List[List[torch.Tensor]],
        weights: List[float]
    ) -> Tuple[List[List[torch.Tensor]], List[float]]:
        """
        Filter client updates to remove Byzantine ones
        
        Args:
            client_updates: List of client model updates
            weights: Weights for each client
        
        Returns:
            Tuple of (filtered_updates, filtered_weights)
        """
        if self.method == "krum":
            return self.krum_filter(client_updates, weights)
        elif self.method == "multi_krum":
            return self.multi_krum_filter(client_updates, weights)
        elif self.method == "trimmed_mean":
            return self.trimmed_mean_filter(client_updates, weights)
        elif self.method == "median":
            return client_updates, weights  # Median is inherently robust
        else:
            logger.warning(f"Unknown Byzantine defense method: {self.method}")
            return client_updates, weights
    
    def krum_filter(
        self,
        client_updates: List[List[torch.Tensor]],
        weights: List[float]
    ) -> Tuple[List[List[torch.Tensor]], List[float]]:
        """
        Krum-based filtering
        Select the single most representative update
        
        Args:
            client_updates: List of client updates
            weights: Client weights
        
        Returns:
            Filtered updates and weights
        """
        num_clients = len(client_updates)
        num_byzantine = int(num_clients * self.tolerance)
        
        if num_clients <= 2 * num_byzantine + 2:
            logger.warning("Not enough clients for Krum, returning all updates")
            return client_updates, weights
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(client_updates)
        
        # Compute Krum scores
        scores = self._compute_krum_scores(distances, num_byzantine)
        
        # Select best client
        best_idx = torch.argmin(scores).item()
        
        logger.info(f"Krum selected client {best_idx}")
        
        return [client_updates[best_idx]], [1.0]
    
    def multi_krum_filter(
        self,
        client_updates: List[List[torch.Tensor]],
        weights: List[float],
        num_selected: Optional[int] = None
    ) -> Tuple[List[List[torch.Tensor]], List[float]]:
        """
        Multi-Krum filtering
        Select multiple representative updates
        
        Args:
            client_updates: List of client updates
            weights: Client weights
            num_selected: Number of updates to select
        
        Returns:
            Filtered updates and weights
        """
        num_clients = len(client_updates)
        num_byzantine = int(num_clients * self.tolerance)
        
        if num_selected is None:
            num_selected = max(1, num_clients - num_byzantine - 1)
        
        if num_clients <= 2 * num_byzantine + 2:
            logger.warning("Not enough clients for Multi-Krum, returning all updates")
            return client_updates, weights
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(client_updates)
        
        # Compute Krum scores
        scores = self._compute_krum_scores(distances, num_byzantine)
        
        # Select top-k clients
        _, selected_indices = torch.topk(scores, num_selected, largest=False)
        selected_indices = selected_indices.tolist()
        
        filtered_updates = [client_updates[i] for i in selected_indices]
        filtered_weights = [weights[i] for i in selected_indices]
        
        # Renormalize weights
        weight_sum = sum(filtered_weights)
        filtered_weights = [w / weight_sum for w in filtered_weights]
        
        logger.info(f"Multi-Krum selected {num_selected} clients: {selected_indices}")
        
        return filtered_updates, filtered_weights
    
    def trimmed_mean_filter(
        self,
        client_updates: List[List[torch.Tensor]],
        weights: List[float],
        trim_ratio: Optional[float] = None
    ) -> Tuple[List[List[torch.Tensor]], List[float]]:
        """
        Filter using statistical outlier detection
        Remove updates that are statistical outliers
        
        Args:
            client_updates: List of client updates
            weights: Client weights
            trim_ratio: Fraction to trim (uses tolerance if None)
        
        Returns:
            Filtered updates and weights
        """
        if trim_ratio is None:
            trim_ratio = self.tolerance
        
        num_clients = len(client_updates)
        
        # Compute update norms
        norms = []
        for update in client_updates:
            norm = sum(torch.sum(param ** 2).item() for param in update)
            norms.append(norm)
        
        norms = torch.tensor(norms)
        
        # Compute median and MAD (Median Absolute Deviation)
        median_norm = torch.median(norms)
        mad = torch.median(torch.abs(norms - median_norm))
        
        # Define outliers using modified z-score
        threshold = 3.5  # Standard threshold for outlier detection
        modified_z_scores = 0.6745 * (norms - median_norm) / (mad + 1e-6)
        
        # Keep updates within threshold
        keep_mask = torch.abs(modified_z_scores) < threshold
        selected_indices = torch.where(keep_mask)[0].tolist()
        
        if len(selected_indices) < num_clients // 2:
            logger.warning("Too many outliers detected, keeping all updates")
            return client_updates, weights
        
        filtered_updates = [client_updates[i] for i in selected_indices]
        filtered_weights = [weights[i] for i in selected_indices]
        
        # Renormalize weights
        weight_sum = sum(filtered_weights)
        filtered_weights = [w / weight_sum for w in filtered_weights]
        
        num_removed = num_clients - len(selected_indices)
        logger.info(f"Trimmed mean: removed {num_removed} outliers, kept {len(selected_indices)} updates")
        
        return filtered_updates, filtered_weights
    
    def _compute_pairwise_distances(
        self,
        client_updates: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute pairwise L2 distances between client updates
        
        Args:
            client_updates: List of client updates
        
        Returns:
            Distance matrix
        """
        num_clients = len(client_updates)
        distances = torch.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = 0.0
                for pi, pj in zip(client_updates[i], client_updates[j]):
                    dist += torch.sum((pi - pj) ** 2).item()
                
                distances[i, j] = np.sqrt(dist)
                distances[j, i] = distances[i, j]
        
        return distances
    
    def _compute_krum_scores(
        self,
        distances: torch.Tensor,
        num_byzantine: int
    ) -> torch.Tensor:
        """
        Compute Krum scores for each client
        
        Args:
            distances: Pairwise distance matrix
            num_byzantine: Number of Byzantine clients
        
        Returns:
            Krum scores
        """
        num_clients = distances.shape[0]
        scores = torch.zeros(num_clients)
        n_neighbors = num_clients - num_byzantine - 2
        
        for i in range(num_clients):
            # Sum of distances to n closest neighbors
            sorted_distances, _ = torch.sort(distances[i])
            scores[i] = torch.sum(sorted_distances[1:n_neighbors+1])
        
        return scores
    
    def detect_model_poisoning(
        self,
        client_update: List[torch.Tensor],
        global_model: List[torch.Tensor],
        threshold: float = 10.0
    ) -> bool:
        """
        Detect potential model poisoning attack
        
        Args:
            client_update: Client's model update
            global_model: Current global model
            threshold: Distance threshold for poisoning detection
        
        Returns:
            True if poisoning detected
        """
        # Compute L2 distance between update and global model
        distance = 0.0
        for client_param, global_param in zip(client_update, global_model):
            distance += torch.sum((client_param - global_param) ** 2).item()
        
        distance = np.sqrt(distance)
        
        is_poisoning = distance > threshold
        
        if is_poisoning:
            logger.warning(f"Potential model poisoning detected! Distance: {distance:.2f}")
        
        return is_poisoning
    
    def detect_gradient_attack(
        self,
        gradients: List[torch.Tensor],
        max_norm: float = 10.0
    ) -> bool:
        """
        Detect gradient-based attacks (e.g., gradient explosion)
        
        Args:
            gradients: Gradient tensors
            max_norm: Maximum allowed gradient norm
        
        Returns:
            True if attack detected
        """
        # Compute gradient norm
        total_norm = torch.sqrt(
            sum(torch.sum(g ** 2) for g in gradients)
        ).item()
        
        is_attack = total_norm > max_norm
        
        if is_attack:
            logger.warning(f"Potential gradient attack detected! Norm: {total_norm:.2f}")
        
        return is_attack
