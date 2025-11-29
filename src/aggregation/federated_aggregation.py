"""
Federated Aggregation Strategies
Implements various aggregation methods for federated learning
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FederatedAggregator:
    """
    Federated Learning Aggregation Strategies
    
    Implements various methods for aggregating client model updates.
    
    Args:
        strategy: Aggregation strategy name
    """
    
    def __init__(self, strategy: str = "weighted_average"):
        self.strategy = strategy
        logger.info(f"Initialized aggregator with strategy: {strategy}")
    
    def aggregate(
        self,
        client_updates: List[List[torch.Tensor]],
        weights: Optional[List[float]] = None
    ) -> List[torch.Tensor]:
        """
        Aggregate client updates using specified strategy
        
        Args:
            client_updates: List of client model updates
            weights: Optional weights for each client
        
        Returns:
            Aggregated model parameters
        """
        if self.strategy == "weighted_average":
            return self.weighted_average(client_updates, weights)
        elif self.strategy == "median":
            return self.coordinate_median(client_updates)
        elif self.strategy == "trimmed_mean":
            return self.trimmed_mean(client_updates, trim_ratio=0.1)
        elif self.strategy == "fedavg":
            return self.fedavg(client_updates, weights)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.strategy}")
    
    def weighted_average(
        self,
        client_updates: List[List[torch.Tensor]],
        weights: Optional[List[float]] = None
    ) -> List[torch.Tensor]:
        """
        Weighted average aggregation (FedAvg)
        
        Args:
            client_updates: List of client model updates
            weights: Weights for each client (typically based on data size)
        
        Returns:
            Aggregated parameters
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        num_clients = len(client_updates)
        
        # Default to equal weights
        if weights is None:
            weights = [1.0 / num_clients] * num_clients
        
        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # Initialize aggregated parameters
        aggregated = []
        num_params = len(client_updates[0])
        
        for param_idx in range(num_params):
            # Weighted sum of parameters
            weighted_sum = torch.zeros_like(client_updates[0][param_idx])
            
            for client_idx, weight in enumerate(weights):
                param = client_updates[client_idx][param_idx]
                weighted_sum += weight * param
            
            aggregated.append(weighted_sum)
        
        logger.debug(f"Weighted average: aggregated {len(aggregated)} parameters from {num_clients} clients")
        return aggregated
    
    def fedavg(
        self,
        client_updates: List[List[torch.Tensor]],
        weights: Optional[List[float]] = None
    ) -> List[torch.Tensor]:
        """
        FedAvg algorithm (McMahan et al., 2017)
        
        Same as weighted_average but explicitly named.
        """
        return self.weighted_average(client_updates, weights)
    
    def coordinate_median(
        self,
        client_updates: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Coordinate-wise median aggregation
        Robust to Byzantine attacks
        
        Args:
            client_updates: List of client model updates
        
        Returns:
            Aggregated parameters using median
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        aggregated = []
        num_params = len(client_updates[0])
        
        for param_idx in range(num_params):
            # Stack parameters from all clients
            stacked = torch.stack([
                client_updates[client_idx][param_idx]
                for client_idx in range(len(client_updates))
            ])
            
            # Compute coordinate-wise median
            median_param = torch.median(stacked, dim=0)[0]
            aggregated.append(median_param)
        
        logger.debug(f"Median aggregation: aggregated {len(aggregated)} parameters")
        return aggregated
    
    def trimmed_mean(
        self,
        client_updates: List[List[torch.Tensor]],
        trim_ratio: float = 0.1
    ) -> List[torch.Tensor]:
        """
        Trimmed mean aggregation
        Remove top and bottom percentiles before averaging
        
        Args:
            client_updates: List of client model updates
            trim_ratio: Fraction of values to trim from each end
        
        Returns:
            Aggregated parameters using trimmed mean
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        num_clients = len(client_updates)
        num_trim = int(num_clients * trim_ratio)
        
        aggregated = []
        num_params = len(client_updates[0])
        
        for param_idx in range(num_params):
            # Stack parameters from all clients
            stacked = torch.stack([
                client_updates[client_idx][param_idx]
                for client_idx in range(len(client_updates))
            ])
            
            # Sort along client dimension
            sorted_params, _ = torch.sort(stacked, dim=0)
            
            # Trim and compute mean
            if num_trim > 0:
                trimmed = sorted_params[num_trim:-num_trim]
            else:
                trimmed = sorted_params
            
            mean_param = torch.mean(trimmed, dim=0)
            aggregated.append(mean_param)
        
        logger.debug(f"Trimmed mean: aggregated {len(aggregated)} parameters, trimmed {num_trim} from each end")
        return aggregated
    
    def fedprox(
        self,
        client_updates: List[List[torch.Tensor]],
        global_model: List[torch.Tensor],
        mu: float = 0.01,
        weights: Optional[List[float]] = None
    ) -> List[torch.Tensor]:
        """
        FedProx aggregation with proximal term
        
        Args:
            client_updates: List of client model updates
            global_model: Current global model parameters
            mu: Proximal term coefficient
            weights: Client weights
        
        Returns:
            Aggregated parameters
        """
        # First do weighted average
        avg_update = self.weighted_average(client_updates, weights)
        
        # Add proximal term
        aggregated = []
        for avg_param, global_param in zip(avg_update, global_model):
            prox_param = avg_param + mu * (global_param - avg_param)
            aggregated.append(prox_param)
        
        return aggregated
    
    def krum(
        self,
        client_updates: List[List[torch.Tensor]],
        num_byzantine: int = 0
    ) -> List[torch.Tensor]:
        """
        Krum aggregation (Blanchard et al., 2017)
        Select the most representative update
        
        Args:
            client_updates: List of client model updates
            num_byzantine: Number of Byzantine clients to tolerate
        
        Returns:
            Selected client update
        """
        num_clients = len(client_updates)
        
        if num_clients <= 2 * num_byzantine + 2:
            raise ValueError(f"Not enough clients: need > {2 * num_byzantine + 2}, got {num_clients}")
        
        # Compute pairwise distances
        distances = torch.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                # Compute L2 distance between updates
                dist = 0.0
                for pi, pj in zip(client_updates[i], client_updates[j]):
                    dist += torch.sum((pi - pj) ** 2).item()
                
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Compute Krum scores
        scores = torch.zeros(num_clients)
        n_neighbors = num_clients - num_byzantine - 2
        
        for i in range(num_clients):
            # Sum of distances to n closest neighbors
            sorted_distances, _ = torch.sort(distances[i])
            scores[i] = torch.sum(sorted_distances[1:n_neighbors+1])
        
        # Select client with minimum score
        selected_idx = torch.argmin(scores).item()
        
        logger.debug(f"Krum: selected client {selected_idx}")
        return client_updates[selected_idx]
    
    def multi_krum(
        self,
        client_updates: List[List[torch.Tensor]],
        num_byzantine: int = 0,
        num_selected: int = None
    ) -> List[torch.Tensor]:
        """
        Multi-Krum aggregation
        Average top-m Krum scores
        
        Args:
            client_updates: List of client model updates
            num_byzantine: Number of Byzantine clients to tolerate
            num_selected: Number of clients to select (default: m = n - f - 2)
        
        Returns:
            Aggregated parameters
        """
        num_clients = len(client_updates)
        
        if num_selected is None:
            num_selected = num_clients - num_byzantine - 2
        
        # Compute pairwise distances
        distances = torch.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = 0.0
                for pi, pj in zip(client_updates[i], client_updates[j]):
                    dist += torch.sum((pi - pj) ** 2).item()
                
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Compute Krum scores
        scores = torch.zeros(num_clients)
        n_neighbors = num_clients - num_byzantine - 2
        
        for i in range(num_clients):
            sorted_distances, _ = torch.sort(distances[i])
            scores[i] = torch.sum(sorted_distances[1:n_neighbors+1])
        
        # Select top-m clients
        _, selected_indices = torch.topk(scores, num_selected, largest=False)
        selected_indices = selected_indices.tolist()
        
        # Average selected updates
        selected_updates = [client_updates[i] for i in selected_indices]
        aggregated = self.weighted_average(selected_updates)
        
        logger.debug(f"Multi-Krum: selected {num_selected} clients: {selected_indices}")
        return aggregated
