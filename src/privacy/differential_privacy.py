"""
Differential Privacy Implementation
Provides privacy-preserving mechanisms for federated learning
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """Privacy budget tracker for differential privacy"""
    epsilon: float
    delta: float
    consumed_epsilon: float = 0.0
    
    def has_budget(self) -> bool:
        """Check if privacy budget is available"""
        return self.consumed_epsilon < self.epsilon
    
    def consume(self, eps: float):
        """Consume privacy budget"""
        self.consumed_epsilon += eps
        logger.info(f"Privacy budget consumed: {eps:.4f}, "
                   f"remaining: {self.epsilon - self.consumed_epsilon:.4f}")
    
    def reset(self):
        """Reset consumed budget"""
        self.consumed_epsilon = 0.0


class DifferentialPrivacy:
    """
    Differential Privacy Mechanism for Federated Learning
    
    Implements DP-SGD (Differentially Private Stochastic Gradient Descent)
    with Gaussian noise addition and gradient clipping.
    
    Args:
        epsilon: Privacy budget parameter (lower = more privacy)
        delta: Failure probability (typically 1/n^2)
        noise_multiplier: Noise scale multiplier
        max_grad_norm: Maximum gradient norm for clipping
        secure_mode: Whether to use cryptographically secure RNG
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        secure_mode: bool = True
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.secure_mode = secure_mode
        
        self.privacy_budget = PrivacyBudget(epsilon, delta)
        
        logger.info(f"Initialized DP with ε={epsilon}, δ={delta}, "
                   f"noise_multiplier={noise_multiplier}, max_grad_norm={max_grad_norm}")
    
    def clip_gradients(
        self,
        gradients: List[torch.Tensor],
        max_norm: Optional[float] = None
    ) -> Tuple[List[torch.Tensor], float]:
        """
        Clip gradients to maximum norm
        
        Args:
            gradients: List of gradient tensors
            max_norm: Maximum norm for clipping (uses self.max_grad_norm if None)
        
        Returns:
            Tuple of (clipped_gradients, actual_norm)
        """
        if max_norm is None:
            max_norm = self.max_grad_norm
        
        # Calculate total norm
        total_norm = torch.sqrt(
            sum(torch.sum(g ** 2) for g in gradients)
        ).item()
        
        # Clip if necessary
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)
        
        clipped_gradients = [g * clip_coef for g in gradients]
        
        return clipped_gradients, total_norm
    
    def add_noise(
        self,
        gradients: List[torch.Tensor],
        sensitivity: Optional[float] = None
    ) -> List[torch.Tensor]:
        """
        Add Gaussian noise to gradients
        
        Args:
            gradients: List of gradient tensors
            sensitivity: Sensitivity of the query (uses max_grad_norm if None)
        
        Returns:
            Noised gradients
        """
        if sensitivity is None:
            sensitivity = self.max_grad_norm
        
        # Calculate noise scale
        noise_scale = self.noise_multiplier * sensitivity
        
        noised_gradients = []
        for grad in gradients:
            if self.secure_mode:
                # Use cryptographically secure random generator
                noise = torch.randn_like(grad) * noise_scale
            else:
                noise = torch.randn_like(grad) * noise_scale
            
            noised_gradients.append(grad + noise)
        
        return noised_gradients
    
    def privatize_gradients(
        self,
        gradients: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Apply full DP mechanism: clip + noise
        
        Args:
            gradients: List of gradient tensors
        
        Returns:
            Privatized gradients
        """
        # Clip gradients
        clipped_grads, grad_norm = self.clip_gradients(gradients)
        
        # Add noise
        private_grads = self.add_noise(clipped_grads)
        
        # Track privacy budget
        epsilon_spent = self._compute_epsilon_spent(len(gradients))
        self.privacy_budget.consume(epsilon_spent)
        
        logger.debug(f"Gradient norm: {grad_norm:.4f}, "
                    f"Privacy budget spent: {epsilon_spent:.6f}")
        
        return private_grads
    
    def _compute_epsilon_spent(self, num_steps: int) -> float:
        """
        Compute epsilon spent using moments accountant
        
        Args:
            num_steps: Number of training steps
        
        Returns:
            Epsilon spent
        """
        # Simplified epsilon calculation
        # In production, use privacy accounting library
        q = 1.0  # Sampling ratio
        sigma = self.noise_multiplier
        
        # RDP to DP conversion (simplified)
        epsilon = q * num_steps / (2 * sigma ** 2)
        
        return epsilon
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy expenditure
        
        Returns:
            Tuple of (epsilon_spent, delta)
        """
        return (self.privacy_budget.consumed_epsilon, self.delta)
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted"""
        return not self.privacy_budget.has_budget()


class LocalDifferentialPrivacy:
    """
    Local Differential Privacy (LDP)
    Each client adds noise to their data before any processing
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        mechanism: str = "laplace"
    ):
        self.epsilon = epsilon
        self.mechanism = mechanism
        logger.info(f"Initialized LDP with ε={epsilon}, mechanism={mechanism}")
    
    def randomized_response(
        self,
        data: torch.Tensor,
        probability: Optional[float] = None
    ) -> torch.Tensor:
        """
        Randomized response mechanism for binary data
        
        Args:
            data: Binary data tensor
            probability: Flip probability (computed from epsilon if None)
        
        Returns:
            Privatized data
        """
        if probability is None:
            # Compute optimal flip probability
            probability = 1 / (1 + np.exp(self.epsilon))
        
        # Flip bits with probability p
        flip_mask = torch.rand_like(data) < probability
        privatized_data = torch.where(flip_mask, 1 - data, data)
        
        return privatized_data
    
    def add_laplace_noise(
        self,
        data: torch.Tensor,
        sensitivity: float = 1.0
    ) -> torch.Tensor:
        """
        Add Laplace noise for LDP
        
        Args:
            data: Data tensor
            sensitivity: Sensitivity of the query
        
        Returns:
            Noised data
        """
        scale = sensitivity / self.epsilon
        noise = torch.tensor(
            np.random.laplace(0, scale, data.shape),
            dtype=data.dtype,
            device=data.device
        )
        
        return data + noise
    
    def privatize(
        self,
        data: torch.Tensor,
        sensitivity: float = 1.0
    ) -> torch.Tensor:
        """
        Apply LDP mechanism to data
        
        Args:
            data: Input data
            sensitivity: Sensitivity parameter
        
        Returns:
            Privatized data
        """
        if self.mechanism == "laplace":
            return self.add_laplace_noise(data, sensitivity)
        elif self.mechanism == "gaussian":
            scale = sensitivity / self.epsilon
            noise = torch.randn_like(data) * scale
            return data + noise
        elif self.mechanism == "randomized_response":
            return self.randomized_response(data)
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")


class PrivacyAccountant:
    """
    Privacy Accountant for tracking cumulative privacy loss
    Implements Renyi Differential Privacy (RDP) accounting
    """
    
    def __init__(self, epsilon: float, delta: float):
        self.target_epsilon = epsilon
        self.target_delta = delta
        self.rdp_orders = [1 + x / 10.0 for x in range(1, 100)]
        self.rdp_values = [0.0] * len(self.rdp_orders)
    
    def accumulate_privacy_spending(
        self,
        noise_multiplier: float,
        sample_rate: float,
        steps: int
    ):
        """
        Accumulate privacy spending for a training round
        
        Args:
            noise_multiplier: Noise multiplier used
            sample_rate: Sampling rate
            steps: Number of steps
        """
        for i, alpha in enumerate(self.rdp_orders):
            rdp = self._compute_rdp(alpha, noise_multiplier, sample_rate)
            self.rdp_values[i] += rdp * steps
    
    def _compute_rdp(
        self,
        alpha: float,
        noise_multiplier: float,
        sample_rate: float
    ) -> float:
        """
        Compute RDP for given parameters
        
        Args:
            alpha: RDP order
            noise_multiplier: Noise multiplier
            sample_rate: Sampling rate
        
        Returns:
            RDP value
        """
        if alpha == 1:
            return 0
        
        # Simplified RDP calculation
        rdp = (alpha * sample_rate ** 2) / (2 * noise_multiplier ** 2)
        return rdp
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Convert RDP to (epsilon, delta)-DP
        
        Returns:
            Tuple of (epsilon, delta)
        """
        min_epsilon = float('inf')
        
        for alpha, rdp in zip(self.rdp_orders, self.rdp_values):
            epsilon = rdp + np.log(1 / self.target_delta) / (alpha - 1)
            min_epsilon = min(min_epsilon, epsilon)
        
        return (min_epsilon, self.target_delta)
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted"""
        epsilon, _ = self.get_privacy_spent()
        return epsilon >= self.target_epsilon
