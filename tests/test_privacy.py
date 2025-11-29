"""
Test suite for differential privacy module
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.privacy.differential_privacy import (
    DifferentialPrivacy,
    LocalDifferentialPrivacy,
    PrivacyBudget,
    PrivacyAccountant
)


class TestPrivacyBudget:
    """Test privacy budget tracking"""
    
    def test_initialization(self):
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert budget.epsilon == 1.0
        assert budget.delta == 1e-5
        assert budget.consumed_epsilon == 0.0
    
    def test_has_budget(self):
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert budget.has_budget() is True
        
        budget.consumed_epsilon = 1.0
        assert budget.has_budget() is False
    
    def test_consume_budget(self):
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        budget.consume(0.3)
        assert budget.consumed_epsilon == 0.3
        
        budget.consume(0.4)
        assert budget.consumed_epsilon == 0.7


class TestDifferentialPrivacy:
    """Test differential privacy mechanisms"""
    
    def test_initialization(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        assert dp.epsilon == 1.0
        assert dp.delta == 1e-5
    
    def test_clip_gradients(self):
        dp = DifferentialPrivacy(max_grad_norm=1.0)
        
        # Create gradients with large norm
        gradients = [torch.randn(10, 10) * 10 for _ in range(3)]
        
        clipped, norm = dp.clip_gradients(gradients)
        
        # Check that gradients are clipped
        clipped_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in clipped)).item()
        assert clipped_norm <= dp.max_grad_norm + 1e-5
    
    def test_add_noise(self):
        dp = DifferentialPrivacy(noise_multiplier=1.0)
        
        gradients = [torch.ones(5, 5) for _ in range(2)]
        noised = dp.add_noise(gradients)
        
        # Check that noise was added
        for original, noised_grad in zip(gradients, noised):
            assert not torch.allclose(original, noised_grad)
    
    def test_privatize_gradients(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        
        gradients = [torch.randn(3, 3) for _ in range(2)]
        private_grads = dp.privatize_gradients(gradients)
        
        # Check shapes are preserved
        for orig, priv in zip(gradients, private_grads):
            assert orig.shape == priv.shape
        
        # Check privacy budget was consumed
        assert dp.privacy_budget.consumed_epsilon > 0


class TestLocalDifferentialPrivacy:
    """Test local differential privacy"""
    
    def test_initialization(self):
        ldp = LocalDifferentialPrivacy(epsilon=1.0)
        assert ldp.epsilon == 1.0
    
    def test_add_laplace_noise(self):
        ldp = LocalDifferentialPrivacy(epsilon=1.0, mechanism="laplace")
        
        data = torch.ones(10, 10)
        noised = ldp.add_laplace_noise(data)
        
        assert data.shape == noised.shape
        assert not torch.allclose(data, noised)
    
    def test_privatize(self):
        ldp = LocalDifferentialPrivacy(epsilon=1.0, mechanism="laplace")
        
        data = torch.randn(5, 5)
        privatized = ldp.privatize(data)
        
        assert data.shape == privatized.shape


class TestPrivacyAccountant:
    """Test privacy accounting"""
    
    def test_initialization(self):
        accountant = PrivacyAccountant(epsilon=1.0, delta=1e-5)
        assert accountant.target_epsilon == 1.0
        assert accountant.target_delta == 1e-5
    
    def test_accumulate_privacy_spending(self):
        accountant = PrivacyAccountant(epsilon=1.0, delta=1e-5)
        
        # Simulate privacy spending
        accountant.accumulate_privacy_spending(
            noise_multiplier=1.0,
            sample_rate=0.1,
            steps=100
        )
        
        epsilon, delta = accountant.get_privacy_spent()
        assert epsilon > 0
        assert delta == accountant.target_delta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
