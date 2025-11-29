"""
Test suite for aggregation strategies
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aggregation.federated_aggregation import FederatedAggregator


class TestFederatedAggregator:
    """Test federated aggregation strategies"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create sample client updates
        self.client_updates = [
            [torch.randn(5, 5), torch.randn(3, 3)],
            [torch.randn(5, 5), torch.randn(3, 3)],
            [torch.randn(5, 5), torch.randn(3, 3)]
        ]
        self.weights = [0.3, 0.5, 0.2]
    
    def test_weighted_average(self):
        aggregator = FederatedAggregator(strategy="weighted_average")
        result = aggregator.aggregate(self.client_updates, self.weights)
        
        # Check shapes
        assert len(result) == len(self.client_updates[0])
        assert result[0].shape == self.client_updates[0][0].shape
    
    def test_median_aggregation(self):
        aggregator = FederatedAggregator(strategy="median")
        result = aggregator.aggregate(self.client_updates)
        
        # Check shapes
        assert len(result) == len(self.client_updates[0])
        assert result[0].shape == self.client_updates[0][0].shape
    
    def test_trimmed_mean(self):
        aggregator = FederatedAggregator(strategy="trimmed_mean")
        result = aggregator.aggregate(self.client_updates)
        
        # Check shapes
        assert len(result) == len(self.client_updates[0])
        assert result[0].shape == self.client_updates[0][0].shape
    
    def test_krum(self):
        aggregator = FederatedAggregator()
        
        # Need more clients for Krum
        large_updates = [
            [torch.randn(5, 5)] for _ in range(10)
        ]
        
        result = aggregator.krum(large_updates, num_byzantine=2)
        
        # Krum returns one client's update
        assert len(result) == len(large_updates[0])
    
    def test_multi_krum(self):
        aggregator = FederatedAggregator()
        
        # Need more clients for Multi-Krum
        large_updates = [
            [torch.randn(5, 5)] for _ in range(10)
        ]
        
        result = aggregator.multi_krum(large_updates, num_byzantine=2, num_selected=5)
        
        # Check shape
        assert len(result) == len(large_updates[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
