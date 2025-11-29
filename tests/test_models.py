"""
Test suite for fraud detection models
"""

import pytest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.fraud_detection_model import (
    FraudDetectionNN,
    AttentionFraudDetector,
    LSTMFraudDetector,
    get_model
)


class TestFraudDetectionNN:
    """Test basic fraud detection neural network"""
    
    def test_initialization(self):
        model = FraudDetectionNN(
            input_size=30,
            hidden_layers=[128, 64, 32],
            output_size=1
        )
        
        assert model.input_size == 30
        assert model.output_size == 1
    
    def test_forward_pass(self):
        model = FraudDetectionNN(input_size=30)
        
        # Create sample input
        batch_size = 16
        x = torch.randn(batch_size, 30)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 1)
        
        # Check output range (sigmoid)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_get_embedding(self):
        model = FraudDetectionNN(input_size=30, hidden_layers=[64, 32])
        
        x = torch.randn(10, 30)
        embedding = model.get_embedding(x)
        
        # Embedding should have shape (batch_size, last_hidden_size)
        assert embedding.shape[0] == 10
        assert embedding.shape[1] == 32  # Last hidden layer size


class TestAttentionFraudDetector:
    """Test attention-based fraud detector"""
    
    def test_initialization(self):
        model = AttentionFraudDetector(input_size=30)
        assert model is not None
    
    def test_forward_pass(self):
        model = AttentionFraudDetector(input_size=30)
        
        # Single transaction
        x = torch.randn(16, 30)
        output = model(x)
        
        assert output.shape == (16, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_sequence_input(self):
        model = AttentionFraudDetector(input_size=30)
        
        # Sequence of transactions
        x = torch.randn(8, 10, 30)  # (batch, sequence, features)
        output = model(x)
        
        assert output.shape == (8, 1)


class TestLSTMFraudDetector:
    """Test LSTM-based fraud detector"""
    
    def test_initialization(self):
        model = LSTMFraudDetector(input_size=30)
        assert model is not None
    
    def test_forward_pass(self):
        model = LSTMFraudDetector(input_size=30)
        
        x = torch.randn(16, 30)
        output = model(x)
        
        assert output.shape == (16, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_bidirectional(self):
        model = LSTMFraudDetector(input_size=30, bidirectional=True)
        
        x = torch.randn(8, 10, 30)
        output = model(x)
        
        assert output.shape == (8, 1)


class TestModelFactory:
    """Test model factory function"""
    
    def test_get_model_basic(self):
        model = get_model("FraudDetectionNN", input_size=30)
        assert isinstance(model, FraudDetectionNN)
    
    def test_get_model_attention(self):
        model = get_model("AttentionFraudDetector", input_size=30)
        assert isinstance(model, AttentionFraudDetector)
    
    def test_get_model_lstm(self):
        model = get_model("LSTMFraudDetector", input_size=30)
        assert isinstance(model, LSTMFraudDetector)
    
    def test_invalid_model(self):
        with pytest.raises(ValueError):
            get_model("InvalidModel")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
