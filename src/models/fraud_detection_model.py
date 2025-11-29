"""
Fraud Detection Neural Network Model
Implements a deep learning architecture for detecting fraudulent transactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class FraudDetectionNN(nn.Module):
    """
    Neural Network for Fraud Detection
    
    Architecture:
        - Input Layer
        - Multiple Hidden Layers with Batch Normalization and Dropout
        - Output Layer with Sigmoid Activation
    
    Args:
        input_size: Number of input features
        hidden_layers: List of hidden layer sizes
        output_size: Number of output classes (1 for binary classification)
        dropout: Dropout probability
        activation: Activation function name
        batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_size: int = 30,
        hidden_layers: List[int] = [128, 64, 32],
        output_size: int = 1,
        dropout: float = 0.3,
        activation: str = "relu",
        batch_norm: bool = True
    ):
        super(FraudDetectionNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.dropout_prob = dropout
        self.batch_norm = batch_norm
        
        # Build the network layers
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_layers):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            elif activation.lower() == "elu":
                layers.append(nn.ELU())
            elif activation.lower() == "selu":
                layers.append(nn.SELU())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        self.hidden_layers_seq = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Hidden layers
        x = self.hidden_layers_seq(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Sigmoid activation for binary classification
        x = torch.sigmoid(x)
        
        return x
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding from the last hidden layer
        
        Args:
            x: Input tensor
        
        Returns:
            Embedding tensor
        """
        return self.hidden_layers_seq(x)


class AttentionFraudDetector(nn.Module):
    """
    Fraud Detection Model with Attention Mechanism
    Useful for capturing temporal patterns in transaction sequences
    """
    
    def __init__(
        self,
        input_size: int = 30,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super(AttentionFraudDetector, self).__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Multi-head attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
               or (batch_size, input_size) for single transactions
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Handle single transaction input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Project input
        x = self.input_projection(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x


class LSTMFraudDetector(nn.Module):
    """
    LSTM-based Fraud Detection Model
    Captures sequential patterns in transaction history
    """
    
    def __init__(
        self,
        input_size: int = 30,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(LSTMFraudDetector, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
               or (batch_size, input_size) for single transactions
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Handle single transaction input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        x = lstm_out[:, -1, :]
        
        # Output layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get a fraud detection model
    
    Args:
        model_name: Name of the model architecture
        **kwargs: Model-specific parameters
    
    Returns:
        Initialized model
    """
    models = {
        "FraudDetectionNN": FraudDetectionNN,
        "AttentionFraudDetector": AttentionFraudDetector,
        "LSTMFraudDetector": LSTMFraudDetector
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)
