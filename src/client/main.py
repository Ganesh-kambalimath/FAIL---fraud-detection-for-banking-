"""
Federated Learning Client
Local training and model update submission
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fraud_detection_model import get_model
from privacy.differential_privacy import DifferentialPrivacy, LocalDifferentialPrivacy
from utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Federated Learning Client
    
    Handles local model training and secure update submission.
    
    Args:
        config: Client configuration dictionary
        client_id: Unique client identifier
    """
    
    def __init__(self, config: Dict, client_id: str):
        self.config = config
        self.client_id = client_id
        self.client_name = config['client']['name']
        
        # Server connection
        self.server_host = config['server']['host']
        self.server_port = config['server']['port']
        
        # Initialize local model
        self.model = self._initialize_model()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Privacy mechanisms
        self.differential_privacy = None
        if config['privacy']['differential_privacy']['enabled']:
            self.differential_privacy = DifferentialPrivacy(
                epsilon=config['privacy']['differential_privacy']['epsilon'],
                delta=config['privacy']['differential_privacy']['delta'],
                noise_multiplier=config['privacy']['differential_privacy']['noise_multiplier'],
                max_grad_norm=config['privacy']['differential_privacy']['max_grad_norm']
            )
        
        self.local_dp = None
        if config['privacy'].get('local_differential_privacy', {}).get('enabled'):
            self.local_dp = LocalDifferentialPrivacy(
                epsilon=config['privacy']['local_differential_privacy']['epsilon_local']
            )
        
        # Training parameters
        self.local_epochs = config['training']['local_epochs']
        self.batch_size = config['training']['batch_size']
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() and config['resources']['gpu'] else "cpu")
        self.model.to(self.device)
        
        # Data
        self.train_loader = None
        self.val_loader = None
        
        logger.info(f"Initialized client {self.client_name} ({self.client_id}) on {self.device}")
    
    def _initialize_model(self) -> nn.Module:
        """Initialize local model"""
        model_config = self.config['model']
        model = get_model(
            model_name=model_config['architecture'],
            input_size=model_config['input_size'],
            hidden_layers=model_config['hidden_layers'],
            output_size=model_config['output_size']
        )
        return model
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        training_config = self.config['training']
        optimizer_name = training_config['optimizer'].lower()
        lr = training_config['learning_rate']
        weight_decay = training_config.get('weight_decay', 0.0)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            momentum = training_config.get('momentum', 0.9)
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def load_data(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Load local training data
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        # Apply local differential privacy if enabled
        if self.local_dp:
            X_train_tensor = self.local_dp.privatize(X_train_tensor)
        
        # Create training dataset
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Create validation dataset if provided
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        logger.info(f"Loaded {len(train_dataset)} training samples")
        if self.val_loader:
            logger.info(f"Loaded {len(val_dataset)} validation samples")
    
    def set_model_parameters(self, parameters: List[torch.Tensor]):
        """
        Set model parameters from global model
        
        Args:
            parameters: List of parameter tensors
        """
        with torch.no_grad():
            for param, new_value in zip(self.model.parameters(), parameters):
                param.copy_(new_value)
        
        logger.debug("Updated local model with global parameters")
    
    def get_model_parameters(self) -> List[torch.Tensor]:
        """
        Get current model parameters
        
        Returns:
            List of parameter tensors
        """
        return [param.data.clone() for param in self.model.parameters()]
    
    def train_local_model(self) -> Dict:
        """
        Train model on local data
        
        Returns:
            Training statistics
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Apply gradient clipping if DP enabled
                if self.differential_privacy:
                    # Clip gradients
                    gradients = [param.grad for param in self.model.parameters()]
                    clipped_grads, _ = self.differential_privacy.clip_gradients(gradients)
                    
                    # Set clipped gradients
                    for param, clipped_grad in zip(self.model.parameters(), clipped_grads):
                        param.grad = clipped_grad
                
                self.optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
            
            avg_epoch_loss = epoch_loss / total_samples
            total_loss += avg_epoch_loss
            
            logger.debug(f"Epoch {epoch+1}/{self.local_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        avg_loss = total_loss / self.local_epochs
        
        # Evaluate on validation set if available
        val_metrics = {}
        if self.val_loader:
            val_metrics = self.evaluate()
        
        stats = {
            'train_loss': avg_loss,
            'num_samples': total_samples,
            **val_metrics
        }
        
        logger.info(f"Local training complete. Loss: {avg_loss:.4f}")
        
        return stats
    
    def evaluate(self) -> Dict:
        """
        Evaluate model on validation set
        
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                
                all_predictions.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader.dataset)
        
        # Compute metrics
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()
        
        metrics = compute_metrics(targets, predictions)
        metrics['val_loss'] = avg_loss
        
        logger.info(f"Validation - Loss: {avg_loss:.4f}, "
                   f"Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def get_model_update(self, global_params: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute model update (difference from global model)
        
        Args:
            global_params: Global model parameters
        
        Returns:
            Model update
        """
        local_params = self.get_model_parameters()
        
        update = []
        for local, global_p in zip(local_params, global_params):
            update.append(local - global_p)
        
        # Apply differential privacy to update if enabled
        if self.differential_privacy:
            update = self.differential_privacy.privatize_gradients(update)
        
        return update
    
    def participate_in_round(self, global_params: List[torch.Tensor]) -> Tuple[List[torch.Tensor], Dict]:
        """
        Participate in one training round
        
        Args:
            global_params: Global model parameters
        
        Returns:
            Tuple of (model_update, training_stats)
        """
        # Set global parameters
        self.set_model_parameters(global_params)
        
        # Train locally
        stats = self.train_local_model()
        
        # Get update
        update = self.get_model_update(global_params)
        
        return update, stats


def load_config(config_path: str) -> Dict:
    """Load client configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main client entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--client-id', type=str, required=True, help='Client ID')
    parser.add_argument('--config', type=str, default='configs/client_config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = load_config(args.config)
    config['client']['id'] = args.client_id
    
    # Create client
    client = FederatedClient(config, args.client_id)
    
    logger.info(f"Client {args.client_id} ready and waiting for server...")


if __name__ == "__main__":
    main()
