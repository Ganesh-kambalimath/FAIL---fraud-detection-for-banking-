"""
Federated Learning Server
Central aggregation server for coordinating federated training
"""

import asyncio
import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import yaml

# Import project modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fraud_detection_model import get_model
from privacy.differential_privacy import DifferentialPrivacy, PrivacyAccountant
from encryption.homomorphic_encryption import HomomorphicEncryption, SecureAggregation
from aggregation.federated_aggregation import FederatedAggregator
from security.byzantine_defense import ByzantineDefense

logger = logging.getLogger(__name__)


@dataclass
class ClientInfo:
    """Information about a connected client"""
    client_id: str
    name: str
    connected_at: datetime
    num_samples: int
    last_update: Optional[datetime] = None


class FederatedServer:
    """
    Federated Learning Server
    
    Coordinates training across multiple clients, aggregates updates,
    and maintains the global model.
    
    Args:
        config: Server configuration dictionary
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Server settings
        self.host = config['server']['host']
        self.port = config['server']['port']
        self.num_rounds = config['server']['num_rounds']
        self.min_clients = config['server']['min_clients']
        self.max_clients = config['server']['max_clients']
        
        # Initialize global model
        self.global_model = self._initialize_model()
        
        # Initialize privacy mechanisms
        self.differential_privacy = None
        if config['privacy']['differential_privacy']['enabled']:
            self.differential_privacy = DifferentialPrivacy(
                epsilon=config['privacy']['differential_privacy']['epsilon'],
                delta=config['privacy']['differential_privacy']['delta'],
                noise_multiplier=config['privacy']['differential_privacy']['noise_multiplier'],
                max_grad_norm=config['privacy']['differential_privacy']['max_grad_norm']
            )
        
        # Initialize homomorphic encryption
        self.homomorphic_encryption = None
        if config['privacy']['homomorphic_encryption']['enabled']:
            self.homomorphic_encryption = HomomorphicEncryption(
                poly_modulus_degree=config['privacy']['homomorphic_encryption']['poly_modulus_degree'],
                coeff_mod_bit_sizes=config['privacy']['homomorphic_encryption']['coeff_mod_bit_sizes'],
                scale=config['privacy']['homomorphic_encryption']['scale']
            )
        
        # Initialize secure aggregation
        self.secure_aggregation = None
        if config['privacy']['secure_aggregation']['enabled']:
            self.secure_aggregation = SecureAggregation(
                threshold=config['privacy']['secure_aggregation']['threshold']
            )
        
        # Initialize Byzantine defense
        self.byzantine_defense = None
        if config['security']['byzantine_defense']['enabled']:
            self.byzantine_defense = ByzantineDefense(
                method=config['security']['byzantine_defense']['method'],
                tolerance=config['security']['byzantine_defense']['tolerance']
            )
        
        # Initialize aggregator
        self.aggregator = FederatedAggregator(
            strategy=config['federated']['aggregation_strategy']
        )
        
        # Client management
        self.connected_clients: Dict[str, ClientInfo] = {}
        self.client_weights: Dict[str, float] = {}
        
        # Training state
        self.current_round = 0
        self.training_history = []
        
        # Privacy accounting
        self.privacy_accountant = PrivacyAccountant(
            epsilon=config['privacy']['differential_privacy']['epsilon'],
            delta=config['privacy']['differential_privacy']['delta']
        )
        
        logger.info(f"Initialized Federated Server on {self.host}:{self.port}")
        logger.info(f"Training for {self.num_rounds} rounds with {self.min_clients}-{self.max_clients} clients")
    
    def _initialize_model(self) -> torch.nn.Module:
        """Initialize the global model"""
        model_config = self.config['model']
        model = get_model(
            model_name=model_config['architecture'],
            input_size=model_config['input_size'],
            hidden_layers=model_config['hidden_layers'],
            output_size=model_config['output_size'],
            dropout=model_config.get('dropout', 0.3),
            activation=model_config.get('activation', 'relu'),
            batch_norm=model_config.get('batch_norm', True)
        )
        
        logger.info(f"Initialized global model: {model_config['architecture']}")
        return model
    
    def register_client(
        self,
        client_id: str,
        client_name: str,
        num_samples: int
    ) -> bool:
        """
        Register a new client
        
        Args:
            client_id: Unique client identifier
            client_name: Client display name
            num_samples: Number of training samples
        
        Returns:
            True if registration successful
        """
        if len(self.connected_clients) >= self.max_clients:
            logger.warning(f"Client {client_id} rejected: max clients reached")
            return False
        
        if client_id in self.connected_clients:
            logger.warning(f"Client {client_id} already registered")
            return False
        
        client_info = ClientInfo(
            client_id=client_id,
            name=client_name,
            connected_at=datetime.now(),
            num_samples=num_samples
        )
        
        self.connected_clients[client_id] = client_info
        
        # Calculate client weight based on data size
        total_samples = sum(c.num_samples for c in self.connected_clients.values())
        self.client_weights[client_id] = num_samples / total_samples
        
        logger.info(f"Registered client {client_name} ({client_id}) with {num_samples} samples")
        return True
    
    def select_clients(self, round_num: int) -> List[str]:
        """
        Select clients for training round
        
        Args:
            round_num: Current round number
        
        Returns:
            List of selected client IDs
        """
        selection_strategy = self.config['federated']['client_selection']
        available_clients = list(self.connected_clients.keys())
        
        if selection_strategy == "random":
            # Random selection
            num_selected = min(len(available_clients), self.max_clients)
            selected = np.random.choice(
                available_clients,
                size=num_selected,
                replace=False
            ).tolist()
        
        elif selection_strategy == "proportional":
            # Proportional to data size
            weights = [self.client_weights[cid] for cid in available_clients]
            num_selected = min(len(available_clients), self.max_clients)
            selected = np.random.choice(
                available_clients,
                size=num_selected,
                replace=False,
                p=weights
            ).tolist()
        
        else:
            # Select all
            selected = available_clients
        
        logger.info(f"Round {round_num}: Selected {len(selected)} clients: {selected}")
        return selected
    
    def aggregate_updates(
        self,
        client_updates: Dict[str, List[torch.Tensor]],
        round_num: int
    ) -> List[torch.Tensor]:
        """
        Aggregate client model updates
        
        Args:
            client_updates: Dictionary mapping client IDs to their updates
            round_num: Current round number
        
        Returns:
            Aggregated model parameters
        """
        # Get client weights
        weights = [self.client_weights[cid] for cid in client_updates.keys()]
        updates_list = list(client_updates.values())
        
        # Apply Byzantine defense if enabled
        if self.byzantine_defense:
            updates_list, weights = self.byzantine_defense.filter_updates(
                updates_list, weights
            )
            logger.info(f"Byzantine defense: kept {len(updates_list)} updates")
        
        # Apply differential privacy if enabled
        if self.differential_privacy:
            updates_list = [
                self.differential_privacy.privatize_gradients(update)
                for update in updates_list
            ]
            logger.info("Applied differential privacy to updates")
        
        # Aggregate
        aggregated = self.aggregator.aggregate(updates_list, weights)
        
        logger.info(f"Aggregated {len(client_updates)} client updates")
        return aggregated
    
    def update_global_model(self, aggregated_params: List[torch.Tensor]):
        """
        Update global model with aggregated parameters
        
        Args:
            aggregated_params: Aggregated model parameters
        """
        with torch.no_grad():
            for param, new_value in zip(self.global_model.parameters(), aggregated_params):
                param.copy_(new_value)
        
        logger.debug("Updated global model")
    
    def get_model_parameters(self) -> List[torch.Tensor]:
        """
        Get current global model parameters
        
        Returns:
            List of model parameter tensors
        """
        return [param.data.clone() for param in self.global_model.parameters()]
    
    async def train_round(self, round_num: int) -> Dict:
        """
        Execute one round of federated training
        
        Args:
            round_num: Current round number
        
        Returns:
            Round statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Round {round_num}/{self.num_rounds}")
        logger.info(f"{'='*60}")
        
        # Check if enough clients are available
        if len(self.connected_clients) < self.min_clients:
            logger.warning(f"Not enough clients: {len(self.connected_clients)} < {self.min_clients}")
            return None
        
        # Select clients for this round
        selected_clients = self.select_clients(round_num)
        
        # Send global model to selected clients
        global_params = self.get_model_parameters()
        
        # Simulate receiving updates from clients
        # In production, this would be actual network communication
        client_updates = {}
        for client_id in selected_clients:
            # TODO: Send global_params to client and receive update
            # For now, simulate with dummy update
            pass
        
        # Aggregate updates
        if client_updates:
            aggregated = self.aggregate_updates(client_updates, round_num)
            
            # Update global model
            self.update_global_model(aggregated)
        
        # Track privacy budget
        epsilon_spent, delta = self.privacy_accountant.get_privacy_spent()
        
        # Collect round statistics
        round_stats = {
            'round': round_num,
            'num_clients': len(selected_clients),
            'epsilon_spent': epsilon_spent,
            'delta': delta,
            'timestamp': datetime.now()
        }
        
        self.training_history.append(round_stats)
        
        logger.info(f"Round {round_num} complete. Privacy budget: ε={epsilon_spent:.4f}, δ={delta:.2e}")
        
        return round_stats
    
    async def start_training(self):
        """Start federated training process"""
        logger.info("Starting federated training...")
        
        for round_num in range(1, self.num_rounds + 1):
            round_stats = await self.train_round(round_num)
            
            if round_stats is None:
                logger.warning("Round failed, waiting for more clients...")
                await asyncio.sleep(10)
                continue
            
            # Check privacy budget
            if self.privacy_accountant.is_budget_exhausted():
                logger.warning("Privacy budget exhausted, stopping training")
                break
        
        logger.info("Federated training complete!")
        self.save_final_model()
    
    def save_final_model(self, path: str = "models/global_model.pth"):
        """
        Save the final global model
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }, path)
        
        logger.info(f"Saved final model to {path}")


def load_config(config_path: str) -> Dict:
    """Load server configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


async def main():
    """Main server entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = load_config('configs/server_config.yaml')
    
    # Create and start server
    server = FederatedServer(config)
    
    # Start training
    await server.start_training()


if __name__ == "__main__":
    asyncio.run(main())
