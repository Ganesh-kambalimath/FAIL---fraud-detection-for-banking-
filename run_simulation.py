"""
Main entry point for running federated training
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from examples.demo_federated_training import simulate_federated_training


def main():
    parser = argparse.ArgumentParser(
        description='Secure Federated Learning Framework for Fraud Detection'
    )
    
    parser.add_argument(
        '--num-clients',
        type=int,
        default=3,
        help='Number of participating clients (default: 3)'
    )
    
    parser.add_argument(
        '--num-rounds',
        type=int,
        default=10,
        help='Number of federated training rounds (default: 10)'
    )
    
    parser.add_argument(
        '--local-epochs',
        type=int,
        default=5,
        help='Local training epochs per round (default: 5)'
    )
    
    parser.add_argument(
        '--no-dp',
        action='store_true',
        help='Disable differential privacy'
    )
    
    parser.add_argument(
        '--no-byzantine',
        action='store_true',
        help='Disable Byzantine defense'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to the credit card fraud dataset (CSV)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run simulation
    simulate_federated_training(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        use_dp=not args.no_dp,
        use_byzantine_defense=not args.no_byzantine,
        data_path=args.data_path
    )


if __name__ == "__main__":
    main()
