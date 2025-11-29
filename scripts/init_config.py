"""
Initialize project configuration and setup
"""

import os
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory_structure():
    """Create necessary project directories"""
    directories = [
        "data/client_data",
        "data/bank_a",
        "data/bank_b",
        "data/bank_c",
        "logs",
        "models",
        "runs/server",
        "runs/client",
        "certs",
        ".azure"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def generate_sample_data():
    """Generate sample fraud detection data for clients"""
    import sys
    sys.path.append('src')
    
    try:
        from utils.data_utils import generate_synthetic_fraud_data, create_federated_splits
        import numpy as np
        
        logger.info("Generating synthetic fraud detection data...")
        
        # Generate data
        X, y = generate_synthetic_fraud_data(n_samples=30000, fraud_ratio=0.02)
        
        # Create federated splits
        client_data = create_federated_splits(X, y, num_clients=3, split_strategy="non_iid")
        
        # Save for each client
        clients = ['bank_a', 'bank_b', 'bank_c']
        for client_name, (X_client, y_client) in zip(clients, client_data):
            data_dir = f"data/{client_name}"
            np.save(f"{data_dir}/features.npy", X_client)
            np.save(f"{data_dir}/labels.npy", y_client)
            logger.info(f"Saved data for {client_name}: {len(X_client)} samples")
        
        logger.info("Sample data generation complete!")
        
    except ImportError as e:
        logger.warning(f"Could not generate sample data: {e}")
        logger.info("Install dependencies first: pip install -r requirements.txt")


def create_env_file():
    """Create .env file with default settings"""
    env_content = """# Environment Configuration
# DO NOT commit this file to version control

# Database
DB_PASSWORD=secure_password_change_me

# Client Tokens
CLIENT_TOKEN_BANK_A=token_bank_a_change_me
CLIENT_TOKEN_BANK_B=token_bank_b_change_me
CLIENT_TOKEN_BANK_C=token_bank_c_change_me

# Security
SECRET_KEY=your_secret_key_here_change_me

# Logging
LOG_LEVEL=INFO

# WandB (optional)
WANDB_API_KEY=your_wandb_api_key_here
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(env_content)
        logger.info("Created .env file")
    else:
        logger.info(".env file already exists")


def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
venv/
ENV/
env/

# PyTorch
*.pth
*.pt
*.ckpt

# Data
data/
*.csv
*.npy
*.npz

# Logs
logs/
*.log

# Models
models/*.pth
models/*.pt

# TensorBoard
runs/
events.out.tfevents.*

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Environment
.env
.env.local

# Certificates
certs/*.key
certs/*.crt
certs/*.pem

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Docker
*.tar

# Misc
.azure/
temp/
tmp/
"""
    
    gitignore_path = Path(".gitignore")
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    logger.info("Created .gitignore file")


def main():
    """Main initialization function"""
    logger.info("="*60)
    logger.info("Secure Federated Fraud Detection - Project Initialization")
    logger.info("="*60)
    
    # Create directories
    logger.info("\n[1/4] Creating directory structure...")
    create_directory_structure()
    
    # Create .env file
    logger.info("\n[2/4] Creating environment configuration...")
    create_env_file()
    
    # Create .gitignore
    logger.info("\n[3/4] Creating .gitignore...")
    create_gitignore()
    
    # Generate sample data
    logger.info("\n[4/4] Generating sample data...")
    generate_sample_data()
    
    logger.info("\n" + "="*60)
    logger.info("Initialization complete!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Install dependencies: pip install -r requirements.txt")
    logger.info("2. Update .env file with your credentials")
    logger.info("3. Run demo: python examples/demo_federated_training.py")
    logger.info("4. See QUICKSTART.md for more information")


if __name__ == "__main__":
    main()
