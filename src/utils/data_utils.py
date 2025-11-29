"""
Data generation and preprocessing utilities
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def generate_synthetic_fraud_data(
    n_samples: int = 10000,
    fraud_ratio: float = 0.02,
    n_features: int = 30,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic fraud detection dataset
    
    Args:
        n_samples: Number of samples to generate
        fraud_ratio: Ratio of fraudulent transactions
        n_features: Number of features
        random_state: Random seed
    
    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(random_state)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Generate normal transactions
    normal_transactions = np.random.randn(n_normal, n_features)
    
    # Generate fraudulent transactions (with different distribution)
    fraud_transactions = np.random.randn(n_fraud, n_features) * 1.5 + 0.5
    
    # Combine
    X = np.vstack([normal_transactions, fraud_transactions])
    y = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    logger.info(f"Generated {n_samples} samples ({n_fraud} fraud, {n_normal} normal)")
    
    return X, y


def load_credit_card_fraud_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load credit card fraud dataset (e.g., Kaggle Credit Card Fraud)
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        Tuple of (features, labels)
    """
    try:
        df = pd.read_csv(filepath)
        
        # Separate features and labels
        if 'Class' in df.columns:
            y = df['Class'].values
            X = df.drop('Class', axis=1).values
        elif 'isFraud' in df.columns:
            y = df['isFraud'].values
            X = df.drop('isFraud', axis=1).values
        else:
            raise ValueError("Label column not found (expected 'Class' or 'isFraud')")
        
        logger.info(f"Loaded {len(X)} samples from {filepath}")
        logger.info(f"Fraud ratio: {y.mean():.4f}")
        
        return X, y
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.info("Generating synthetic data instead...")
        return generate_synthetic_fraud_data()


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    normalize: bool = True,
    balance: bool = False,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess fraud detection data
    
    Args:
        X: Features
        y: Labels
        normalize: Whether to normalize features
        balance: Whether to balance classes using SMOTE
        test_size: Test set ratio
        val_size: Validation set ratio
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split train+val into train and val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=random_state, stratify=y_trainval
    )
    
    # Normalize features
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        logger.info("Normalized features")
    
    # Balance classes using SMOTE
    if balance:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"Balanced training set using SMOTE: {len(X_train)} samples")
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_federated_splits(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int = 3,
    split_strategy: str = "iid",
    alpha: float = 0.5,
    random_state: int = 42
) -> list:
    """
    Split data for federated learning clients
    
    Args:
        X: Features
        y: Labels
        num_clients: Number of clients
        split_strategy: 'iid' or 'non_iid'
        alpha: Dirichlet distribution parameter for non-IID split
        random_state: Random seed
    
    Returns:
        List of (X_client, y_client) tuples
    """
    np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    if split_strategy == "iid":
        # IID split - each client gets random samples
        splits = np.array_split(indices, num_clients)
        client_data = [(X[split], y[split]) for split in splits]
        
        logger.info(f"Created {num_clients} IID splits")
    
    elif split_strategy == "non_iid":
        # Non-IID split using Dirichlet distribution
        labels = y[indices]
        unique_labels = np.unique(labels)
        
        # Assign samples to clients based on Dirichlet distribution
        label_distributions = np.random.dirichlet([alpha] * num_clients, len(unique_labels))
        
        client_indices = [[] for _ in range(num_clients)]
        
        for label_idx, label in enumerate(unique_labels):
            label_mask = labels == label
            label_indices = indices[label_mask]
            
            # Split according to Dirichlet distribution
            distribution = label_distributions[label_idx]
            split_points = (np.cumsum(distribution) * len(label_indices)).astype(int)
            
            start = 0
            for client_idx in range(num_clients):
                end = split_points[client_idx] if client_idx < num_clients - 1 else len(label_indices)
                client_indices[client_idx].extend(label_indices[start:end])
                start = end
        
        # Shuffle each client's data
        client_data = []
        for client_idx in range(num_clients):
            client_idx_array = np.array(client_indices[client_idx])
            np.random.shuffle(client_idx_array)
            client_data.append((X[client_idx_array], y[client_idx_array]))
        
        logger.info(f"Created {num_clients} non-IID splits (alpha={alpha})")
    
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    
    # Log statistics
    for i, (X_client, y_client) in enumerate(client_data):
        fraud_ratio = y_client.mean()
        logger.info(f"Client {i}: {len(X_client)} samples, fraud ratio: {fraud_ratio:.4f}")
    
    return client_data


def simulate_data_heterogeneity(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int = 3,
    heterogeneity_level: str = "medium"
) -> list:
    """
    Simulate data heterogeneity across clients
    
    Args:
        X: Features
        y: Labels
        num_clients: Number of clients
        heterogeneity_level: 'low', 'medium', or 'high'
    
    Returns:
        List of (X_client, y_client) tuples
    """
    alpha_map = {
        "low": 10.0,    # More IID
        "medium": 0.5,  # Moderate heterogeneity
        "high": 0.1     # High heterogeneity
    }
    
    alpha = alpha_map.get(heterogeneity_level, 0.5)
    
    return create_federated_splits(
        X, y,
        num_clients=num_clients,
        split_strategy="non_iid",
        alpha=alpha
    )


def load_ieee_cis_data(directory_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load IEEE-CIS Fraud Detection dataset
    
    Args:
        directory_path: Directory containing train_transaction.csv
    
    Returns:
        Tuple of (features, labels)
    """
    import os
    
    transaction_path = os.path.join(directory_path, 'train_transaction.csv')
    identity_path = os.path.join(directory_path, 'train_identity.csv')
    
    if not os.path.exists(transaction_path):
        raise FileNotFoundError(f"train_transaction.csv not found in {directory_path}")
    
    logger.info(f"Loading IEEE-CIS data from {transaction_path}...")
    
    # Load transaction data (optimize types to save memory)
    df = pd.read_csv(transaction_path)
    
    # Merge with identity if available
    if os.path.exists(identity_path):
        logger.info(f"Found identity data, merging...")
        df_id = pd.read_csv(identity_path)
        df = pd.merge(df, df_id, on='TransactionID', how='left')
    
    logger.info(f"Loaded dataset shape: {df.shape}")
    
    # target label
    if 'isFraud' in df.columns:
        y = df['isFraud'].values
        X_df = df.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
    else:
        raise ValueError("isFraud column not found")
        
    # Handle categorical columns
    # Identify categorical columns (object type)
    cat_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    
    # Also some numeric columns are actually categorical in this dataset (e.g. card1-6, addr1-2)
    # For simplicity in this adaptation, we'll treat object columns as categorical
    # and fill missing values
    
    logger.info("Preprocessing IEEE-CIS data (this may take a moment)...")
    
    # 1. Drop columns with too many missing values (>90%)
    missing_ratio = X_df.isnull().sum() / len(X_df)
    drop_cols = missing_ratio[missing_ratio > 0.9].index.tolist()
    X_df = X_df.drop(columns=drop_cols)
    logger.info(f"Dropped {len(drop_cols)} columns with >90% missing values")
    
    # 2. Fill missing values
    # Numeric: median
    num_cols = X_df.select_dtypes(include=['number']).columns
    X_df[num_cols] = X_df[num_cols].fillna(X_df[num_cols].median())
    
    # Categorical: 'Unknown'
    cat_cols = X_df.select_dtypes(include=['object']).columns
    X_df[cat_cols] = X_df[cat_cols].fillna('Unknown')
    
    # 3. Encode categorical variables
    # Using frequency encoding for high cardinality, label encoding for low
    for col in cat_cols:
        if X_df[col].nunique() < 10:
            # One-hot encoding could be better but increases dimensionality too much
            # Using simple label encoding mapping
            labels = X_df[col].astype('category').cat.codes
            X_df[col] = labels
        else:
            # Frequency encoding
            freq_encoding = X_df[col].value_counts(normalize=True)
            X_df[col] = X_df[col].map(freq_encoding)
            
    # Final check for any remaining NaNs (e.g. if median was NaN)
    X_df = X_df.fillna(0)
    
    X = X_df.values
    
    logger.info(f"Final feature shape: {X.shape}")
    logger.info(f"Fraud ratio: {y.mean():.4f}")
    
    return X, y
