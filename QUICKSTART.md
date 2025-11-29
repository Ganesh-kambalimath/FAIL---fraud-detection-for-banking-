# Quick Start Guide

## Installation

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Demo

### 1. Single Machine Simulation

```powershell
# Run complete demo
python examples\demo_federated_training.py
```

### 2. Multi-Process Simulation

**Terminal 1 - Start Server:**
```powershell
python src\server\main.py --config configs\server_config.yaml
```

**Terminal 2, 3, 4 - Start Clients:**
```powershell
python src\client\main.py --client-id bank_a --config configs\client_config.yaml
python src\client\main.py --client-id bank_b --config configs\client_config.yaml
python src\client\main.py --client-id bank_c --config configs\client_config.yaml
```

## Running Tests

```powershell
# Run all tests
pytest tests\ -v

# Run specific test file
pytest tests\test_privacy.py -v

# Run with coverage
pytest --cov=src tests\
```

## Key Features Demonstrated

### 1. Differential Privacy
- Gradient clipping
- Gaussian noise addition
- Privacy budget tracking
- Epsilon: 0.5, Delta: 1e-5

### 2. Homomorphic Encryption
- CKKS scheme implementation
- Encrypted model aggregation
- Secure computation

### 3. Byzantine Defense
- Multi-Krum filtering
- Outlier detection
- Malicious client protection

### 4. Fraud Detection
- 95%+ accuracy target
- Real-time inference
- Financial impact analysis

## Configuration

Edit `configs/server_config.yaml` and `configs/client_config.yaml` to customize:
- Privacy parameters (epsilon, delta)
- Model architecture
- Training hyperparameters
- Security settings

## Expected Output

```
[Step 1] Generating synthetic fraud data...
Generated 50000 samples (1000 fraud, 49000 normal)

[Step 2] Preprocessing data...
Train: 28000, Val: 6000, Test: 10000

[Step 3] Creating 3 federated client datasets...
Client 0: 9333 samples, fraud ratio: 0.0189
Client 1: 9333 samples, fraud ratio: 0.0201
Client 2: 9334 samples, fraud ratio: 0.0198

[Step 8] Starting 10 rounds of federated training...

Round 1/10
================================================================================
Client 0 training...
Client 1 training...
Client 2 training...
Aggregating 3 client updates...

FINAL EVALUATION
================================================================================
Test Set Evaluation

Performance Metrics:
  Accuracy:    0.9812
  Precision:   0.8954
  Recall:      0.9123
  F1 Score:    0.9038
  AUC-ROC:     0.9891

[Privacy Budget]
Epsilon spent: 0.4123 / 0.5000
Delta: 1.00e-05
Privacy preserved: ✓
```

## Troubleshooting

### Issue: Import errors
**Solution:** Make sure you're in the project root and virtual environment is activated

### Issue: CUDA not available
**Solution:** Set `gpu: false` in config files or install CUDA-enabled PyTorch

### Issue: Memory error
**Solution:** Reduce batch_size or model size in configs

## Next Steps

1. **Customize Data:** Replace synthetic data with real fraud detection dataset
2. **Tune Privacy:** Adjust epsilon/delta for your privacy requirements
3. **Scale Up:** Increase number of clients and training rounds
4. **Deploy:** Use Docker containers for production deployment

## Documentation

- [Architecture Guide](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Privacy Mechanisms](docs/privacy.md)
- [Security Analysis](docs/security.md)
