# Project Documentation

## Architecture Overview

### System Components

The Secure Federated Learning Framework consists of the following key components:

#### 1. **Central Aggregation Server**
- Coordinates federated training rounds
- Aggregates encrypted model updates
- Manages privacy budget
- Implements Byzantine defense mechanisms
- Provides secure communication protocols

#### 2. **Client Nodes (Financial Institutions)**
- Local model training on private data
- Differential privacy implementation
- Secure model update submission
- Data preprocessing and validation

#### 3. **Privacy Layer**
- **Differential Privacy**: ε-DP guarantees with Gaussian noise
- **Homomorphic Encryption**: CKKS scheme for encrypted computation
- **Secure Aggregation**: Multi-party computation protocols
- **Privacy Accounting**: Renyi Differential Privacy (RDP) tracking

#### 4. **Security Layer**
- **Byzantine Defense**: Multi-Krum, Trimmed Mean, Median aggregation
- **Attack Detection**: Gradient and model poisoning detection
- **Authentication**: Token-based client authentication
- **Encryption**: SSL/TLS for communication

#### 5. **Model Layer**
- **Fraud Detection Networks**: Deep neural networks, LSTM, Attention models
- **Loss Functions**: Binary cross-entropy for fraud classification
- **Optimization**: Adam, SGD, AdamW optimizers

### Data Flow

```
┌────────────────────────────────────────────────────────────┐
│                      Training Round Flow                    │
└────────────────────────────────────────────────────────────┘

1. Server broadcasts global model parameters
   ↓
2. Clients receive global model
   ↓
3. Local training on private data
   - Forward pass
   - Compute loss
   - Backward pass
   - Gradient clipping (DP)
   ↓
4. Apply differential privacy
   - Clip gradients
   - Add Gaussian noise
   ↓
5. Encrypt model updates (optional HE)
   ↓
6. Submit encrypted updates to server
   ↓
7. Server collects updates
   ↓
8. Byzantine defense filtering
   ↓
9. Secure aggregation
   - Weighted average
   - Multi-Krum
   - Trimmed mean
   ↓
10. Update global model
    ↓
11. Track privacy budget
    ↓
12. Repeat for next round
```

### Privacy Guarantees

#### Differential Privacy (DP)

The system implements (ε, δ)-differential privacy:

- **Epsilon (ε)**: Privacy budget parameter
  - Lower ε = stronger privacy
  - Typical values: 0.1 - 10.0
  - Our default: 0.5

- **Delta (δ)**: Failure probability
  - Typically: 1/n² where n = dataset size
  - Our default: 1e-5

**DP-SGD Algorithm:**
```
For each training round:
  1. Clip gradients: g_clip = g / max(1, ||g||₂ / C)
  2. Add noise: g_private = g_clip + N(0, σ²C²I)
  3. Update model: θ_t+1 = θ_t - η * g_private
```

#### Homomorphic Encryption

Uses CKKS (Cheon-Kim-Kim-Song) scheme:

- **Encryption**: E(x)
- **Addition**: E(x + y) = E(x) + E(y)
- **Multiplication**: E(x * y) = E(x) * E(y)
- **Scalar multiplication**: E(α * x) = α * E(x)

**Aggregation on encrypted data:**
```
Encrypted_Aggregate = Σ(weight_i * E(update_i))
Decrypted_Aggregate = D(Encrypted_Aggregate)
```

### Security Mechanisms

#### Byzantine Defense

**Multi-Krum Algorithm:**
```
1. Compute pairwise distances between all client updates
2. For each client i, compute score:
   score_i = Σ(distances to n-f-2 nearest neighbors)
3. Select m clients with lowest scores
4. Average selected updates
```

**Trimmed Mean:**
```
1. Stack all client updates
2. Sort along client dimension
3. Remove top β% and bottom β% (outliers)
4. Compute mean of remaining updates
```

#### Attack Detection

1. **Model Poisoning**: Detect updates with L2 distance > threshold
2. **Gradient Explosion**: Detect gradients with ||g|| > max_norm
3. **Membership Inference**: Privacy accounting prevents information leakage
4. **Model Inversion**: Differential privacy protects individual records

### Performance Optimization

#### Communication Efficiency

- **Gradient Compression**: Quantization, sparsification
- **Update Frequency**: Configurable local epochs
- **Batch Optimization**: Efficient batching strategies

#### Computational Efficiency

- **GPU Acceleration**: CUDA support for training
- **Parallel Processing**: Multi-threaded data loading
- **Model Compression**: Pruning, quantization (optional)

### Scalability

The system is designed to scale:

- **Horizontal**: Support for 10-1000+ clients
- **Vertical**: Large models (millions of parameters)
- **Geographic**: Distributed across regions

**Scaling Considerations:**
- Network bandwidth
- Synchronization overhead
- Stragglers (slow clients)
- Dynamic client participation

### Compliance

The framework ensures compliance with:

- **GDPR**: Data minimization, purpose limitation, privacy by design
- **CCPA**: Consumer privacy rights, data protection
- **HIPAA**: Healthcare data protection (if applicable)
- **PCI DSS**: Payment card industry standards

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Production Deployment                  │
└─────────────────────────────────────────────────────────┘

Load Balancer
     │
     ├─── Server Instance 1 (Primary)
     ├─── Server Instance 2 (Backup)
     └─── Server Instance 3 (Backup)
          │
          ├─── PostgreSQL (State)
          ├─── Redis (Cache)
          └─── TensorBoard (Monitoring)

Client Nodes (Behind Firewalls):
     ├─── Bank A (Region 1)
     ├─── Bank B (Region 2)
     └─── Bank C (Region 3)
```

### Monitoring & Logging

- **Metrics**: Accuracy, precision, recall, F1, AUC-ROC
- **Privacy**: Epsilon spent, delta
- **Performance**: Training time, communication overhead
- **Security**: Attack attempts, filtered updates
- **System**: CPU, memory, network usage

### API Endpoints

**Server API:**
- `POST /register` - Register new client
- `POST /start_round` - Start training round
- `POST /submit_update` - Submit client update
- `GET /get_model` - Get current global model
- `GET /status` - Get server status

**Client API:**
- `GET /health` - Health check
- `POST /train` - Trigger local training
- `GET /metrics` - Get local metrics

## Further Reading

- [Privacy Mechanisms](privacy.md)
- [Security Analysis](security.md)
- [API Reference](api_reference.md)
- [Deployment Guide](deployment.md)
