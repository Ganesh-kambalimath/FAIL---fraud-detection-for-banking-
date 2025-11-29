# 🎯 GETTING STARTED - Complete Setup Guide

This guide will walk you through setting up and running the Secure Federated Learning Framework in **5 minutes**.

---

## ⚡ Quick Setup (Recommended)

### Step 1: Clone/Navigate to Project
```powershell
cd "e:\Secure ai"
```

### Step 2: Run Initialization Script
```powershell
python scripts\init_config.py
```

**What this does:**
- ✅ Creates all necessary directories
- ✅ Generates synthetic fraud detection data for 3 banks
- ✅ Sets up configuration files
- ✅ Creates .env for secrets
- ✅ Generates .gitignore

**Expected output:**
```
============================================================
Secure Federated Fraud Detection - Project Initialization
============================================================

[1/4] Creating directory structure...
Created directory: data/client_data
Created directory: data/bank_a
Created directory: data/bank_b
...

[4/4] Generating sample data...
Generated 30000 samples (600 fraud, 29400 normal)
Saved data for bank_a: 10000 samples
Saved data for bank_b: 10000 samples
Saved data for bank_c: 10000 samples

============================================================
Initialization complete!
============================================================
```

### Step 3: Install Dependencies
```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**This will install:**
- PyTorch (deep learning)
- TensorFlow (federated learning)
- Cryptographic libraries
- Data processing tools
- Testing frameworks

**Note:** Installation may take 5-10 minutes depending on your internet connection.

### Step 4: Run Your First Simulation! 🎉
```powershell
python run_simulation.py
```

**What happens:**
1. Creates 3 virtual banks with private fraud data
2. Initializes global fraud detection model
3. Runs 10 rounds of federated training
4. Applies differential privacy (ε=0.5)
5. Uses Byzantine defense (Multi-Krum)
6. Evaluates on test set
7. Shows privacy budget spent

**Expected output:**
```
================================================================================
FEDERATED FRAUD DETECTION SYSTEM - SIMULATION
================================================================================

[Step 1] Generating synthetic fraud data...
Generated 50000 samples (1000 fraud, 49000 normal)

[Step 2] Preprocessing data...
Train: 28000, Val: 6000, Test: 10000

[Step 3] Creating 3 federated client datasets...
Client 0: 9333 samples, fraud ratio: 0.0189
Client 1: 9333 samples, fraud ratio: 0.0201
Client 2: 9334 samples, fraud ratio: 0.0198

[Step 8] Starting 10 rounds of federated training...

================================================================================
Round 1/10
================================================================================
Client 0 training...
Client 1 training...
Client 2 training...
Aggregating 3 client updates...

...

================================================================================
FINAL EVALUATION
================================================================================

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

================================================================================
SIMULATION COMPLETE
================================================================================
```

**🎉 Congratulations! You've just run your first secure federated learning simulation!**

---

## 🧪 Verify Installation

### Run Tests
```powershell
pytest tests\ -v
```

**Expected output:**
```
tests/test_models.py::TestFraudDetectionNN::test_initialization PASSED
tests/test_models.py::TestFraudDetectionNN::test_forward_pass PASSED
tests/test_privacy.py::TestDifferentialPrivacy::test_clip_gradients PASSED
...

==================== 20 passed in 5.43s ====================
```

---

## 🎮 Try Different Configurations

### More Clients
```powershell
python run_simulation.py --num-clients 5 --num-rounds 15
```

### Without Privacy (to see the difference)
```powershell
python run_simulation.py --no-dp
```

### Debug Mode
```powershell
python run_simulation.py --log-level DEBUG
```

### Full Configuration
```powershell
python run_simulation.py --num-clients 5 --num-rounds 20 --local-epochs 10
```

---

## 🐳 Docker Setup (Alternative)

If you prefer Docker:

### Build and Run
```powershell
docker-compose up --build
```

This starts:
- PostgreSQL database
- Redis cache
- Federated learning server
- 3 client nodes (Bank A, B, C)

### View Logs
```powershell
docker-compose logs -f fl_server
docker-compose logs -f fl_client_a
```

### Stop
```powershell
docker-compose down
```

---

## 🔧 Advanced Setup

### Multi-Process Mode

For a more realistic simulation with separate processes:

**Terminal 1 - Start Server:**
```powershell
python src\server\main.py --config configs\server_config.yaml
```

**Terminal 2 - Start Bank A:**
```powershell
python src\client\main.py --client-id bank_a --config configs\client_config.yaml
```

**Terminal 3 - Start Bank B:**
```powershell
python src\client\main.py --client-id bank_b --config configs\client_config.yaml
```

**Terminal 4 - Start Bank C:**
```powershell
python src\client\main.py --client-id bank_c --config configs\client_config.yaml
```

---

## 📝 Customize Configuration

### Edit Server Config
Open `configs\server_config.yaml`:

```yaml
# Change privacy settings
privacy:
  differential_privacy:
    epsilon: 1.0  # Less privacy, more accuracy
    delta: 1e-5

# Change model
model:
  architecture: "AttentionFraudDetector"  # Use attention model
  hidden_layers: [256, 128, 64]  # Bigger model

# Change training
training:
  local_epochs: 10  # More local training
  batch_size: 64  # Bigger batches
```

### Edit Client Config
Open `configs\client_config.yaml`:

```yaml
# Change client settings
client:
  id: "my_bank"
  name: "My Financial Institution"

# Change resources
resources:
  gpu: true  # Enable GPU
  max_memory: 16384  # 16GB RAM
```

---

## 📊 Understanding the Output

### Training Progress
```
Round 1/10
================================================================================
Client 0 training...  ← Each client trains locally
Client 1 training...
Client 2 training...
Aggregating 3 client updates...  ← Server combines updates securely
```

### Performance Metrics
```
Performance Metrics:
  Accuracy:    0.9812  ← 98.12% accuracy
  Precision:   0.8954  ← 89.54% of fraud predictions are correct
  Recall:      0.9123  ← 91.23% of actual fraud is caught
  F1 Score:    0.9038  ← Harmonic mean of precision & recall
  AUC-ROC:     0.9891  ← Excellent discrimination ability
```

### Privacy Budget
```
[Privacy Budget]
Epsilon spent: 0.4123 / 0.5000  ← Used 82% of privacy budget
Delta: 1.00e-05  ← Failure probability
Privacy preserved: ✓  ← Still within budget!
```

### Financial Impact
```
[Financial Impact Analysis]
Prevented Fraud: $182,300.00  ← Fraud caught
Missed Fraud: $17,400.00  ← Fraud missed
False Alarm Cost: $4,560.00  ← Investigation costs
Net Benefit: $160,340.00  ← Total savings
ROI: 40.00x  ← Return on investment
```

---

## 🎓 Learn More

### Documentation
- **Architecture**: `docs\architecture.md` - System design
- **Quick Start**: `QUICKSTART.md` - Fast overview
- **Project Guide**: `PROJECT_GUIDE.md` - Complete guide
- **API Reference**: In code docstrings

### Examples
- **Demo Script**: `examples\demo_federated_training.py`
- **Main Runner**: `run_simulation.py`

### Code Structure
- **Models**: `src\models\` - Neural networks
- **Privacy**: `src\privacy\` - DP mechanisms
- **Security**: `src\security\` - Byzantine defense
- **Aggregation**: `src\aggregation\` - Federated averaging

---

## ❓ Troubleshooting

### Issue: "Module not found"
**Solution:**
```powershell
pip install -r requirements.txt
```

### Issue: "No module named 'torch'"
**Solution:**
```powershell
pip install torch torchvision
```

### Issue: "CUDA not available"
**Solution:** Either:
1. Install CUDA-enabled PyTorch, OR
2. Set `gpu: false` in configs (uses CPU)

### Issue: "Out of memory"
**Solution:** Reduce batch size in configs:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Issue: "Permission denied"
**Solution:** Run PowerShell as Administrator or adjust folder permissions

### Issue: Tests failing
**Solution:**
```powershell
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Run individual test
pytest tests\test_models.py -v
```

---

## 🚀 Next Steps

Now that you have it running:

1. **Explore the code** - Check out `src/` directory
2. **Modify configs** - Try different settings
3. **Run tests** - Verify functionality
4. **Read docs** - Understand the architecture
5. **Customize models** - Add your own neural networks
6. **Use real data** - Replace synthetic data with actual fraud data
7. **Deploy** - Use Docker for production

---

## 📞 Get Help

- **Documentation**: Check `docs/` folder
- **Examples**: See `examples/` folder
- **Issues**: File a GitHub issue
- **Discussions**: Join community discussions

---

## ✅ Checklist

Before you start developing:

- [ ] Project initialized (`python scripts\init_config.py`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Tests passing (`pytest tests\ -v`)
- [ ] Demo running (`python run_simulation.py`)
- [ ] Configurations reviewed (`configs/*.yaml`)
- [ ] Documentation read (`README.md`, `QUICKSTART.md`)

---

## 🎉 You're Ready!

You now have a fully functional secure federated learning framework for fraud detection!

**What you can do:**
- ✅ Train fraud detection models across multiple institutions
- ✅ Preserve privacy with differential privacy (ε-DP)
- ✅ Secure aggregation with homomorphic encryption
- ✅ Defend against malicious clients (Byzantine defense)
- ✅ Comply with regulations (GDPR, CCPA, HIPAA)
- ✅ Deploy at scale with Docker

**Happy federated learning! 🚀**

---

For detailed information, see:
- [PROJECT_GUIDE.md](PROJECT_GUIDE.md) - Complete project guide
- [QUICKSTART.md](QUICKSTART.md) - Quick reference
- [docs/architecture.md](docs/architecture.md) - Technical details
