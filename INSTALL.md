# Installation and Setup Instructions

Run the initialization script to set up the project:

```powershell
python scripts\init_config.py
```

This will:
1. ✅ Create all necessary directories
2. ✅ Generate sample fraud detection data
3. ✅ Set up configuration files
4. ✅ Create .env for environment variables

## Manual Installation

If you prefer manual setup:

### 1. Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

```powershell
python -c "import torch; import numpy; import sklearn; print('✓ All dependencies installed')"
```

### 4. Run Tests

```powershell
pytest tests\ -v
```

### 5. Run Demo

```powershell
python examples\demo_federated_training.py
```

## Troubleshooting

### PyTorch Installation Issues

If you encounter issues with PyTorch:

**For CPU-only:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**For CUDA 11.8:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### TenSEAL Installation Issues

TenSEAL (for homomorphic encryption) may require additional steps:

```powershell
pip install tenseal
```

If this fails, you can disable homomorphic encryption in the config files by setting:
```yaml
homomorphic_encryption:
  enabled: false
```

### Memory Issues

If you run into memory issues:

1. Reduce batch size in configs:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

2. Reduce model size:
```yaml
model:
  hidden_layers: [64, 32]  # Reduce from [128, 64, 32]
```

### Import Errors

Make sure you're running from the project root directory:

```powershell
cd "e:\Secure ai"
python examples\demo_federated_training.py
```

## Docker Installation

Alternatively, use Docker:

```powershell
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Next Steps

After successful installation:

1. Read [QUICKSTART.md](QUICKSTART.md) for usage examples
2. Review [docs/architecture.md](docs/architecture.md) for system design
3. Customize configs in `configs/` directory
4. Run your first federated training session!

## Support

If you encounter issues:
- Check [GitHub Issues](https://github.com/your-org/secure-federated-fraud-detection/issues)
- Review troubleshooting section above
- Contact the development team
