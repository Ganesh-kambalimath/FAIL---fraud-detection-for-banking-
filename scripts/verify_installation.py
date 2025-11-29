"""
Verify Installation and System Requirements
Run this script to check if everything is set up correctly
"""

import sys
import os
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_check(name, status, message=""):
    """Print check result"""
    symbol = "✓" if status else "✗"
    status_text = "OK" if status else "FAIL"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    
    print(f"{color}{symbol} {name:.<50} {status_text}{reset}")
    if message:
        print(f"  → {message}")


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    required = (3, 8)
    
    is_valid = version >= required
    message = f"Python {version.major}.{version.minor}.{version.micro}"
    
    if not is_valid:
        message += f" (Required: >= {required[0]}.{required[1]})"
    
    return is_valid, message


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, f"{package_name} installed"
    except ImportError:
        return False, f"{package_name} NOT installed - run: pip install {package_name}"


def check_directory(dir_path):
    """Check if directory exists"""
    path = Path(dir_path)
    exists = path.exists() and path.is_dir()
    
    if not exists:
        return False, f"Directory missing: {dir_path}"
    
    # Count files in directory
    try:
        file_count = len(list(path.rglob('*')))
        return True, f"Found with {file_count} items"
    except:
        return True, "Exists"


def check_file(file_path):
    """Check if file exists"""
    path = Path(file_path)
    exists = path.exists() and path.is_file()
    
    if not exists:
        return False, f"File missing: {file_path}"
    
    # Get file size
    size = path.stat().st_size
    size_kb = size / 1024
    return True, f"Found ({size_kb:.1f} KB)"


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, f"CUDA available: {device_name}"
        else:
            return False, "CUDA not available (CPU only mode)"
    except:
        return False, "Cannot check CUDA (PyTorch not installed)"


def main():
    """Main verification function"""
    print_header("🔍 SECURE FEDERATED LEARNING - INSTALLATION VERIFICATION")
    
    all_checks_passed = True
    
    # Python Version
    print_header("1. Python Environment")
    status, msg = check_python_version()
    print_check("Python Version", status, msg)
    all_checks_passed &= status
    
    # Core Dependencies
    print_header("2. Core Dependencies")
    
    packages = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("PyYAML", "yaml"),
    ]
    
    for pkg_name, import_name in packages:
        status, msg = check_package(pkg_name, import_name)
        print_check(pkg_name, status, msg)
        all_checks_passed &= status
    
    # Optional Dependencies
    print_header("3. Optional Dependencies")
    
    optional_packages = [
        ("tenseal", "tenseal"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("pytest", "pytest"),
    ]
    
    for pkg_name, import_name in optional_packages:
        status, msg = check_package(pkg_name, import_name)
        print_check(f"{pkg_name} (optional)", status, msg)
    
    # Project Structure
    print_header("4. Project Structure")
    
    directories = [
        "src",
        "src/models",
        "src/server",
        "src/client",
        "src/privacy",
        "src/encryption",
        "src/aggregation",
        "src/security",
        "src/utils",
        "configs",
        "tests",
        "examples",
        "docs",
    ]
    
    for directory in directories:
        status, msg = check_directory(directory)
        print_check(directory, status, msg)
        if not status:
            all_checks_passed = False
    
    # Configuration Files
    print_header("5. Configuration Files")
    
    config_files = [
        "configs/server_config.yaml",
        "configs/client_config.yaml",
        "requirements.txt",
        "setup.py",
        "README.md",
    ]
    
    for file_path in config_files:
        status, msg = check_file(file_path)
        print_check(file_path, status, msg)
        if not status:
            all_checks_passed = False
    
    # GPU Support
    print_header("6. GPU Support")
    status, msg = check_cuda()
    print_check("CUDA/GPU", status, msg)
    
    # Data Directories
    print_header("7. Data Directories")
    
    data_dirs = [
        "data",
        "logs",
        "models",
    ]
    
    for directory in data_dirs:
        status, msg = check_directory(directory)
        print_check(directory, status, msg)
        if not status:
            print(f"  → Run: python scripts\\init_config.py")
    
    # Summary
    print_header("📊 VERIFICATION SUMMARY")
    
    if all_checks_passed:
        print("\n✅ ALL CRITICAL CHECKS PASSED!")
        print("\nYour installation is complete and ready to use.")
        print("\nNext steps:")
        print("  1. Run demo: python run_simulation.py")
        print("  2. Run tests: pytest tests\\ -v")
        print("  3. See QUICKSTART.md for more information")
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\nPlease fix the issues above:")
        print("  1. Install missing dependencies: pip install -r requirements.txt")
        print("  2. Initialize project: python scripts\\init_config.py")
        print("  3. Run this verification again: python scripts\\verify_installation.py")
    
    print("\n" + "=" * 70)
    
    # Additional Information
    print("\n📋 System Information:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")
    print(f"  Working Directory: {os.getcwd()}")
    
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
    except:
        pass
    
    try:
        import numpy
        print(f"  NumPy: {numpy.__version__}")
    except:
        pass
    
    print("\n" + "=" * 70)
    
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
