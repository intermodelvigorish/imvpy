#!/usr/bin/env python3
"""
Dependency checker for SHAP-IMV Adult Income analysis
Run this before the main script to ensure all dependencies are available
"""

import sys

def check_dependencies():
    """Check if all required packages are installed"""
    
    print("="*60)
    print("Checking Dependencies for SHAP-IMV Analysis")
    print("="*60)
    
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
    }
    
    optional = {
        'ucimlrepo': 'ucimlrepo',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
    }
    
    # Check required packages
    print("\n✓ Required Packages:")
    missing_required = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing_required.append(package)
    
    # Check optional packages
    print("\n✓ Optional Packages:")
    missing_optional = []
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ⚠ {package} - not installed (optional)")
            missing_optional.append(package)
    
    # Check IMV package
    print("\n✓ IMV Package:")
    try:
        sys.path.insert(0, '../..')
        from imv import BinaryIMV
        print("  ✓ imv.BinaryIMV")
    except ImportError as e:
        print(f"  ✗ IMV package - ERROR: {e}")
        missing_required.append('imv')
    
    # Summary
    print("\n" + "="*60)
    if missing_required:
        print("⚠ MISSING REQUIRED PACKAGES:")
        for pkg in missing_required:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing_required)}")
        return False
    elif missing_optional:
        print("✓ All required packages installed!")
        print("\n⚠ Optional packages not installed:")
        for pkg in missing_optional:
            print(f"  - {pkg}")
        print("\nTo enable all features:")
        print(f"  pip install {' '.join(missing_optional)}")
        print("\nYou can run the analysis but some models may be skipped.")
        return True
    else:
        print("✓ All packages installed!")
        print("✓ Ready to run full analysis")
        return True

if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)
