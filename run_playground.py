#!/usr/bin/env python3
"""
Neural Network Playground Launcher
This script checks dependencies and launches the neural network playground.
"""

import sys
import subprocess
import importlib

def check_dependency(module_name, package_name=None):
    """Check if a Python module is available."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        print(f"❌ Missing dependency: {package_name or module_name}")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies. Please install them manually:")
        print("pip install -r requirements.txt")
        return False

def main():
    print("🚀 Neural Network Playground Launcher")
    print("=" * 40)
    
    # Check dependencies
    dependencies = [
        ("PyQt6", "PyQt6"),
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("sklearn", "scikit-learn"),
        ("networkx", "networkx")
    ]
    
    missing_deps = []
    for module, package in dependencies:
        if not check_dependency(module, package):
            missing_deps.append(package)
    
    if missing_deps:
        print(f"\n❌ Missing {len(missing_deps)} dependencies.")
        response = input("Would you like to install them automatically? (y/n): ")
        if response.lower() in ['y', 'yes']:
            if not install_dependencies():
                return
        else:
            print("Please install the missing dependencies manually and try again.")
            return
    else:
        print("✅ All dependencies are available!")
    
    # Launch the playground
    print("\n🎮 Launching Neural Network Playground...")
    try:
        from neural_network_playground import main as launch_playground
        launch_playground()
    except ImportError as e:
        print(f"❌ Error importing playground: {e}")
        print("Make sure neural_network_playground.py is in the same directory.")
    except Exception as e:
        print(f"❌ Error launching playground: {e}")

if __name__ == "__main__":
    main() 