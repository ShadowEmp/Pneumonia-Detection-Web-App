"""
Quick Start Script for Pneumonia Detection System
Automated setup and verification
"""
import os
import sys
import subprocess


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = [
        'tensorflow',
        'keras',
        'opencv-python',
        'flask',
        'numpy',
        'matplotlib',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed")
    return True


def check_directories():
    """Check if required directories exist"""
    print_header("Checking Directory Structure")
    
    required_dirs = ['data', 'models', 'results', 'uploads']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ exists")
        else:
            print(f"⚠️  {dir_name}/ does not exist - creating...")
            os.makedirs(dir_name, exist_ok=True)
            print(f"✅ Created {dir_name}/")
    
    return True


def check_dataset():
    """Check if dataset is available"""
    print_header("Checking Dataset")
    
    data_dirs = [
        'data/train/Normal',
        'data/train/Pneumonia',
        'data/test/Normal',
        'data/test/Pneumonia'
    ]
    
    dataset_exists = all(os.path.exists(d) for d in data_dirs)
    
    if dataset_exists:
        # Count images
        train_normal = len([f for f in os.listdir('data/train/Normal') 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        train_pneumonia = len([f for f in os.listdir('data/train/Pneumonia') 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"✅ Dataset found")
        print(f"   Training - Normal: {train_normal} images")
        print(f"   Training - Pneumonia: {train_pneumonia} images")
        return True
    else:
        print("❌ Dataset not found")
        print("\nTo set up the dataset:")
        print("1. Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("2. Extract to data/ directory")
        print("3. Organize as:")
        print("   data/train/Normal/")
        print("   data/train/Pneumonia/")
        print("   data/test/Normal/")
        print("   data/test/Pneumonia/")
        return False


def check_model():
    """Check if trained model exists"""
    print_header("Checking Trained Model")
    
    model_paths = ['models/pneumonia_model.h5', 'models/best_pneumonia_model.h5']
    
    model_exists = any(os.path.exists(p) for p in model_paths)
    
    if model_exists:
        for path in model_paths:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"✅ Model found: {path} ({size_mb:.2f} MB)")
        return True
    else:
        print("⚠️  No trained model found")
        print("\nTo train the model:")
        print("  python train.py")
        return False


def check_frontend():
    """Check if frontend is set up"""
    print_header("Checking Frontend")
    
    if not os.path.exists('frontend'):
        print("❌ Frontend directory not found")
        return False
    
    if not os.path.exists('frontend/node_modules'):
        print("⚠️  Frontend dependencies not installed")
        print("\nTo install frontend dependencies:")
        print("  cd frontend")
        print("  npm install")
        return False
    
    print("✅ Frontend is set up")
    return True


def print_next_steps(checks):
    """Print next steps based on check results"""
    print_header("Summary and Next Steps")
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("🎉 All checks passed! Your system is ready to use.\n")
        print("To start the application:")
        print("\n1. Start the backend (Terminal 1):")
        print("   python app.py")
        print("\n2. Start the frontend (Terminal 2):")
        print("   cd frontend")
        print("   npm run dev")
        print("\n3. Open your browser:")
        print("   http://localhost:3000")
    else:
        print("⚠️  Some checks failed. Please complete the following:\n")
        
        if not checks['dependencies']:
            print("1. Install Python dependencies:")
            print("   pip install -r requirements.txt\n")
        
        if not checks['dataset']:
            print("2. Download and set up the dataset")
            print("   See SETUP_GUIDE.md for instructions\n")
        
        if not checks['model']:
            print("3. Train the model:")
            print("   python train.py\n")
        
        if not checks['frontend']:
            print("4. Set up the frontend:")
            print("   cd frontend")
            print("   npm install\n")
    
    print("\nFor detailed instructions, see:")
    print("  - README.md")
    print("  - SETUP_GUIDE.md")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("  PNEUMONIA DETECTION SYSTEM - QUICK START")
    print("="*70)
    
    checks = {
        'python': check_python_version(),
        'dependencies': check_dependencies(),
        'directories': check_directories(),
        'dataset': check_dataset(),
        'model': check_model(),
        'frontend': check_frontend()
    }
    
    print_next_steps(checks)
    
    print("\n" + "="*70)
    print("  Quick Start Complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
