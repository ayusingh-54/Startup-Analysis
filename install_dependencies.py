import subprocess
import sys

def install_requirements():
    """Install required packages"""
    packages = [
        'streamlit>=1.28.0',
        'pandas>=1.5.0',
        'plotly>=5.15.0',
        'numpy>=1.24.0'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    print("\n🎉 All dependencies installed!")
    print("You can now run: streamlit run app.py")

if __name__ == "__main__":
    install_requirements()
