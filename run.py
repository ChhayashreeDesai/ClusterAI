#!/usr/bin/env python3
"""
ClusterAI - Customer Segmentation Web Application
Startup script for development and deployment
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Ensure Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def install_requirements():
    """Install required packages"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def check_dependencies():
    """Check if all required packages are available"""
    required_packages = [
        ('flask', 'flask'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'), 
        ('plotly', 'plotly'),
        ('kneed', 'kneed'),
        ('openpyxl', 'openpyxl')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("🔧 Run with --install to install missing packages")
        return False
    
    print("✅ All dependencies satisfied")
    return True

def start_server(host='127.0.0.1', port=5000, debug=True):
    """Start the Flask development server"""
    print(f"🚀 Starting ClusterAI server...")
    print(f"🌐 Server will be available at: http://{host}:{port}")
    print("📊 Upload your CSV/Excel files to start clustering analysis")
    print("⭐ Try the demo with sample data to see ClusterAI in action")
    print("\n" + "="*60)
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development' if debug else 'production'
    
    # Import and run the app
    try:
        from app import app
        app.run(host=host, port=port, debug=debug)
    except ImportError as e:
        print(f"❌ Failed to import app: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    print("🤖 ClusterAI - Advanced Customer Segmentation Platform")
    print("="*60)
    
    # Parse command line arguments
    install_deps = '--install' in sys.argv
    production = '--production' in sys.argv
    
    # Check Python version
    check_python_version()
    
    # Install dependencies if requested
    if install_deps:
        if not install_requirements():
            sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        if not install_deps:
            print("\n💡 Tip: Run 'python run.py --install' to install dependencies")
        sys.exit(1)
    
    # Start the server
    host = '0.0.0.0' if production else '127.0.0.1'
    debug = not production
    
    try:
        start_server(host=host, debug=debug)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Startup error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()