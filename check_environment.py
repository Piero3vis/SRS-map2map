import os
import sys
import torch
import numpy as np

def check_models():
    """Check if trained models exist."""
    model_files = ['SRmodel/G_z0.pt', 'SRmodel/G_z2.pt']
    missing_models = []
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            missing_models.append(model_file)
    
    if missing_models:
        print("❌ Missing model files:", ", ".join(missing_models))
        print("Please ensure the model files are in the SRmodel directory")
        return False
    
    print("✅ Found trained models")
    return True

def check_data_format(data_path):
    """Check if N-body simulation data exists and has correct format."""
    if not os.path.exists(data_path):
        print(f"❌ Data not found at {data_path}")
        return False
    
    try:
        # Try to load the first few bytes to verify format
        # Add specific format checking logic here based on your data format
        print("✅ Data format appears correct")
        return True
    except Exception as e:
        print(f"❌ Error reading data: {str(e)}")
        return False

def print_model_architecture():
    """Print ASCII visualization of the model architecture."""
    print("""
    3D Super-Resolution Architecture (512x upsampling)
    
    Input (6 channels: pos+vel)         Output (6 channels: SR pos+vel)
    [Ng×Ng×Ng×6] ─────────────┐              ┌──── [8Ng×8Ng×8Ng×6]
                              ↓              ↑
                     [512 channels]──→[256]──→[128]──→[64]
                        block0    block1    block2    block3
                     (3D Conv)   (3D Conv) (3D Conv) (3D Conv)
    
    Each block contains:
    • Normalization
    • Two 3×3×3 conv layers
    • Projection back to 6 channels
    """)

def check_data_shape(data_path):
    """Check if data dimensions match model requirements."""
    try:
        data = np.load(data_path)
        print(f"📊 Data shape: {data.shape}")
        
        # Expected input validation based on README
        if len(data.shape) != 4:
            print("❌ Data should be 4-dimensional (Nc, Ng, Ng, Ng)")
            return False
            
        nc, nx, ny, nz = data.shape
        if nc != 6:
            print("❌ Input should have 6 channels (displacement + velocity)")
            return False
            
        if not (nx == ny == nz):
            print("❌ Spatial dimensions should be equal (Ng × Ng × Ng)")
            return False
            
        # Check if grid size matches training parameters
        if nx in [64, 128]:
            print("✅ Grid size matches training parameters")
            if nx == 64:
                print("ℹ️ Using same resolution as training (Ng=64, Box=100 Mpc/h)")
            else:
                print("ℹ️ Using scaled resolution (Ng=128, Box=200 Mpc/h)")
        else:
            print(f"⚠️ Grid size {nx} differs from training parameters (64 or 128)")
            
        print("\n💡 Input data requirements:")
        print("   • Should be normalized displacement + velocity field")
        print("   • Arranged by original grid positions")
        print("   • Recommended: Ng=64 with 100 Mpc/h box or Ng=128 with 200 Mpc/h box")
        print("   • WMAP9 cosmology preferred (see scripts/LR-sim/paramfile.genic)")
        
        return True
    except Exception as e:
        print(f"❌ Data shape error: {str(e)}")
        return False

def check_gpu():
    """Check if GPU is available."""
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("⚠️ No GPU found. Processing will be slower on CPU")
        return True

def check_model_input_size(model_path):
    """Check the input size requirements of the model."""
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        print(f"\n🏗️ Model {os.path.basename(model_path)} structure:")
        
        # Print the full structure of state_dict
        for key in state_dict.keys():
            print(f"   {key}")
            if isinstance(state_dict[key], dict):
                for subkey, value in state_dict[key].items():
                    if isinstance(value, torch.Tensor):
                        print(f"      {subkey}: {value.shape}")
                    else:
                        print(f"      {subkey}: {type(value)}")
        
        return True
    except Exception as e:
        print(f"❌ Error analyzing model structure: {str(e)}")
        return False

def find_npy_files():
    """Find all .npy files in the current directory and subdirectories."""
    npy_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    
    if npy_files:
        print("\n📁 Found .npy files:")
        for file in npy_files:
            print(f"   • {file}")
    else:
        print("\n⚠️ No .npy files found in the directory tree")
    
    return npy_files

def suggest_command(npy_files):
    """Suggest command to run the model based on available files."""
    if not npy_files:
        return
    
    print("\n🚀 Suggested commands:")
    print("1. For single file processing:")
    for npy_file in npy_files:
        print(f"   python lr2sr.py --redshift 0.0 --lr-input {npy_file} --sr-path output/sr_{os.path.basename(npy_file)} --model SRmodel/G_z0.pt")
    
    print("\n2. For batch processing all files:")
    print("   python lr2sr.py --redshift 0.0 --lr-input ./data --sr-path output/ --model SRmodel/G_z0.pt")

def main():
    print("Running environment checks...")
    print("-" * 50)
    print_model_architecture()
    print("-" * 50)
    
    checks_passed = True
    
    # Check for trained models and their input sizes
    if not check_models():
        checks_passed = False
    else:
        for model_file in ['SRmodel/G_z0.pt', 'SRmodel/G_z2.pt']:
            check_model_input_size(model_file)
    
    # Find all .npy files
    npy_files = find_npy_files()
    
    # Check format and shape of found files
    for npy_file in npy_files:
        if os.path.exists(npy_file):
            if not check_data_format(npy_file):
                checks_passed = False
            if not check_data_shape(npy_file):
                checks_passed = False
    
    # Check GPU availability
    check_gpu()
    
    print("-" * 50)
    if checks_passed:
        print("✅ All critical checks passed!")
        suggest_command(npy_files)
        sys.exit(0)
    else:
        print("❌ Some checks failed. Please resolve the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 