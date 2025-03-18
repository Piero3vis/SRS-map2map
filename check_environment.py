import os
import sys
import importlib.util
import torch
import numpy as np

def check_script_exists(script_name):
    """Check if a Python script exists."""
    if os.path.exists(script_name):
        print(f"✅ Found {script_name}")
        return True
    else:
        print(f"❌ Missing {script_name}")
        return False

def check_script_imports(script_name, required_modules):
    """Check if a script can import its required modules."""
    if not os.path.exists(script_name):
        return False
    
    missing_modules = []
    for module in required_modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"⚠️ {script_name} may have issues: missing modules {', '.join(missing_modules)}")
        return False
    else:
        print(f"✅ {script_name} dependencies satisfied")
        return True

def check_models():
    """Check if trained models exist."""
    model_files = [
        'SRmodel/G_z0.pt', 
        'SRmodel/G_z2.pt',
        'SRmodel/G_z0_modified_2x.pt',
        'SRmodel/G_z0_modified_4x.pt'
    ]
    missing_models = []
    found_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            found_models.append(model_file)
        else:
            missing_models.append(model_file)
    
    if missing_models:
        print("⚠️ Some model files not found:", ", ".join(missing_models))
        if found_models:
            print("✅ Found models:", ", ".join(found_models))
        else:
            print("❌ No model files found in the SRmodel directory")
            print("Please ensure at least one model file is available")
            return False
    else:
        print("✅ Found all model files")
    
    return True

def check_data_format(data_path):
    """Check if N-body simulation data exists and has correct format."""
    if not os.path.exists(data_path):
        print(f"❌ Data not found at {data_path}")
        return False
    
    try:
        # Try to load the first few bytes to verify format
        if data_path.endswith('.npy'):
            data = np.load(data_path)
            print(f"✅ Data format appears correct: {data_path}")
            return True
        elif os.path.isdir(data_path) and os.path.exists(os.path.join(data_path, '000')):
            print(f"✅ BigFile format detected: {data_path}")
            return True
        else:
            print(f"⚠️ Unknown data format: {data_path}")
            return False
    except Exception as e:
        print(f"❌ Error reading data: {str(e)}")
        return False

def check_workflow_scripts():
    """Check if all required workflow scripts exist and have dependencies."""
    scripts = {
        "preproc.py": ["numpy", "argparse", "os"],
        "lr2sr.py": ["numpy", "torch", "argparse", "os"],
        "down_sample.py": ["numpy", "argparse", "os", "random"],
        "field2bigfile.py": ["numpy", "argparse", "os"],
        "generator.py": ["torch", "argparse", "os"],
        "visualize_sr.py": ["numpy", "matplotlib", "argparse", "os"],
        "visualize_lr.py": ["numpy", "matplotlib", "argparse", "os"],
        "visualize_hr.py": ["numpy", "matplotlib", "argparse", "os"]
    }
    
    all_scripts_found = True
    all_dependencies_met = True
    
    print("\n📋 Checking workflow scripts:")
    for script, dependencies in scripts.items():
        script_found = check_script_exists(script)
        all_scripts_found &= script_found
        
        if script_found:
            deps_met = check_script_imports(script, dependencies)
            all_dependencies_met &= deps_met
    
    return all_scripts_found and all_dependencies_met

def check_down_sample_functionality():
    """Check if down_sample.py has random sampling capability."""
    if not os.path.exists("down_sample.py"):
        return False
    
    try:
        with open("down_sample.py", "r") as f:
            content = f.read()
            
        if "random" in content and "sample" in content:
            print("✅ down_sample.py appears to have random sampling capability")
            return True
        else:
            print("⚠️ down_sample.py may not have random sampling capability")
            return False
    except Exception as e:
        print(f"❌ Error checking down_sample.py: {str(e)}")
        return False

def check_field2bigfile_functionality():
    """Check if field2bigfile.py has required functionality."""
    if not os.path.exists("field2bigfile.py"):
        return False
    
    try:
        with open("field2bigfile.py", "r") as f:
            content = f.read()
            
        if "BigFile" in content or "write" in content:
            print("✅ field2bigfile.py appears to have BigFile writing capability")
            return True
        else:
            print("⚠️ field2bigfile.py may not have proper BigFile writing capability")
            return False
    except Exception as e:
        print(f"❌ Error checking field2bigfile.py: {str(e)}")
        return False

def check_visualization_scripts():
    """Check if visualization scripts have plotting capability."""
    vis_scripts = ["visualize_sr.py", "visualize_lr.py", "visualize_hr.py"]
    all_functional = True
    
    for script in vis_scripts:
        if not os.path.exists(script):
            all_functional = False
            continue
            
        try:
            with open(script, "r") as f:
                content = f.read()
                
            if "matplotlib" in content and "plot" in content:
                print(f"✅ {script} appears to have plotting capability")
            else:
                print(f"⚠️ {script} may not have proper plotting capability")
                all_functional = False
        except Exception as e:
            print(f"❌ Error checking {script}: {str(e)}")
            all_functional = False
    
    return all_functional

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
        
        # Count total parameters
        total_params = 0
        for key in state_dict.keys():
            print(f"   {key}")
            if isinstance(state_dict[key], dict):
                for subkey, value in state_dict[key].items():
                    if isinstance(value, torch.Tensor):
                        params = value.numel()
                        total_params += params
                        print(f"      {subkey}: {value.shape} ({params:,} parameters)")
                    else:
                        print(f"      {subkey}: {type(value)}")
        
        print(f"\n📊 Total parameters: {total_params:,}")
        return True
    except Exception as e:
        print(f"❌ Error analyzing model structure: {str(e)}")
        return False

def print_model_architecture(model_path):
    """Print ASCII visualization of the model architecture based on model type."""
    model_name = os.path.basename(model_path)
    
    if model_name == 'G_z0.pt' or model_name == 'G_z2.pt':
        print(f"\nModel: {model_name} (Original 8× upsampling)")
        print("""
    3D Super-Resolution Architecture (8× upsampling)
    
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
    elif 'modified_2x' in model_name:
        print(f"\nModel: {model_name} (Modified 2× upsampling)")
        print("""
    3D Super-Resolution Architecture (2× upsampling)
    
    Input (6 channels: pos+vel)         Output (6 channels: SR pos+vel)
    [Ng×Ng×Ng×6] ─────────────┐              ┌──── [2Ng×2Ng×2Ng×6]
                              ↓              ↑
                     [512 channels]──→[256]
                        block0    block1
                     (3D Conv)   (3D Conv)
    
    Each block contains:
    • Normalization
    • Two 3×3×3 conv layers
    • Projection back to 6 channels
        """)
    elif 'modified_4x' in model_name:
        print(f"\nModel: {model_name} (Modified 4× upsampling)")
        print("""
    3D Super-Resolution Architecture (4× upsampling)
    
    Input (6 channels: pos+vel)         Output (6 channels: SR pos+vel)
    [Ng×Ng×Ng×6] ─────────────┐              ┌──── [4Ng×4Ng×4Ng×6]
                              ↓              ↑
                     [512 channels]──→[256]──→[128]
                        block0    block1    block2
                     (3D Conv)   (3D Conv) (3D Conv)
    
    Each block contains:
    • Normalization
    • Two 3×3×3 conv layers
    • Projection back to 6 channels
        """)
    else:
        print(f"\nUnknown model type: {model_name}")

def check_data_shape(data_path):
    """Check if data dimensions match model requirements."""
    try:
        if not data_path.endswith('.npy'):
            print("⚠️ Skipping shape check for non-NPY file")
            return True
            
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
        if nx in [32, 64, 128]:
            print("✅ Grid size matches training parameters")
            if nx == 32:
                print("ℹ️ Suitable for 2× or 4× upsampling models")
            elif nx == 64:
                print("ℹ️ Suitable for 8× upsampling model (Ng=64, Box=100 Mpc/h)")
            else:
                print("ℹ️ Using scaled resolution (Ng=128, Box=200 Mpc/h)")
        else:
            print(f"⚠️ Grid size {nx} differs from training parameters (32, 64, or 128)")
            
        print("\n💡 Input data requirements:")
        print("   • Should be normalized displacement + velocity field")
        print("   • Arranged by original grid positions")
        print("   • Recommended: Ng=32 for 2×/4× models, Ng=64 for 8× model")
        print("   • WMAP9 cosmology preferred (see scripts/LR-sim/paramfile.genic)")
        
        return True
    except Exception as e:
        print(f"❌ Data shape error: {str(e)}")
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

def find_bigfile_dirs():
    """Find all potential BigFile directories."""
    bigfile_dirs = []
    for root, dirs, files in os.walk('.'):
        if '000' in files and '001' in files:
            bigfile_dirs.append(root)
    
    if bigfile_dirs:
        print("\n📁 Found potential BigFile directories:")
        for directory in bigfile_dirs:
            print(f"   • {directory}")
    else:
        print("\n⚠️ No BigFile directories found")
    
    return bigfile_dirs

def suggest_workflow(npy_files, bigfile_dirs):
    """Suggest workflow based on available files."""
    print("\n🚀 Suggested workflow:")
    
    if bigfile_dirs:
        print("\n1. Starting with BigFile simulation:")
        bigfile_dir = bigfile_dirs[0]
        print(f"   python preproc.py --inpath {bigfile_dir} --outpath field_64.npy")
        print(f"   python lr2sr.py --lr-input field_64.npy --sr-path output/sr_512 --model SRmodel/G_z0.pt")
        print(f"   python visualize_sr.py --input output/sr_512")
    
    if npy_files:
        for npy_file in npy_files[:1]:  # Just use the first one as example
            try:
                data = np.load(npy_file)
                if len(data.shape) == 4 and data.shape[0] == 6:
                    grid_size = data.shape[1]
                    
                    print(f"\n2. Starting with field data ({npy_file}):")
                    if grid_size == 32:
                        print(f"   python lr2sr.py --lr-input {npy_file} --sr-path output/sr_128 --model SRmodel/G_z0_modified_4x.pt")
                    elif grid_size == 64:
                        print(f"   python lr2sr.py --lr-input {npy_file} --sr-path output/sr_512 --model SRmodel/G_z0.pt")
                    print(f"   python visualize_sr.py --input output/sr_512")
            except:
                pass
    
    print("\n3. Starting with raw simulation (example):")
    print("   python down_sample.py --input sim/PART010 --output field_32.npy --downsample-factor 32")
    print("   python preproc.py --inpath field_32.npy --outpath field_32_processed.npy")
    print("   python lr2sr.py --lr-input field_32_processed.npy --output sr_128 --model SRmodel/G_z0_modified_4x.pt")
    print("   python visualize_sr.py --input sr_128")

def main():
    print("Running environment checks...")
    print("-" * 50)
    
    checks_passed = True
    
    # Check for trained models
    if not check_models():
        checks_passed = False
    
    # Check for required workflow scripts
    if not check_workflow_scripts():
        checks_passed = False
    
    # Check specific script functionality
    check_down_sample_functionality()
    check_field2bigfile_functionality()
    check_visualization_scripts()
    
    # Find all .npy files and BigFile directories
    npy_files = find_npy_files()
    bigfile_dirs = find_bigfile_dirs()
    
    # Check GPU availability
    check_gpu()
    
    print("-" * 50)
    if checks_passed:
        print("✅ All critical checks passed!")
        suggest_workflow(npy_files, bigfile_dirs)
        sys.exit(0)
    else:
        print("❌ Some checks failed. Please resolve the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 