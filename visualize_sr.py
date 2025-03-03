import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys, argparse
from bigfile import BigFile

W = 5.8
plt.rcParams.update({
    'figure.figsize': (W, W/(4/3)),     # 4:3 aspect ratio
    'font.size' : 14,                   # Set font size to 11pt
    'axes.labelsize': 14,               # -> axis labels
    'legend.fontsize': 14,              # -> legends
    'font.family': 'lmodern',
    'text.usetex': True,
    'text.latex.preamble': (            # LaTeX preamble
        r'\usepackage{lmodern}'
        # ... more packages if needed
    )
})


def load_sr_positions(inpath):
    """Load positions from SR simulation data."""
    if not os.path.exists(inpath):
        raise FileNotFoundError(f"Input path does not exist: {inpath}")
    
    print(f"[INFO] Loading SR data from: {inpath}")
    bf = BigFile(inpath)
    
    if 'Position' not in bf:
        raise KeyError("'Position' dataset not found in BigFile")
    
    pos = bf['Position'][:]
    print(f"[INFO] Loaded positions with shape: {pos.shape}")
    return pos

def is_cube(n):
    cube_root = n**(1./3.)
    if round(cube_root) ** 3 == n:
        print(True, "Particle number is a cube number, its cubed root is", round(cube_root))
        return cube_root
    else:
        print(False, "Particle number is not a cube number, its cubed root is", cube_root)

def check_shape(inpath):
    pos = load_sr_positions(inpath)
    cube_root = is_cube(pos.shape[0])
    if cube_root is not None:
        return round(cube_root)
    else:
        return None

def check_latex_installed():
    """Check if LaTeX is available in the system."""
    try:
        # Try to create a simple LaTeX string
        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots()
        ax.text(0, 0, r'$\LaTeX$')
        plt.close(fig)
        return True
    except Exception as e:
        plt.rcParams['text.usetex'] = False
        return False

def visualize_sr_3d(inpath, downsample_factor, box_size=100.0):
    """Visualize 3D positions of SR simulation data."""
    # Check LaTeX availability
    has_latex = check_latex_installed()
    
    if has_latex:
        # LaTeX configuration
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
            'text.latex.preamble': r'\usepackage{amsmath}'
        })
        print("[INFO] Using LaTeX for text rendering")
    else:
        # Regular text configuration
        plt.rcParams.update({
            'text.usetex': False,
            'font.family': 'DejaVu Sans'
        })
        print("[INFO] LaTeX not available, using standard text rendering")

    # Load positions from file
    pos = load_sr_positions(inpath)
    pos = pos/1000.0
    sample_3d = downsample_factor**3
    
    # Basic data validation
    if pos.size == 0:
        raise ValueError("Empty position array loaded")
    
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected position array of shape (N, 3), got {pos.shape}")
    
    print(f"[DEBUG] Position stats:")
    print(f"  - Shape: {pos.shape}")
    print(f"  - Range X: [{pos[:, 0].min():.2f}, {pos[:, 0].max():.2f}]")
    print(f"  - Range Y: [{pos[:, 1].min():.2f}, {pos[:, 1].max():.2f}]")
    print(f"  - Range Z: [{pos[:, 2].min():.2f}, {pos[:, 2].max():.2f}]")
    
    # Sample a subset of particles if specified

    if downsample_factor is not None:
        if sample_3d > pos.shape[0]:
            print(f"[WARNING] Requested sample size ({sample_3d}) larger than data size ({pos.shape[0]})")
            sample_3d = pos.shape[0]
        
        random_indices = np.random.choice(pos.shape[0], sample_3d, replace=False)
        pos = pos[random_indices]
        print(f"[INFO] Sampled {sample_3d}  points for visualization. Sampling_size: {sample_3d==downsample_factor**3}")

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot with error handling
    try:
        scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=0.8, alpha=0.008)
        print(f"[INFO] If it looks empty, it's because the alpha is too low for the number of particles")
    except Exception as e:
        print(f"[ERROR] Failed to create scatter plot: {str(e)}")
        raise
    
    # Get Ng values
    ng_sr = check_shape(inpath)
    ng_lr = int(ng_sr/8)  # LR is half of SR in the case of the corrected NN, the original is an eighth
    
    # Add labels and title with conditional formatting
    if has_latex:
        ax.set_xlabel(r'$X [Mpc]$', fontsize=13)
        ax.set_ylabel(r'$Y [Mpc]$', fontsize=13)
        ax.set_zlabel(r'$Z [Mpc]$', fontsize=13)
        
        title = r'$\mathrm{SR\ Simulation: \ Particle\ Distribution}$' + '\n' + \
                r'$N_{\mathrm{g,sr}}: ' + f'{ng_sr}' + \
                r'\ (N_{\mathrm{g,lr}}: ' + f'{ng_lr}'+ r',\ \mathrm{super\ resolution:\ 8\times})$'
        
        ax.set_title(title, fontsize=20, pad=15)
    else:
        ax.set_xlabel('X [Kpc]', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y [Kpc]', fontsize=14, fontweight='bold')
        ax.set_zlabel('Z [Kpc]', fontsize=14, fontweight='bold')
        
        title = f'SR Simulation: Particle Distribution\n' + \
                f'Ng_sr: {ng_sr} (Ng_lr: {ng_lr}, super_resolution: 2x)'
        
        ax.set_title(title, 
                     fontsize=14, 
                     fontweight='bold', 
                     family='DejaVu Sans',
                     pad=20)
    
    # # Set axis limits based on data range
    # data_min = pos.min()
    # data_max = pos.max()
    # print(f"[INFO] Setting plot limits from {data_min:.2f} to {data_max:.2f}")

    data_min = 0.0
    data_max = 100
    margin = 10
    ax.set_xlim(data_min - margin, data_max + margin)
    ax.set_ylim(data_min - margin, data_max + margin)
    ax.set_zlim(data_min - margin, data_max + margin)
    
    # Update save filename to include both Ng values
    plt.savefig(f'plot/sr_3d_{downsample_factor}_from_sim_lr{ng_lr}.png', dpi=200, bbox_inches='tight', pad_inches=0.3)
    print(f"[INFO] Saved figure to plot/sr_3d_{downsample_factor}_from_sim_lr{ng_lr}.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize SR simulation data')
    parser.add_argument('--inpath', type=str, required=True, help='Path to the SR simulation data')
    parser.add_argument('--downsample_factor', '--downsample-factor', type=int, default=64, help='Downsample factor for visualization')
    args = parser.parse_args()
    print(f"Downsample Factor: {args.downsample_factor}")
    load_sr_positions(args.inpath)
    visualize_sr_3d(args.inpath, args.downsample_factor)