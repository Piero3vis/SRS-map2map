import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys, argparse
from bigfile import BigFile

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

def visualize_sr_3d(inpath, box_size=100.0, downsample_factor=64):
    """Visualize 3D positions of SR simulation data."""
    # Load positions from file
    pos = load_sr_positions(inpath)
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
        print(f"[INFO] Sampled {sample_3d} points for visualization")

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot with error handling
    try:
        scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=0.8, alpha=0.2)
    except Exception as e:
        print(f"[ERROR] Failed to create scatter plot: {str(e)}")
        raise
    
    # Add labels and title
    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    ax.set_title('SR Simulation Particle Distribution with downsample factor = {}'.format(downsample_factor))
    
    # # Set axis limits based on data range
    # data_min = pos.min()
    # data_max = pos.max()
    # print(f"[INFO] Setting plot limits from {data_min:.2f} to {data_max:.2f}")

    data_min = 0.0
    data_max = 100000
    margin = 5000
    ax.set_xlim(data_min - margin, data_max + margin)
    ax.set_ylim(data_min - margin, data_max + margin)
    ax.set_zlim(data_min - margin, data_max + margin)
    
    plt.savefig('sr_3d.png', dpi=300, bbox_inches='tight')
    print("[INFO] Saved figure to sr_3d.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize SR simulation data')
    parser.add_argument('--inpath', type=str, required=True, help='Path to the SR simulation data')
    parser.add_argument('--downsample_factor', type=int, default=64, help='Downsample factor for visualization')
    args = parser.parse_args()
    visualize_sr_3d(args.inpath, args.downsample_factor)


