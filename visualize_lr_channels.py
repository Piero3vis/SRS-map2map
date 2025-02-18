import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def visualize_lr_channels(lr_path, box_size=100.0):
    """
    Visualize the LR simulation data channel by channel in 3D space.
    
    Parameters:
    - lr_path: path to the LR input .npy file
    - box_size: simulation box size in Mpc/h (default 100.0)
    """
    print(f"[INFO] Loading LR data from: {lr_path}")
    
    # Load LR data and validate
    lr_data = np.load(lr_path)
    print(f"[INFO] Data shape: {lr_data.shape}")
    lr_pos = lr_data[:3]  # First 3 channels are positions
    _, nx, ny, nz = lr_pos.shape
    print(f"[INFO] Grid size: {nx}x{ny}x{nz}")
    
    # Create physical coordinate grids
    x = np.linspace(0, box_size, nx)
    y = np.linspace(0, box_size, ny)
    z = np.linspace(0, box_size, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Calculate displacement magnitude
    displacement_magnitude = np.sqrt(
        lr_pos[0]**2 + lr_pos[1]**2 + lr_pos[2]**2
    )
    print(f"[INFO] Displacement magnitude range: [{displacement_magnitude.min():.3f}, {displacement_magnitude.max():.3f}]")
    
    # Create a figure for the 4 plots (3 components + magnitude)
    fig = plt.figure(figsize=(20, 5))
    
    # Channel 1: Displacement in X direction
    ax1 = fig.add_subplot(141, projection='3d')
    sc1 = ax1.scatter(X.flatten(), Y.flatten(), Z.flatten(), 
                     c=lr_pos[0].flatten(), cmap='viridis', alpha=0.5, s=1)
    ax1.set_title('X Displacement [Mpc/h]')
    fig.colorbar(sc1, ax=ax1)
    ax1.set_xlabel('X [Mpc/h]')
    ax1.set_ylabel('Y [Mpc/h]')
    ax1.set_zlabel('Z [Mpc/h]')
    
    # Channel 2: Displacement in Y direction
    ax2 = fig.add_subplot(142, projection='3d')
    sc2 = ax2.scatter(X.flatten(), Y.flatten(), Z.flatten(), 
                     c=lr_pos[1].flatten(), cmap='viridis', alpha=0.5, s=1)
    ax2.set_title('Y Displacement [Mpc/h]')
    fig.colorbar(sc2, ax=ax2)
    ax2.set_xlabel('X [Mpc/h]')
    ax2.set_ylabel('Y [Mpc/h]')
    ax2.set_zlabel('Z [Mpc/h]')
    
    # Channel 3: Displacement in Z direction
    ax3 = fig.add_subplot(143, projection='3d')
    sc3 = ax3.scatter(X.flatten(), Y.flatten(), Z.flatten(), 
                     c=lr_pos[2].flatten(), cmap='viridis', alpha=0.5, s=1)
    ax3.set_title('Z Displacement [Mpc/h]')
    fig.colorbar(sc3, ax=ax3)
    ax3.set_xlabel('X [Mpc/h]')
    ax3.set_ylabel('Y [Mpc/h]')
    ax3.set_zlabel('Z [Mpc/h]')
    
    # Channel 4: Displacement magnitude
    ax4 = fig.add_subplot(144, projection='3d')
    sc4 = ax4.scatter(X.flatten(), Y.flatten(), Z.flatten(), 
                     c=displacement_magnitude.flatten(), cmap='viridis', alpha=0.5, s=1)
    ax4.set_title('Displacement Magnitude [Mpc/h]')
    fig.colorbar(sc4, ax=ax4)
    ax4.set_xlabel('X [Mpc/h]')
    ax4.set_ylabel('Y [Mpc/h]')
    ax4.set_zlabel('Z [Mpc/h]')
    
    # Set consistent limits for all plots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_zlim(0, box_size)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize LR simulation data channel by channel in 3D space')
    parser.add_argument('--lr-input', required=True, type=str, 
                        help='path to LR input .npy file')
    parser.add_argument('--box-size', type=float, default=100.0,
                        help='simulation box size in Mpc/h (default: 100.0)')
    
    args = parser.parse_args()
    
    visualize_lr_channels(args.lr_input, box_size=args.box_size) 