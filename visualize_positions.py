import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def visualize_positions(lr_path, box_size=100.0):
    """
    Visualize the original and theoretical final positions of particles in the LR simulation data.
    
    Parameters:
    - lr_path: path to the LR input .npy file
    - box_size: simulation box size in Mpc/h (default 100.0)
    """
    print(f"[INFO] Loading LR data from: {lr_path}")
    
    # Load LR data
    lr_data = np.load(lr_path)
    lr_pos = lr_data[:3]  # First 3 channels are positions
    _, nx, ny, nz = lr_pos.shape
    print(f"Grid size: {nx}x{ny}x{nz}")
    
    # Create physical coordinate grids
    x = np.linspace(0, box_size, nx)
    y = np.linspace(0, box_size, ny)
    z = np.linspace(0, box_size, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Extract displacement vectors
    displacement_x = lr_pos[0]
    displacement_y = lr_pos[1]
    displacement_z = lr_pos[2]
    
    # Calculate final positions
    final_x = X + displacement_x
    final_y = Y + displacement_y
    final_z = Z + displacement_z
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the original grid points
 #   ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c='gray', alpha=0.1, s=1, label='Original Positions')
    
    # Plot final positions of particles
    ax.scatter(final_x.flatten(), final_y.flatten(), final_z.flatten(),
                alpha=0.2, s=1, label='Positions')
    
    # Set labels and title
    ax.set_xlabel('X Position [Mpc/h]')
    ax.set_ylabel('Y Position [Mpc/h]')
    ax.set_zlabel('Z Position [Mpc/h]')
    ax.set_title('3D Visualization of Positions')
    ax.legend()
    
    # Set axis limits
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_zlim(0, box_size)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize original and theoretical final positions of particles in LR simulation data')
    parser.add_argument('--lr-input', required=True, type=str, 
                        help='path to LR input .npy file')
    parser.add_argument('--box-size', type=float, default=100.0,
                        help='simulation box size in Mpc/h (default: 100.0)')
    
    args = parser.parse_args()
    
    visualize_positions(args.lr_input, box_size=args.box_size) 