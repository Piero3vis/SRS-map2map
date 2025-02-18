import numpy as np
import matplotlib.pyplot as plt
import argparse
from bigfile import BigFile

def visualize_lr_sr_comparison(lr_path, sr_path, box_size=100.0):
    """
    Visualize the LR and SR data in a 6-panel comparison.
    
    Parameters:
    - lr_path: path to the LR input .npy file
    - sr_path: path to the SR input .npy file
    - box_size: simulation box size in Mpc/h (default 100.0)
    """
    print(f"[INFO] Loading LR data from: {lr_path}")
    lr_data = np.load(lr_path)
    lr_pos = lr_data[:3]
    print(f"[INFO] LR data shape: {lr_data.shape}")  # First 3 channels are positions
    _, nx, ny, nz = lr_pos.shape
    
    print(f"[INFO] Loading SR data from: {sr_path}")
    
     # Load SR data
    try:
        bf = BigFile(sr_path)
        sr_pos = bf['Position'][:]
        print(f"SR particles: {len(sr_pos):,}")
        Ng = 512
        sr_pos = sr_pos.reshape(Ng, Ng, Ng, 3)
        
        
    except Exception as e:
        print(f"[ERROR] Could not load SR data: {str(e)}")
        return
      # Assuming SR data has the same structure
    print(f"[INFO] SR data shape: {sr_pos.shape}")
    
    sr_subsample = [sr_pos[random.randint(0, Ng-1), random.randint(0, Ng-1), random.randint(0, Ng-1), :] for _ in range(100000)]

    # Create a figure for the 6-panel comparison
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    fig.suptitle('Comparison of LR and SR Data', fontsize=16)

    # Displacement in X Direction
    axes[0, 0].imshow(lr_pos[0, :, :, nz//2].T, extent=[0, box_size, 0, box_size], origin='lower', cmap='viridis')
    axes[0, 0].set_title('LR: Displacement in X')
    axes[0, 0].set_xlabel('X Position [Mpc/h]')
    axes[0, 0].set_ylabel('Y Position [Mpc/h]')

    axes[0, 1].imshow(sr_pos[0, :, :, nz//2].T, extent=[0, box_size, 0, box_size], origin='lower', cmap='viridis')
    axes[0, 1].set_title('SR: Displacement in X')
    axes[0, 1].set_xlabel('X Position [Mpc/h]')
    axes[0, 1].set_ylabel('Y Position [Mpc/h]')

    # Displacement in Y Direction
    axes[1, 0].imshow(lr_pos[1, nx//2, :, :].T, extent=[0, box_size, 0, box_size], origin='lower', cmap='viridis')
    axes[1, 0].set_title('LR: Displacement in Y')
    axes[1, 0].set_xlabel('Y Position [Mpc/h]')
    axes[1, 0].set_ylabel('Z Position [Mpc/h]')

    axes[1, 1].imshow(sr_pos[1, nx//2, :, :].T, extent=[0, box_size, 0, box_size], origin='lower', cmap='viridis')
    axes[1, 1].set_title('SR: Displacement in Y')
    axes[1, 1].set_xlabel('Y Position [Mpc/h]')
    axes[1, 1].set_ylabel('Z Position [Mpc/h]')

    # Displacement in Z Direction
    axes[2, 0].imshow(lr_pos[2, :, :, nz//2].T, extent=[0, box_size, 0, box_size], origin='lower', cmap='viridis')
    axes[2, 0].set_title('LR: Displacement in Z')
    axes[2, 0].set_xlabel('X Position [Mpc/h]')
    axes[2, 0].set_ylabel('Y Position [Mpc/h]')

    axes[2, 1].imshow(sr_pos[2, :, :, nz//2].T, extent=[0, box_size, 0, box_size], origin='lower', cmap='viridis')
    axes[2, 1].set_title('SR: Displacement in Z')
    axes[2, 1].set_xlabel('X Position [Mpc/h]')
    axes[2, 1].set_ylabel('Y Position [Mpc/h]')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize LR and SR data in a 6-panel comparison')
    parser.add_argument('--lr-input', required=True, type=str, 
                        help='path to LR input .npy file')
    parser.add_argument('--sr-input', required=True, type=str, 
                        help='path to SR input .npy file')
    parser.add_argument('--box-size', type=float, default=100.0,
                        help='simulation box size in Mpc/h (default: 100.0)')
    
    args = parser.parse_args()
    
    visualize_lr_sr_comparison(args.lr_input, args.sr_input, box_size=args.box_size) 