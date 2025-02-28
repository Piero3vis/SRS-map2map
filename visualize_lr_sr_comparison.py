import numpy as np
import matplotlib.pyplot as plt
from bigfile import File, BigFile
import os
from map2map.norms import cosmology

def pos2dis(pos, boxsize, Ng):
    """Assume `pos` is ordered in `pid` that aligns with the Lagrangian lattice,
    and all displacement must not exceed half box size.
    """
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize + 0.5 * cellsize

    pos[..., 0] -= lattice.reshape(-1, 1, 1)
    pos[..., 1] -= lattice.reshape(-1, 1)
    pos[..., 2] -= lattice

    pos -= np.rint(pos / boxsize) * boxsize

    return pos

def dis2pos(dis_field, boxsize, Ng):
    """Assume 'dis_field' is in order of `pid` that aligns with the Lagrangian lattice,
    and dis_field.shape = (3,Ng,Ng,Ng)
    """
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize + 0.5 * cellsize

    pos = dis_field.copy()

    pos[2] += lattice
    pos[1] += lattice.reshape(-1, 1)
    pos[0] += lattice.reshape(-1, 1, 1)

    pos[pos < 0] += boxsize
    pos[pos > boxsize] -= boxsize

    return pos

def load_lr_data(file_path):
    """Load LR data from either BigFile or .npy format."""
    if file_path.endswith('.npy'):
        print(f'loading {file_path} from .npy file')
        lr_data = np.load(file_path)
        if lr_data.shape[0] < 3:
            raise ValueError("The .npy file must have at least 3 channels for positions.")
        return lr_data[:3]  # Return only the first three channels (positions)
    else:
        print(f'loading {file_path} from bigfile')
        bigf = File(file_path)
        header = bigf.open('Header')
        boxsize = header.attrs['BoxSize'][0]
        redshift = 1./header.attrs['Time'][0] - 1
        
        Ng = header.attrs['TotNumPart'][1] ** (1/3)
        Ng = int(np.rint(Ng))

        pid_ = bigf.open('1/ID')[:] - 1   # so that particle id starts from 0
        pos_ = bigf.open('1/Position')[:]
        pos = np.empty_like(pos_)
        pos[pid_] = pos_
        pos = pos.reshape(Ng, Ng, Ng, 3)

        dis = pos2dis(pos, boxsize, Ng)
        del pos

        dis = dis.astype('f4')
        dis = np.moveaxis(dis,-1,0)
        disp = dis.astype('f4')
        print(f"z={redshift:.1f}, disp shape: {np.shape(disp)}")
        return disp

def load_sr_positions(inpath, downsample_factor=64):
    """Load positions from SR simulation data."""
    if not os.path.exists(inpath):
        raise FileNotFoundError(f"Input path does not exist: {inpath}")
    
    print(f"[INFO] Loading SR data from: {inpath}")
    bf = BigFile(inpath)
    
    if 'Position' not in bf:
        raise KeyError("'Position' dataset not found in BigFile")
    
    pos = bf['Position'][:]
    print(f"[INFO] Loaded positions with shape: {pos.shape}")
    
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
        sample_3d = downsample_factor**3
        if sample_3d > pos.shape[0]:
            print(f"[WARNING] Requested sample size ({sample_3d}) larger than data size ({pos.shape[0]})")
            sample_3d = pos.shape[0]
        
        random_indices = np.random.choice(pos.shape[0], sample_3d, replace=False)
        pos = pos[random_indices]
        print(f"[INFO] Sampled {sample_3d} points for visualization")
    
    return pos

def create_positions(lr_data, Lbox=100000, Ng_lr=64):
    lr_pos = load_lr_data(lr_data)
    lr_pos = dis2pos(lr_pos, Lbox, Ng_lr)
    lr_pos = lr_pos.reshape(3, Ng_lr*Ng_lr*Ng_lr).transpose()
    print(f'shape of lr_pos after dis2pos: {lr_pos.shape}')
    return lr_pos

def visualize_comparison(lr_pos, sr_pos, Lbox=100000, margin=5000, downsample_factor=64):
    """Visualize LR and SR data side by side with their difference."""
    fig = plt.figure(figsize=(20, 7))
    
    # LR Plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(lr_pos[:, 0], lr_pos[:, 1], lr_pos[:, 2], s=0.8, alpha=0.2)
    ax1.set_xlabel('X [Mpc/h]')
    ax1.set_ylabel('Y [Mpc/h]')
    ax1.set_zlabel('Z [Mpc/h]')
    ax1.set_title('LR Simulation')
    ax1.set_xlim(0 - margin, Lbox + margin)
    ax1.set_ylim(0 - margin, Lbox + margin)
    ax1.set_zlim(0 - margin, Lbox + margin)
    
    # SR Plot
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(sr_pos[:, 0], sr_pos[:, 1], sr_pos[:, 2], s=0.8, alpha=0.2)
    ax2.set_xlabel('X [Mpc/h]')
    ax2.set_ylabel('Y [Mpc/h]')
    ax2.set_zlabel('Z [Mpc/h]')
    ax2.set_title('SR Simulation')
    ax2.set_xlim(0 - margin, Lbox + margin)
    ax2.set_ylim(0 - margin, Lbox + margin)
    ax2.set_zlim(0 - margin, Lbox + margin)
    
    # Difference Plot (if possible to match particles)
    ax3 = fig.add_subplot(133, projection='3d')
    try:
        # Downsample SR positions to match LR grid
        sr_downsampled = sr_pos[:lr_pos.shape[0]]
        diff = sr_downsampled - lr_pos
        
        # Plot difference vectors
        scatter = ax3.scatter(lr_pos[:, 0], lr_pos[:, 1], lr_pos[:, 2], 
                            c=np.linalg.norm(diff, axis=1),
                            cmap='viridis', s=0.8, alpha=0.2)
        plt.colorbar(scatter, ax=ax3, label='Displacement magnitude [Mpc/h]')
        
    except Exception as e:
        print(f"[WARNING] Could not compute differences: {str(e)}")
        ax3.text(0.5, 0.5, 0.5, "Could not compute differences", 
                 horizontalalignment='center', verticalalignment='center')
    
    ax3.set_xlabel('X [Mpc/h]')
    ax3.set_ylabel('Y [Mpc/h]')
    ax3.set_zlabel('Z [Mpc/h]')
    ax3.set_title('Position Differences')
    ax3.set_xlim(0 - margin, Lbox + margin)
    ax3.set_ylim(0 - margin, Lbox + margin)
    ax3.set_zlim(0 - margin, Lbox + margin)
    
    plt.tight_layout()
    plt.savefig(f'plots/lr_sr_comparison_{downsample_factor}.png', dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved comparison plot to plots/lr_sr_comparison_{downsample_factor}.png")
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare LR and SR simulation data')
    parser.add_argument('--lr-input', required=True, type=str, help='Path to LR input file')
    parser.add_argument('--sr-input', required=True, type=str, help='Path to SR input file')
    parser.add_argument('--box-size', type=float, default=100000, help='Box size in Mpc/h')
    parser.add_argument('--margin', type=float, default=5000, help='Margin for plots in Mpc/h')
    parser.add_argument('--downsample-factor', type=int, default=64, help='Downsample factor for SR data')
    
    args = parser.parse_args()
    
    # Load and process LR data
    lr_pos = create_positions(args.lr_input, args.box_size)
    
    # Load and process SR data with downsampling
    sr_pos = load_sr_positions(args.sr_input, args.downsample_factor)
    
    # Visualize comparison
    visualize_comparison(lr_pos, sr_pos, args.box_size, args.margin, args.downsample_factor)

