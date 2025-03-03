from bigfile import File
import numpy as np
import os, argparse
from map2map.norms import cosmology
from bigfile import BigFile

def load_lr_data(file_path):
    """Load LR data from either BigFile or .npy format."""
    if file_path.endswith('.npy'):
        print(f'loading {file_path} from .npy file')
        lr_data = np.load(file_path)
        if lr_data.shape[0] < 3:
            raise ValueError("The .npy file must have at least 3 channels for positions.")
        return lr_data  # Return only positions for now
    else:
        print(f'loading {file_path} from bigfile')
        bigf = File(file_path)
        header = bigf.open('Header')
        boxsize = header.attrs['BoxSize'][0]
        redshift = 1./header.attrs['Time'][0] - 1
        
        Ng = header.attrs['TotNumPart'][1] ** (1/3)
        Ng = int(np.rint(Ng))
        
        pid_ = bigf.open('1/ID')[:] - 1
        pos_ = bigf.open('1/Position')[:]
        pos = np.empty_like(pos_)
        pos[pid_] = pos_
        pos = pos.reshape(Ng, Ng, Ng, 3)
        
        dis = pos2dis(pos, boxsize, Ng)
        dis = dis.astype('f4')
        dis = np.moveaxis(dis, -1, 0)
        
        return dis

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

def down_sample(data, downsample_factor):
    """Downsample the data randomly."""
    if downsample_factor is None:
        return data
        
    sample_3d = downsample_factor**3
    total_points = np.prod(data.shape[1:])
    
    if sample_3d > total_points:
        print(f"[WARNING] Requested sample size ({sample_3d}) larger than data size ({total_points})")
        sample_3d = total_points
    
    # Reshape to (3, N) format
    data_reshaped = data.reshape(data.shape[0], -1)
    
    # Get random indices
    random_indices = np.random.choice(data_reshaped.shape[1], sample_3d, replace=False)
    
    # Sample the data
    downsampled = data_reshaped[:, random_indices]
    
    print(f"[INFO] Sampled {sample_3d} points")
    return downsampled

def save_bigfile(pos, vel, output_path):
    """Save position and velocity data in BigFile format."""
    print(f"[INFO] Saving BigFile format to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    dest = BigFile(output_path, create=1)
    
    dest.create_from_array('Position', pos)
    dest.create_from_array('Velocity', vel)
    print(f"[INFO] Saved Position and Velocity data to {output_path}")

def save_field(field_data, output_path, downsample_factor):
    """Save field data in .npy format with shape (6, N, N, N)."""
    print(f"[INFO] Saving field data to {output_path}")
    # Reshape to (6, N, N, N)
    N = downsample_factor
    field_data = field_data.reshape(6, N, N, N)
    np.save(output_path, field_data)
    print(f"[INFO] Saved field data with shape {field_data.shape} to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downsample field data')
    parser.add_argument('--input', required=True, type=str, help='Input field data (.npy)')
    parser.add_argument('--output', required=True, type=str, help='Output path for downsampled field (.npy)')
    parser.add_argument('--downsample-factor', type=int, default=32, help='Target grid size')
    
    args = parser.parse_args()

    # Load input field (6, Ng, Ng, Ng)
    print(f"[INFO] Loading field from {args.input}")
    field = np.load(args.input)
    
    # Get random indices for downsampling
    N = args.downsample_factor
    total_samples = N**3
    original_size = np.prod(field.shape[1:])
    
    # Generate grid indices and randomly sample them
    grid_indices = np.arange(original_size)
    random_indices = np.random.choice(grid_indices, total_samples, replace=False)
    
    # Reshape and downsample while keeping the channel dimension
    field_flat = field.reshape(6, -1)
    field_downsampled = field_flat[:, random_indices]
    
    # Reshape to final grid format (6, N, N, N)
    field_grid = field_downsampled.reshape(6, N, N, N)
    
    # Save downsampled field
    np.save(args.output, field_grid)
    print(f"[INFO] Saved downsampled field with shape {field_grid.shape} to {args.output}")

