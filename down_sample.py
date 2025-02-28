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

def save_downsampled_data(data, output_path, Lbox=100000, Ng_sr=64):
    """Save downsampled data in BigFile format with Position block."""
    if output_path is None:
        return
        
    print(data.shape)
    sr_pos = dis2pos(data,Lbox,Ng_sr)
    sr_pos = sr_pos.reshape(3,Ng_sr*Ng_sr*Ng_sr).transpose()
    vel_field = vel_field.reshape(3,Ng_sr*Ng_sr*Ng_sr).transpose()
    
    print(f"[INFO] Saving downsampled data to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # Create BigFile
    dest = BigFile(output_path, create=1)
    
    # Save position block
    blockname = 'Position'
    dest.create_from_array(blockname, data.transpose())

    blockname = 'Velocity'
    dest.create_from_array(blockname, data.transpose())
    
    print(f"[INFO] Saved Position data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='lr2sr')
    parser.add_argument('--lr-input', required=True, type=str, help='path of the lr input')
    parser.add_argument('--ds-path', required=True, type=str, help='path to save ds output')
    parser.add_argument('--redshift', required=True, type=float, help='redshift of the data')
    parser.add_argument('--Lbox-kpc', default=100000, type=float, help='LR/HR/SR Boxsize, in kpc/h')
    parser.add_argument('--downsample-factor', type=int, default=32, help='Downsample factor')

    args = parser.parse_args()

    # Load the normalized field (6 channels: 3 displacement, 3 velocity)
    lr_box = np.load(args.lr_input)
    
    # Unnormalize the fields
    disp_field = cosmology.disnorm(lr_box[0:3,], z=args.redshift, undo=True)
    vel_field = cosmology.velnorm(lr_box[3:6,], z=args.redshift, undo=True)

    # Convert displacement to position
    Lbox = args.Lbox_kpc
    Ng_ds = lr_box.shape[1]  # assuming cubic box
    ds_pos = dis2pos(disp_field, Lbox, Ng_ds)
    
    # Reshape for downsampling
    ds_pos = ds_pos.reshape(3, Ng_ds*Ng_ds*Ng_ds)
    vel_field = vel_field.reshape(3, Ng_ds*Ng_ds*Ng_ds)
    
    # Downsample both position and velocity fields using the same indices
    if args.downsample_factor:
        sample_3d = args.downsample_factor**3
        total_points = ds_pos.shape[1]
        
        if sample_3d > total_points:
            print(f"[WARNING] Requested sample size ({sample_3d}) larger than data size ({total_points})")
            sample_3d = total_points
        
        # Use same random indices for both fields
        random_indices = np.random.choice(total_points, sample_3d, replace=False)
        ds_pos = ds_pos[:, random_indices]
        vel_field = vel_field[:, random_indices]
        print(f"[INFO] Sampled {sample_3d} points")
    
    # Transpose for BigFile format
    ds_pos = ds_pos.transpose()
    vel_field = vel_field.transpose()

    # Save to BigFile format
    path = args.ds_path
    os.makedirs(path, exist_ok=True)

    dest = BigFile(path, create=1)

    blockname = 'Position'
    dest.create_from_array(blockname, ds_pos)

    blockname = 'Velocity'
    dest.create_from_array(blockname, vel_field)

    print(f"Generated downsampled column in {path}")

