"""
This script converts a field data (.npy) with shape (6, N, N, N) into BigFile format,
creating Position and Velocity subfolders. The first 3 channels are displacement,
the last 3 are velocity.
"""

import numpy as np
from bigfile import BigFile
import argparse
import os
from map2map.norms import cosmology

def dis2pos(dis_field, boxsize, Ng):
    """Convert displacement field to positions.
    Assume 'dis_field' is in order of `pid` that aligns with the Lagrangian lattice,
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

def field2bigfile(field_path, output_path, redshift, Lbox=100000):
    """Convert field data to BigFile format."""
    print(f"[INFO] Loading field from {field_path}")
    field = np.load(field_path)
    
    if field.shape[0] != 6:
        raise ValueError(f"Expected 6 channels, got {field.shape[0]}")
    
    Ng = field.shape[1]
    print(f"[INFO] Grid size: {Ng}")
    
    # Unnormalize the fields
    disp_field = cosmology.disnorm(field[0:3], z=redshift, undo=True)
    vel_field = cosmology.velnorm(field[3:6], z=redshift, undo=True)
    
    # Convert displacement to position
    pos = dis2pos(disp_field, Lbox, Ng)
    
    # Reshape to (N³, 3) format for BigFile
    pos = pos.reshape(3, -1).T
    vel = vel_field.reshape(3, -1).T
    
    # Save to BigFile format
    print(f"[INFO] Saving to BigFile format in {output_path}")
    os.makedirs(output_path, exist_ok=True)
    dest = BigFile(output_path, create=1)
    
    dest.create_from_array('Position', pos)
    dest.create_from_array('Velocity', vel)
    
    print(f"[INFO] Saved {pos.shape[0]} particles to {output_path}")
    print(f"[INFO] Position range: [{pos.min():.1f}, {pos.max():.1f}]")
    print(f"[INFO] Velocity range: [{vel.min():.1f}, {vel.max():.1f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert field data to BigFile format')
    parser.add_argument('--input', required=True, type=str, help='Input field data (.npy)')
    parser.add_argument('--output', required=True, type=str, help='Output path for BigFile')
    parser.add_argument('--redshift', required=True, type=float, help='Redshift of the data')
    parser.add_argument('--Lbox-kpc', default=100000, type=float, help='Box size in kpc/h')
    
    args = parser.parse_args()
    
    field2bigfile(args.input, args.output, args.redshift, args.Lbox_kpc) 