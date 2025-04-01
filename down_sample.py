"""
This script preprocess the snapshot of N-body simulation into 3D image with 6 channels,
output is in shape of (Nc,Ng,Ng,Ng), giving the normalized {displacement + velocity} field arranged by the original grid of the tracer particles
"""

import numpy as np
from bigfile import File
import argparse
import os, sys
from map2map.norms import cosmology
import readsnap_mod as rs

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


def down_sample_snapshot_nonlin_fields(inpath, outpath, downsample_factor=None):
    """
    inpath is LR simulation snapshot in Gadget format
    outpath is numpy array in shape (Nc,Ng,Ng,Ng)
    
    Parameters:
    -----------
    inpath : str
        Path to the input snapshot (without file number)
    outpath : str
        Path to save the output
    downsample_factor : int, optional
        New grid size after downsampling (e.g., downsample_factor=512 for a 512³ grid)
    """
    header = rs.snapshot_header(inpath+'.0')
    boxsize = header.boxsize
    redshift = header.redshift
    
    Ng = int(np.round(header.npart[1] ** (1/3)))
    print(f'Original grid size: {Ng}')
    print(f'Boxsize, redshift: {boxsize}, {redshift}')
    
    part_type = 1

    
    pos_ = rs.read_block(inpath, "POS ", parttype=part_type, verbose=True)
    vel_ = rs.read_block(inpath, "VEL ", parttype=part_type, verbose=True)
    pid_ = rs.read_block(inpath, "ID  ", parttype=part_type, verbose=True) - 1
    
    # Random sampling if requested
    if downsample_factor is not None:
        sample_3d = downsample_factor**3
        if sample_3d > pos_.shape[0]:
            print(f"[WARNING] Requested sample size ({sample_3d}) larger than data size ({pos_.shape[0]})")
            sample_3d = pos_.shape[0]
        
        
        random_indices = np.random.choice(pos_.shape[0], sample_3d, replace=False)
        random_indices.sort()  
        
        
        pos_ = pos_[random_indices]
        vel_ = vel_[random_indices]
        
        # Create new sequential IDs for the downsampled grid
        pid_ = np.arange(sample_3d)
        
        # Update Ng for the new grid size
        Ng = downsample_factor
        print(f"[INFO] Downsampled to {sample_3d} particles, new grid size: {Ng}")

    print(f'Shape of pos_: {np.shape(pos_)}')
    
    # Arrange particles on grid
    pos = np.empty_like(pos_)
    pos[pid_] = pos_
    pos = pos.reshape(Ng, Ng, Ng, 3)
    
    vel = np.empty_like(vel_)
    vel[pid_] = vel_
    vel = vel.reshape(Ng, Ng, Ng, 3)
    del pid_, pos_, vel_

    
    dis = pos2dis(pos, boxsize, Ng)
    del pos

    dis = dis.astype('f4')
    vel = vel.astype('f4')
    
    dis = np.moveaxis(dis,-1,0)
    vel = np.moveaxis(vel,-1,0)
    
    disp = cosmology.disnorm(dis,z=redshift)
    velocity = cosmology.velnorm(vel,z=redshift)
    catnorm = np.concatenate([disp,velocity],axis=0)
    catnorm = catnorm.astype('f4')
    print(f"z={redshift:.1f} catnorm shape:", np.shape(catnorm))
    
    np.save(outpath, catnorm)

#-------------------------------------------------------------------    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--inpath',required=True,type=str,help='path of the LR input snapshot')
    parser.add_argument('--outpath',required=True,type=str,help='path of the output')
    parser.add_argument('--downsample-factor',required=False, default=None,type=int,help='downsample factor')
    parser.add_argument('--snapshot-format',required=False,action='store_true',help='read data in Gadget snapshot format instead of BigFile')
    
    args = parser.parse_args()
    
    if args.snapshot_format:
        down_sample_snapshot_nonlin_fields(args.inpath, args.outpath, args.downsample_factor)
    else:
        down_sample_snapshot_nonlin_fields(args.inpath, args.outpath, args.downsample_factor)
    
    
    
    
