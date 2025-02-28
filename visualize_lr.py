import numpy as np
import matplotlib.pyplot as plt
from bigfile import File
import os
from map2map.norms import cosmology

def check_latex_installed():
    """Check if LaTeX is available in the system."""
    try:
        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots()
        ax.text(0, 0, r'$\LaTeX$')
        plt.close(fig)
        return True
    except Exception as e:
        plt.rcParams['text.usetex'] = False
        return False

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

        cellsize = boxsize / Ng

        pid_ = bigf.open('1/ID')[:] - 1   # so that particle id starts from 0
        pos_ = bigf.open('1/Position')[:]
        pos = np.empty_like(pos_)
        pos[pid_] = pos_
        pos = pos.reshape(Ng, Ng, Ng, 3)

        dis = pos2dis(pos, boxsize, Ng)
        del pos

        dis = dis.astype('f4')
        
        dis = np.moveaxis(dis,-1,0)
        
        #disp = cosmology.disnorm(dis,z=redshift) when removing this it removes the grid
        disp = dis
        disp = disp.astype('f4')
        print ("z=%.1f"%redshift,"disp shape:",np.shape(disp))
        return disp

def create_positions(lr_data, Lbox=100000, Ng_lr=64):
    lr_pos = load_lr_data(lr_data)
    lr_pos = dis2pos(lr_pos,Lbox,Ng_lr)
    lr_pos = lr_pos.reshape(3,Ng_lr*Ng_lr*Ng_lr).transpose()
    print(f'shape of lr_pos after dis2pos: {lr_pos.shape}')
    return lr_pos

def visualize_lr(pos, Lbox=100000, Ng_lr=64):
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

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot with error handling
    try:
        scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=0.8, alpha=0.2)
    except Exception as e:
        print(f"[ERROR] Failed to create scatter plot: {str(e)}")
        raise
    
    # Add labels and title with conditional formatting
    if has_latex:
        ax.set_xlabel(r'$X$', fontsize=12)
        ax.set_ylabel(r'$Y$', fontsize=12)
        ax.set_zlabel(r'$Z$', fontsize=12)
        
        title = r'$\mathrm{LR\ Simulation:\ Particle\ Distribution}$' + '\n' + \
                r'$N_{\mathrm{g,lr}}: ' + f'{Ng_lr}$'
        
        ax.set_title(title, fontsize=14, pad=20)
    else:
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z', fontsize=12, fontweight='bold')
        
        title = f'LR Simulation: Particle Distribution\n' + \
                f'Ng_lr: {Ng_lr}'
        
        ax.set_title(title, 
                     fontsize=14, 
                     fontweight='bold', 
                     family='DejaVu Sans',
                     pad=20)
    
    data_min = 0.0
    data_max = Lbox
    margin = 5000
    ax.set_xlim(data_min - margin, data_max + margin)
    ax.set_ylim(data_min - margin, data_max + margin)
    ax.set_zlim(data_min - margin, data_max + margin)
    
    plt.savefig(f'plot/lr_3d_Ng{Ng_lr}.png', dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved figure to plot/lr_3d_Ng{Ng_lr}.png")
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize LR simulation data')
    parser.add_argument('--input', required=True, type=str, help='Path to LR input file (BigFile or .npy)')
    
    args = parser.parse_args()
   
    lr_pos = create_positions(args.input, 100000, 64)
    visualize_lr(lr_pos, 100000, 64)

