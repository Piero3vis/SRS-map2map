import numpy as np
import matplotlib.pyplot as plt
from bigfile import File
import os
from map2map.norms import cosmology
import argparse

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

def load_hr_data(file_path):
    """Load HR data from either BigFile or .npy format."""
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

        cellsize = boxsivisualize_hr.visualize_hr(hr_pos, 100, 64, s=0.8, alpha=0.08)ze / Ng

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

def create_positions(hr_data, Lbox=100000, Ng_hr=64):
    hr_pos = load_hr_data(hr_data)
    hr_pos = dis2pos(hr_pos,Lbox,Ng_hr)
    hr_pos = hr_pos.reshape(3,Ng_hr*Ng_hr*Ng_hr).transpose()
    print(f'shape of hr_pos after dis2pos: {hr_pos.shape}')
    return hr_pos

def visualize_hr(pos, Lbox=100, Ng_hr=64, s=0.8, alpha=0.008):
    pos = pos/1000.0 # Convert to Mpc
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
        scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=s, alpha=alpha)
    except Exception as e:
        print(f"[ERROR] Failed to create scatter plot: {str(e)}")
        raise
    
    # Add labels and title with conditional formatting
    if has_latex:
        ax.set_xlabel(r'$X [Mpc]$', fontsize=13)
        ax.set_ylabel(r'$Y [Mpc]$', fontsize=13)
        ax.set_zlabel(r'$Z [Mpc]$', fontsize=13)
        
        title = r'$\mathrm{HR\ Simulation:\ 3D\ Particle\ Distribution}$' + '\n' + \
                r'$N_{\mathrm{g,hr}}: ' + f'{Ng_hr}$'
        
        ax.set_title(title, fontsize=20, pad=15)
    else:
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z', fontsize=12, fontweight='bold')
        
        title = f'HR Simulation: Particle Distribution\n' + \
                f'Ng_hr: {Ng_hr}'
        
        ax.set_title(title, 
                     fontsize=14, 
                     fontweight='bold', 
                     family='DejaVu Sans',
                     pad=20)
    
    data_min = 0.0
    data_max = Lbox
    margin = 5
    ax.set_xlim(data_min - margin, data_max + margin)
    ax.set_ylim(data_min - margin, data_max + margin)
    ax.set_zlim(data_min - margin, data_max + margin)
    
    plt.savefig(f'plot/hr_3d_Ng{Ng_hr}.png', dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved figure to plot/hr_3d_Ng{Ng_hr}.png")
    plt.show()

def main():
  

    parser = argparse.ArgumentParser(description='Visualize HR simulation data')
    parser.add_argument('--input', required=True, type=str, help='Path to HR input file (BigFile or .npy)')

    args = parser.parse_args()

    hr_pos = create_positions(args.input, 100000, 64)
    visualize_hr(hr_pos, 100, 64, s=0.8, alpha=0.08)

if __name__ == "__main__":
    main()



