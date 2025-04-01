#!/bin/bash                                                                               
                                                                                          
#SBATCH --job-name=preproc             # Nome del lavoro                              
#SBATCH --output=output.%j.out            # Nome file di output                           
#SBATCH --error=error.%j.err              # Nome file di errori                           
#SBATCH --nodes=4                         # Numero di nodi richiesti                       
#SBATCH --ntasks-per-node=32              # Numero di task per nodo (Leonardo ha fino a 64 core per nodo CPU)                                                                       
#SBATCH --time=12:00:00                   # Tempo massimo richiesto (hh:mm:ss)            
#SBATCH --partition=boost_usr_prod                  # Partizione (main per CPU)
#SBATCH --mail-user=piero.trevisan@inaf.it
#SBATCH --mail-type=ALL

#hostname; pwd; date

echo "Job started at $(date)"
echo "Running on $SLURM_JOB_NODELIST"
echo "Running on $SLURM_NNODES nodes"
echo "Running on $SLURM_NTASKS tasks"
echo "Running on $SLURM_CPUS_PER_TASK cores per task"


inpath="g100_scratch/userexternal/ptrevis1/LCDM/seed_123456/snapdir_062/snapdir_062"
outpath="g100_scratch/userexternal/ptrevis1/LCDM/seed_123456/catnorm_062"

python preproc.py --inpath "$inpath" --outpath "$outpath"

date

