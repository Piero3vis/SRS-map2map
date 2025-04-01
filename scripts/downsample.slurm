#!/bin/bash

#SBATCH -A IscrC_KP-JC-6
#SBATCH --partition=g100_usr_dbg
#SBATCH --nodes=1
#SBATCH --mem=200000MB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=SRS_demnuni
#SBATCH --out=%x_%j.out
#SBATCH --err=%x_%j.err
#SBATCH --mail-user=piero.trevisan@inaf.it
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00

cd $PWD

export mEXEC=/g100/home/userexternal/ptrevis1/SRS-map2map/preproc.py
env_path=/g100/home/userexternal/ptrevis1/train_sr_env


module purge
module load profile/base
module load profile/deeplrn
module load python/3.8.12--gcc--10.2.0
source ${env_path}/bin/activate

#python_exec=${env_path}/bin/python

inpath="/g100_scratch/userexternal/ccarbone/DEMNUniiCOV/LCDM/seed_145236/snapdir_062/snap_062"
outpath="/g100_scratch/userexternal/ptrevis1/field_data/lr_seed_145236.npy"

srun python ${mEXEC} --inpath "$inpath" --outpath "$outpath"

deactivate
