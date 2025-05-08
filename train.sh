#!/bin/bash 
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --exclusive
#SBATCH --output=workdir/test2/slurm.out
#SBATCH --error=workdir/test2/slurm.err  
#SBATCH --mem 32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -J train_st

module purge
module list
module load CUDA/12.4

conda init
conda activate pytorch
python main.py --config config/default_configs.py --mode train --workdir workdir/mshoot