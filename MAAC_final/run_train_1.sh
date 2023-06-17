#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=20
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --job-name=mnisttrain
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=gpu
# cd $SLURM_SUBMIT_DIR
module load python/conda-python/3.7_new
source activate maac
