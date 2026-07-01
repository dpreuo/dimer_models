#!/bin/bash
#SBATCH --job-name=pachner
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --array=0-19
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=16:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --output=/scratch/perudornellas/logs/outputs/%x_%A_%a.out
#SBATCH --error=/scratch/perudornellas/logs/errors/%x_%A_%a.err

echo "Job started on $(hostname) at $(date)"

# Load Python (the system python does not include numpy)
module load Miniforge3

# perform the computation
/scratch/perudornellas/conda-envs/phys/bin/python /scratch/perudornellas/dimer_models/pachner/full_run/run.py

echo "Job finished at $(date)"