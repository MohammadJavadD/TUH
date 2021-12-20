#!/bin/bash
#SBATCH --array=12
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32000M
#SBATCH --time=08:00:00
#SBATCH --job-name=beetl
#SBATCH --wait

# bash download.sh

module load anaconda/3
conda activate eeg

date
python main.py
date