#!/bin/bash
#SBATCH --gres=gpu:h100_80gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=inpainting
#SBATCH --partition=blanca-blast-lecs
#SBATCH --account=blanca-blast-lecs
#SBATCH --qos=blanca-blast-lecs
#SBATCH --mail-type=END,FAIL

module load uv
uv sync
uv run train.py --batch-size 16 --model google/flan-t5-xl
