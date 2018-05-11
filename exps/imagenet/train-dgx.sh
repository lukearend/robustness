#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=500000
#SBATCH --job-name=train
#SBATCH --output=out/%a.out
#SBATCH --mail-user=larend@mit.edu
#SBATCH --mail-type=FAIL

singularity exec --nv -B /raid:/raid /raid/poggio/home/larend/localtensorflow.img \
python /raid/poggio/home/larend/robust/exps/imagenet/train.py \
--model_index="${SLURM_ARRAY_TASK_ID}" --host_filesystem=/raid
