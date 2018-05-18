#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=50000
#SBATCH --cpus-per-task=80
#SBATCH --job-name=imagenet
#SBATCH --output=out/imagenet-0000%a.out
#SBATCH --mail-user=larend@mit.edu
#SBATCH --mail-type=FAIL

SCALE_FACTOR=('0.25' '0.5' '1' '2' '4')

singularity exec --nv -B /raid:/raid /raid/poggio/home/larend/localtensorflow.img \
python /raid/poggio/home/larend/robust/exps/analysis/activations.py \
--model_dir=/raid/poggio/home/larend/models/robust/imagenet/0000${SLURM_ARRAY_TASK_ID} \
--scale_factor=${SCALE_FACTOR[$SLURM_ARRAY_TASK_ID]} \
--dataset=imagenet \
--pickle_dir=/raid/poggio/home/larend/pickles2/imagenet/0000${SLURM_ARRAY_TASK_ID} \
--host_filesystem=/raid

singularity exec --nv -B /raid:/raid /raid/poggio/home/larend/localtensorflow.img \
python /raid/poggio/home/larend/robust/exps/analysis/redundancy.py \
--pickle_dir=/raid/poggio/home/larend/pickles2/imagenet/0000${SLURM_ARRAY_TASK_ID}

singularity exec --nv -B /raid:/raid /raid/poggio/home/larend/localtensorflow.img \
python /raid/poggio/home/larend/robust/exps/analysis/robustness.py \
--model_dir=/raid/poggio/home/larend/models/robust/imagenet/0000${SLURM_ARRAY_TASK_ID} \
--scale_factor=${SCALE_FACTOR[$SLURM_ARRAY_TASK_ID]} \
--dataset=imagenet \
--pickle_dir=/raid/poggio/home/larend/pickles2/imagenet/0000${SLURM_ARRAY_TASK_ID} \
--host_filesystem=/raid
