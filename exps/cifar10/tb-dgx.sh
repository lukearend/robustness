#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=10000
#SBATCH --job-name=tb
#SBATCH --output=out/tb.out
#SBATCH --mail-user=larend@mit.edu
#SBATCH --mail-type=FAIL

singularity exec --nv -B /raid:/raid /raid/poggio/home/larend/localtensorflow.img \
tensorboard \
--logdir=/raid/poggio/home/larend/models/robust/cifar10/00000 \
--port=${1:-6050}
