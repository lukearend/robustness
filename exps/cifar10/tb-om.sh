#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=2000
#SBATCH --job-name=tb
#SBATCH --output=out/tb.out
#SBATCH --mail-user=larend@mit.edu
#SBATCH --mail-type=FAIL

module load openmind/singularity/2.4

singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img \
tensorboard \
--logdir=/om/user/larend/models/robust/cifar10/00000 \
--port=${1:-6050}
