#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:tesla-k80:8
#SBATCH --job-name=train
#SBATCH --output=out/%a.out
#SBATCH --mail-user=larend@mit.edu
#SBATCH --mail-type=FAIL

module load openmind/singularity/older_versions/2.4

singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img \
python /om/user/larend/robust/exps/cifar10-no-bn/train.py \
--model_index="${SLURM_ARRAY_TASK_ID}" --host_filesystem=/om
