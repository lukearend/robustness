#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --mem=64000
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --job-name=imagenet-00000
#SBATCH --output=out/imagenet-00000.out
#SBATCH --mail-user=larend@mit.edu
#SBATCH --mail-type=FAIL

module load openmind/singularity/older_versions/2.4

# singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img \
# python /om/user/larend/robust/exps/analysis/activations.py \
# --model_dir=/om/user/larend/models/robust/imagenet/00000 \
# --scale_factor=0.25 \
# --dataset=imagenet \
# --pickle_dir=/om/user/larend/pickles/imagenet/00000 \
# --host_filesystem=/om

# singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img \
# python /om/user/larend/robust/exps/analysis/redundancy.py \
# --pickle_dir=/om/user/larend/pickles/imagenet/00000

singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img \
python /om/user/larend/robust/exps/analysis/robustness.py \
--model_dir=/om/user/larend/models/robust/imagenet/00000 \
--scale_factor=0.25 \
--dataset=imagenet \
--pickle_dir=/om/user/larend/pickles/imagenet/00000 \
--host_filesystem=/om
