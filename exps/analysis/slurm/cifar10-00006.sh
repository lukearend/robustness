#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=64000
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --job-name=ci6
#SBATCH --output=out/cifar10-00006.out
#SBATCH --mail-user=larend@mit.edu
#SBATCH --mail-type=FAIL

module load openmind/singularity/older_versions/2.4

# singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img \
# python /om/user/larend/robust/exps/analysis/activations.py \
# --model_dir=/om/user/larend/models/robust/cifar10/00006 \
# --disable_batch_norm \
# --scale_factor=0.5 \
# --dataset=cifar10 \
# --pickle_dir=/om/user/larend/pickles/cifar10/00006 \
# --host_filesystem=/om \
# --rush

# singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img \
# python /om/user/larend/robust/exps/analysis/redundancy.py \
# --pickle_dir=/om/user/larend/pickles/cifar10/00006

singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img \
python /om/user/larend/robust/exps/analysis/robustness.py \
--model_dir=/om/user/larend/models/robust/cifar10/00006 \
--disable_batch_norm \
--scale_factor=0.5 \
--dataset=cifar10 \
--pickle_dir=/om/user/larend/pickles/cifar10/00006 \
--host_filesystem=/om