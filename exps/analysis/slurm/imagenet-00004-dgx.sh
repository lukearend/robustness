#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=500000
#SBATCH --cpus-per-task=80
#SBATCH --job-name=imagenet-00004
#SBATCH --output=out/imagenet-00004.out
#SBATCH --mail-user=larend@mit.edu
#SBATCH --mail-type=FAIL

module load openmind/singularity/older_versions/2.4

singularity exec --nv -B /om:/om /raid/poggio/home/larend/localtensorflow.img \
python /raid/poggio/home/larend/robust/exps/analysis/activations.py \
--model_dir=/raid/poggio/home/larend/models/robust/imagenet/00004 \
--scale_factor=4 \
--dataset=imagenet \
--pickle_dir=/raid/poggio/home/larend/pickles/imagenet/00004 \
--host_filesystem=/raid

singularity exec --nv -B /om:/om /raid/poggio/home/larend/localtensorflow.img \
python /raid/poggio/home/larend/robust/exps/analysis/redundancy.py \
--pickle_dir=/raid/poggio/home/larend/pickles/imagenet/00004

singularity exec --nv -B /om:/om /raid/poggio/home/larend/localtensorflow.img \
python /raid/poggio/home/larend/robust/exps/analysis/robustness.py \
--model_dir=/raid/poggio/home/larend/models/robust/imagenet/00004 \
--scale_factor=4 \
--dataset=imagenet \
--pickle_dir=/raid/poggio/home/larend/pickles/imagenet/00004 \
--host_filesystem=/raid
