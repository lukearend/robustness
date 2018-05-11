#!/bin/bash

module load openmind/singularity/older_versions/2.4

singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img \
python /om/user/larend/robust/exps/imagenet/train.py \
--model_index=$1 --host_filesystem=/om
