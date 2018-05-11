#!/bin/bash

source /cbcl/cbcl01/larend/envs/robust/bin/activate
module add cuda/8.0
module add cudnn/8.0-v5.1

python /cbcl/cbcl01/larend/robust/exps/imagenet/train.py \
--model_index=$1 --host_filesystem=/cbcl
