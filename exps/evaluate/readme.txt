Script for building the graph and evaluating a model.


example usage:
    for models with batch norm:
        `evaluate.py --model_dir=/om/user/larend/models/robust/cifar10/00000 --scale_factor=0.25 --dataset=cifar10`
    for models without batch norm:
        `evaluate.py --model_dir=/om/user/larend/models/robust/cifar10-no-bn/00000 --scale_factor=0.25 --dataset=cifar10 --disable_batch_norm`


arguments:
`--model_dir`: path to model directory to load
    for cifar10 models:
        `/om/user/larend/models/robust/cifar10/00000`
        `/om/user/larend/models/robust/cifar10/00001`
        `/om/user/larend/models/robust/cifar10/00002`
        `/om/user/larend/models/robust/cifar10/00003`
        `/om/user/larend/models/robust/cifar10/00004`
    for cifar10-no-bn (no batch norm) models:
        /om/user/larend/models/robust/cifar10-no-bn/00000
        ...
    for imagenet models:
        /om/user/larend/models/robust/imagenet/00000
        ...

`--scale_factor`: scale factor to use; should be
    `0.25` if model index is 00000
    `0.5`                    00001
    `1`                      00002
    `2`                      00003
    `4`                      00004

`--disable_batch_norm`: set this flag for the cifar10-no-bn to disable batch norm; otherwise, do not set this flag.
`--dataset`: `cifar10` or `imagenet`


example usage with slurm:
    `sbatch --qos=cbmm --mem=16000 --gres=gpu:tesla-k80:1 --job-name=robustness --output=out/cifar10-00000.out singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img python evaluate.py --model_dir=/om/user/larend/models/robust/cifar10/00000 --scale_factor=0.25 --dataset=cifar10`
