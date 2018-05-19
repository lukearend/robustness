Training ResNet on ImageNet.

Training notes ----------------------------------------------------------------
Training 4 on dgx; 3 on polestar; 0,1 on polestar-old 16,17

dgx:               (/raid/poggio/home/larend/robust/exps/imagenet/)
screen -R tf       (0: git, 1: tb, 2: train, 3: tail, 4: squeue, 5: models)
git pull origin master
bash tb.sh
sbatch --array=4 train-dgx.sh
tail -f out/4.out
watch -n 30 squeue

polestar:          (/om/user/larend/robust/exps/imagenet/)
screen -R tf       (0: git, 1: tb, 2: train, 3: nvidia-smi, 4: models)
git pull origin master
bash tb.sh
bash train-polestar.sh 3
watch -n 30 nvidia-smi


polestar-old:      (/cbcl/cbcl01/larend/robust/exps/imagenet/)
screen -R tf       (0: git, 1: tb, 2: train0, 3: train1, 4: nvidia-smi0, 5: nvidia-smi1, 6: models)
bash tb.sh                             (on gpu-16)
bash train-polestar-old.sh 0           (on gpu-16)
bash train-polestar-old.sh 1           (on gpu-17)
watch -n 30 nvidia-smi                 (on gpu-16)
watch -n 30 nvidia-smi                 (on gpu-17)

local:
screen -R tf       (0: dgx, 1: polestar, 2: polestar-old)
ssh -L 6006:dgx1:6050 larend@dgx1.mit.edu
ssh -L 6007:polestar:6050 larend@polestar.mit.edu
ssh -L 6008:gpu-16:6050 larend@polestar-old.mit.edu
