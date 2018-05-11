Training ResNet on CIFAR-10.

Testing notes -----------------------------------------------------------------
Debugging on dgx.

dgx:               (/raid/poggio/home/larend/robust/exps/cifar10/)
screen -R debug    (0: git, 1: tb, 2: train, 3: tail, 4: squeue, 5: models)
git pull origin master
bash tb.sh 6051
sbatch --array=0 train-dgx.sh
tail -f out/0.out
watch -n 30 squeue

local:
screen -R debug    (0: dgx)
ssh -L 6009:dgx1:6051 larend@dgx1.mit.edu
