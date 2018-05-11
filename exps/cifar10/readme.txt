Training ResNet on CIFAR-10.

Testing notes -----------------------------------------------------------------
Debugging on dgx.

dgx:               (/raid/poggio/home/larend/robust/exps/cifar10/)
screen -R tf       (0: git, 1: tb, 2: train, 3: tail, 4: squeue, 5: models)
git pull origin master
bash tb.sh
sbatch --array=4 train-dgx.sh
tail -f out/4.out
watch -n 30 squeue