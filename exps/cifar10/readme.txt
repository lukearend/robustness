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


Debugging notes:
-problem: slightly above chance on validation set, while perfectly fitting training set.
-printed out labels to see if it's a data problem - didn't appear to be anything unusual about the labels despite the fact that I thought I saw repeat images
-tried LeNet instead of ResNet with this same framework and it worked - suggests the issue is within the ResNet implementation. my suspicion is it's batch norm.
-tried ResNet with bypassing the batch norm function using tf.identity. the model instantly diverged with NaN loss when I tried training.
-another reason to suspect batch norm: I'm not using the batch norm function provided with the implementation of the network, as I'm in an older version of tensorflow. I've been using tf.contrib.layers.batch_norm(), but will try something else.
-discussed with xavier. probably an issue with the batch norm decay parameter (a regularizer). should see if it works for the 1x model but not the 4x model, for instance.
