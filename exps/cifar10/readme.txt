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
-online suggests that people typically set batch norm decay to 0.9, not 0.997 as our script has by default. there are two possible alternatives:
    1) 0.997 is optimal for scale_factor = 1.0, but not for other scale factors
        -this would be bad because we would have to adjust regularization while changing redundancy
    2) 0.9 is optimal for scale_factor = 1.0, and the given parameter was just a bad one.
        -this would be good because it leaves open the possibility that we don't have to adjust this parameter (adjust regularization while changing redundancy).
-tried v1, scale_factor = 1.0 with decay = 0.997. demonstrated overfitting problem.
-tried v1, scale_factor = 1.0 with decay = 0.9. demonstrated overfitting problem.
-tried v2, scale_factor = 1.0 with decay = 0.9. demonstrated overfitting problem.
-set decay back to 0.997 and version back to v1. read on the docs for tf.layers.batch_normalization() something about update_ops and realized I may have missed a step in the implementation. maybe that's the issue? haven't investigated yet.
-could be that I'm using the contrib version of batch norm instead of the proper version (which doesn't allow fusing, sadly). trying the proper version without fusing to see if that works. v1, 0.997, scale factor 1.0.