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
-could be that I'm using the contrib version of batch norm instead of the proper version (which doesn't allow fusing, sadly). trying the proper version without fusing to see if that works.
-v1, 0.997, scale factor 1.0. demonstrated overfitting problem.
-v1, 0.9, scale factor 1.0. demonstrated overfitting problem.
-maybe the best way to go about this is to try to replicate as precisely as possible the original script. so I'm using v2, resnet 50, with decay = 0.997, and the only difference as far as I can see is that i'm training on cifar and using the contrib version of fused batch norm available with 1.3. still had overfitting problem.
-let's be extreme. I went back to resnet-18 and set bn decay = 0.1. didn't seem to make a difference.
-tried setting training to True for batch norm function, even while evaluating (I saw someone mention this online). set bn decay back to 0.997. evaluation worked! but this can't be the way to fix the problem...
-saw some more stuff online about how you have to set tf.control dependencies and whatnot... trying this. seemed to improve evaluation...! still seems to overfit a bit but maybe not so badly.
-tried setting bn decay to 0.9, will see if this improves things even more. worked!!!!!!

More notes:
-going to train cifar with and without batch norm.
-when bypassing batch norm, the model instantly diverges with nan loss (lr still 0.1). so going to try reducing learning rate. found that 0.001 is good learning rate when batch norm is off.
-going to make sure v1 works with batch norm and lr back to 0.1. if it does, I'll do all experiments with v1 instead since it's supposedly better for resnet-18 (or not super deep resnets).
