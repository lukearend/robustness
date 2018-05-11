Training ResNet on ImageNet.

Training notes ----------------------------------------------------------------
Training 4 on dgx; 3 on polestar; 0,1 on polestar-old 16,17

dgx:
bash tb.sh
sbatch --array=4 train-dgx.sh

polestar:
screen -R tf       (0: git, 1: tb, 2: train)
bash tb.sh
bash train-polestar.sh 3

polestar-old:
screen -R tf       (0: git, 1: tb, 2: train0 [gpu-16], 3: train1 [gpu-17])
bash tb.shj                            (on gpu-16)
bash train-polestar-old.sh 0           (on gpu-16)
bash train-polestar-old.sh 1           (on gpu-17)

om:
sbatch tb.sh
sbatch --array=2 --qos=cbmm train-om.sh

local:
screen -R tf       (0: dgx, 1: polestar, 2: polestar-old0, 3: polestar-old1)
ssh -L 6006:dgx1:6050 larend@dgx1.mit.edu
ssh -L 6007:polestar:6050 larend@polestar.mit.edu
ssh -L 6008:gpu-16:6050 larend@polestar-old.mit.edu
