# DCP++

DCP++ performs almost twice as well as DCP and provides more robust ICP initializations.

## Prerequisites 

PyTorch>=1.0: https://pytorch.org

scipy>=1.2 

numpy

h5pytqdm

TensorboardX: https://github.com/lanpa/tensorboardX

## Training

### DCP

python main.py --exp_name=dcp --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd

### DCP++

python main.py --exp_name=dcp++ --model=dcp++ --emb_nn=dgcnn --pointer=transformer --head=svd

## Testing

### DCP

python main.py --exp_name=dcp --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval

### DCP++

python main.py --exp_name=dcp++ --model=dcp++ --emb_nn=dgcnn --pointer=transformer --head=svd --eval
