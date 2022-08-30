# DCP++

DCP++ performs almost twice as well as DCP and provides more robust ICP initializations.

## Recursive Clone

```
git clone --recursive https://github.com/stoianmihail/DCP2.0.git
cd DCP2.0
```

## Requirements 

```
pip3 install -r requirements.txt
```

## Training

### DCP

```
python main.py --exp_name=dcp --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd
```

### DCP++

```
python main.py --exp_name=dcp++ --model=dcp++ --emb_nn=dgcnn --pointer=transformer --head=svd
```

## Testing

### DCP

```
python main.py --exp_name=dcp --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval
```

### DCP++

```
python main.py --exp_name=dcp++ --model=dcp++ --emb_nn=dgcnn --pointer=transformer --head=svd --eval
```
