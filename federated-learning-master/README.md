# Federated Learning for Malicious PDF Detection


Only experiments on MNIST and CIFAR10 (both IID and non-IID) is produced by far.

Note: The scripts will be slow without the implementation of parallel computing. 

## Requirements
python>=3.6  
pytorch>=0.4

## Run

The CNN models is produced by:
> python [main_nn.py](main_nn.py)

Federated learning with CNN is produced by:
> python [main_fed.py](main_fed.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  

`--all_clients` for averaging over all client models

Datasets: PdfRep

## Results
### PdfRep
Results are shown in Table 1 cmpared with centralized CNN, SVM and RF.
Table 1. results of 10 epochs training with the learning rate of 0.01

                       Accuracy%                   DR%(Detection Rate) FP%              (False Positive)
FL                        98.9                              99.52                             1.71
CNN                       99.54                             99.61                             0.53
SVM                       99.32                             99.46                             0.82
Random Forest             99.86                             99.77                             0.04


## Ackonwledgements
Acknowledgements give to [Junjie Zhang](https://github.com/***).


M
## Cite As



