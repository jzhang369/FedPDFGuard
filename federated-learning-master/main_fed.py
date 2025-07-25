import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP
from models.Fed import FedAvg
from models.test import test_img

# Custom Dataset class for PdfRep
class PdfDataset(Dataset):
    def __init__(self, csv_path, indices=None):
        # Load the CSV data
        self.data = pd.read_csv(csv_path)
        self.features = self.data.iloc[:, 1:-3].values  # Exclude filename, label, and source columns
        self.labels = self.data['label'].values
        
        # Use only specified indices (for train/test split)
        if indices is not None:
            self.features = self.features[indices]
            self.labels = self.labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# Function for IID data split
def custom_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    all_idxs = np.arange(len(dataset))
    dict_users = {i: set(np.random.choice(all_idxs, num_items, replace=False)) for i in range(num_users)}
    return dict_users

# Function for non-IID data split
def custom_noniid(dataset, num_users):
    dict_users = {i: np.array([]) for i in range(num_users)}
    all_idxs = np.arange(len(dataset))
    
    # Partition based on labels for non-IID (example with shards)
    labels = dataset.labels
    num_shards = num_users * 2  # Adjust the number of shards per user
    shards = np.array_split(all_idxs, num_shards)
    
    for i in range(num_users):
        dict_users[i] = np.concatenate([shards[i], shards[num_users + i]])
    
    return dict_users

if __name__ == '__main__':
    # parse args
    args = args_parser()
    
    # Set the device to GPU if available, otherwise use CPU
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and split into 80% training and 20% testing
    if args.dataset == 'pdfrep':
        full_dataset = PdfDataset('data/mnist/MNIST/feature_file.csv')
        
        # Split indices for 80% train and 20% test
        indices = list(range(len(full_dataset)))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=args.seed)
        
        # Create train and test datasets based on the split indices
        dataset_train = PdfDataset('data/mnist/MNIST/feature_file.csv', indices=train_indices)
        dataset_test = PdfDataset('data/mnist/MNIST/feature_file.csv', indices=test_indices)
        
        

        # Sample users
        if args.iid:
            dict_users = custom_iid(dataset_train, args.num_users)
        else:
            dict_users = custom_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    
    # Determine input size for MLP
    input_size = dataset_train.features.shape[1]

    # Build model with the correct input size
    if args.model == 'mlp':
        net_glob = MLP(dim_in=input_size, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: only MLP is currently supported for the custom dataset')
    
    print(net_glob)
    net_glob.train()

    # Copy initial weights
    w_glob = net_glob.state_dict()

    # Training variables
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    
    # Federated Training
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # Update global weights
        w_glob = FedAvg(w_locals)

        # Copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # Print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # Ensure the save directory exists
    if not os.path.exists('./save'):
        os.makedirs('./save')

    # Plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # Testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    print(dataset_test)

    