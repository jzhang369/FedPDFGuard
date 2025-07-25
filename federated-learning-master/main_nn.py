import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar
from torch.utils.data import Dataset
import pandas as pd

# Custom Dataset class for PdfRep
class PdfDataset(Dataset):
    def __init__(self, csv_path):
        # Load the CSV data
        self.data = pd.read_csv(csv_path)
        self.features = self.data.iloc[:, 1:-3].values  # Exclude filename, label, and source columns
        self.labels = self.data['label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)

    # Load dataset and determine input size
    if args.dataset == 'pdfrep':
        full_dataset = PdfDataset('data/mnist/MNIST/feature_file.csv')
        img_size = full_dataset.features.shape[1]  # 12 features in your dataset
    else:
        exit('Error: unrecognized dataset')

    # Split dataset: 80% train, 20% test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    dataset_train, dataset_test = random_split(full_dataset, [train_size, test_size])

    # Build model
    if args.model == 'mlp':
        net_glob = MLP(dim_in=img_size, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: only MLP is currently supported for the custom dataset')
    
    print(net_glob)

    # Training setup
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

    list_loss = []
    net_glob.train()
    for epoch in range(args.epochs):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)

    # Plot training loss
    plt.figure()
    plt.plot(range(len(list_loss)), list_loss)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.savefig('./log/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))

    # Testing
    test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    print('Test on', len(dataset_test), 'samples')
    test_acc, test_loss = test(net_glob, test_loader)
