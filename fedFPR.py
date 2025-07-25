import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP
from models.Fed import FedAvg
from models.test import test_img

# Custom Dataset class for PdfRep
class PdfDataset(Dataset):
    def __init__(self, csv_path, indices=None):
        self.data = pd.read_csv(csv_path)
        self.features = self.data.iloc[:, 1:-3].values
        self.labels = self.data['label'].values
        if indices is not None:
            self.features = self.features[indices]
            self.labels = self.labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def custom_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    all_idxs = np.arange(len(dataset))
    dict_users = {i: set(np.random.choice(all_idxs, num_items, replace=False)) for i in range(num_users)}
    return dict_users

def custom_noniid(dataset, num_users):
    dict_users = {i: np.array([]) for i in range(num_users)}
    all_idxs = np.arange(len(dataset))
    labels = dataset.labels
    num_shards = num_users * 2
    shards = np.array_split(all_idxs, num_shards)
    for i in range(num_users):
        dict_users[i] = np.concatenate([shards[i], shards[num_users + i]])
    return dict_users

def calculate_detection_metrics(model, dataset, device):
    model.eval()
    all_preds, all_labels = [], []
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        detection_rate = false_positive_rate = 0.0
    return detection_rate, false_positive_rate

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'pdfrep':
        full_dataset = PdfDataset('data/mnist/MNIST/feature_file.csv')
        total_len = len(full_dataset)
        ten_percent = total_len // 10
        test_indices = list(range(ten_percent)) + list(range(total_len - ten_percent, total_len))
        train_indices = list(range(ten_percent, total_len - ten_percent))
        dataset_train = PdfDataset('data/mnist/MNIST/feature_file.csv', indices=train_indices)
        dataset_test = PdfDataset('data/mnist/MNIST/feature_file.csv', indices=test_indices)
        if args.iid:
            dict_users = custom_iid(dataset_train, args.num_users)
        else:
            dict_users = custom_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')

    input_size = dataset_train.features.shape[1]
    if args.model == 'mlp':
        net_glob = MLP(dim_in=input_size, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: only MLP is currently supported for the custom dataset')

    print(net_glob)
    net_glob.train()
    w_glob = net_glob.state_dict()

    loss_train = []
    test_accuracies = []
    fpr_list = []
    fpr_epochs = []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

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

        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # Accuracy after each epoch
        net_glob.eval()
        acc_test, _ = test_img(net_glob, dataset_test, args)
        test_accuracies.append(acc_test)

        if not os.path.exists('./save'):
            os.makedirs('./save')

        with open('./save/accuracy_log.txt', 'w') as f:
            for epoch, acc in enumerate(test_accuracies, 1):
                f.write(f"{epoch}\t{acc:.4f}\n")
        print("Accuracy log saved to ./save/accuracy_log.txt")

        # FPR after each epoch
        _, fpr = calculate_detection_metrics(net_glob, dataset_test, args.device)
        fpr_list.append(fpr)
        fpr_epochs.append(iter + 1)
        with open('./save/fpr_log.txt', 'a') as f:
            f.write(f"{iter + 1}\t{fpr:.4f}\n")
        print(f"Epoch {iter + 1} - FPR: {fpr:.4f}")

    # Final overall evaluation
    net_glob.eval()
    acc_train, _ = test_img(net_glob, dataset_train, args)
    acc_test, _ = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    detection_rate, false_positive_rate = calculate_detection_metrics(net_glob, dataset_test, args.device)
    print("Detection Rate (Recall): {:.2f}".format(detection_rate * 100))
    print("False Positive Rate: {:.2f}".format(false_positive_rate * 100))

    # Find epoch with lowest FPR
    if fpr_list:
        min_fpr = min(fpr_list)
        min_index = fpr_list.index(min_fpr)
        best_epoch = fpr_epochs[min_index]
        print(f"\n✅ Lowest FPR is at Epoch {best_epoch}: FPR = {min_fpr:.4f}")