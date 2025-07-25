import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

# 自定义Dataset
class PdfDataset(Dataset):
    def __init__(self, csv_path, indices):
        data = pd.read_csv(csv_path)
        self.features = data.iloc[:, 1:-3].values.astype(np.float32)
        self.labels = data['label'].values.astype(np.int64)
        self.features = self.features[indices]
        self.labels = self.labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.labels[idx])
        return x.unsqueeze(0), y  # 增加channel维度 (B, 1, D)

# Simple CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self, input_len, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear((input_len // 4) * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # shape: (B, 8, L/2)
        x = self.pool(self.relu(self.conv2(x)))  # shape: (B, 16, L/4)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 训练函数
def train(model, loader, criterion, optimizer, device):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

# 测试函数
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    TP, FP, FN = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            TP += ((pred == 1) & (target == 1)).sum().item()
            FP += ((pred == 1) & (target == 0)).sum().item()
            FN += ((pred == 0) & (target == 1)).sum().item()

    acc = 100. * correct / total
    detection_rate = 100. * TP / (TP + FN + 1e-8)
    false_positive = 100. * FP / (FP + (total - TP - FP - FN) + 1e-8)
    return acc, detection_rate, false_positive

# 主函数
def main():
    csv_path = 'data/mnist/MNIST/feature_file.csv'
    df = pd.read_csv(csv_path)
    total_len = len(df)
    input_len = df.shape[1] - 4
    num_classes = len(set(df['label']))

    ten_percent = total_len // 10
    test_indices = list(range(ten_percent)) + list(range(total_len - ten_percent, total_len))
    train_indices = list(range(ten_percent, total_len - ten_percent))

    dataset_train = PdfDataset(csv_path, train_indices)
    dataset_test = PdfDataset(csv_path, test_indices)

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(input_len=input_len, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    os.makedirs('./save', exist_ok=True)
    acc_log_file = open('./save/cnn_accuracy_log.txt', 'w')

    for epoch in range(100):
        train(model, train_loader, criterion, optimizer, device)
        acc, detect, fp = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: Test Acc={acc:.2f}%, Detection Rate={detect:.2f}%, False Positive={fp:.2f}%")
        acc_log_file.write(f"{epoch+1}\t{acc:.4f}\n")

    acc_log_file.close()
    print("Accuracy log saved to ./save/cnn_accuracy_log.txt")

if __name__ == '__main__':
    main()
