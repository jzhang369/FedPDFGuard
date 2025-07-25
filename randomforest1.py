import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from joblib import Parallel, delayed
import os

# 加载 CSV 数据
csv_path = 'data/mnist/MNIST/feature_file.csv'
data = pd.read_csv(csv_path)

# 特征与标签
features = data.iloc[:, 1:-3].values
labels = data['label'].values

# 划分数据集
total_len = len(data)
ten_percent = total_len // 10
test_indices = list(range(ten_percent)) + list(range(total_len - ten_percent, total_len))
train_indices = list(range(ten_percent, total_len - ten_percent))

X_train, y_train = features[train_indices], labels[train_indices]
X_test, y_test = features[test_indices], labels[test_indices]

# 保存目录
os.makedirs('./save', exist_ok=True)

# 单次 Random Forest 训练测试函数
def run_rf_once(run_id):
    clf = RandomForestClassifier(n_estimators=100, random_state=None)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, pos_label=1)

    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn + 1e-8)
    else:
        fpr = 0.0

    print(f"✅ Finished run {run_id}")
    return run_id, acc, recall, fpr

# 并行执行 100 次
results = Parallel(n_jobs=-1)(delayed(run_rf_once)(i) for i in range(1, 101))

# 保存到 txt
with open('./save/rf_parallel_accuracy.txt', 'w') as f:
    for run_id, acc, _, _ in results:
        f.write(f"{run_id}\t{acc:.4f}\n")

# 统计平均值
all_acc = [r[1] for r in results]
all_recall = [r[2] for r in results]
all_fpr = [r[3] for r in results]

print("\n🎯 100 runs complete (Random Forest + multithreading).")
print(f"Average Accuracy: {np.mean(all_acc) * 100:.2f}%")
print(f"Average Detection Rate (Recall): {np.mean(all_recall) * 100:.2f}%")
print(f"Average False Positive Rate: {np.mean(all_fpr) * 100:.2f}%")
