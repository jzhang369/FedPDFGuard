import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler

# 加载数据
csv_path = 'data/mnist/MNIST/feature_file.csv'  # ← 修改为你的实际路径
data = pd.read_csv(csv_path)

# 选择特征：排除 filename, label, type, source (假设这些是最后三列)
features = data.iloc[:, 1:-3].values
labels = data['label'].values

# 划分数据（前10% + 后10%测试，其余训练）
total_len = len(data)
ten_percent = total_len // 10
test_indices = list(range(ten_percent)) + list(range(total_len - ten_percent, total_len))
train_indices = list(range(ten_percent, total_len - ten_percent))

X_train = features[train_indices]
y_train = labels[train_indices]
X_test = features[test_indices]
y_test = labels[test_indices]

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义并训练 SVM 模型
clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# 计算指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, pos_label=1)  # Detection Rate
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

# 打印结果
print("✅ SVM Evaluation Results:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Detection Rate (Recall): {recall * 100:.2f}%")
print(f"False Positive Rate: {fpr * 100:.2f}%")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
