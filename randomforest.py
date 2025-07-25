import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 加载 CSV 数据
csv_path = 'data/mnist/MNIST/feature_file.csv'  # 根据你的实际路径修改
data = pd.read_csv(csv_path)

# 特征：跳过 filename、label、source 三列
features = data.iloc[:, 1:-3].values
labels = data['label'].values

# 划分：前10% + 后10% 测试，中间80%训练
total_len = len(data)
ten_percent = total_len // 10

test_indices = list(range(ten_percent)) + list(range(total_len - ten_percent, total_len))
train_indices = list(range(ten_percent, total_len - ten_percent))

X_train, y_train = features[train_indices], labels[train_indices]
X_test, y_test = features[test_indices], labels[test_indices]

# 训练 Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 基本评估
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# 从混淆矩阵提取 TP, TN, FP, FN
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp + 1e-8)
    detection_rate = tp / (tp + fn + 1e-8)
    false_positive_rate = fp / (fp + tn + 1e-8)

    print(f"\n✅ Accuracy: {acc:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Detection Rate (Recall): {detection_rate:.4f}")
    print(f"✅ False Positive Rate: {false_positive_rate:.4f}")
else:
    print("\n⚠️ 当前任务不是二分类，无法计算 Detection Rate 和 FPR。")

# 补充详细报告
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", cm)
