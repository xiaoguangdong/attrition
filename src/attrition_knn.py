import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib

# 数据加载
train = pd.read_csv('../data/train.csv', index_col=0)
test = pd.read_csv('../data/test.csv', index_col=0)

# 数据探索
print(train['Attrition'].value_counts())

# 处理Attrition字段
train['Attrition'] = train['Attrition'].map(lambda x: 1 if x == 'Yes' else 0)
print(train['Attrition'].value_counts())

# 查看数据是否有空值
print(train.isnull().sum())
# 如果方差为0, 没有意义
print(train['StandardHours'].value_counts())

# 去掉没用的列 员工号码，标准工时（=80）
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)
print(train.info())

# 对于分类特征进行特征值编码
attr = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
lbe_list = []
for feature in attr:
    lbe = LabelEncoder()
    train[feature] = lbe.fit_transform(train[feature])
    test[feature] = lbe.transform(test[feature])
    lbe_list.append(lbe)

train.to_csv('../data/train_label_encoder.csv')

# 建模环节，KNN分类模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 数据集进行切分，20%用于测试
X_train, X_valid, y_train, y_valid = train_test_split(
    train.drop('Attrition', axis=1), 
    train['Attrition'], 
    test_size=0.2, 
    random_state=2025
)

# 特征标准化，因为KNN基于距离，对特征尺度敏感
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
test_scaled = scaler.transform(test)

# KNN分类模型
model = KNeighborsClassifier(
    n_neighbors=5,      # 邻居数量
    weights='uniform',  # 权重方式：'uniform'或'distance'
    algorithm='auto',   # 算法：'auto', 'ball_tree', 'kd_tree', 'brute'
    leaf_size=30,       # 叶子大小，用于ball_tree和kd_tree
    p=2,                # 距离度量：1为曼哈顿，2为欧几里得
    metric='minkowski', # 距离度量
    n_jobs=-1           # 并行计算
)

# 模型训练
model.fit(X_train_scaled, y_train)

# 二分类结果，0或1
predict = model.predict(test_scaled)
print('标签Label：')
print(predict)

# KNN也可以输出概率（基于邻居投票）
print('标签概率')
predict_proba = model.predict_proba(test_scaled)[:, 1]
print(predict_proba)

# 使用概率进行预测，并应用阈值调整
target_ratio = 0.16  # 目标比例
n_samples = len(test)
target_positive = int(n_samples * target_ratio)

# 根据概率排序，选择前 target_positive 个样本作为正类
threshold = sorted(predict_proba)[-target_positive] if target_positive > 0 else 0.5

# 使用调整后的阈值进行预测
predict_adjusted = (predict_proba >= threshold).astype(int)

# 输出结果统计
print('\n预测结果统计：')
print('预测的 Attrition 数量：', sum(predict_adjusted))
print('预测的 Attrition 比例：', sum(predict_adjusted) / len(predict_adjusted))
print('使用的阈值：', threshold)

# 保存结果 - 重新加载原始测试集以确保ID正确
test_copy = pd.read_csv('../data/test.csv', index_col=0)
test_copy['Attrition'] = predict_adjusted
test_copy[['Attrition']].to_csv('../data/submit_knn.csv')
print('\n结果已保存到 ../data/submit_knn.csv')