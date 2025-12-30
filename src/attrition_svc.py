import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from model_utils import ModelManager

# 数据加载
train = pd.read_csv('../data/train.csv', index_col=0)
test1 = pd.read_csv('../data/test.csv', index_col=0)
test = test1.copy()

# 数据探索
print('训练集 Attrition 分布：')
print(train['Attrition'].value_counts())
print('\n训练集 Attrition 比例：')
print(train['Attrition'].value_counts(normalize=True))

# 处理Attrition字段
train['Attrition'] = train['Attrition'].map(lambda x: 1 if x == 'Yes' else 0)

# 去掉没用的列 员工号码，标准工时（=80）
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)

# 对于分类特征进行特征值编码
attr = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'Gender', 
        'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
lbe_list = []
for feature in attr:
    lbe = LabelEncoder()
    train[feature] = lbe.fit_transform(train[feature])
    test[feature] = lbe.transform(test[feature])
    lbe_list.append(lbe)

# 数据集切分
X_train, X_valid, y_train, y_valid = train_test_split(
    train.drop('Attrition', axis=1), 
    train['Attrition'], 
    test_size=0.2, 
    random_state=42
)

# 归一化
mms = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = mms.fit_transform(X_train)
X_valid_scaled = mms.transform(X_valid)
test_scaled = mms.transform(test)

# 高维映射，启用概率预测
model = SVC(
    kernel='rbf',
    gamma="auto",
    max_iter=1000,
    random_state=33,
    verbose=True,
    tol=1e-5,
    cache_size=50000,
    probability=True  # 启用概率预测
)

print("开始训练SVC模型...")
model.fit(X_train_scaled, y_train)

# 模型评估
valid_pred = model.predict(X_valid_scaled)
print(f"\n验证集准确率: {accuracy_score(y_valid, valid_pred):.4f}")
print("分类报告:")
print(classification_report(y_valid, valid_pred))

# 保存模型
model_manager = ModelManager()
feature_names = X_train.columns.tolist()
additional_info = {
    'model_type': 'SVC',
    'kernel': 'rbf',
    'gamma': 'auto',
    'max_iter': 1000,
    'scaler': 'MinMaxScaler',
    'validation_accuracy': accuracy_score(y_valid, valid_pred)
}

saved_paths = model_manager.save_model(model, 'svc', feature_names, additional_info, preprocessor=mms)

# 获取预测概率
proba = model.predict_proba(test_scaled)[:, 1]

# 计算目标阈值
target_ratio = 0.16  # 目标比例
n_samples = len(test_scaled)
target_positive = int(n_samples * target_ratio)

# 根据概率排序，选择前 target_positive 个样本作为正类
threshold = np.sort(proba)[-target_positive]

# 使用调整后的阈值进行预测
predict = (proba >= threshold).astype(int)

# 输出结果统计
print('\n预测结果统计：')
print('预测的 Attrition 数量：', np.sum(predict))
print('预测的 Attrition 比例：', np.sum(predict) / len(predict))
print('使用的阈值：', threshold)

# 保存结果
test1['Attrition'] = predict
test1[['Attrition']].to_csv('../data/submit_svc.csv')
print('\n结果已保存到 ../data/submit_svc.csv')

print(f"\nSVC模型训练完成！模型已保存至: {saved_paths['model']}")

# 高维映射，启用概率预测
model = SVC(
    kernel='rbf',
    gamma="auto",
    max_iter=1000,
    random_state=33,
    verbose=True,
    tol=1e-5,
    cache_size=50000,
    probability=True  # 启用概率预测
)
#print(X_train)
#print(y_train)
#print(sum(y_train))
"""
model = LinearSVC(
			max_iter=1000,
			random_state=33,
			verbose=True,
		   )
"""
model.fit(X_train_scaled, y_train)

# 模型评估
valid_pred = model.predict(X_valid_scaled)
print(f"\n验证集准确率: {accuracy_score(y_valid, valid_pred):.4f}")
print("分类报告:")
print(classification_report(y_valid, valid_pred))

# 保存模型
model_manager = ModelManager()
feature_names = X_train.columns.tolist()
additional_info = {
    'model_type': 'SVC',
    'kernel': 'rbf',
    'gamma': 'auto',
    'max_iter': 1000,
    'scaler': 'MinMaxScaler',
    'validation_accuracy': accuracy_score(y_valid, valid_pred)
}

saved_paths = model_manager.save_model(model, 'svc', feature_names, additional_info, preprocessor=mms)

# 获取预测概率
proba = model.predict_proba(test_scaled)[:, 1]

# 计算目标阈值
target_ratio = 0.16  # 目标比例
n_samples = len(test_scaled)
target_positive = int(n_samples * target_ratio)

# 根据概率排序，选择前 target_positive 个样本作为正类
threshold = np.sort(proba)[-target_positive]

# 使用调整后的阈值进行预测
predict = (proba >= threshold).astype(int)

# 输出结果统计
print('\n预测结果统计：')
print('预测的 Attrition 数量：', np.sum(predict))
print('预测的 Attrition 比例：', np.sum(predict) / len(predict))
print('使用的阈值：', threshold)

# 保存结果
test1['Attrition'] = predict
test1[['Attrition']].to_csv('../data/submit_svc.csv')
print('\n结果已保存到 submit_svc.csv')

print(f"\nSVC模型训练完成！模型已保存至: {saved_paths['model']}")
