import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
from model_utils import ModelManager

# 数据加载
train = pd.read_csv('../data/train.csv', index_col=0)
test = pd.read_csv('../data/test.csv', index_col=0)

# 数据探索
print('训练集 Attrition 分布：')
print(train['Attrition'].value_counts())
print('\n训练集 Attrition 比例：')
print(train['Attrition'].value_counts(normalize=True))

# 处理Attrition字段
train['Attrition'] = train['Attrition'].map(lambda x: 1 if x == 'Yes' else 0)

# 去掉没用的列
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)

# One-hot编码分类特征（移除'Age'，因其为数值型）
categorical_features = ['BusinessTravel', 'Department', 'Education', 'EducationField',
                       'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']

train_encoded = pd.get_dummies(train, columns=categorical_features)
test_encoded = pd.get_dummies(test, columns=categorical_features)

# 确保训练和测试数据具有相同的列
for col in train_encoded.columns:
    if col not in test_encoded.columns and col != 'Attrition':
        test_encoded[col] = 0

for col in test_encoded.columns:
    if col not in train_encoded.columns and col != 'Attrition':
        train_encoded = train_encoded.drop(col, axis=1)

# 数据集切分
X_train, X_valid, y_train, y_valid = train_test_split(
    train_encoded.drop('Attrition', axis=1), train_encoded['Attrition'],
    test_size=0.2, random_state=42
)

# 模型参数
param = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'max_depth': 7,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 8,
    'lambda_l1': 0.6,
    'lambda_l2': 0,
    'is_unbalance': True  # 处理类别不平衡
}

# 创建数据集
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)

print("开始训练LightGBM (OneHot编码) 模型...")
# 训练模型
callbacks = [
    lgb.early_stopping(stopping_rounds=200),
    lgb.log_evaluation(period=25)
]

model = lgb.train(
    params=param,
    train_set=train_data,
    valid_sets=[train_data, valid_data],
    num_boost_round=10000,
    callbacks=callbacks
)

# 模型评估
valid_pred = model.predict(X_valid)
valid_pred_binary = (valid_pred >= 0.5).astype(int)
print(f"\n验证集准确率: {accuracy_score(y_valid, valid_pred_binary):.4f}")
print("分类报告:")
print(classification_report(y_valid, valid_pred_binary))

# 保存模型
model_manager = ModelManager()
feature_names = X_train.columns.tolist()
additional_info = {
    'model_type': 'LightGBM with OneHot Encoding',
    'learning_rate': 0.01,
    'max_depth': 7,
    'validation_accuracy': accuracy_score(y_valid, valid_pred_binary)
}

saved_paths = model_manager.save_model(model, 'lgb_onehot', feature_names, additional_info)

# 预测测试集
# 确保测试集特征与训练集一致
test_features = test_encoded.drop(columns=[col for col in test_encoded.columns if col not in X_train.columns])
predict_proba = model.predict(test_features)
predict = (predict_proba >= 0.5).astype(int)

# 保存预测结果
test_copy = pd.read_csv('../data/test.csv', index_col=0)
test_copy['Attrition'] = predict
test_copy[['Attrition']].to_csv('../data/submit_lgb_onehot.csv')
print('结果已保存到 ../data/submit_lgb_onehot.csv')

print(f"\nLightGBM-OneHot模型训练完成！模型已保存至: {saved_paths['model']}")
