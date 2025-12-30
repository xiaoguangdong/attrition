import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from model_utils import ModelManager

# 数据加载
train = pd.read_csv('../data/train.csv', index_col=0)
test = pd.read_csv('../data/test.csv', index_col=0)

# 处理Attrition字段
train['Attrition'] = train['Attrition'].map(lambda x: 1 if x == 'Yes' else 0)

# 去掉没用的列 员工号码，标准工时（=80）
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)

# 对于分类特征进行特征值编码
attr = ['Age', 'BusinessTravel', 'Department', 'Education', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
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

# GBDT分类器
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

print("开始训练GBDT模型...")
model.fit(X_train, y_train)

# 模型评估
valid_pred = model.predict(X_valid)
print(f"\n验证集准确率: {accuracy_score(y_valid, valid_pred):.4f}")
print("分类报告:")
print(classification_report(y_valid, valid_pred))

# 保存模型
model_manager = ModelManager()
feature_names = X_train.columns.tolist()
additional_info = {
    'model_type': 'GradientBoosting',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'validation_accuracy': accuracy_score(y_valid, valid_pred)
}

saved_paths = model_manager.save_model(model, 'gbdt', feature_names, additional_info)

# 预测测试集
predict_proba = model.predict_proba(test)[:, 1]
print("预测概率分布:", predict_proba[:10])

# 使用0.5作为阈值进行二分类预测
predict = (predict_proba >= 0.5).astype(int)

test_copy = pd.read_csv('../data/test.csv', index_col=0)
test_copy['Attrition'] = predict
test_copy[['Attrition']].to_csv('../data/submit_gbdt.csv')
print('结果已保存到 ../data/submit_gbdt.csv')

print(f"\nGBDT模型训练完成！模型已保存至: {saved_paths['model']}")
