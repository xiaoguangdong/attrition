import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
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

# 对于分类特征进行特征值编码
attr = ['BusinessTravel', 'Department', 'Education', 'EducationField',
        'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
for feature in attr:
    lbe = LabelEncoder()
    train[feature] = lbe.fit_transform(train[feature])
    test[feature] = lbe.transform(test[feature])

# 数据集切分
X_train, X_valid, y_train, y_valid = train_test_split(
    train.drop('Attrition', axis=1), train['Attrition'],
    test_size=0.2, random_state=42
)

# 模型训练
model = CatBoostClassifier(
    iterations=5000,
    depth=7,
    learning_rate=0.01,
    loss_function='Logloss',
    verbose=False  # 关闭训练过程输出
)
model.fit(X_train, y_train, cat_features=attr)

# 模型评估
y_pred = model.predict(X_valid)
print(f"\n验证集准确率: {accuracy_score(y_valid, y_pred):.4f}")
print("分类报告:")
print(classification_report(y_valid, y_pred))

# 保存模型
model_manager = ModelManager()
feature_names = X_train.columns.tolist()
additional_info = {
    'model_type': 'CatBoost',
    'iterations': 5000,
    'depth': 7,
    'learning_rate': 0.01,
    'validation_accuracy': accuracy_score(y_valid, y_pred)
}

saved_paths = model_manager.save_model(model, 'catboost', feature_names, additional_info)

# 预测测试集
predict_proba = model.predict_proba(test)[:, 1]
predict = (predict_proba >= 0.5).astype(int)

# 保存预测结果
test_copy = pd.read_csv('../data/test.csv', index_col=0)
test_copy['Attrition'] = predict
test_copy[['Attrition']].to_csv('../data/submit_cb.csv')
print('结果已保存到 ../data/submit_cb.csv')

print(f"\nCatBoost模型训练完成！模型已保存至: {saved_paths['model']}")