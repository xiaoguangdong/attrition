import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from model_utils import ModelManager

# 数据加载
train=pd.read_csv('../data/train.csv',index_col=0)
test=pd.read_csv('../data/test.csv',index_col=0)

# 处理Attrition字段
train['Attrition']=train['Attrition'].map(lambda x:1 if x=='Yes' else 0)

# 去掉没用的列 员工号码，标准工时（=80）
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)

# 对于分类特征进行特征值编码
attr=['Age','BusinessTravel','Department','Education','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']
lbe_list=[]
for feature in attr:
    lbe=LabelEncoder()
    train[feature]=lbe.fit_transform(train[feature])
    test[feature]=lbe.transform(test[feature])
    lbe_list.append(lbe)

# 数据集切分
X_train, X_valid, y_train, y_valid = train_test_split(
    train.drop('Attrition',axis=1), train['Attrition'],
    test_size=0.2, random_state=42
)

# XGBoost参数设置
param = {
    'boosting_type':'gbdt',
    'objective' : 'binary:logistic', #目标函数
    'eval_metric' : 'auc', # 评价指标
    'eta' : 0.01,
    'max_depth' : 6,
    'colsample_bytree':0.8,
    'subsample': 0.9,
    'subsample_freq': 8,
    'alpha': 0.6,
    'lambda': 0
}

# 转换为XGBoost数据格式
train_data = xgb.DMatrix(X_train, label=y_train)
valid_data = xgb.DMatrix(X_valid, label=y_valid)
test_data = xgb.DMatrix(test)

# 模型训练
print("开始训练XGBoost模型...")
model = xgb.train(
    param, train_data,
    evals=[(train_data, 'train'), (valid_data, 'valid')],
    num_boost_round=1000,
    early_stopping_rounds=10,
    verbose_eval=25
)

# 模型评估
valid_pred_prob = model.predict(valid_data)
valid_pred = (valid_pred_prob >= 0.5).astype(int)
print(f"\n验证集准确率: {accuracy_score(y_valid, valid_pred):.4f}")
print("分类报告:")
print(classification_report(y_valid, valid_pred))

# 保存模型
model_manager = ModelManager()
feature_names = X_train.columns.tolist()
additional_info = {
    'model_type': 'XGBoost',
    'parameters': param,
    'best_iteration': model.best_iteration,
    'best_score': model.best_score,
    'validation_accuracy': accuracy_score(y_valid, valid_pred)
}

# 自定义模型保存路径
saved_paths = model_manager.save_model(
    model,
    'xgboost',
    feature_names,
    additional_info,
    model_path='../models/xgboost_model.pkl'
)

# 预测测试集
predict = model.predict(test_data)
test_copy = pd.read_csv('../data/test.csv', index_col=0)
test_copy['Attrition'] = predict
test_copy[['Attrition']].to_csv('../data/submit_xgb.csv')
print('结果已保存到 ../data/submit_xgb.csv')

print(f"\nXGBoost模型训练完成！模型已保存至: {saved_paths['model']}")
print(f"预测结果已保存至: ../data/submit_xgb.csv")