import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from model_utils import ModelManager
import matplotlib.pyplot as plt
import numpy as np

# 数据加载
train = pd.read_csv('../data/train.csv', index_col=0)
test = pd.read_csv('../data/test.csv', index_col=0)

# 数据探索
print('训练集 Attrition 分布：')
print(train['Attrition'].value_counts())
print('\n训练集 Attrition 比例：')
print(train['Attrition'].value_counts(normalize=True))

# 处理Attrition字段, 可以使用map 进行自定义，也可以使用LabelEncoder进行自动的标签编码
train['Attrition']=train['Attrition'].map(lambda x:1 if x=='Yes' else 0)
print(train['Attrition'].value_counts())

from sklearn.preprocessing import LabelEncoder
# 查看数据是否有空值
print(train.isnull().sum())
# 如果方差为0, 没有意义
print(train['StandardHours'].value_counts())

# 去掉没用的列 员工号码，标准工时（=80）
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)
print(train.info())

# 对于分类特征进行特征值编码
attr=['BusinessTravel','Department','Education','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']
lbe_list=[]
# 在这个数据集中，测试集出现的标签，在训练集中都出现过
# 一般还可以，将训练集和测试集统一起来，一起进行fit_transform
for feature in attr:
    # 标签编码： 如果有10个类别，会编码成0-9
    lbe=LabelEncoder()
    # fit_transform = 先fit 再 transform
    # fit就是指定 LabelEncoder的关系，transform是应用这种LabelEncoder的关系进行编码
    train[feature]=lbe.fit_transform(train[feature])
    # 对测试集的特征值 不需要进行fit
    # 如果对测试集进行了fit, 训练集和测试集的lbe标准 就不一样了
    test[feature]=lbe.transform(test[feature])
    lbe_list.append(lbe)
#print(train)
train.to_csv('../data/train_label_encoder.csv')

# 建模环节，分类模型
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# 数据集进行切分，20%用于测试
X_train, X_valid, y_train, y_valid = train_test_split(train.drop('Attrition',axis=1), train['Attrition'], test_size=0.2, random_state=2025)

# 分类模型 二分类
# 为什么写random_state？如果不写random_state 每次运行的结果不同
model = LogisticRegression(max_iter=1000, random_state=42)
# 模型训练
model.fit(X_train, y_train)

# 模型评估
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_valid)
print(f"\n验证集准确率: {accuracy_score(y_valid, y_pred):.4f}")
print("分类报告:")
print(classification_report(y_valid, y_pred))

# 保存模型
model_manager = ModelManager()
feature_names = X_train.columns.tolist()
additional_info = {
    'max_iter': 1000,
    'penalty': 'l2',
    'C': 1.0,
    'validation_accuracy': accuracy_score(y_valid, y_pred)
}

saved_paths = model_manager.save_model(model, 'lr', feature_names, additional_info)

# To DO 还可以使用验证集，提前了解模型的效果
# 二分类结果，0或1
predict = model.predict(test)
print('标签Label：')
print(predict)
# 二分类任务 有2个概率值，label=0的概率， label=1的概率
print('标签概率')
predict = model.predict_proba(test)[:, 1]
print(predict)

# 保存预测结果
test_copy = pd.read_csv('../data/test.csv', index_col=0)
test_copy['Attrition'] = predict
test_copy[['Attrition']].to_csv('../data/submit_lr.csv')
print('结果已保存到 ../data/submit_lr.csv')

# 分析系数
import matplotlib.pyplot as plt

# 获取系数
coef = model.coef_[0]
feature_names = X_train.columns

# 打印系数
print("\nLogistic Regression Coefficients:")
for name, c in zip(feature_names, coef):
    print(f"{name}: {c:.4f}")

# 可视化系数
plt.figure(figsize=(12, 8))
colors = ['red' if c > 0 else 'blue' for c in coef]
plt.barh(feature_names, coef, color=colors)
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.title('Logistic Regression Coefficients\n(Red: Positive impact on attrition, Blue: Negative impact)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('../images/lr_coefficients.png', dpi=300, bbox_inches='tight')
print("\nCoefficient visualization saved as '../images/lr_coefficients.png'")

print(f"\nLR模型训练完成！模型已保存至: {saved_paths['model']}")