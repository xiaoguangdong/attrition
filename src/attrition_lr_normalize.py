import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import warnings

# 忽略所有warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(train_path='train.csv', test_path='test.csv', scaler_type='standard'):
    """
    加载数据并进行预处理，包括标签编码和特征归一化

    参数:
    train_path: 训练数据路径
    test_path: 测试数据路径
    scaler_type: 归一化类型，'standard' 或 'minmax'

    返回:
    X_train_scaled, X_test_scaled, y_train, feature_names
    """
    # 加载数据
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)

    print("原始数据形状:")
    print(f"训练集: {train.shape}")
    print(f"测试集: {test.shape}")

    # 处理Attrition字段
    train['Attrition'] = train['Attrition'].map(lambda x: 1 if x == 'Yes' else 0)

    # 去掉没用的列
    train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
    test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)

    # 分类特征列表
    categorical_features = ['BusinessTravel', 'Department', 'Education', 'EducationField',
                           'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']

    # 标签编码分类特征
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        train[feature] = le.fit_transform(train[feature])
        test[feature] = le.transform(test[feature])
        label_encoders[feature] = le

    # 分离特征和标签
    X_train = train.drop('Attrition', axis=1)
    y_train = train['Attrition']
    X_test = test.copy()

    # 选择归一化方法
    if scaler_type == 'standard':
        scaler = StandardScaler()
        print("使用StandardScaler进行标准化")
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
        print("使用MinMaxScaler进行归一化")
    else:
        raise ValueError("scaler_type必须是'standard'或'minmax'")

    # 拟合和转换训练数据
    X_train_scaled = scaler.fit_transform(X_train)
    # 只转换测试数据（不重新拟合）
    X_test_scaled = scaler.transform(X_test)

    # 获取特征名称
    feature_names = X_train.columns.tolist()

    print(f"\n归一化后数据形状:")
    print(f"训练集: {X_train_scaled.shape}")
    print(f"测试集: {X_test_scaled.shape}")
    print(f"特征数量: {len(feature_names)}")

    # 打印归一化统计信息
    if scaler_type == 'standard':
        print("\n标准化统计:")
        print(f"均值: {scaler.mean_}")
        print(f"方差: {scaler.var_}")
    elif scaler_type == 'minmax':
        print("\n归一化范围:")
        print(f"最小值: {scaler.data_min_}")
        print(f"最大值: {scaler.data_max_}")

    return X_train_scaled, X_test_scaled, y_train.values, feature_names, scaler

def train_lr_and_analyze(X_train_scaled, X_test_scaled, y_train, feature_names):
    """
    训练Logistic Regression模型并分析特征重要性
    """
    # 训练LR模型
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = lr_model.predict(X_test_scaled)
    y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    # 评估模型
    print("\nLogistic Regression模型评估:")
    print(f"准确率: {accuracy_score(y_train, lr_model.predict(X_train_scaled)):.4f}")
    print("分类报告:")
    print(classification_report(y_train, lr_model.predict(X_train_scaled)))

    # 分析特征系数
    coefficients = lr_model.coef_[0]
    feature_importance = pd.DataFrame({
        '特征': feature_names,
        '系数': coefficients,
        '绝对系数': np.abs(coefficients)
    }).sort_values('绝对系数', ascending=False)

    print("\n特征重要性分析（按绝对系数排序）:")
    print(feature_importance.head(10))

    # 可视化前10个最重要的特征
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(10)
    plt.barh(top_features['特征'], top_features['系数'])
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.title('Logistic Regression Feature Coefficients (Normalized)')
    plt.tight_layout()
    plt.savefig('lr_normalized_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以避免显示

    return lr_model, feature_importance, y_pred_proba

def save_normalized_data(X_train_scaled, X_test_scaled, y_train, feature_names,
                        train_output='train_normalized.csv', test_output='test_normalized.csv'):
    """
    保存归一化后的数据
    """
    # 创建训练集DataFrame
    train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    train_df['Attrition'] = y_train

    # 创建测试集DataFrame
    test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    # 保存到CSV
    train_df.to_csv(train_output)
    test_df.to_csv(test_output)

    print(f"\n数据已保存:")
    print(f"训练集: {train_output}")
    print(f"测试集: {test_output}")

if __name__ == "__main__":
    # 示例使用
    print("开始数据归一化处理...")

    # 使用StandardScaler
    X_train_scaled, X_test_scaled, y_train, feature_names, scaler = load_and_preprocess_data(
        scaler_type='standard'
    )

    # 保存处理后的数据
    save_normalized_data(X_train_scaled, X_test_scaled, y_train, feature_names)

    # 训练LR模型并分析特征
    lr_model, feature_importance, y_pred_proba = train_lr_and_analyze(
        X_train_scaled, X_test_scaled, y_train, feature_names
    )

    print("\n处理完成！")

    # 可选：显示前5个特征的归一化前后对比
    print("\n特征归一化前后对比（前5个特征）:")
    original_train = pd.read_csv('../data/train.csv', index_col=0)
    original_train['Attrition'] = original_train['Attrition'].map(lambda x: 1 if x == 'Yes' else 0)
    original_train = original_train.drop(['EmployeeNumber', 'StandardHours', 'Attrition'], axis=1)

    print("原始数据统计:")
    print(original_train.iloc[:, :5].describe())

    print("\n归一化后数据统计:")
    normalized_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    print(normalized_df.iloc[:, :5].describe())