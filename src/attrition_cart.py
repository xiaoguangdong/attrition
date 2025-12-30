import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree
import warnings
import os
from datetime import datetime

# 忽略所有warnings
warnings.filterwarnings('ignore')

# 导入模型管理器
from model_utils import ModelManager

def load_and_preprocess_data(train_path='../data/train.csv', test_path='../data/test.csv'):
    """
    加载数据并进行预处理，包括标签编码

    参数:
    train_path: 训练数据路径
    test_path: 测试数据路径

    返回:
    X_train, X_test, y_train, feature_names
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

    # 对分类特征进行标签编码
    attr = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
    for feature in attr:
        lbe = LabelEncoder()
        train[feature] = lbe.fit_transform(train[feature])
        test[feature] = lbe.transform(test[feature])

    # 分离特征和标签
    X_train = train.drop('Attrition', axis=1)
    y_train = train['Attrition']
    X_test = test

    return X_train, X_test, y_train, X_train.columns.tolist()

def main():
    """
    主函数
    """
    print("开始训练CART决策树模型...")
    
    # 加载和预处理数据
    X_train, X_test, y_train, feature_names = load_and_preprocess_data()
    
    # 数据集切分
    X_train_split, X_valid, y_train_split, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_split, y_train_split)
    
    # 模型评估
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"\n验证集准确率: {accuracy:.4f}")
    print("分类报告:")
    print(classification_report(y_valid, y_pred))
    
    # 可视化
    os.makedirs('../images', exist_ok=True)
    
    # 决策树可视化
    plt.figure(figsize=(20, 10))
    tree.plot_tree(model, 
                   feature_names=feature_names,
                   class_names=['No Attrition', 'Attrition'],
                   filled=True,
                   rounded=True,
                   fontsize=10)
    plt.title('CART Decision Tree for Employee Attrition Prediction')
    plt.savefig('../images/cart_decision_tree.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"决策树图已保存至: ../images/cart_decision_tree.png")
    
    # 特征重要性可视化
    feature_importance = model.feature_importances_
    indices = sorted(range(len(feature_importance)), key=lambda i: feature_importance[i], reverse=True)
    
    plt.figure(figsize=(10, 6))
    plt.title("Top Feature Importances in CART Model")
    plt.bar(range(len(feature_importance)), [feature_importance[i] for i in indices])
    plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('../images/cart_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征重要性图已保存至: ../images/cart_feature_importance.png")
    
    # 保存模型
    model_manager = ModelManager()
    additional_info = {
        'model_type': 'CART Decision Tree',
        'max_depth': 5,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'validation_accuracy': accuracy
    }
    
    saved_paths = model_manager.save_model(model, 'cart', feature_names, additional_info)
    
    # 预测测试集
    predict_proba = model.predict_proba(X_test)[:, 1]
    predict = (predict_proba >= 0.5).astype(int)
    
    # 保存预测结果
    test_copy = pd.read_csv('../data/test.csv', index_col=0)
    test_copy['Attrition'] = predict
    test_copy[['Attrition']].to_csv('../data/submit_cart.csv')
    print('\n结果已保存到 ../data/submit_cart.csv')
    
    print(f"\nCART模型训练完成！模型已保存至: {saved_paths['model']}")

if __name__ == '__main__':
    main()