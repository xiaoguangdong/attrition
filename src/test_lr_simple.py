"""
简单测试LR模型加载和预测
"""

from model_utils import ModelManager
import pandas as pd

def test_lr_model():
    # 加载测试数据
    test = pd.read_csv('../data/test.csv', index_col=0)

    # 去掉没用的列
    test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)

    # 对于分类特征进行特征值编码（与训练时保持一致）
    attr = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'Gender',
            'JobRole', 'MaritalStatus', 'Over18', 'OverTime']

    from sklearn.preprocessing import LabelEncoder
    lbe_list = []
    for feature in attr:
        lbe = LabelEncoder()
        # 注意：这里需要使用训练时的编码器，实际使用时应该保存编码器
        # 这里为了演示，使用测试数据的唯一值进行编码
        test[feature] = lbe.fit_transform(test[feature])
        lbe_list.append(lbe)

    # 初始化模型管理器
    manager = ModelManager()

    try:
        # 加载模型
        print("正在加载LR模型...")
        model, feature_names, metadata = manager.load_model('lr')

        print("✓ LR模型加载成功")
        print(f"模型类型: {metadata.get('model_type', 'Unknown') if metadata else 'Unknown'}")
        print(f"验证集准确率: {metadata.get('validation_accuracy', 'N/A') if metadata else 'N/A'}")

        # 进行预测
        print("正在进行预测...")
        predictions, probabilities = manager.predict(model, test, feature_names)

        print(f"✓ 预测完成，预测离职数: {predictions.sum()}")
        print(f"预测离职比例: {predictions.mean():.4f}")

        # 保存结果
        test_copy = pd.read_csv('../data/test.csv', index_col=0)
        test_copy['Attrition'] = predictions
        test_copy[['Attrition']].to_csv('test_lr_predictions.csv')
        print("✓ 结果已保存至: test_lr_predictions.csv")

    except Exception as e:
        print(f"✗ 错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lr_model()