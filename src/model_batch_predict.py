"""
员工离职预测模型 - 批量预测示例
展示如何加载所有保存的模型并进行批量预测
"""

import pandas as pd
import numpy as np
from model_utils import ModelManager
import os

def load_and_predict_all_models():
    """
    加载所有保存的模型并对测试数据进行预测
    """
    # 初始化模型管理器
    model_manager = ModelManager()

    # 数据加载和预处理
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

    # 模型列表
    model_names = ['lr', 'lr_threshold', 'xgboost', 'lgb', 'lgb_onehot', 'catboost', 'svc', 'gbdt', 'ngboost']

    # 存储所有预测结果
    all_predictions = {}

    print("开始加载模型并进行预测...\n")

    for model_name in model_names:
        try:
            print(f"正在加载 {model_name} 模型...")

            # 加载模型
            loaded_data = model_manager.load_model(model_name)
            model = loaded_data[0]
            metadata = loaded_data[1]

            # 根据模型类型进行不同的预处理
            if model_name == 'svc':
                # SVC需要归一化
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(0, 1))
                test_scaled = scaler.fit_transform(test)  # 注意：实际应该使用训练时的scaler
                predictions = model.predict(test_scaled)
                pred_proba = model.predict_proba(test_scaled)[:, 1]

            elif model_name == 'lgb_onehot':
                # OneHot编码的LightGBM
                from sklearn.feature_extraction import DictVectorizer
                dvec = DictVectorizer(sparse=False)
                test_encoded = dvec.fit_transform(test.to_dict(orient='record'))  # 注意：实际应该使用训练时的encoder
                pred_proba = model.predict(test_encoded)
                predictions = (pred_proba >= 0.5).astype(int)

            else:
                # 其他模型直接预测
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(test)[:, 1]
                    predictions = (pred_proba >= 0.5).astype(int)
                else:
                    predictions = model.predict(test)
                    pred_proba = predictions  # 对于没有概率的模型

            # 存储预测结果
            all_predictions[model_name] = {
                'predictions': predictions,
                'probabilities': pred_proba,
                'metadata': metadata
            }

            # 输出模型信息
            print(f"✓ {model_name} 模型加载成功")
            print(f"  模型类型: {metadata.get('model_type', 'Unknown')}")
            print(f"  验证集准确率: {metadata.get('validation_accuracy', 'N/A'):.4f}")
            print(f"  预测离职比例: {predictions.mean():.4f}")
            print()

        except Exception as e:
            print(f"✗ {model_name} 模型加载失败: {str(e)}")
            print()

    return all_predictions

def create_comparison_report(predictions):
    """
    创建模型对比报告
    """
    print("=== 模型预测对比报告 ===\n")

    # 创建结果DataFrame
    results_df = pd.DataFrame()

    for model_name, pred_data in predictions.items():
        results_df[f'{model_name}_pred'] = pred_data['predictions']
        results_df[f'{model_name}_prob'] = pred_data['probabilities']

    # 计算统计信息
    summary_stats = []
    for model_name, pred_data in predictions.items():
        stats = {
            '模型': model_name,
            '预测离职数': int(pred_data['predictions'].sum()),
            '预测离职比例': pred_data['predictions'].mean(),
            '平均预测概率': pred_data['probabilities'].mean(),
            '验证集准确率': pred_data['metadata'].get('validation_accuracy', 'N/A')
        }
        summary_stats.append(stats)

    summary_df = pd.DataFrame(summary_stats)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    print()

    # 保存详细结果
    results_df.to_csv('../data/model_predictions_comparison.csv', index=False)
    print("详细预测结果已保存至: ../data/model_predictions_comparison.csv")

    return summary_df

def main():
    """
    主函数
    """
    print("员工离职预测 - 批量模型预测示例")
    print("=" * 50)

    # 检查模型文件是否存在
    model_dir = '../models'
    if not os.path.exists(model_dir):
        print(f"错误：模型目录 '{model_dir}' 不存在！请先运行训练脚本保存模型。")
        return

    # 加载并预测所有模型
    predictions = load_and_predict_all_models()

    if predictions:
        # 创建对比报告
        create_comparison_report(predictions)

        print("\n批量预测完成！")
        print(f"共加载了 {len(predictions)} 个模型")
        print("所有模型的预测结果已保存")
    else:
        print("没有成功加载任何模型，请检查模型文件。")

if __name__ == "__main__":
    main()