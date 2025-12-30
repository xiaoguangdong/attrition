"""
跨模型特征重要性对比分析
比较不同模型的特征重要性排名
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_utils import ModelManager
import os
import glob
import joblib

def get_feature_importance(model, model_type, feature_names):
    """
    获取模型的特征重要性
    
    参数:
    - model: 训练好的模型
    - model_type: 模型类型字符串
    - feature_names: 特征名称列表
    
    返回:
    - 特征重要性字典
    """
    if model_type in ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting', 'NGBoost']:
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'booster'):
            # XGBoost
            importances = model.get_booster().get_score(importance_type='weight')
            # 将特征重要性映射到feature_names的顺序
            importances = [importances.get(name, 0) for name in feature_names]
        else:
            return None
        
        return dict(zip(feature_names, importances))
    
    elif model_type == 'LogisticRegression' or 'LogisticRegression' in model_type:
        # 对于逻辑回归，使用系数的绝对值
        if hasattr(model, 'coef_'):
            coef_abs = np.abs(model.coef_[0])
            return dict(zip(feature_names, coef_abs))
        else:
            return None
    
    elif model_type == 'CART Decision Tree':
        # 决策树的特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(feature_names, importances))
        else:
            return None
    
    elif model_type == 'SVC':
        # 线性SVM可以使用系数
        if hasattr(model, 'coef_'):
            coef_abs = np.abs(model.coef_[0])
            return dict(zip(feature_names, coef_abs))
        else:
            return None
    
    else:
        print(f"模型类型 {model_type} 不支持特征重要性提取")
        return None

def load_all_models():
    """
    加载所有模型
    """
    model_manager = ModelManager()
    
    # 获取所有模型名称
    model_names = model_manager.list_models()
    print(f"找到模型: {model_names}")
    
    models_data = {}
    
    for model_name in model_names:
        try:
            model, metadata = model_manager.load_model(model_name)
            model_type = metadata.get('model_type', 'Unknown')
            feature_names = metadata.get('feature_names', [])
            
            if feature_names:
                importance = get_feature_importance(model, model_type, feature_names)
                if importance:
                    models_data[model_name] = {
                        'importance': importance,
                        'model_type': model_type,
                        'feature_names': feature_names
                    }
                    print(f"成功加载 {model_name} 模型的特征重要性")
                else:
                    print(f"{model_name} 模型不支持特征重要性提取")
            else:
                print(f"{model_name} 模型没有保存特征名称")
        except Exception as e:
            print(f"加载 {model_name} 模型失败: {e}")
    
    return models_data

def create_comparison_report(models_data):
    """
    创建特征重要性对比报告
    """
    if not models_data:
        print("没有模型数据可分析")
        return
    
    # 创建特征重要性DataFrame
    all_features = set()
    for model_name, data in models_data.items():
        all_features.update(data['importance'].keys())
    
    all_features = sorted(list(all_features))
    
    importance_df = pd.DataFrame(index=all_features)
    
    for model_name, data in models_data.items():
        model_importance = data['importance']
        # 填充缺失特征的重要性为0
        model_values = [model_importance.get(feature, 0) for feature in all_features]
        importance_df[model_name] = model_values
    
    # 归一化处理
    for col in importance_df.columns:
        if importance_df[col].max() != 0:
            importance_df[col] = importance_df[col] / importance_df[col].max()
    
    # 输出表格
    print("\n=== 特征重要性对比表 ===")
    print("(数值已归一化，1.0表示该模型中该特征最重要)")
    print(importance_df.round(3))
    
    # 为每个模型创建特征重要性排名
    print("\n=== 各模型特征重要性排名 ===")
    for model_name, data in models_data.items():
        print(f"\n{model_name} ({data['model_type']}) - 前10重要特征:")
        importance = data['importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance_value) in enumerate(sorted_features[:10], 1):
            print(f"  {i:2d}. {feature}: {importance_value:.4f}")
    
    # 保存对比报告
    importance_df.to_csv('../data/feature_importance_comparison.csv')
    print(f"\n特征重要性对比已保存到: ../data/feature_importance_comparison.csv")
    
    return importance_df

def visualize_feature_importance(models_data):
    """
    可视化特征重要性对比
    """
    if not models_data:
        print("没有模型数据可分析")
        return
    
    # 创建特征重要性DataFrame
    all_features = set()
    for model_name, data in models_data.items():
        all_features.update(data['importance'].keys())
    
    all_features = sorted(list(all_features))
    
    importance_df = pd.DataFrame(index=all_features)
    
    for model_name, data in models_data.items():
        model_importance = data['importance']
        model_values = [model_importance.get(feature, 0) for feature in all_features]
        importance_df[model_name] = model_values
    
    # 为每个模型归一化
    for col in importance_df.columns:
        if importance_df[col].max() != 0:
            importance_df[col] = importance_df[col] / importance_df[col].max()
    
    # 可视化
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 使用matplotlib创建热力图
    im = ax.imshow(importance_df.T.values, cmap='YlOrRd', aspect='auto')
    
    # 设置坐标轴标签
    ax.set_xticks(range(len(all_features)))
    ax.set_yticks(range(len(importance_df.columns)))
    ax.set_xticklabels(all_features, rotation=45, ha="right")
    ax.set_yticklabels(importance_df.columns)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('归一化重要性')
    
    plt.title('跨模型特征重要性对比热力图')
    plt.xlabel('特征')
    plt.ylabel('模型')
    plt.tight_layout()
    plt.savefig('../images/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以避免显示
    
    print("特征重要性对比热力图已保存到: ../images/feature_importance_comparison.png")
    
    # 为每个模型单独绘制特征重要性
    n_models = len(models_data)
    if n_models > 0:
        n_cols = 2
        n_rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_models == 2:
            axes = [axes[0], axes[1]]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, data) in enumerate(models_data.items()):
            importance = data['importance']
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]  # 取前10个
            
            features, importances = zip(*sorted_features)
            
            axes[idx].barh(range(len(features)), importances)
            axes[idx].set_yticks(range(len(features)))
            axes[idx].set_yticklabels(features)
            axes[idx].set_xlabel('归一化重要性')
            axes[idx].set_title(f'{model_name} - 特征重要性 (Top 10)')
            axes[idx].invert_yaxis()
        
        # 隐藏多余的子图
        for idx in range(len(models_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('../images/model_specific_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以避免显示
        
        print("各模型特征重要性图已保存到: ../images/model_specific_feature_importance.png")

def main():
    """
    主函数
    """
    print("开始跨模型特征重要性对比分析...")
    
    # 检查模型目录是否存在
    model_dir = '../models'
    if not os.path.exists(model_dir):
        print(f"错误：模型目录 '{model_dir}' 不存在！请先训练模型并保存。")
        return
    
    # 加载所有模型
    models_data = load_all_models()
    
    if not models_data:
        print("没有找到支持特征重要性分析的模型")
        return
    
    # 创建对比报告
    importance_df = create_comparison_report(models_data)
    
    # 可视化
    visualize_feature_importance(models_data)
    
    print("\n跨模型特征重要性对比分析完成！")

if __name__ == "__main__":
    main()