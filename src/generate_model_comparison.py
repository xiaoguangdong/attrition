"""
生成模型比较报告
"""

import pandas as pd
import numpy as np
import os
from model_utils import ModelManager
import matplotlib.pyplot as plt

def load_predictions():
    """
    加载所有模型的预测结果
    """
    prediction_files = []
    for file in os.listdir('../data/'):
        if file.startswith('submit_') and file.endswith('.csv'):
            prediction_files.append(file)
    
    predictions = {}
    for file in prediction_files:
        model_name = file.replace('submit_', '').replace('.csv', '')
        df = pd.read_csv(f'../data/{file}', index_col=0)
        predictions[model_name] = df['Attrition']
    
    return predictions

def load_model_info():
    """
    加载模型信息
    """
    model_manager = ModelManager()
    model_names = model_manager.list_models()
    
    model_info = {}
    for name in model_names:
        try:
            model, metadata = model_manager.load_model(name)
            model_info[name] = metadata
        except Exception as e:
            print(f"加载模型 {name} 信息失败: {e}")
    
    return model_info

def calculate_metrics(predictions):
    """
    计算预测指标
    """
    metrics = {}
    
    # 假设我们有真实标签作为基准
    # 在实际应用中，我们可能需要验证集上的真实标签
    if os.path.exists('../data/train.csv'):
        train_df = pd.read_csv('../data/train.csv', index_col=0)
        train_df['Attrition'] = train_df['Attrition'].map(lambda x: 1 if x == 'Yes' else 0)
        
        # 这里我们使用训练集的一部分作为验证（实际应用中应该有独立的验证集）
        # 为演示目的，我们假设提交结果是概率值，计算一些基本指标
        for model_name, pred in predictions.items():
            # 由于我们没有测试集的真实标签，我们只能计算一些基本统计信息
            metrics[model_name] = {
                '预测离职比例': pred.mean(),
                '预测离职人数': pred.sum(),
                '预测标准差': pred.std()
            }
    
    return metrics

def get_feature_importance():
    """
    获取特征重要性
    """
    model_manager = ModelManager()
    model_names = model_manager.list_models()
    
    feature_importance = {}
    
    for name in model_names:
        try:
            model, metadata = model_manager.load_model(name)
            feature_names = metadata.get('feature_names', [])
            
            if name in ['xgboost', 'lgb', 'lgb_onehot', 'catboost', 'gbdt', 'ngboost', 'cart']:
                # 这些模型有feature_importances_属性
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    if len(feature_names) == len(importances):
                        feature_importance[name] = dict(zip(feature_names, importances))
                    else:
                        print(f"模型 {name} 的特征数量与重要性数量不匹配")
                        
            elif name in ['lr', 'lr_threshold', 'svc']:
                # 对于线性模型，使用系数的绝对值
                if hasattr(model, 'coef_'):
                    coef_abs = np.abs(model.coef_[0])
                    if len(feature_names) == len(coef_abs):
                        feature_importance[name] = dict(zip(feature_names, coef_abs))
                    else:
                        print(f"模型 {name} 的特征数量与系数数量不匹配")
        
        except Exception as e:
            print(f"获取模型 {name} 特征重要性失败: {e}")
    
    return feature_importance

def create_comparison_report():
    """
    创建比较报告
    """
    print("正在生成模型比较报告...")
    
    # 加载预测结果
    predictions = load_predictions()
    print(f"加载了 {len(predictions)} 个模型的预测结果")
    
    # 加载模型信息
    model_info = load_model_info()
    print(f"加载了 {len(model_info)} 个模型的信息")
    
    # 计算指标
    metrics = calculate_metrics(predictions)
    
    # 获取特征重要性
    feature_importance = get_feature_importance()
    
    # 生成Markdown报告
    report = []
    report.append("# 模型比较报告\n")
    report.append("## 模型性能比较\n")
    
    if metrics:
        report.append("| 模型名称 | 预测离职比例 | 预测离职人数 | 预测标准差 |\n")
        report.append("|---------|------------|------------|------------|\n")
        for model_name, metric in metrics.items():
            report.append(f"| {model_name} | {metric['预测离职比例']:.4f} | {int(metric['预测离职人数'])} | {metric['预测标准差']:.4f} |\n")
    
    report.append("\n## 模型信息\n")
    for model_name, info in model_info.items():
        report.append(f"### {model_name}\n")
        report.append(f"- 模型类型: {info.get('model_type', 'Unknown')}\n")
        if 'validation_accuracy' in info:
            report.append(f"- 验证集准确率: {info['validation_accuracy']:.4f}\n")
        if 'max_iter' in info:
            report.append(f"- 最大迭代次数: {info['max_iter']}\n")
        report.append("\n")
    
    report.append("## 特征重要性排名\n")
    for model_name, importance in feature_importance.items():
        report.append(f"### {model_name} 特征重要性排名 (Top 10)\n")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        report.append("| 排名 | 特征名 | 重要性 |\n")
        report.append("|------|--------|--------|\n")
        for i, (feature, imp) in enumerate(sorted_features[:10], 1):
            report.append(f"| {i} | {feature} | {imp:.4f} |\n")
        report.append("\n")
    
    # 保存报告
    report_content = "".join(report)
    with open('../model_comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("模型比较报告已保存至 ../model_comparison_report.md")
    
    # 生成可视化图表
    generate_visualizations(feature_importance, metrics)
    
    return report_content

def generate_visualizations(feature_importance, metrics):
    """
    生成可视化图表
    """
    if feature_importance:
        # 特征重要性对比图
        plt.figure(figsize=(14, 10))
        
        # 选择前10个最重要的特征进行可视化
        all_features = set()
        for importance in feature_importance.values():
            all_features.update(importance.keys())
        
        # 为每个模型选择前10重要特征
        for model_name, importance in feature_importance.items():
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_features = dict(sorted_features[:10])
            
            plt.figure(figsize=(12, 8))
            features = list(top_features.keys())
            importances = list(top_features.values())
            
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('重要性')
            plt.title(f'{model_name} 模型 - Top 10 特征重要性')
            plt.gca().invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(f'../images/{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"已生成 {len(feature_importance)} 个模型的特征重要性图表")
    
    if metrics:
        # 预测离职比例对比图
        model_names = list(metrics.keys())
        attrition_ratios = [metrics[name]['预测离职比例'] for name in model_names]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, attrition_ratios)
        plt.title('各模型预测离职比例对比')
        plt.ylabel('离职比例')
        plt.xticks(rotation=45)
        
        # 在柱子上添加数值标签
        for bar, ratio in zip(bars, attrition_ratios):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{ratio:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('../images/model_attrition_ratio_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("已生成预测离职比例对比图表")

def main():
    """
    主函数
    """
    report = create_comparison_report()
    print("模型比较报告生成完成！")

if __name__ == "__main__":
    main()