"""
主运行脚本 - 运行所有模型并生成比较报告
"""

import os
import sys
import subprocess
import pandas as pd
from datetime import datetime

def run_model_script(script_name):
    """
    运行模型训练脚本
    """
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    try:
        print(f"正在运行 {script_name}...")
        result = subprocess.run([sys.executable, script_path], 
                                capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print(f"✓ {script_name} 运行成功")
            return True
        else:
            print(f"✗ {script_name} 运行失败")
            print(f"错误信息: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ {script_name} 运行出错: {str(e)}")
        return False

def run_all_models():
    """
    运行所有模型训练脚本
    """
    model_scripts = [
        'attrition_lr.py',
        'attrition_lr_threshold.py', 
        'attrition_cart.py',
        'attrition_xgboost.py',
        'attrition_lgb.py',
        'attrition_lgb_onehot.py',
        'attrition_catboost.py',
        'attrition_svc.py',
        'attrition_gbdt.py',
        'attrition_ngboost.py',
        'attrition_knn.py'
    ]
    
    results = {}
    for script in model_scripts:
        success = run_model_script(script)
        results[script] = success
    
    return results

def run_comparison_analysis():
    """
    运行比较分析
    """
    try:
        # 运行特征重要性对比脚本
        print("正在运行特征重要性对比分析...")
        feature_importance_script = os.path.join(os.path.dirname(__file__), 'feature_importance_comparison.py')
        result = subprocess.run([sys.executable, feature_importance_script], 
                                capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print("✓ 特征重要性对比分析运行成功")
            return True
        else:
            print(f"✗ 特征重要性对比分析运行失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ 特征重要性对比分析运行出错: {str(e)}")
        return False

def run_model_comparison():
    """
    运行模型比较
    """
    try:
        print("正在生成模型比较报告...")
        comparison_script = os.path.join(os.path.dirname(__file__), 'generate_model_comparison.py')
        result = subprocess.run([sys.executable, comparison_script], 
                                capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print("✓ 模型比较报告生成成功")
            return True
        else:
            print(f"✗ 模型比较报告生成失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ 模型比较报告生成出错: {str(e)}")
        return False

def main():
    """
    主函数
    """
    print("开始运行员工离职预测项目完整流程...")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行所有模型
    print("\n1. 运行所有模型...")
    model_results = run_all_models()
    
    print("\n模型运行结果汇总:")
    success_count = 0
    for script, success in model_results.items():
        status = "成功" if success else "失败"
        print(f"  {script}: {status}")
        if success:
            success_count += 1
    
    print(f"\n成功运行 {success_count}/{len(model_results)} 个模型")
    
    # 运行比较分析
    print("\n2. 运行特征重要性对比分析...")
    feature_success = run_comparison_analysis()
    
    # 运行模型比较
    print("\n3. 生成模型比较报告...")
    comparison_success = run_model_comparison()
    
    print("\n所有任务完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count > 0 and feature_success and comparison_success:
        print("\n✓ 项目运行成功！")
        print("- 模型文件保存在: ../models/")
        print("- 预测结果保存在: ../data/submit_*.csv")
        print("- 特征重要性对比结果: ../data/feature_importance_comparison.csv")
        print("- 模型比较报告: ../model_comparison_report.md")
        print("- 可视化图表保存在: ../images/")
    else:
        print("\n⚠ 项目运行存在部分问题，请检查错误信息")

if __name__ == "__main__":
    main()