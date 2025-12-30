"""
批量运行所有模型并收集结果
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

def generate_comparison_report():
    """
    生成模型对比报告
    """
    try:
        # 运行特征重要性对比脚本
        print("正在运行特征重要性对比分析...")
        feature_importance_script = os.path.join(os.path.dirname(__file__), 'feature_importance_comparison.py')
        result = subprocess.run([sys.executable, feature_importance_script], 
                                capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print("✓ 特征重要性对比分析运行成功")
        else:
            print(f"✗ 特征重要性对比分析运行失败: {result.stderr}")
    except Exception as e:
        print(f"✗ 特征重要性对比分析运行出错: {str(e)}")

def main():
    """
    主函数
    """
    print("开始批量运行所有模型...")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行所有模型
    results = run_all_models()
    
    print("\n模型运行结果汇总:")
    for script, success in results.items():
        status = "成功" if success else "失败"
        print(f"  {script}: {status}")
    
    # 生成对比报告
    generate_comparison_report()
    
    print("\n所有模型运行完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()