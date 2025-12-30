"""
CART决策树模型预测示例

此脚本演示如何加载保存的CART决策树模型并进行预测
"""

import pandas as pd
import numpy as np
from attrition_cart import load_cart_model, predict_with_saved_model, load_and_preprocess_data
import os
import glob

def find_latest_model(model_dir='models'):
    """
    查找最新的模型文件

    参数:
    model_dir: 模型目录

    返回:
    model_path, feature_path, info_path
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    # 查找最新的模型文件
    model_files = glob.glob(os.path.join(model_dir, 'cart_model_*.pkl'))
    if not model_files:
        raise FileNotFoundError(f"在 {model_dir} 中未找到模型文件")

    # 按时间戳排序，获取最新的
    latest_model = max(model_files, key=os.path.getctime)
    timestamp = os.path.basename(latest_model).replace('cart_model_', '').replace('.pkl', '')

    model_path = latest_model
    feature_path = os.path.join(model_dir, f'feature_names_{timestamp}.pkl')
    info_path = os.path.join(model_dir, f'model_info_{timestamp}.pkl')

    return model_path, feature_path, info_path

def create_sample_prediction_data():
    """
    创建示例预测数据

    返回:
    sample_data: 示例数据DataFrame
    """
    # 创建示例员工数据 (包含所有训练特征)
    sample_data = pd.DataFrame({
        'Age': [30, 45, 25, 35],
        'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel', 'Travel_Rarely'],
        'DailyRate': [800, 1200, 600, 900],
        'Department': ['Sales', 'Research & Development', 'Human Resources', 'Sales'],
        'DistanceFromHome': [5, 20, 2, 15],
        'Education': [3, 4, 2, 3],  # 1-5级别
        'EducationField': ['Marketing', 'Life Sciences', 'Human Resources', 'Technical Degree'],
        'EmployeeCount': [1, 1, 1, 1],  # 通常都是1
        'EnvironmentSatisfaction': [3, 4, 2, 3],
        'Gender': ['Male', 'Female', 'Female', 'Male'],
        'HourlyRate': [50, 80, 40, 60],
        'JobInvolvement': [3, 4, 2, 3],
        'JobLevel': [2, 3, 1, 2],
        'JobRole': ['Sales Executive', 'Research Scientist', 'Human Resources', 'Sales Representative'],
        'JobSatisfaction': [4, 3, 2, 4],
        'MaritalStatus': ['Single', 'Married', 'Divorced', 'Married'],
        'MonthlyIncome': [5000, 8000, 3000, 4500],
        'MonthlyRate': [20000, 30000, 15000, 18000],
        'NumCompaniesWorked': [1, 3, 0, 2],
        'Over18': ['Y', 'Y', 'Y', 'Y'],
        'OverTime': ['No', 'Yes', 'No', 'Yes'],
        'PercentSalaryHike': [15, 12, 18, 20],
        'PerformanceRating': [3, 4, 3, 3],
        'RelationshipSatisfaction': [4, 3, 2, 4],
        'StandardHours': [80, 80, 80, 80],  # 通常都是80
        'StockOptionLevel': [1, 2, 0, 1],
        'TotalWorkingYears': [5, 15, 1, 8],
        'TrainingTimesLastYear': [2, 3, 1, 4],
        'WorkLifeBalance': [3, 2, 4, 3],
        'YearsAtCompany': [3, 10, 1, 5],
        'YearsInCurrentRole': [2, 7, 0, 3],
        'YearsSinceLastPromotion': [1, 5, 0, 2],
        'YearsWithCurrManager': [2, 8, 0, 3]
    })

    return sample_data

def preprocess_prediction_data(sample_data, feature_names):
    """
    对预测数据进行预处理，使其与训练数据格式一致

    参数:
    sample_data: 原始预测数据
    feature_names: 训练时的特征名称列表

    返回:
    processed_data: 处理后的数据
    """
    # 复制数据避免修改原数据
    processed_data = sample_data.copy()

    # 分类特征列表 (与训练时保持一致)
    categorical_features = ['BusinessTravel', 'Department', 'EducationField',
                           'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']

    # 注意：这里我们需要使用训练时保存的标签编码器
    # 在实际应用中，应该保存和加载标签编码器
    # 这里为了演示，我们使用简单的映射

    # 简单映射示例 (实际应用中应该使用保存的编码器)
    business_travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    department_map = {'Human Resources': 0, 'Research & Development': 1, 'Sales': 2}
    education_field_map = {
        'Human Resources': 0, 'Life Sciences': 1, 'Marketing': 2,
        'Medical': 3, 'Other': 4, 'Technical Degree': 5
    }
    gender_map = {'Female': 0, 'Male': 1}
    job_role_map = {
        'Healthcare Representative': 0, 'Human Resources': 1, 'Laboratory Technician': 2,
        'Manager': 3, 'Manufacturing Director': 4, 'Research Director': 5,
        'Research Scientist': 6, 'Sales Executive': 7, 'Sales Representative': 8
    }
    marital_status_map = {'Divorced': 0, 'Married': 1, 'Single': 2}
    over18_map = {'N': 0, 'Y': 1}
    overtime_map = {'No': 0, 'Yes': 1}

    # 应用编码
    processed_data['BusinessTravel'] = processed_data['BusinessTravel'].map(business_travel_map)
    processed_data['Department'] = processed_data['Department'].map(department_map)
    processed_data['EducationField'] = processed_data['EducationField'].map(education_field_map)
    processed_data['Gender'] = processed_data['Gender'].map(gender_map)
    processed_data['JobRole'] = processed_data['JobRole'].map(job_role_map)
    processed_data['MaritalStatus'] = processed_data['MaritalStatus'].map(marital_status_map)
    processed_data['Over18'] = processed_data['Over18'].map(over18_map)
    processed_data['OverTime'] = processed_data['OverTime'].map(overtime_map)

    # Education已经是数值型，不需要编码
    # 数值型特征保持不变

    # 确保列的顺序与训练时一致
    processed_data = processed_data[feature_names]

    return processed_data

def main():
    """
    主函数：演示模型加载和预测
    """
    print("=== CART决策树模型预测演示 ===\n")

    try:
        # 1. 查找最新的模型文件
        print("1. 查找最新保存的模型...")
        model_path, feature_path, info_path = find_latest_model()

        # 2. 加载模型
        print("\n2. 加载模型...")
        cart_model, feature_names, model_info = load_cart_model(model_path, feature_path, info_path)

        if model_info:
            print(f"模型信息: {model_info['model_type']} (max_depth={model_info['max_depth']})")
            print(f"特征数量: {model_info['n_features']}")

        # 3. 创建示例预测数据
        print("\n3. 创建示例预测数据...")
        sample_data = create_sample_prediction_data()
        print(f"示例数据包含 {len(sample_data)} 个员工记录")
        print("员工基本信息:")
        for i, row in sample_data.iterrows():
            print(f"  员工{i+1}: 年龄{row['Age']}, 部门{row['Department']}, "
                  f"月收入{row['MonthlyIncome']}, 加班{row['OverTime']}")

        # 4. 预处理预测数据
        print("\n4. 预处理预测数据...")
        processed_data = preprocess_prediction_data(sample_data, feature_names)
        print(f"预处理完成，数据形状: {processed_data.shape}")

        # 5. 进行预测
        print("\n5. 进行离职概率预测...")
        predictions, probabilities = predict_with_saved_model(model_path, processed_data, feature_names)

        # 6. 显示预测结果
        print("\n6. 预测结果:")
        print("-" * 60)
        print(f"{'员工':<4} {'年龄':<4} {'部门':<20} {'月收入':<8} {'加班':<4} {'离职预测':<8} {'离职概率':<8}")
        print("-" * 60)

        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            attrition_prob = prob[1] * 100  # 离职概率 (百分比)
            attrition_pred = "是" if pred == 1 else "否"
            row = sample_data.iloc[i]
            print(f"{i+1:<4} {row['Age']:<4} {row['Department']:<20} "
                  f"{row['MonthlyIncome']:<8} {row['OverTime']:<4} {attrition_pred:<8} {attrition_prob:<8.1f}%")

        print("-" * 60)

        # 7. 风险分析
        print("\n7. 风险分析:")
        high_risk_count = sum(predictions)
        total_count = len(predictions)
        high_risk_rate = high_risk_count / total_count * 100

        print(f"高风险员工数量: {high_risk_count}/{total_count} ({high_risk_rate:.1f}%)")

        # 找出高风险员工
        high_risk_indices = [i for i, pred in enumerate(predictions) if pred == 1]
        if high_risk_indices:
            print("高风险员工详情:")
            for idx in high_risk_indices:
                row = sample_data.iloc[idx]
                prob = probabilities[idx][1] * 100
                print(f"  员工{idx+1}: 年龄{row['Age']}, 部门{row['Department']}, "
                      f"月收入{row['MonthlyIncome']}, 加班{row['OverTime']}, 离职概率{prob:.1f}%")

        print("\n=== 预测演示完成 ===")

    except Exception as e:
        print(f"错误: {e}")
        print("\n请确保:")
        print("1. 已运行 attrition_cart.py 训练并保存模型")
        print("2. models/ 目录存在且包含模型文件")
        print("3. 所有依赖包已安装")

if __name__ == "__main__":
    main()