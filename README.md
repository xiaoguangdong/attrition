# 员工离职预测项目 (Employee Attrition Prediction)

基于IBM HR Analytics数据集，实现完整的员工离职概率预测系统，为人力资源管理提供数据驱动的决策支持。

## 项目概述

本项目使用多种机器学习算法对员工离职进行预测，包含完整的模型训练、保存、加载和预测功能。支持9种主流机器学习算法，并提供详细的性能比较和部署指南。

## 快速开始

### 环境准备
```bash
pip install -r requirements.txt
```

### 运行所有模型
```bash
cd src
python run_all_models.py
```

### 单个模型训练
```bash
python src/attrition_lgb.py    # LightGBM
python src/attrition_xgb.py    # XGBoost
python src/attrition_cart.py   # 决策树
python src/attrition_lr.py     # 逻辑回归
# 更多模型请查看 src/ 目录
```

### 批量预测
```bash
python src/model_batch_predict.py
```

## 支持的模型算法

| 算法 | 文件名 | 特点 | 验证集准确率 |
|------|--------|------|------------|
| LightGBM | attrition_lgb.py | 高效梯度提升，大数据集 | 83.90% |
| XGBoost | attrition_xgboost.py | 强大梯度提升 | - |
| CatBoost | attrition_catboost.py | 类别特征处理优秀 | 82.63% |
| 逻辑回归 | attrition_lr.py | 解释性强，线性关系 | 88.98% |
| 决策树 | attrition_cart.py | 可视化好，易理解 | 70.76% |
| SVM | attrition_svc.py | 核函数支持 | 79.24% |
| GBDT | attrition_gbdt.py | 经典梯度提升 | 83.47% |
| NGBoost | attrition_ngboost.py | 自然梯度提升 | - |
| KNN | attrition_knn.py | 基于实例的学习 | - |

## 项目结构

```
attrition/
├── data/                    # 数据文件 (训练/测试数据)
├── src/                     # 源代码 (模型训练脚本)
├── docs/                    # 详细文档
├── images/                  # 可视化结果
├── models/                  # 训练好的模型 (.pkl文件)
└── requirements.txt         # 依赖包列表
```

## 模型性能分析

### 基础模型验证集准确率

| 模型类型 | 验证集准确率 | 特点 |
|---------|------------|------|
| 逻辑回归 (lr) | 88.98% | 解释性强，适合线性关系 |
| LightGBM (lgb) | 83.90% | 高效，处理大数据集 |
| GBDT | 83.47% | 经典梯度提升 |
| CatBoost | 82.63% | 处理类别特征优秀 |
| SVM | 79.24% | 核函数支持 |
| 决策树 (cart) | 70.76% | 可视化好，易解释 |

### 特殊配置模型性能

| 模型名称 | 配置特点 | 验证集准确率 |
|---------|----------|------------|
| lr_threshold | 带阈值调整的逻辑回归 | 86.86% |
| lgb_onehot | LightGBM + OneHot编码 | 81.78% |

### 预测结果比较

| 模型名称 | 预测离职比例 | 预测离职人数 | 预测标准差 | 推荐指数 |
|---------|------------|------------|-----------|----------|
| KNN | 41.50% | 122 | 0.4936 | ⭐⭐ |
| 决策树 (CART) | 35.37% | 104 | 0.4789 | ⭐⭐⭐ |
| XGBoost | 18.85% | 55 | 0.0185 | ⭐⭐⭐⭐⭐ |
| 逻辑回归 | 16.45% | 48 | 0.1727 | ⭐⭐⭐⭐⭐ |
| SVM | 15.99% | 47 | 0.3671 | ⭐⭐⭐⭐ |
| LightGBM | 15.99% | 47 | 0.3671 | ⭐⭐⭐⭐ |
| LightGBM-OneHot | 11.90% | 35 | 0.3244 | ⭐⭐⭐⭐ |
| GBDT | 6.80% | 20 | 0.2522 | ⭐⭐⭐ |
| CatBoost | 5.78% | 17 | 0.2338 | ⭐⭐⭐⭐ |
| NGBoost | 3.74% | 11 | 0.1901 | ⭐⭐⭐ |

### 预测稳定性分析

**最稳定的模型** (标准差最小):
- XGBoost: 0.0185
- 逻辑回归: 0.1727

**最不稳定的模型** (标准差最大):
- KNN: 0.4936
- 决策树: 0.4789

### 模型选择建议

1. **追求最高准确率**: 选择逻辑回归 (88.98%)
2. **需要稳定性**: 选择XGBoost (标准差最小: 0.0185)
3. **需要可解释性**: 选择逻辑回归或决策树
4. **处理复杂特征**: 选择CatBoost
5. **大数据集**: 选择LightGBM

### 实际部署建议

1. **主要模型**: 逻辑回归 (平衡准确性和解释性)
2. **备选模型**: XGBoost (稳定性和性能)
3. **特殊情况**: CatBoost (类别特征多时)

## 特征重要性分析

### 关键离职因素 (各模型Top 3特征)

| 特征名称 | 在多少个模型中排前3 | 平均重要性排名 |
|---------|-------------------|---------------|
| OverTime (加班) | 4/5 | 1.5 |
| MonthlyIncome (月收入) | 3/5 | 2.0 |
| YearsWithCurrManager (与经理共事年数) | 3/5 | 3.0 |
| StockOptionLevel (股票期权等级) | 2/5 | 4.0 |
| EnvironmentSatisfaction (环境满意度) | 2/5 | 4.5 |

### 模型特征重要性特点

**逻辑回归模型**:
- 最重视: OverTime (0.5079), MaritalStatus (0.3741)
- 线性特征权重，易于解释

**树模型 (CART/GBDT)**:
- 更关注: MonthlyIncome, YearsWithCurrManager
- 能够捕获非线性关系

**CatBoost模型**:
- 强调: OverTime (6.7437), MonthlyIncome (5.7350)
- 对类别特征处理优秀

### 离职风险因素总结

根据所有模型的综合分析，员工离职的主要风险因素为:

1. **加班情况** - 最关键的预测因子
2. **薪酬水平** - 月收入直接影响离职意愿
3. **管理关系** - 与直属经理的关系质量
4. **工作环境满意度** - 工作环境满意度
5. **职业发展** - 股票期权、晋升机会等

## 高级功能

### 自定义模型参数

修改各模型脚本中的参数配置:

```python
# LightGBM参数示例
params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'random_state': 42
}

# XGBoost参数优化示例
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}
```

### 批量预测功能

使用 `model_batch_predict.py` 可以:
- 同时加载多个已训练的模型
- 生成综合预测结果
- 输出模型性能对比报告

### 特征工程高级技巧

```python
def create_interaction_features(df):
    """创建特征交互项"""
    # 收入与工作年限的交互
    df['Income_Years_Interaction'] = df['MonthlyIncome'] * df['YearsAtCompany']
    
    # 满意度交互
    df['Satisfaction_Interaction'] = (
        df['JobSatisfaction'] * df['EnvironmentSatisfaction'] * df['WorkLifeBalance']
    )
    
    # 年龄收入比
    df['Age_Income_Ratio'] = df['MonthlyIncome'] / (df['Age'] + 1)
    
    return df
```

### FastAPI 部署示例

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="员工离职预测API")

class EmployeeData(BaseModel):
    Age: int
    OverTime: str
    MonthlyIncome: float
    YearsAtCompany: int
    # ... 其他特征

@app.post("/predict")
async def predict_attrition(employee: EmployeeData):
    try:
        # 加载模型
        model = joblib.load("../models/lgb_model_20251230_120449.pkl")
        
        # 预处理数据
        data = pd.DataFrame([employee.dict()])
        processed_data = preprocess_employee_data(data)
        
        # 预测
        probability = model.predict_proba(processed_data)[0, 1]
        prediction = "高风险" if probability > 0.5 else "低风险"
        
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": "high" if probability > 0.5 else "low"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"预测失败: {str(e)}")
```

### 性能优化策略

```python
# 内存优化
def optimize_memory_usage(df):
    """优化DataFrame内存使用"""
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0:
            df[col] = df[col].astype('uint16')
        elif df[col].min() >= -32768 and df[col].max() <= 32767:
            df[col] = df[col].astype('int16')
        else:
            df[col] = df[col].astype('int32')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    return df

# ONNX模型转换加速推理
import skl2onnx
from skl2onnx import convert_sklearn

# 转换模型为ONNX格式
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# 保存ONNX模型
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

## 输出文件说明

### 预测结果文件
- `submit_*.csv`: 各模型的预测结果
- `test_lr_predictions.csv`: 逻辑回归预测详情

### 模型文件
- `*_model_*.pkl`: 训练好的模型
- `*_features_*.pkl`: 特征预处理对象
- `*_info_*.pkl`: 模型训练信息

### 可视化文件
本项目生成的可视化图表包括:
- `cart_decision_tree.png` - 决策树可视化
- `feature_importance_comparison.png` - 特征重要性对比
- `model_attrition_ratio_comparison.png` - 模型预测比例对比
- `*_feature_importance.png` - 各模型特征重要性图
- `lr_coefficients.png` - 逻辑回归系数图
- `lr_normalized_coefficients.png` - 标准化逻辑回归系数图

## 故障排除

### 常见问题

1. **模型文件不存在**: 确保已运行对应的训练脚本
2. **内存不足**: 减少数据量或使用轻量级模型
3. **依赖包冲突**: 重新安装 requirements.txt
4. **预测失败**: 检查数据格式是否与训练时一致

### 性能优化建议

1. 使用LightGBM处理大数据集
2. 逻辑回归适合快速原型和可解释性需求
3. 决策树适合结果解释和规则发现
4. XGBoost适合追求稳定性和准确性

### 依赖包安装

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost ngboost joblib matplotlib seaborn
```

## 模型保存和加载

### 保存功能
所有模型都支持统一的保存和加载功能：

```python
import joblib
import pandas as pd
from datetime import datetime

# 保存模型
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"../models/{model_name}_model_{timestamp}.pkl"
features_path = f"../models/{model_name}_features_{timestamp}.pkl"

joblib.dump(model, model_path)
joblib.dump(feature_names, features_path)
```

### 加载和使用模型

```python
# 加载已训练模型
model = joblib.load('models/lgb_model_20251230_120449.pkl')
features = joblib.load('models/lgb_features_20251230_120449.pkl')

# 预测
predictions = model.predict(features)
```

## 项目结构说明

### 目录详解
- **data/** - 训练和测试数据集
- **src/** - 模型训练脚本源代码
- **models/** - 训练好的模型文件 (.pkl格式)
- **images/** - 可视化结果和图表
- **docs/** - 可留空，所有文档已整合到本README

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进项目！

## 更新日志

- **v1.0** (2025-12-30): 初始版本，支持9种机器学习算法
- 添加FastAPI部署示例
- 完善特征工程和性能优化指南
- 详细的模型比较和选择建议