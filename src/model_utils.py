"""
通用模型保存和加载工具

提供统一的模型保存、加载和预测接口
"""

import os
import joblib
import pickle
import glob
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report


class ModelManager:
    """模型管理器：负责模型的保存、加载和预测"""

    def __init__(self, model_dir='../models'):
        """
        初始化模型管理器

        参数:
            model_dir: 模型保存目录
        """
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def save_model(self, model, model_name, feature_names=None, additional_info=None, preprocessor=None):
        """
        保存模型和相关信息

        参数:
            model: 训练好的模型
            model_name: 模型名称 (如 'lr', 'xgboost', 'cart')
            feature_names: 特征名称列表
            additional_info: 额外信息字典
            preprocessor: 预处理器（如scaler, encoder等）

        返回:
            保存的文件路径字典
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存模型
        model_path = os.path.join(self.model_dir, f'{model_name}_model_{timestamp}.pkl')
        joblib.dump(model, model_path)

        # 保存特征名称
        feature_path = None
        if feature_names is not None:
            feature_path = os.path.join(self.model_dir, f'{model_name}_features_{timestamp}.pkl')
            joblib.dump(feature_names, feature_path)
        else:
            # 即使没有特征名称也创建一个空列表来保存
            feature_path = os.path.join(self.model_dir, f'{model_name}_features_{timestamp}.pkl')
            joblib.dump([], feature_path)

        # 保存额外信息
        info_path = None
        if additional_info is not None:
            info_path = os.path.join(self.model_dir, f'{model_name}_info_{timestamp}.pkl')
            joblib.dump(additional_info, info_path)
        else:
            # 创建基本的附加信息
            basic_info = {
                'model_name': model_name,
                'save_time': timestamp,
                'model_type': str(type(model).__name__)
            }
            info_path = os.path.join(self.model_dir, f'{model_name}_info_{timestamp}.pkl')
            joblib.dump(basic_info, info_path)

        # 保存预处理器
        preprocessor_path = None
        if preprocessor is not None:
            preprocessor_path = os.path.join(self.model_dir, f'{model_name}_preprocessor_{timestamp}.pkl')
            joblib.dump(preprocessor, preprocessor_path)

        # 返回保存路径信息
        saved_paths = {
            'model': model_path,
            'features': feature_path,
            'info': info_path,
            'preprocessor': preprocessor_path
        }

        print(f"\n{model_name.upper()} 模型保存完成:")
        print(f"- 模型文件: {model_path}")
        print(f"- 特征文件: {feature_path}")
        print(f"- 信息文件: {info_path}")
        if preprocessor_path:
            print(f"- 预处理器文件: {preprocessor_path}")

        return saved_paths

    def load_model(self, model_name):
        """
        加载模型和相关信息

        参数:
            model_name: 模型名称 (如 'lr', 'xgboost', 'cart')

        返回:
            (model, metadata) 元组，metadata包含所有附加信息
        """
        # 查找最新的模型文件
        model_files = glob.glob(os.path.join(self.model_dir, f'{model_name}_model_*.pkl'))
        if not model_files:
            raise FileNotFoundError(f"未找到模型文件: {model_name}")

        # 获取最新的模型文件
        latest_model = max(model_files, key=os.path.getctime)
        
        # 从模型文件名中提取时间戳
        model_filename = os.path.basename(latest_model)
        # 格式是 {model_name}_model_{timestamp}.pkl
        timestamp = model_filename.replace(f'{model_name}_model_', '').replace('.pkl', '')

        # 加载模型
        model = joblib.load(latest_model)

        # 初始化元数据
        metadata = {
            'model_name': model_name,
            'source_file': latest_model
        }

        # 加载额外信息
        try:
            info_path = os.path.join(self.model_dir, f'{model_name}_info_{timestamp}.pkl')
            info_data = joblib.load(info_path)
            metadata.update(info_data)
        except FileNotFoundError:
            metadata['load_warning'] = 'No info file found'

        # 加载特征名称
        try:
            feature_path = os.path.join(self.model_dir, f'{model_name}_features_{timestamp}.pkl')
            feature_names = joblib.load(feature_path)
            metadata['feature_names'] = feature_names
        except FileNotFoundError:
            metadata['feature_names'] = []
            metadata['load_warning'] = 'No feature names file found'

        # 加载预处理器
        try:
            preprocessor_path = os.path.join(self.model_dir, f'{model_name}_preprocessor_{timestamp}.pkl')
            preprocessor = joblib.load(preprocessor_path)
            metadata['preprocessor'] = preprocessor
        except FileNotFoundError:
            metadata['preprocessor'] = None

        return model, metadata

    def predict(self, model, data, feature_names=None, preprocess_func=None):
        """
        使用模型进行预测

        参数:
            model: 训练好的模型
            data: 预测数据 (DataFrame)
            feature_names: 特征名称列表
            preprocess_func: 数据预处理函数

        返回:
            predictions, probabilities
        """
        # 数据预处理
        if preprocess_func:
            data = preprocess_func(data)

        # 特征选择
        if feature_names and feature_names:  # 确保feature_names不为空
            if isinstance(feature_names, list) and len(feature_names) > 0:
                missing_features = set(feature_names) - set(data.columns)
                if missing_features:
                    raise ValueError(f"数据缺少以下特征: {missing_features}")
                data = data[feature_names]

        # 预测
        predictions = model.predict(data)

        # 概率预测 (如果模型支持)
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data)

        return predictions, probabilities

    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        评估模型性能

        参数:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            model_name: 模型名称

        返回:
            评估结果字典
        """
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)

        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_0': report['0']['precision'],  # 不离职类的精确率
            'precision_1': report['1']['precision'],  # 离职类的精确率
            'recall_0': report['0']['recall'],        # 不离职类的召回率
            'recall_1': report['1']['recall'],        # 离职类的召回率
            'f1_0': report['0']['f1-score'],          # 不离职类的F1
            'f1_1': report['1']['f1-score'],          # 离职类的F1
        }

        print(f"\n{model_name} 评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"离职类精确率: {results['precision_1']:.4f}")
        print(f"离职类召回率: {results['recall_1']:.4f}")
        print(f"离职类F1分数: {results['f1_1']:.4f}")

        return results

    def list_models(self):
        """
        列出所有保存的模型
        """
        model_files = glob.glob(os.path.join(self.model_dir, '*_model_*.pkl'))
        model_names = set()
        for file in model_files:
            model_name = os.path.basename(file).split('_model_')[0]
            model_names.add(model_name)
        return sorted(list(model_names))  # 排序以便更容易查看

    def delete_model(self, model_name):
        """
        删除指定名称的所有模型文件

        参数:
            model_name: 要删除的模型名称
        """
        files_to_delete = []
        for file in os.listdir(self.model_dir):
            if file.startswith(f'{model_name}_') and file.endswith('.pkl'):
                files_to_delete.append(os.path.join(self.model_dir, file))

        for file_path in files_to_delete:
            os.remove(file_path)

        print(f"已删除 {len(files_to_delete)} 个与 {model_name} 相关的文件")