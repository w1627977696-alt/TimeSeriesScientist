# TimeSeriesScientist 代码文件详细注释

本文档为 TimeSeriesScientist 项目的核心代码文件提供详细的中文注释说明，帮助初学者理解代码结构和实现逻辑。

---

## 目录

1. [main.py - 程序入口](#1-mainpy---程序入口)
2. [graph/agent_graph.py - 工作流编排](#2-graphagent_graphpy---工作流编排)
3. [agents/preprocess_agent.py - 预处理Agent](#3-agentspreprocess_agentpy---预处理agent)
4. [agents/analysis_agent.py - 分析Agent](#4-agentsanalysis_agentpy---分析agent)
5. [agents/validation_agent.py - 验证Agent](#5-agentsvalidation_agentpy---验证agent)
6. [agents/forecast_agent.py - 预测Agent](#6-agentsforecast_agentpy---预测agent)
7. [agents/report_agent.py - 报告Agent](#7-agentsreport_agentpy---报告agent)
8. [agents/memory.py - 内存管理](#8-agentsmemorypy---内存管理)
9. [config/default_config.py - 默认配置](#9-configdefault_configpy---默认配置)
10. [utils/data_utils.py - 数据工具](#10-utilsdata_utilspy---数据工具)
11. [utils/model_library.py - 模型库](#11-utilsmodel_librarypy---模型库)

---

## 1. main.py - 程序入口

```python
"""
main.py - 时间序列预测Agent系统的主入口

【文件作用】
这是整个系统的入口点，负责：
1. 初始化配置
2. 检查环境变量（API密钥）
3. 创建并运行Agent工作流
4. 保存和展示结果

【执行流程】
用户运行 -> 加载配置 -> 检查API密钥 -> 初始化工作流 -> 执行预测 -> 保存结果

【关键依赖】
- graph.agent_graph: 工作流编排器
- config.default_config: 默认配置
"""

# 标准库导入
import os          # 操作系统接口，用于读取环境变量
import sys         # 系统相关功能，用于退出程序
import json        # JSON处理，用于保存结果
from datetime import datetime  # 日期时间处理
from pathlib import Path       # 路径处理（推荐的现代方式）

# 项目模块导入
from graph.agent_graph import TimeSeriesAgentGraph  # 核心！Agent工作流编排器
from config.default_config import DEFAULT_CONFIG    # 默认配置字典

if __name__ == "__main__":
    """
    主程序入口
    使用 if __name__ == "__main__" 确保只有直接运行此文件时才执行
    这是Python的标准做法，便于模块被其他文件导入
    """
    
    # ==================== 阶段1: 显示欢迎信息 ====================
    print("=" * 60)
    print("TimeSeriesScientist")  # 项目名称（注意原代码有拼写错误 "Sciensist"）
    print("=" * 60)

    # ==================== 阶段2: 配置初始化 ====================
    # 复制默认配置，避免修改原始配置对象
    config = DEFAULT_CONFIG.copy()
    
    # 自定义配置项（根据需求修改）
    config["num_slices"] = 25       # 数据切片数量
                                    # 解释：将数据分成多个切片进行独立测试
                                    # 更多切片 = 更稳定的评估，但更慢
    
    config["input_length"] = 512    # 输入序列长度（历史数据点数）
                                    # 解释：用多少个历史数据点来预测未来
    
    config["horizon"] = 96          # 预测步长（向前预测多少个时间点）
                                    # 例如：96小时 = 4天
    
    config["data_path"] = "../dataset/ETT-small/ETTh1.csv"  # 数据文件路径
    
    config["debug"] = False         # 调试模式：True会打印更多日志
    config["verbose"] = False       # 详细输出模式
    
    config["date_column"] = "date"  # 数据中的日期列名
    config["value_column"] = "OT"   # 数据中的目标值列名（OT = Oil Temperature）

    # ==================== 阶段3: API密钥检查 ====================
    # OpenAI API密钥是必需的，因为系统使用GPT模型进行智能决策
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Please set the OPENAI_API_KEY environment variable before running.")
        sys.exit(1)  # 退出程序，返回错误码1

    # ==================== 阶段4: 初始化Agent工作流 ====================
    print("Initializing Time Series Agent Graph...")
    
    # 创建工作流编排器实例
    # 这会初始化所有5个Agent（Preprocess, Analysis, Validation, Forecast, Report）
    graph = TimeSeriesAgentGraph(
        config=config,                    # 传递配置
        model=config["llm_model"],        # LLM模型名称（如 "gpt-4o"）
        debug=config["debug"]             # 是否开启调试
    )

    # ==================== 阶段5: 执行预测工作流 ====================
    print("Running the time series agent workflow...")
    print(f"Configuration: {config['num_slices']} slices, {config['horizon']} horizon steps")
    print("=" * 60)
    
    # 导入时间模块用于计时和延迟
    import time
    
    delay_between_slices = 5  # 切片之间的延迟（秒），避免API限速
    delay_between_agents = 2  # Agent之间的延迟（秒）
    
    start_time = time.time()
    
    # 核心执行！运行整个Agent工作流
    results = graph.run()
    
    end_time = time.time()
    
    # 显示执行时间统计
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"Average time per slice: {execution_time/config['num_slices']:.2f} seconds")
    
    # 最后延迟，确保API请求完成
    print("Adding final delay to ensure API rate limit compliance...")
    time.sleep(delay_between_slices)

    # ==================== 阶段6: 保存结果 ====================
    print("\n" + "=" * 60)
    print("Workflow execution completed!")
    print("=" * 60)
    
    # 创建结果目录
    results_dir = Path("results/reports")
    results_dir.mkdir(parents=True, exist_ok=True)  # 递归创建目录
    
    # 生成时间戳用于文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存完整结果（JSON格式）
    complete_report_filename = f"complete_time_series_report_{timestamp}.json"
    complete_report_path = results_dir / complete_report_filename
    
    try:
        with open(complete_report_path, 'w', encoding='utf-8') as f:
            # default=str 处理不能直接序列化的对象（如datetime）
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Complete results saved to: {complete_report_path}")
    except Exception as e:
        print(f"Error saving complete results: {e}")
    
    # 保存聚合结果（最终预测）
    if results.get("aggregated_results"):
        aggregated_report_filename = f"aggregated_forecast_results_{timestamp}.json"
        aggregated_report_path = results_dir / aggregated_report_filename
        
        # 构建聚合结果摘要
        aggregated_summary = {
            "timestamp": timestamp,
            "aggregation_info": results["aggregated_results"]["aggregation_info"],
            "final_individual_predictions": results["aggregated_results"]["individual_predictions"],
            "final_ensemble_predictions": results["aggregated_results"]["ensemble_predictions"],
            "final_test_metrics": results["aggregated_results"]["test_metrics"],
            "final_forecast_metrics": results["aggregated_results"]["forecast_metrics"]
        }
        
        try:
            with open(aggregated_report_path, 'w', encoding='utf-8') as f:
                json.dump(aggregated_summary, f, indent=2, ensure_ascii=False, default=str)
            print(f"Aggregated forecast results saved to: {aggregated_report_path}")
        except Exception as e:
            print(f"Error saving aggregated results: {e}")

    # ==================== 阶段7: 打印结果摘要 ====================
    if results.get("aggregated_results"):
        print("\n" + "=" * 60)
        print("FINAL AGGREGATED FORECAST RESULTS")  # 最终聚合预测结果
        print("=" * 60)
        
        agg_info = results["aggregated_results"]["aggregation_info"]
        print(f"Number of slices processed: {agg_info['num_slices']}")  # 处理的切片数
        print(f"Aggregation method: {agg_info['aggregation_method']}")  # 聚合方法
        
        # 打印集成模型性能指标
        if results["aggregated_results"]["test_metrics"].get("ensemble"):
            ensemble_metrics = results["aggregated_results"]["test_metrics"]["ensemble"]
            print(f"\nFinal Ensemble Performance:")
            print(f"  MSE: {ensemble_metrics['mse']:.4f}")   # 均方误差
            print(f"  MAE: {ensemble_metrics['mae']:.4f}")   # 平均绝对误差
            print(f"  MAPE: {ensemble_metrics['mape']:.2f}%") # 平均绝对百分比误差
        
        # 打印各个模型的性能
        print(f"\nIndividual Model Performance (averaged across slices):")
        for model_name, metrics in results["aggregated_results"]["test_metrics"].items():
            if model_name != "ensemble":
                print(f"  {model_name}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%")

    print("\nExperiment completed!")
```

---

## 2. graph/agent_graph.py - 工作流编排

```python
"""
agent_graph.py - 基于LangGraph的Agent工作流编排器

【文件作用】
这是整个系统的"指挥中心"，负责：
1. 初始化所有Agent
2. 定义Agent之间的执行顺序和数据流
3. 管理工作流状态
4. 聚合多个数据切片的结果

【核心概念】
- StateGraph: LangGraph的状态图，定义工作流结构
- Node: 工作流中的节点，每个Agent是一个节点
- Edge: 节点之间的连接，定义执行顺序
- State: 在节点之间传递的状态字典

【执行流程】
preprocess -> analyze -> validate -> forecast -> report -> END

【关键类】
- TimeSeriesAgentGraph: 主编排器类
"""

import os
import pandas as pd
import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END  # LangGraph核心组件
import numpy as np

logger = logging.getLogger(__name__)

# 导入所有Agent
from agents.preprocess_agent import PreprocessAgent   # 数据预处理
from agents.analysis_agent import AnalysisAgent       # 数据分析
from agents.validation_agent import ValidationAgent   # 模型验证
from agents.forecast_agent import ForecastAgent       # 预测生成
from agents.report_agent import ReportAgent           # 报告生成

# 导入数据工具
from utils.data_utils import DataLoader, DataSplitter, DataPreprocessor
from utils.file_utils import FileManager


class TimeSeriesAgentGraph:
    """
    基于LangGraph的时间序列预测Agent编排器
    
    【设计模式】
    - 使用组合模式：包含多个Agent实例
    - 使用状态模式：通过状态字典在Agent之间传递数据
    
    【职责】
    - 仅负责编排和状态转换，不负责具体业务逻辑
    - 具体业务逻辑由各个Agent实现
    """
    
    def __init__(self, config: Dict[str, Any], model: str = "gpt-4o", debug: bool = False):
        """
        初始化Agent工作流编排器
        
        Args:
            config: 配置字典，包含所有运行参数
            model: LLM模型名称（如 "gpt-4o"）
            debug: 是否开启调试模式
        """
        self.config = config
        self.model = model
        self.debug = debug
        
        # 初始化文件管理器，用于保存结果
        self.file_manager = FileManager(config.get('output_dir', 'results'))
        self.path_manager = self.file_manager.path_manager
        
        # ===== 初始化所有Agent =====
        # 每个Agent都接收同样的model和config参数
        # API密钥通过环境变量自动获取
        self.preprocess_agent = PreprocessAgent(model, config)  # 数据预处理
        self.analysis_agent = AnalysisAgent(model, config)      # 数据分析
        self.validation_agent = ValidationAgent(model, config)  # 模型验证
        self.forecast_agent = ForecastAgent(model, config)      # 预测生成
        self.report_agent = ReportAgent(model, config)          # 报告生成
        
        # 构建LangGraph工作流
        self.graph = self._build_graph()

    def _create_agent_nodes(self):
        """
        创建Agent节点映射
        
        Returns:
            dict: 节点名称到节点函数的映射
            
        【设计说明】
        将Agent的run方法封装成节点函数，便于LangGraph调用
        """
        return {
            "preprocess": self._preprocess_node,  # 预处理节点
            "analyze": self._analyze_node,        # 分析节点
            "validate": self._validate_node,      # 验证节点
            "forecast": self._forecast_node,      # 预测节点
            "report": self._report_node,          # 报告节点
        }

    def _preprocess_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理节点函数
        
        【职责】
        - 调用PreprocessAgent进行数据预处理
        - 将结果存入状态字典
        
        Args:
            state: 当前状态字典，包含 validation_data 等
            
        Returns:
            更新后的状态字典，添加 preprocessed_data 和 preprocess_result
        """
        # 调用Agent的run方法
        result = self.preprocess_agent.run(state["validation_data"])
        
        # 处理返回结果
        # 如果结果是DataFrame，直接使用；否则从字典中提取
        state["preprocessed_data"] = result if isinstance(result, pd.DataFrame) else result.get("cleaned_data", state["validation_data"])
        state["preprocess_result"] = result
        
        return state

    def _analyze_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析节点函数
        
        【职责】
        - 调用AnalysisAgent分析预处理后的数据
        - 生成数据特征分析报告
        """
        # 获取预处理阶段生成的可视化
        visualizations = state["preprocess_result"]["visualizations"]
        
        # 调用分析Agent
        result = self.analysis_agent.run(state["preprocessed_data"], visualizations)
        state["analysis_result"] = result
        
        print("analysis_result: ", state["analysis_result"])  # 调试输出
        return state

    def _validate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证节点函数
        
        【职责】
        - 获取可用模型列表
        - 调用ValidationAgent进行模型选择和超参数优化
        - 返回最优模型及其参数
        """
        # 获取可用模型列表（在配置中定义）
        available_models = self.config.get('models').get('available_models')
        print(f"{len(available_models)} available models: {available_models}")
        
        # 调用验证Agent
        # 传入分析结果、可用模型列表和验证数据
        validation_data = state["preprocessed_data"]
        result = self.validation_agent.run(
            state["analysis_result"], 
            available_models, 
            validation_data
        )
        
        state["validation_result"] = result
        
        # 从结果中提取关键信息
        # result是一个模型列表，每个元素包含模型名、超参数和验证分数
        state["selected_models"] = [m['model'] for m in result]
        state["best_hyperparameters"] = {m['model']: m['hyperparameters'] for m in result}
        state["model_validation_scores"] = {m['model']: m['validation_score'] for m in result}
        
        return state

    def _forecast_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        预测节点函数
        
        【职责】
        - 使用选定的模型和最优超参数进行预测
        - 生成个体预测和集成预测
        - 计算预测评估指标
        """
        print(f"Forecast node: Processing slice {state.get('slice_info', {}).get('slice_id', 'unknown')}")
        print(f"Forecast node: Selected models: {state.get('selected_models', [])}")
        print(f"Forecast node: Test data shape: {state.get('test_data', pd.DataFrame()).shape}")
        
        # 调用预测Agent
        result = self.forecast_agent.run(
            state["selected_models"],        # 选定的模型列表
            state["best_hyperparameters"],   # 最优超参数
            state["test_data"]               # 测试数据
        )
        
        state["forecast_result"] = result
        return state

    def _report_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        报告节点函数
        
        【职责】
        - 汇总所有Agent的结果
        - 生成综合实验报告
        """
        print(f"Report node: Processing slice {state.get('slice_info', {}).get('slice_id', 'unknown')+1}")
        
        # 构建实验摘要字典
        experiment_summary = {
            'slice_info': state.get('slice_info', {}),
            'preprocess_result': {
                'cleaned_data_shape': state.get('preprocessed_data', pd.DataFrame()).shape,
                'analysis_report': state.get('preprocess_result', {}).get('analysis_report', {}),
                'visualizations': state.get('preprocess_result', {}).get('visualizations', {}),
                'outlier_info': state.get('preprocess_result', {}).get('outlier_info', {}),
                'preprocess_config': state.get('preprocess_result', {}).get('preprocess_config', {})
            },
            'analysis_result': state.get('analysis_result', {}),
            'validation_result': {
                'selected_models': state.get('selected_models', []),
                'best_hyperparameters': state.get('best_hyperparameters', {}),
                'model_validation_scores': state.get('model_validation_scores', {})
            },
            'forecast_result': {
                'individual_predictions': state.get('forecast_result', {}).get('individual_predictions', {}),
                'ensemble_predictions': state.get('forecast_result', {}).get('ensemble_predictions', {}),
                'test_metrics': state.get('forecast_result', {}).get('test_metrics', {}),
                'forecast_metrics': state.get('forecast_result', {}).get('forecast_metrics', {}),
                'confidence_intervals': state.get('forecast_result', {}).get('confidence_intervals', {}),
                'visualizations': state.get('forecast_result', {}).get('visualizations', {})
            },
            'config': state.get('config', {})
        }
        
        # 调用报告Agent生成报告
        report = self.report_agent.run(experiment_summary)
        state["report"] = report
        
        return state

    def _build_graph(self):
        """
        构建LangGraph工作流
        
        【核心方法】
        定义Agent之间的执行顺序和数据流
        
        Returns:
            编译后的工作流图
        """
        nodes = self._create_agent_nodes()
        
        # 创建状态图，使用dict作为状态类型
        workflow = StateGraph(dict)
        
        # ===== 添加节点 =====
        # 每个节点对应一个Agent
        workflow.add_node("preprocess", nodes["preprocess"])
        workflow.add_node("analyze", nodes["analyze"])
        workflow.add_node("validate", nodes["validate"])
        workflow.add_node("forecast", nodes["forecast"])
        workflow.add_node("report", nodes["report"])
        
        # ===== 添加边（定义执行顺序）=====
        # 线性顺序：preprocess -> analyze -> validate -> forecast -> report -> END
        workflow.add_edge("preprocess", "analyze")
        workflow.add_edge("analyze", "validate")
        workflow.add_edge("validate", "forecast")
        workflow.add_edge("forecast", "report")
        workflow.add_edge("report", END)  # END是LangGraph的特殊节点，表示结束
        
        # 设置入口点
        workflow.set_entry_point("preprocess")
        
        # 编译工作流（使其可执行）
        return workflow.compile()

    def run(self) -> dict:
        """
        执行完整的预测工作流
        
        【执行流程】
        1. 加载数据
        2. 创建数据切片
        3. 对每个切片执行Agent工作流
        4. 聚合所有切片的结果
        
        Returns:
            包含所有结果和聚合结果的字典
        """
        # ===== 步骤1: 加载和预处理数据 =====
        print("Start loading data...")
        data_path = self.config.get('data_path')
        df = DataLoader.load_data(data_path)
        
        # 获取列名配置
        date_column = self.config.get('date_column', 'date')
        value_column = self.config.get('value_column', 'OT')
        
        # 转换为时间序列格式
        df_ts = DataPreprocessor.convert_to_time_series(df, date_column, value_column)
        
        # ===== 步骤2: 创建数据切片 =====
        num_slices = self.config.get('num_slices', 10)
        input_length = self.config.get('input_length', 512)
        horizon = self.config.get('horizon', 96)
        
        slices = DataSplitter.create_slices(df_ts, num_slices, input_length, horizon)
        
        all_results = []
        
        # 设置切片之间的延迟，避免API限速
        import time
        delay_between_slices = 3
        
        print(f"Processing {len(slices)} slices with {delay_between_slices}s delay...")
        print("=" * 60)
        
        # ===== 步骤3: 对每个切片执行工作流 =====
        for i, s in enumerate(slices):
            slice_start_time = time.time()
            print(f"Processing slice {i+1}/{len(slices)} (ID: {s['slice_id']})...")
            
            # 提取切片数据
            validation_data = s['validation']  # 验证数据
            test_data = s['test']              # 测试数据
            
            # 切片信息
            slice_info = {
                'slice_id': s['slice_id'],
                'validation_start': s['validation_start'],
                'validation_end': s['validation_end'],
                'test_start': s['test_start'],
                'test_end': s['test_end'],
            }
            
            # 构建初始状态
            state = {
                "validation_data": validation_data,
                "test_data": test_data,
                "slice_info": slice_info,
                "config": self.config
            }
            
            # 执行工作流
            if self.debug:
                # 调试模式：流式执行，记录每个步骤
                trace = []
                for chunk in self.graph.stream(state):
                    trace.append(chunk)
                final_state = trace[-1]
            else:
                # 正常模式：直接执行
                final_state = self.graph.invoke(state)
            
            all_results.append(final_state)
            
            # 打印执行时间
            slice_end_time = time.time()
            print(f"Slice {i+1} completed in {slice_end_time - slice_start_time:.2f} seconds")
            
            # 切片之间添加延迟
            if i < len(slices) - 1:
                print(f"Waiting {delay_between_slices} seconds...")
                time.sleep(delay_between_slices)
            
            print("-" * 40)
        
        print(f"All {len(slices)} slices processed successfully!")
        print("=" * 60)
        
        # ===== 步骤4: 聚合结果 =====
        aggregated_results = self._aggregate_slice_results(all_results)
        
        return {
            "all_results": all_results,              # 所有切片的详细结果
            "aggregated_results": aggregated_results, # 聚合后的结果
            "report": all_results[-1].get("report")  # 最后一个切片的报告
        }

    def _aggregate_slice_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        聚合所有切片的结果
        
        【聚合方法】
        - 个体预测：对各切片的预测取平均
        - 集成预测：对各切片的集成预测取平均
        - 评估指标：对各切片的指标取平均
        
        Args:
            all_results: 所有切片的结果列表
            
        Returns:
            聚合后的结果字典
        """
        print("Aggregating results from all slices...")
        
        if not all_results:
            return {}
        
        # 收集所有预测和指标
        all_individual_predictions = {}  # 个体模型预测
        all_ensemble_predictions = []    # 集成预测
        all_test_metrics = {}            # 测试指标
        all_forecast_metrics = {}        # 预测指标
        
        # 遍历每个切片的结果
        for i, result in enumerate(all_results):
            forecast_result = result.get('forecast_result')
            
            if not forecast_result:
                continue
            
            # 收集个体预测
            individual_predictions = forecast_result.get('individual_predictions', {})
            for model_name, predictions in individual_predictions.items():
                if model_name not in all_individual_predictions:
                    all_individual_predictions[model_name] = []
                all_individual_predictions[model_name].append(predictions)
            
            # 收集集成预测
            ensemble_predictions = forecast_result.get('ensemble_predictions', {})
            if ensemble_predictions and 'predictions' in ensemble_predictions:
                all_ensemble_predictions.append(ensemble_predictions['predictions'])
            
            # 收集测试指标
            test_metrics = forecast_result.get('test_metrics', {})
            for model_name, metrics in test_metrics.items():
                if model_name not in all_test_metrics:
                    all_test_metrics[model_name] = {'mse': [], 'mae': [], 'mape': []}
                all_test_metrics[model_name]['mse'].append(metrics.get('mse', float('inf')))
                all_test_metrics[model_name]['mae'].append(metrics.get('mae', float('inf')))
                all_test_metrics[model_name]['mape'].append(metrics.get('mape', float('inf')))
        
        # ===== 计算平均值 =====
        
        # 平均个体预测
        averaged_individual_predictions = {}
        for model_name, predictions_list in all_individual_predictions.items():
            if predictions_list:
                predictions_array = np.array(predictions_list)
                averaged_predictions = np.mean(predictions_array, axis=0)
                averaged_individual_predictions[model_name] = averaged_predictions.tolist()
        
        # 平均集成预测
        averaged_ensemble_predictions = {}
        if all_ensemble_predictions:
            ensemble_array = np.array(all_ensemble_predictions)
            averaged_ensemble = np.mean(ensemble_array, axis=0)
            averaged_ensemble_predictions = {
                'predictions': averaged_ensemble.tolist(),
                'method_used': 'average_across_slices',
                'num_slices': len(all_results)
            }
        
        # 平均测试指标
        averaged_test_metrics = {}
        for model_name, metrics_list in all_test_metrics.items():
            averaged_test_metrics[model_name] = {
                'mse': np.mean(metrics_list['mse']),
                'mae': np.mean(metrics_list['mae']),
                'mape': np.mean(metrics_list['mape'])
            }
        
        # 构建聚合结果
        aggregated_results = {
            'individual_predictions': averaged_individual_predictions,
            'ensemble_predictions': averaged_ensemble_predictions,
            'test_metrics': averaged_test_metrics,
            'aggregation_info': {
                'num_slices': len(all_results),
                'aggregation_method': 'average',
            }
        }
        
        print(f"Aggregated results from {len(all_results)} slices")
        return aggregated_results
```

---

## 3. agents/preprocess_agent.py - 预处理Agent

```python
"""
preprocess_agent.py - 数据预处理Agent

【文件作用】
负责时间序列数据的清洗和准备工作：
1. 数据验证 - 检查格式和完整性
2. 缺失值处理 - 使用LLM推荐的策略
3. 异常值检测和处理
4. 生成数据可视化
5. 生成数据质量报告

【核心设计】
- 使用LLM分析数据质量并推荐预处理策略
- 将LLM的"智慧"与传统数据处理工具结合
- 为后续Agent提供高质量的清洗数据
"""

# [省略导入部分...]

# ===== 系统提示词 =====
# 这是发送给LLM的角色说明，定义了Agent的能力和职责
PREPROCESS_SYSTEM_PROMPT = """
You are the Data Preprocessing Chief Agent for an advanced time series forecasting system.
你是一个高级时间序列预测系统的数据预处理首席Agent。

Your mission is to ensure that all input data is of the highest possible quality.
你的任务是确保所有输入数据达到最高质量。

Background:
- You have deep expertise in time series data cleaning, anomaly detection, and preparation.
- 你在时序数据清洗、异常检测和准备方面有深厚专业知识。

Your responsibilities:
- Rigorously assess the quality of the input time series
- 严格评估输入时序的质量
- For each issue, recommend the most appropriate handling strategy
- 对每个问题，推荐最合适的处理策略
- Justify your recommendations with clear reasoning
- 用清晰的推理证明你的建议
"""


class PreprocessAgent:
    """
    数据预处理Agent
    
    【职责】
    - 数据加载和验证
    - 数据清洗（缺失值、异常值）
    - 生成可视化和分析报告
    
    【与LLM的交互】
    - 使用LLM分析数据质量
    - 使用LLM推荐预处理策略
    - 使用LLM决定可视化类型
    """
    
    def __init__(self, model: str = "gpt-4o", config: dict = None):
        """
        初始化预处理Agent
        
        Args:
            model: LLM模型名称
            config: 配置字典
        """
        # 初始化LLM客户端
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,  # 低温度 = 更确定性的输出
        )
        
        # 初始化LLM工具类（封装了与LLM交互的方法）
        self.tools = PreprocessLLMTools(self.llm)
        
        self.config = config
        
        # 初始化可视化器
        self.visualizer = TimeSeriesVisualizer(self.config)
        
        # 获取预处理配置
        self.preprocess_config = config.get('preprocess')
        self.outlier_threshold = self.preprocess_config.get('outlier_threshold', 1.5)
        
        # 初始化记忆管理器
        self.memory = ExperimentMemory(self.config)
    
    def process(self, data: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """
        执行完整的预处理工作流
        
        【处理步骤】
        1. 数据验证
        2. LLM分析数据质量并推荐策略
        3. 使用推荐策略清洗数据
        4. 检测和处理异常值
        5. 生成可视化
        6. 生成综合分析报告
        7. 保存结果
        8. 更新记忆
        
        Args:
            data: 原始数据（DataFrame格式）
            output_dir: 输出目录
            
        Returns:
            预处理结果字典
        """
        logger.info("Starting data preprocessing...")
        
        try:
            # ===== 步骤1: 数据验证 =====
            validation_result = self._validate_data(data)
            
            # ===== 步骤2: LLM分析数据质量 =====
            # 这是关键步骤！LLM会分析数据并推荐处理策略
            initial_quality_analysis = self._analyze_data_quality(data, {})
            
            # 从LLM响应中提取推荐策略
            missing_value_strategy = initial_quality_analysis.get(
                'recommended_strategies', {}
            ).get('missing_value_strategy', 'interpolate')
            
            outlier_detect_strategy = initial_quality_analysis.get(
                'recommended_strategies', {}
            ).get('outlier_detect_strategy', 'iqr')
            
            outlier_handle_strategy = initial_quality_analysis.get(
                'recommended_strategies', {}
            ).get('outlier_handle_strategy', 'clip')
            
            logger.info(f"LLM recommended strategies:")
            logger.info(f"  Missing value: {missing_value_strategy}")
            logger.info(f"  Outlier detect: {outlier_detect_strategy}")
            logger.info(f"  Outlier handle: {outlier_handle_strategy}")
            
            # ===== 步骤3: 使用推荐策略清洗数据 =====
            cleaned_data = self._clean_data(data, missing_value_strategy)
            
            # ===== 步骤4: 检测异常值 =====
            outlier_info = self._detect_outliers(cleaned_data, outlier_detect_strategy)
            
            # ===== 步骤5: 处理异常值 =====
            if outlier_info:
                cleaned_data = self._handle_outliers(
                    cleaned_data, outlier_info, outlier_handle_strategy
                )
            
            # ===== 步骤6: 生成可视化 =====
            visualizations = self._generate_visualizations(cleaned_data, output_dir)
            
            # ===== 步骤7: 生成综合分析报告 =====
            analysis_report = self._generate_comprehensive_analysis_report(
                cleaned_data, visualizations
            )
            
            # ===== 步骤8: 保存结果 =====
            self._save_preprocess_results(cleaned_data, analysis_report, output_dir)
            
            # ===== 步骤9: 更新记忆 =====
            self._update_memory(cleaned_data, analysis_report, visualizations)
            
            # 构建返回结果
            result = {
                'cleaned_data': cleaned_data,
                'analysis_report': analysis_report,
                'outlier_info': outlier_info,
                'validation_result': validation_result,
                'visualizations': visualizations,
                'preprocess_config': {
                    'missing_strategy': missing_value_strategy,
                    'outlier_detect_strategy': outlier_detect_strategy,
                    'outlier_handle_strategy': outlier_handle_strategy,
                }
            }
            
            logger.info("Data preprocessing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def _analyze_data_quality(self, data: pd.DataFrame, outlier_info: Dict) -> Dict:
        """
        使用LLM分析数据质量
        
        【LLM交互】
        发送数据统计信息给LLM，让其分析质量并推荐策略
        
        【Prompt设计要点】
        1. 提供数据的结构化表示
        2. 明确要求返回JSON格式
        3. 提供响应格式模板
        """
        # 将数据转换为字典格式发送给LLM
        sample = data.to_dict(orient='list')
        
        prompt = f"""
Given the following time series data (as a Python dict):

{sample}

Please analyze the data quality and provide:

1. Basic statistics (mean, std, min, max, trend)
2. Missing value information
3. Outlier information  
4. Data quality assessment (0-1 score)
5. Recommended preprocessing strategies:
   - missing_value_strategy: choose from 'interpolate', 'forward_fill', 'mean', etc.
   - outlier_detect_strategy: choose from 'iqr', 'zscore', 'percentile'
   - outlier_handle_strategy: choose from 'clip', 'drop', 'interpolate'

IMPORTANT: Return ONLY a JSON object, no markdown formatting.

{{
    "basic_stats": {{...}},
    "missing_info": {{...}},
    "outlier_info": {{...}},
    "quality_assessment": {{...}},
    "recommended_strategies": {{...}}
}}
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # 解析JSON响应
            return json.loads(response.content)
            
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            # 使用回退策略
            return self._generate_fallback_data_quality_analysis(data, outlier_info)
    
    def _generate_visualizations(self, data: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """
        使用LLM决定生成哪些可视化
        
        【创新设计】
        不是硬编码可视化类型，而是让LLM根据数据特征决定
        """
        # 让LLM决定需要哪些可视化
        viz_decision_prompt = f"""
Given the following time series data:

Data shape: {data.shape}
Data columns: {list(data.columns)}

Please decide what visualizations would be most useful.

Choose from:
- time_series: Basic time series plot
- distribution: Histogram, box plot, KDE
- rolling_stats: Rolling mean, std
- autocorrelation: ACF/PACF plots
- seasonal_decomposition: Trend, seasonal, residual

Return as JSON:
{{
    "visualizations": [
        {{"name": "...", "type": "...", "description": "..."}}
    ]
}}
"""
        
        response = self.llm.invoke([HumanMessage(content=viz_decision_prompt)])
        viz_plan = json.loads(response.content)
        
        # 根据LLM的决定生成可视化
        visualizations = {}
        for viz_config in viz_plan.get("visualizations", []):
            viz_type = viz_config.get("type")
            
            if viz_type == "time_series":
                path = self._create_time_series_plot(data, viz_config, output_dir)
            elif viz_type == "distribution":
                path = self._create_distribution_plot(data, viz_config, output_dir)
            # ... 其他可视化类型
            
            if path:
                visualizations[viz_config.get("name")] = path
        
        return visualizations
```

---

## 9. config/default_config.py - 默认配置

```python
"""
default_config.py - 默认配置文件

【文件作用】
集中管理所有配置参数，包括：
1. LLM配置 - 模型选择、温度等
2. 数据预处理配置 - 缺失值处理、异常值处理策略
3. 模型配置 - 可用模型、集成方法
4. 实验配置 - 切片数、序列长度、预测步长
5. 可视化配置 - 图表大小、格式
6. 系统配置 - 调试模式、日志级别

【使用方法】
from config.default_config import DEFAULT_CONFIG
config = DEFAULT_CONFIG.copy()
config["num_slices"] = 10  # 自定义
"""

import os
from typing import Dict, Any, List

# ===== LLM配置 =====
# 支持多个LLM提供商
LLM_CONFIG = {
    "openai": {
        "provider": "openai",
        "model": "gpt-4o",           # 模型名称
        "temperature": 0.2,          # 温度：0=确定性，1=随机性
        "max_tokens": 4000,          # 最大输出token数
        "api_base": "https://api.openai.com/v1"
    },
    "google": {
        "provider": "google",
        "model": "gemini-2.0-flash",
        "temperature": 0.1,
        "max_tokens": 4000,
    },
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.1,
        "max_tokens": 4000,
    }
}

# ===== 数据预处理配置 =====
PREPROCESS_CONFIG = {
    # 缺失值处理策略
    # 可选: interpolate, forward_fill, backward_fill, mean, median, drop, zero
    "missing_value_strategy": "interpolate",
    
    # 异常值处理策略
    # 可选: clip, drop, iqr
    "outlier_strategy": "clip",
    
    # 是否标准化数据
    "normalization": False,
    
    # 标准化方法: minmax, standard, quantile, log, dummy
    "scaler_type": "dummy",
    
    # 是否生成可视化
    "visualization": True,
    
    # 是否保存中间结果
    "save_intermediate": True
}

# ===== 模型配置 =====
MODEL_CONFIG = {
    # 可用模型列表（21个模型！）
    "available_models": [
        "ARIMA",              # 自回归积分滑动平均
        "RandomWalk",         # 随机游走
        "ExponentialSmoothing", # 指数平滑
        "MovingAverage",      # 移动平均
        "LinearRegression",   # 线性回归
        "PolynomialRegression", # 多项式回归
        "RidgeRegression",    # 岭回归
        "LassoRegression",    # Lasso回归
        "ElasticNet",         # 弹性网络
        "SVR",                # 支持向量回归
        "RandomForest",       # 随机森林
        "GradientBoosting",   # 梯度提升
        "XGBoost",            # XGBoost
        "LightGBM",           # LightGBM
        "NeuralNetwork",      # 神经网络
        "LSTM",               # 长短期记忆
        "Prophet",            # Facebook Prophet
        "TBATS",              # TBATS
        "Theta",              # Theta方法
        "Croston",            # Croston方法
        "Transformer"         # Transformer
    ],
    
    # 最终选择的模型数量
    "k_models": 3,
    
    # 候选模型数量（LLM推荐）
    "n_candidates": 5,
    
    # 集成方法: simple_average, weighted_average, trimmed_mean, median
    "ensemble_method": "weighted_average",
    
    # 是否进行超参数优化
    "hyperparameter_optimization": True,
    
    # 交叉验证折数
    "cross_validation_folds": 3
}

# ===== 实验配置 =====
EXPERIMENT_CONFIG = {
    "num_slices": 10,        # 数据切片数量
    "input_length": 512,     # 输入序列长度
    "horizon": 96,           # 预测步长
    "validation_ratio": 0.2, # 验证集比例
    "test_ratio": 0.2,       # 测试集比例
    "random_seed": 42,       # 随机种子（保证可复现）
    "parallel_processing": False,  # 是否并行处理
    "max_workers": 4         # 最大工作进程数
}

# ===== 评估指标配置 =====
METRICS_CONFIG = {
    "primary_metric": "mse",     # 主要指标: mse, mae, mape
    "secondary_metrics": ["mae", "mape"],  # 次要指标
    "confidence_level": 0.95,    # 置信水平
    "bootstrap_samples": 1000    # Bootstrap采样数
}

# ===== 可视化配置 =====
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),      # 图表大小
    "dpi": 300,                  # 分辨率
    "style": "seaborn-v0_8",     # 绘图风格
    "color_palette": "husl",     # 颜色方案
    "save_format": "png",        # 保存格式
    "show_plots": False          # 是否显示图表
}

# ===== 报告配置 =====
REPORT_CONFIG = {
    "generate_comprehensive_report": True,   # 生成综合报告
    "include_confidence_intervals": True,    # 包含置信区间
    "include_model_interpretability": True,  # 包含模型解释性
    "report_format": "json",                 # 报告格式
    "save_individual_reports": True          # 保存单独报告
}

# ===== 系统配置 =====
SYSTEM_CONFIG = {
    "debug": False,          # 调试模式
    "verbose": False,        # 详细输出
    "log_level": "INFO",     # 日志级别
    "save_logs": True,       # 保存日志
    "max_memory_usage": "8GB", # 最大内存使用
    "timeout": 3600,         # 超时时间（秒）
    "retry_attempts": 3      # 重试次数
}

# ===== 路径配置 =====
PATH_CONFIG = {
    "output_dir": "results",
    "log_dir": "logs",
    "cache_dir": "cache",
    "model_dir": "models",
    "visualization_dir": "visualizations",
    "report_dir": "reports"
}

# ===== 默认配置（合并所有子配置）=====
DEFAULT_CONFIG = {
    # 基础配置
    "data_path": None,           # 需要用户指定
    "output_dir": "results",
    
    # LLM配置
    "llm_provider": "openai",
    "llm_model": "gpt-4o",
    "llm_temperature": 0.1,
    "llm_max_tokens": 4000,
    
    # 实验参数
    "num_slices": 10,
    "input_length": 512,
    "horizon": 96,
    "k_models": 3,
    
    # 系统参数
    "debug": False,
    "verbose": False,
    "random_seed": 42,
    
    # 合并子配置
    "preprocess": PREPROCESS_CONFIG,
    "models": MODEL_CONFIG,
    "experiment": EXPERIMENT_CONFIG,
    "metrics": METRICS_CONFIG,
    "visualization": VISUALIZATION_CONFIG,
    "report": REPORT_CONFIG,
    "system": SYSTEM_CONFIG,
    "paths": PATH_CONFIG
}

# ===== 模型超参数配置 =====
# 定义每个模型的超参数搜索空间
MODEL_HYPERPARAMETERS = {
    "ARMA": {
        "p": [1, 2, 3],      # 自回归阶数
        "q": [1, 2, 3],      # 移动平均阶数
        "d": [0, 1]          # 差分阶数
    },
    "LSTM": {
        "units": [50, 100, 200],    # 隐藏单元数
        "layers": [1, 2, 3],        # 层数
        "dropout": [0.1, 0.2, 0.3], # Dropout率
        "batch_size": [32, 64, 128],
        "epochs": [50, 100, 200]
    },
    "RandomForest": {
        "n_estimators": [100, 200, 500],  # 树的数量
        "max_depth": [10, 20, None],       # 最大深度
        "min_samples_split": [2, 5, 10]   # 最小分裂样本数
    },
    # ... 更多模型
}

# ===== 数据质量阈值 =====
DATA_QUALITY_THRESHOLDS = {
    "missing_ratio_threshold": 0.3,   # 缺失率阈值
    "outlier_ratio_threshold": 0.1,   # 异常值比例阈值
    "min_data_points": 100,           # 最小数据点数
    "max_data_points": 100000,        # 最大数据点数
    "stationarity_p_value": 0.05      # 平稳性检验p值
}


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证配置有效性
    
    Args:
        config: 用户提供的配置
        
    Returns:
        验证后的配置
        
    Raises:
        ValueError: 如果必需参数缺失
    """
    validated_config = config.copy()
    
    # 验证必需参数
    if not config.get("data_path"):
        raise ValueError("data_path is required")
    
    # 验证数值范围
    if config.get("num_slices", 0) <= 0:
        validated_config["num_slices"] = 10
    
    if config.get("input_length", 0) <= 0:
        validated_config["input_length"] = 512
    
    if config.get("horizon", 0) <= 0:
        validated_config["horizon"] = 96
    
    return validated_config
```

---

## 更多模块...

由于篇幅限制，其他模块的详细注释请参考源代码文件。每个模块都遵循类似的注释模式：

1. **文件头部注释**：说明文件作用、核心概念、关键类
2. **类注释**：说明类的职责、设计模式、与其他模块的关系
3. **方法注释**：说明参数、返回值、处理步骤
4. **行内注释**：解释关键代码逻辑

---

## 学习建议

1. **从配置文件开始**：理解系统的可配置性
2. **然后是工具模块**：了解基础设施
3. **接着是Agent模块**：理解核心业务逻辑
4. **最后是工作流**：看懂整体架构

**祝学习顺利！** 🎓
