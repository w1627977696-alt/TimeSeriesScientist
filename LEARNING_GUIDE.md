# TimeSeriesScientist 项目学习指南

## 📚 目录

1. [项目概述](#1-项目概述)
2. [项目架构](#2-项目架构)
3. [核心概念](#3-核心概念)
4. [学习路线](#4-学习路线)
5. [模块详解](#5-模块详解)
6. [代码注释](#6-代码注释)
7. [实践建议](#7-实践建议)
8. [常见问题](#8-常见问题)

---

## 1. 项目概述

### 1.1 什么是 TimeSeriesScientist？

**TimeSeriesScientist (TSci)** 是一个基于 LLM（大语言模型）的智能代理框架，专门用于时间序列预测任务。它是第一个将 AI Agent 技术应用于通用时间序列预测的开源框架。

### 1.2 核心特点

- **🤖 多智能体协作**：使用 4 个专门的 Agent 协同工作
- **🧠 LLM 驱动决策**：利用 GPT-4 等大语言模型进行智能决策
- **📊 端到端自动化**：从数据预处理到生成报告全程自动化
- **🔍 可解释性强**：每个决策都有明确的推理过程
- **🔧 高度可扩展**：支持多种预测模型和集成策略

### 1.3 四大核心 Agent

| Agent | 中文名 | 职责 |
|-------|--------|------|
| **Curator (PreprocessAgent)** | 数据策展员 | 数据加载、清洗、质量评估 |
| **Planner (AnalysisAgent)** | 规划分析员 | 数据特征分析、模式识别 |
| **Forecaster (ValidationAgent + ForecastAgent)** | 预测执行员 | 模型选择、训练和预测 |
| **Reporter (ReportAgent)** | 报告生成员 | 汇总结果、生成报告 |

---

## 2. 项目架构

### 2.1 目录结构

```
TimeSeriesScientist/
├── time_series_agent/          # 核心代码目录
│   ├── agents/                 # Agent 模块（核心！）
│   │   ├── preprocess_agent.py # 数据预处理 Agent
│   │   ├── analysis_agent.py   # 数据分析 Agent
│   │   ├── validation_agent.py # 模型验证 Agent
│   │   ├── forecast_agent.py   # 预测 Agent
│   │   ├── report_agent.py     # 报告生成 Agent
│   │   └── memory.py           # Agent 记忆管理
│   │
│   ├── graph/                  # 工作流编排
│   │   └── agent_graph.py      # LangGraph 工作流定义
│   │
│   ├── utils/                  # 工具模块
│   │   ├── data_utils.py       # 数据处理工具
│   │   ├── model_library.py    # 模型库（20+预测模型）
│   │   ├── visualization_utils.py  # 可视化工具
│   │   └── file_utils.py       # 文件管理工具
│   │
│   ├── config/                 # 配置模块
│   │   └── default_config.py   # 默认配置
│   │
│   ├── main.py                 # 程序入口
│   └── requirements.txt        # 依赖列表
│
├── dataset/                    # 数据集目录
│   └── ETTh1.csv              # 示例数据
│
├── assets/                     # 资源文件
├── README.md                   # 项目说明
└── LICENSE                     # 开源协议
```

### 2.2 数据流程图

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  原始数据    │ -> │  PreprocessAgent │ -> │  清洗后的数据    │
│  (CSV文件)  │    │  (数据预处理)     │    │  (DataFrame)    │
└─────────────┘    └──────────────────┘    └─────────────────┘
                                                   │
                                                   ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  分析报告    │ <- │  AnalysisAgent   │ <- │  清洗后的数据    │
│  (JSON)     │    │  (数据分析)      │    │                 │
└─────────────┘    └──────────────────┘    └─────────────────┘
                                                   │
                                                   ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  最优模型    │ <- │ ValidationAgent  │ <- │  分析结果       │
│  (参数)     │    │  (模型验证)      │    │                 │
└─────────────┘    └──────────────────┘    └─────────────────┘
                                                   │
                                                   ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  预测结果    │ <- │  ForecastAgent   │ <- │  验证后的模型    │
│  (数值)     │    │  (预测生成)      │    │                 │
└─────────────┘    └──────────────────┘    └─────────────────┘
                                                   │
                                                   ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  最终报告    │ <- │   ReportAgent    │ <- │  所有结果       │
│  (Markdown) │    │  (报告生成)      │    │                 │
└─────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 3. 核心概念

### 3.1 什么是 AI Agent？

**AI Agent（人工智能代理）** 是一种能够自主感知环境、做出决策并执行行动的智能系统。

**关键特征：**
- **自主性**：能够独立完成任务
- **反应性**：能够响应环境变化
- **目标导向**：有明确的任务目标
- **学习能力**：能够从经验中学习

### 3.2 LangChain 和 LangGraph

#### LangChain
LangChain 是一个用于构建 LLM 应用的框架，提供：
- **模型集成**：统一接口调用各种 LLM
- **Prompt 管理**：模板化的提示词管理
- **链式调用**：将多个操作组合成流程

```python
# LangChain 基本使用示例
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o")
response = llm.invoke([HumanMessage(content="你好")])
```

#### LangGraph
LangGraph 是 LangChain 的扩展，专门用于构建多 Agent 工作流：
- **状态管理**：跟踪工作流状态
- **节点定义**：每个 Agent 是一个节点
- **边连接**：定义 Agent 之间的数据流

```python
# LangGraph 基本使用示例
from langgraph.graph import StateGraph, END

workflow = StateGraph(dict)
workflow.add_node("agent1", agent1_function)
workflow.add_node("agent2", agent2_function)
workflow.add_edge("agent1", "agent2")
workflow.add_edge("agent2", END)
```

### 3.3 时间序列预测基础

**时间序列**是按时间顺序排列的数据点序列，如股票价格、温度记录等。

**常见模型类型：**

| 类型 | 模型 | 特点 |
|------|------|------|
| 统计模型 | ARIMA, 指数平滑 | 简单、可解释性强 |
| 机器学习 | 随机森林, XGBoost | 特征工程灵活 |
| 深度学习 | LSTM, Transformer | 处理复杂模式 |

---

## 4. 学习路线

### 4.1 推荐学习顺序

```
阶段1：基础知识（1-2周）
    ├── Python 基础
    ├── pandas 数据处理
    └── 时间序列基础

阶段2：核心框架（1周）
    ├── LangChain 基础
    ├── LangGraph 工作流
    └── OpenAI API 使用

阶段3：项目代码（2周）
    ├── 配置模块 (config/)
    ├── 工具模块 (utils/)
    ├── Agent 模块 (agents/)
    └── 工作流模块 (graph/)

阶段4：实践应用（1-2周）
    ├── 运行示例
    ├── 修改配置
    └── 自定义数据集
```

### 4.2 每个阶段的学习资源

#### 阶段1：基础知识

**Python 基础：**
- 官方教程：https://docs.python.org/zh-cn/3/tutorial/
- 推荐书籍：《Python编程：从入门到实践》

**pandas 数据处理：**
- 官方文档：https://pandas.pydata.org/docs/
- 实战练习：Kaggle 的 pandas 教程

**时间序列基础：**
- 《使用Python进行时间序列分析》
- statsmodels 官方教程

#### 阶段2：核心框架

**LangChain：**
- 官方文档：https://python.langchain.com/docs/
- 教程：LangChain Cookbook

**LangGraph：**
- 官方文档：https://langchain-ai.github.io/langgraph/
- 示例代码：LangGraph Examples

**OpenAI API：**
- 官方文档：https://platform.openai.com/docs/

### 4.3 学习方法建议

1. **先跑通示例**
   ```bash
   cd time_series_agent
   export OPENAI_API_KEY="your-key"
   python main.py
   ```

2. **阅读日志输出**
   - 观察每个 Agent 的执行顺序
   - 理解数据如何在 Agent 之间流转

3. **逐个模块学习**
   - 从 `config/default_config.py` 开始
   - 然后是 `utils/` 工具模块
   - 最后是 `agents/` 核心模块

4. **动手修改代码**
   - 尝试修改配置参数
   - 添加新的模型
   - 自定义 Prompt

---

## 5. 模块详解

### 5.1 配置模块 (config/)

#### default_config.py - 默认配置

```python
# 核心配置结构
DEFAULT_CONFIG = {
    # 基础配置
    "data_path": None,           # 数据文件路径
    "output_dir": "results",     # 输出目录
    
    # LLM 配置
    "llm_model": "gpt-4o",       # 使用的 LLM 模型
    "llm_temperature": 0.1,      # 温度参数（影响输出随机性）
    
    # 实验参数
    "num_slices": 10,            # 数据切片数量
    "input_length": 512,         # 输入序列长度
    "horizon": 96,               # 预测步长
    "k_models": 3,               # 选择的模型数量
    
    # 子配置
    "preprocess": {...},         # 预处理配置
    "models": {...},             # 模型配置
    "visualization": {...},      # 可视化配置
}
```

**学习要点：**
- 理解每个参数的含义
- 知道如何修改配置来改变行为

### 5.2 工具模块 (utils/)

#### data_utils.py - 数据处理工具

**核心类：**

```python
class DataLoader:
    """数据加载器 - 支持 CSV、JSON、Excel 等格式"""
    
    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        """通用数据加载方法"""
        # 根据文件扩展名自动选择加载方式

class DataPreprocessor:
    """数据预处理器 - 处理缺失值、异常值等"""
    
    @staticmethod
    def handle_missing_values(df, strategy='interpolate'):
        """处理缺失值
        
        策略选项：
        - interpolate: 插值填充
        - forward_fill: 向前填充
        - mean: 均值填充
        - drop: 删除缺失行
        """

class DataSplitter:
    """数据分割器 - 创建训练/验证/测试集"""
    
    @staticmethod
    def create_slices(df, num_slices, input_length, horizon):
        """创建数据切片
        
        每个切片包含：
        - validation: 验证数据（用于模型选择）
        - test: 测试数据（用于最终评估）
        """
```

#### model_library.py - 模型库

**支持的模型：**

```python
MODEL_FUNCTIONS = {
    # 统计模型
    'ARIMA': predict_arima,              # 自回归移动平均
    'ExponentialSmoothing': predict_es,  # 指数平滑
    'Prophet': predict_prophet,          # Facebook Prophet
    
    # 机器学习模型
    'RandomForest': predict_rf,          # 随机森林
    'XGBoost': predict_xgb,              # XGBoost
    'LightGBM': predict_lgb,             # LightGBM
    
    # 深度学习模型
    'LSTM': predict_lstm,                # 长短期记忆网络
    'Transformer': predict_transformer,  # Transformer
    
    # 更多...
}
```

**模型函数接口：**

```python
def predict_xxx(data: Dict, params: Dict, horizon: int) -> List[float]:
    """标准预测函数接口
    
    Args:
        data: 包含 'value' 键的数据字典
        params: 模型超参数
        horizon: 预测步长
        
    Returns:
        预测值列表
    """
```

### 5.3 Agent 模块 (agents/)

#### preprocess_agent.py - 预处理 Agent

**核心职责：**
1. 数据验证 - 检查数据格式和质量
2. 缺失值处理 - 使用 LLM 推荐的策略
3. 异常值检测和处理
4. 生成可视化图表
5. 生成数据质量报告

**关键代码结构：**

```python
class PreprocessAgent:
    def __init__(self, model="gpt-4o", config=None):
        # 初始化 LLM
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        # 初始化工具类
        self.tools = PreprocessLLMTools(self.llm)
        
    def run(self, data: pd.DataFrame, output_dir: str):
        """执行预处理流程"""
        # 1. 数据验证
        validation_result = self._validate_data(data)
        
        # 2. LLM 分析数据质量，推荐策略
        quality_analysis = self._analyze_data_quality(data, {})
        
        # 3. 根据推荐策略清洗数据
        cleaned_data = self._clean_data(data, strategy)
        
        # 4. 处理异常值
        cleaned_data = self._handle_outliers(cleaned_data, ...)
        
        # 5. 生成可视化
        visualizations = self._generate_visualizations(...)
        
        # 6. 生成分析报告
        report = self._generate_comprehensive_analysis_report(...)
        
        return result
```

**LLM Prompt 设计：**

```python
PREPROCESS_SYSTEM_PROMPT = """
You are the Data Preprocessing Chief Agent for an advanced 
time series forecasting system.

Your responsibilities:
- Rigorously assess the quality of the input time series
- Recommend appropriate handling strategies
- Justify your recommendations with clear reasoning
"""
```

#### analysis_agent.py - 分析 Agent

**核心职责：**
1. 趋势分析 - 识别数据趋势方向和强度
2. 季节性分析 - 检测周期性模式
3. 平稳性检验 - 判断数据是否平稳
4. 生成分析洞察

#### validation_agent.py - 验证 Agent

**核心职责：**
1. 模型选择 - LLM 根据数据特征推荐模型
2. 超参数优化 - 网格搜索最优参数
3. 交叉验证 - 评估模型性能
4. 返回最优模型列表

**关键设计模式：**

```python
# 使用 TypedDict 定义结构化输出
class SelectedModel(TypedDict):
    model: str              # 模型名称
    hyperparameters: Dict   # 超参数
    reason: str             # 选择理由

# 使用 structured_output 获取结构化响应
structured_llm = self.llm.with_structured_output(ModelSelectionOutput)
```

#### forecast_agent.py - 预测 Agent

**核心职责：**
1. 训练模型 - 使用最优超参数
2. 生成预测 - 各模型独立预测
3. 集成预测 - LLM 决定集成策略
4. 计算评估指标

**集成策略：**

```python
# LLM 根据各模型验证性能决定权重
ensemble_methods = {
    'simple_average': np.mean(predictions, axis=0),
    'weighted_average': np.average(predictions, weights=weights),
    'median': np.median(predictions, axis=0),
    'trimmed_mean': trimmed_mean(predictions),
}
```

#### report_agent.py - 报告 Agent

**核心职责：**
1. 汇总所有结果
2. 生成综合报告
3. 提供建议和洞察

### 5.4 工作流模块 (graph/)

#### agent_graph.py - LangGraph 工作流

**核心实现：**

```python
class TimeSeriesAgentGraph:
    def __init__(self, config, model="gpt-4o"):
        # 初始化所有 Agent
        self.preprocess_agent = PreprocessAgent(model, config)
        self.analysis_agent = AnalysisAgent(model, config)
        self.validation_agent = ValidationAgent(model, config)
        self.forecast_agent = ForecastAgent(model, config)
        self.report_agent = ReportAgent(model, config)
        
        # 构建工作流
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """构建 LangGraph 工作流"""
        workflow = StateGraph(dict)
        
        # 添加节点（每个 Agent 是一个节点）
        workflow.add_node("preprocess", self._preprocess_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("forecast", self._forecast_node)
        workflow.add_node("report", self._report_node)
        
        # 定义边（执行顺序）
        workflow.add_edge("preprocess", "analyze")
        workflow.add_edge("analyze", "validate")
        workflow.add_edge("validate", "forecast")
        workflow.add_edge("forecast", "report")
        workflow.add_edge("report", END)
        
        # 设置入口点
        workflow.set_entry_point("preprocess")
        
        return workflow.compile()
    
    def run(self):
        """执行工作流"""
        # 加载数据
        df = DataLoader.load_data(self.config['data_path'])
        
        # 创建数据切片
        slices = DataSplitter.create_slices(df, ...)
        
        # 对每个切片执行工作流
        all_results = []
        for slice in slices:
            state = {"validation_data": slice['validation'], ...}
            result = self.graph.invoke(state)
            all_results.append(result)
        
        # 聚合结果
        aggregated = self._aggregate_slice_results(all_results)
        
        return aggregated
```

---

## 6. 代码注释

### 6.1 main.py 详细注释

```python
#!/usr/bin/env python3
"""
Time Series Prediction Agent - 主入口文件

功能：
1. 读取和验证配置
2. 初始化 Agent 工作流
3. 执行预测任务
4. 保存和展示结果
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# 导入核心模块
from graph.agent_graph import TimeSeriesAgentGraph  # 工作流编排器
from config.default_config import DEFAULT_CONFIG     # 默认配置

if __name__ == "__main__":
    print("=" * 60)
    print("TimeSeriesScientist")  # 程序标题
    print("=" * 60)

    # ===== 步骤1: 配置初始化 =====
    config = DEFAULT_CONFIG.copy()  # 复制默认配置
    
    # 自定义配置项
    config["num_slices"] = 25       # 数据切片数（更多切片=更稳定的评估）
    config["input_length"] = 512    # 输入序列长度（历史数据点数）
    config["horizon"] = 96          # 预测步长（向前预测多少步）
    config["data_path"] = "../dataset/ETT-small/ETTh1.csv"  # 数据路径
    
    # ===== 步骤2: API 密钥检查 =====
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: 请设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    # ===== 步骤3: 初始化工作流 =====
    print("正在初始化时间序列 Agent 工作流...")
    graph = TimeSeriesAgentGraph(
        config=config,
        model=config["llm_model"],  # 使用配置中指定的 LLM
        debug=config["debug"]
    )

    # ===== 步骤4: 执行工作流 =====
    print("正在执行时间序列预测...")
    import time
    start_time = time.time()
    
    results = graph.run()  # 核心执行！
    
    end_time = time.time()
    print(f"总执行时间: {end_time - start_time:.2f} 秒")

    # ===== 步骤5: 保存结果 =====
    results_dir = Path("results/reports")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存完整结果
    with open(f"complete_report_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # ===== 步骤6: 打印摘要 =====
    if results.get("aggregated_results"):
        print("\n=== 最终预测结果汇总 ===")
        # 打印集成模型性能
        ensemble_metrics = results["aggregated_results"]["test_metrics"]["ensemble"]
        print(f"MSE: {ensemble_metrics['mse']:.4f}")
        print(f"MAE: {ensemble_metrics['mae']:.4f}")
        print(f"MAPE: {ensemble_metrics['mape']:.2f}%")
```

### 6.2 关键函数注释模板

为便于学习，每个核心函数应包含以下注释结构：

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    函数简述（一句话说明功能）
    
    详细说明（可选）：
    - 这个函数是做什么的
    - 为什么需要这个函数
    - 它在整个系统中的作用
    
    Args:
        param1: 参数1的说明
        param2: 参数2的说明
    
    Returns:
        返回值的说明
    
    Raises:
        可能抛出的异常
    
    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
    
    Notes:
        - 特殊注意事项
        - 性能考虑
        - 依赖关系
    """
    pass
```

---

## 7. 实践建议

### 7.1 环境搭建

```bash
# 1. 克隆项目
git clone https://github.com/Y-Research-SBU/TimeSeriesScientist.git
cd TimeSeriesScientist

# 2. 创建虚拟环境
conda create -n TSci python=3.10
conda activate TSci

# 3. 安装依赖
pip install -r time_series_agent/requirements.txt

# 4. 设置 API 密钥
export OPENAI_API_KEY="your-api-key-here"

# 5. 运行示例
cd time_series_agent
python main.py
```

### 7.2 调试技巧

**1. 开启调试模式：**
```python
config["debug"] = True
config["verbose"] = True
```

**2. 减少切片数量快速测试：**
```python
config["num_slices"] = 2  # 快速测试时用
```

**3. 查看 Agent 输出：**
```python
# 在各 Agent 的 run 方法中添加日志
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 7.3 自定义数据集

**数据格式要求：**
```csv
date,OT
2016-07-01 00:00:00,5.827
2016-07-01 01:00:00,5.760
2016-07-01 02:00:00,5.738
...
```

**配置自定义数据：**
```python
config["data_path"] = "/path/to/your/data.csv"
config["date_column"] = "date"     # 日期列名
config["value_column"] = "OT"      # 目标值列名
```

### 7.4 添加新模型

**步骤1: 在 model_library.py 中添加预测函数**

```python
def predict_my_model(data: Dict, params: Dict, horizon: int) -> List[float]:
    """自定义模型预测函数"""
    # 1. 准备数据
    series = pd.DataFrame(data)['value']
    
    # 2. 训练模型
    # ... 你的模型代码 ...
    
    # 3. 生成预测
    predictions = []
    for i in range(horizon):
        pred = model.predict(...)
        predictions.append(pred)
    
    return predictions
```

**步骤2: 注册到模型映射**

```python
MODEL_FUNCTIONS = {
    # ... 现有模型 ...
    'MyModel': predict_my_model,  # 添加新模型
}
```

**步骤3: 更新配置**

```python
# 在 default_config.py 中
MODEL_CONFIG = {
    "available_models": [
        # ... 现有模型 ...
        "MyModel",  # 添加新模型
    ],
}
```

---

## 8. 常见问题

### Q1: 运行时提示 API 配额不足怎么办？

**解决方案：**
1. 减少切片数量：`config["num_slices"] = 5`
2. 增加请求间隔：修改 `delay_between_slices` 参数
3. 使用更便宜的模型：`config["llm_model"] = "gpt-3.5-turbo"`

### Q2: 如何解读预测结果？

**关键指标：**
- **MSE (均方误差)**：越小越好，对大误差敏感
- **MAE (平均绝对误差)**：越小越好，更稳健
- **MAPE (平均绝对百分比误差)**：百分比形式，便于理解

### Q3: 为什么使用多个 Agent 而不是一个？

**优势：**
1. **模块化**：每个 Agent 专注一个任务
2. **可维护性**：便于独立开发和调试
3. **可扩展性**：容易添加新功能
4. **可解释性**：每步决策都有记录

### Q4: 如何提高预测精度？

**建议：**
1. 增加数据量和历史长度
2. 调整 `input_length` 参数
3. 尝试不同的集成策略
4. 使用更多模型进行集成

### Q5: 项目使用了哪些关键技术？

| 技术 | 用途 | 学习资源 |
|------|------|----------|
| LangChain | LLM 应用框架 | langchain.com |
| LangGraph | 多 Agent 工作流 | LangGraph 文档 |
| pandas | 数据处理 | pandas 官方文档 |
| statsmodels | 统计模型 | statsmodels 文档 |
| scikit-learn | 机器学习 | sklearn 文档 |
| matplotlib | 可视化 | matplotlib 教程 |

---

## 📝 总结

通过学习这个项目，你将掌握：

1. **AI Agent 设计模式**：如何设计和组织多个协作的智能体
2. **LLM 应用开发**：如何利用大语言模型解决实际问题
3. **时间序列预测**：从数据预处理到模型集成的完整流程
4. **工作流编排**：使用 LangGraph 构建复杂的任务流程
5. **Python 工程实践**：模块化设计、配置管理、日志记录等

**祝你学习愉快！** 🚀

---

*最后更新：2024年*

*如有问题，请提交 Issue 或联系项目维护者*
