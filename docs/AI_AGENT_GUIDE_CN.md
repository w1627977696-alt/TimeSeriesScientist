# TimeSeriesScientist AI Agent 架构详解

## 📚 目录

1. [项目概述](#项目概述)
2. [核心架构](#核心架构)
3. [LangChain 与 LangGraph 基础](#langchain-与-langgraph-基础)
4. [Agent 详解](#agent-详解)
5. [工作流程](#工作流程)
6. [代码示例与最佳实践](#代码示例与最佳实践)
7. [升级指南](#升级指南)
8. [常见问题](#常见问题)

---

## 项目概述

**TimeSeriesScientist** (TSci) 是首个基于大语言模型(LLM)驱动的时序预测 AI Agent 框架。该项目采用了最新的 LangChain 1.2.0 和 LangGraph 1.0.5 来构建多智能体协作系统。

### 为什么选择 AI Agent?

传统的时序预测需要：
- 手动选择预处理方法
- 人工挑选合适的模型
- 反复调试超参数
- 编写复杂的分析报告

AI Agent 框架通过：
- ✅ **自动化决策**: LLM 根据数据特征自动选择最佳策略
- ✅ **智能协作**: 多个专业 Agent 分工合作
- ✅ **可解释性**: 每个决策都有清晰的理由
- ✅ **可扩展性**: 易于添加新的模型和功能

---

## 核心架构

### 技术栈

```
┌─────────────────────────────────────────┐
│          应用层 (Application)            │
│      TimeSeriesScientist 主程序          │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│       编排层 (Orchestration)             │
│   LangGraph 1.0.5 - 工作流管理           │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Agent 层 (Agent Layer)          │
│ PreprocessAgent | AnalysisAgent         │
│ ValidationAgent | ForecastAgent         │
│               ReportAgent               │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│        LLM 层 (LLM Layer)               │
│   LangChain 1.2.0 + OpenAI API          │
└─────────────────────────────────────────┘
```

### 五大核心 Agent

1. **PreprocessAgent (预处理智能体)**
   - 职责: 数据清洗、异常值处理、缺失值填充
   - LLM 作用: 分析数据质量，推荐最佳预处理策略
   - 输出: 清洗后的数据、质量报告、可视化

2. **AnalysisAgent (分析智能体)**
   - 职责: 趋势分析、季节性检测、平稳性测试
   - LLM 作用: 解释数据模式，识别关键特征
   - 输出: 统计分析报告、数据洞察

3. **ValidationAgent (验证智能体)**
   - 职责: 模型选择、超参数优化
   - LLM 作用: 根据数据特征选择最合适的模型
   - 输出: 选中的模型列表、优化后的超参数

4. **ForecastAgent (预测智能体)**
   - 职责: 模型训练、预测、集成学习
   - LLM 作用: 决定最佳的集成策略
   - 输出: 预测结果、性能指标、置信区间

5. **ReportAgent (报告智能体)**
   - 职责: 生成综合分析报告
   - LLM 作用: 总结实验结果，提供建议
   - 输出: 完整的 Markdown 格式报告

---

## LangChain 与 LangGraph 基础

### LangChain 是什么？

LangChain 是一个用于开发由大语言模型驱动的应用程序的框架。

**核心概念:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 1. 初始化 LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,  # 控制输出的随机性
)

# 2. 使用消息进行对话
messages = [
    SystemMessage(content="你是一个时序预测专家"),
    HumanMessage(content="请分析这个时序数据的特征")
]

response = llm.invoke(messages)
print(response.content)
```

**关键组件:**

- **Messages**: 系统消息、用户消息、AI 消息
- **Prompts**: 结构化的提示词模板
- **Chains**: 将多个组件链接在一起
- **Output Parsers**: 解析 LLM 的输出

### LangGraph 是什么？

LangGraph 是基于 LangChain 的状态图编排框架，用于构建复杂的多步骤工作流。

**核心概念:**

```python
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# 1. 定义状态结构
class AgentState(TypedDict):
    data: str
    result: str

# 2. 定义节点函数
def process_node(state: AgentState) -> AgentState:
    state["result"] = f"处理完成: {state['data']}"
    return state

# 3. 构建图
workflow = StateGraph(AgentState)
workflow.add_node("process", process_node)
workflow.set_entry_point("process")
workflow.add_edge("process", END)

# 4. 编译并执行
app = workflow.compile()
result = app.invoke({"data": "测试数据"})
```

**关键特性:**

- **状态管理**: 自动管理跨节点的状态
- **条件路由**: 根据状态动态选择路径
- **循环支持**: 支持迭代式工作流
- **检查点**: 可以保存和恢复执行状态

---

## Agent 详解

### 1. PreprocessAgent 实现详解

**文件位置**: `time_series_agent/agents/preprocess_agent.py`

**核心代码结构**:

```python
class PreprocessAgent:
    def __init__(self, model: str = "gpt-4o", config: dict = None):
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
        )
        self.tools = PreprocessLLMTools(self.llm)
        self.config = config
        
    def run(self, data: pd.DataFrame) -> dict:
        """
        执行预处理流程
        
        流程:
        1. 使用 LLM 分析数据质量
        2. 获取预处理建议
        3. 执行数据清洗
        4. 生成可视化
        5. 返回处理结果
        """
        # 1. LLM 分析数据质量
        quality_analysis = self.tools.analyze_data_quality(data)
        
        # 2. 根据 LLM 建议处理数据
        cleaned_data = self._apply_preprocessing(data, quality_analysis)
        
        # 3. 生成报告
        report = self._generate_report(cleaned_data)
        
        return {
            "cleaned_data": cleaned_data,
            "analysis_report": report,
            "visualizations": {...}
        }
```

**LLM 提示词设计**:

```python
PREPROCESS_SYSTEM_PROMPT = """
你是数据预处理首席智能体，负责确保输入数据质量。

背景知识:
- 你精通时序数据清洗、异常检测和机器学习准备工作
- 你理解预处理选择对下游模型性能的影响

职责:
- 严格评估时序数据质量
- 针对每个问题推荐最合适的处理策略
- 用清晰的逻辑解释你的建议
"""
```

### 2. AnalysisAgent 实现详解

**文件位置**: `time_series_agent/agents/analysis_agent.py`

**关键功能**:

```python
class AnalysisAgent:
    def run(self, data: pd.DataFrame, visualizations: dict) -> str:
        """
        分析时序数据特征
        
        分析内容:
        1. 趋势分析 (Trend Analysis)
        2. 季节性分析 (Seasonality Analysis)
        3. 平稳性检验 (Stationarity Test)
        4. 自相关分析 (Autocorrelation)
        """
        
        # 构建分析提示词
        prompt = f"""
        分析以下时序数据:
        数据统计: {data.describe()}
        可视化: {visualizations}
        
        请以 JSON 格式返回分析结果:
        {{
          "trend_analysis": "趋势描述",
          "seasonality_analysis": "季节性描述",
          "stationarity": "平稳性评估",
          "potential_issues": "潜在问题",
          "summary": "总结"
        }}
        """
        
        response = self.llm.invoke([
            SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        
        return response.content
```

### 3. ValidationAgent 实现详解

**文件位置**: `time_series_agent/agents/validation_agent.py`

**模型选择流程**:

```python
class ValidationAgent:
    def run(self, analysis: dict, available_models: list, 
            validation_data: pd.DataFrame) -> list:
        """
        选择最佳模型并优化超参数
        
        步骤:
        1. LLM 根据分析结果选择 top-k 模型
        2. 为每个模型生成超参数搜索空间
        3. 在验证集上评估模型
        4. 返回排序后的模型列表
        """
        
        # 1. LLM 选择模型
        model_selection = self._select_models_with_llm(
            analysis, available_models
        )
        
        # 2. 超参数优化
        optimized_models = []
        for model_config in model_selection:
            best_params = self._optimize_hyperparameters(
                model_config, validation_data
            )
            optimized_models.append({
                'model': model_config['model'],
                'hyperparameters': best_params,
                'validation_score': score
            })
        
        return sorted(optimized_models, 
                     key=lambda x: x['validation_score'])
```

**支持的模型**:

| 类型 | 模型 | 特点 |
|------|------|------|
| 统计模型 | ARMA, ARIMA | 适合平稳时序 |
| | Prophet | 擅长处理季节性 |
| | TBATS | 多季节性建模 |
| 机器学习 | Random Forest | 非线性模式 |
| | XGBoost | 梯度提升 |
| | LightGBM | 快速高效 |
| 深度学习 | LSTM | 长短期记忆 |
| | Transformer | 注意力机制 |

### 4. ForecastAgent 实现详解

**文件位置**: `time_series_agent/agents/forecast_agent.py`

**集成学习策略**:

```python
class ForecastAgent:
    def run(self, selected_models: list, 
            best_hyperparameters: dict, 
            test_data: pd.DataFrame) -> dict:
        """
        生成预测结果
        
        流程:
        1. 训练所有选中的模型
        2. 生成单模型预测
        3. LLM 决定集成策略
        4. 生成最终预测
        """
        
        # 1. 训练模型并预测
        individual_predictions = {}
        for model_name in selected_models:
            predictions = self._train_and_predict(
                model_name, 
                best_hyperparameters[model_name],
                test_data
            )
            individual_predictions[model_name] = predictions
        
        # 2. LLM 决定集成策略
        ensemble_strategy = self._get_ensemble_strategy(
            individual_predictions
        )
        
        # 3. 执行集成
        ensemble_predictions = self._ensemble_predictions(
            individual_predictions,
            ensemble_strategy
        )
        
        return {
            "individual_predictions": individual_predictions,
            "ensemble_predictions": ensemble_predictions,
            "test_metrics": {...}
        }
```

**集成方法**:

- **Simple Average**: 简单平均
- **Weighted Average**: 加权平均（权重由 LLM 决定）
- **Median**: 中位数（鲁棒性强）
- **Trimmed Mean**: 截尾均值
- **Best Model**: 选择单个最佳模型

### 5. ReportAgent 实现详解

**文件位置**: `time_series_agent/agents/report_agent.py`

**报告生成**:

```python
class ReportAgent:
    def run(self, experiment_summary: dict) -> str:
        """
        生成综合报告
        
        报告包含:
        1. 执行摘要
        2. 数据质量评估
        3. 模型性能对比
        4. 预测结果可视化
        5. 建议与总结
        """
        
        prompt = f"""
        生成时序预测实验报告:
        {json.dumps(experiment_summary, indent=2)}
        
        要求:
        1. 简洁的执行摘要
        2. 关键发现和模型性能
        3. 问题和局限性
        
        以 Markdown 格式输出
        """
        
        response = self.llm.invoke([
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        
        return response.content
```

---

## 工作流程

### LangGraph 工作流定义

**文件位置**: `time_series_agent/graph/agent_graph.py`

**状态定义**:

```python
from typing_extensions import TypedDict

class AgentState(TypedDict, total=False):
    """Agent 工作流状态"""
    # 数据相关
    validation_data: pd.DataFrame
    test_data: pd.DataFrame
    preprocessed_data: pd.DataFrame
    
    # Agent 结果
    preprocess_result: Dict[str, Any]
    analysis_result: Any
    validation_result: List[Dict[str, Any]]
    forecast_result: Dict[str, Any]
    report: Any
    
    # 模型相关
    selected_models: List[str]
    best_hyperparameters: Dict[str, Dict[str, Any]]
    
    # 元数据
    slice_info: Dict[str, Any]
    config: Dict[str, Any]
```

**图构建**:

```python
def _build_graph(self):
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("preprocess", self._preprocess_node)
    workflow.add_node("analyze", self._analyze_node)
    workflow.add_node("validate", self._validate_node)
    workflow.add_node("forecast", self._forecast_node)
    workflow.add_node("report", self._report_node)
    
    # 定义边（控制流）
    workflow.add_edge("preprocess", "analyze")
    workflow.add_edge("analyze", "validate")
    workflow.add_edge("validate", "forecast")
    workflow.add_edge("forecast", "report")
    workflow.add_edge("report", END)
    
    # 设置入口点
    workflow.set_entry_point("preprocess")
    
    return workflow.compile()
```

**节点函数示例**:

```python
def _preprocess_node(self, state: AgentState) -> AgentState:
    """预处理节点"""
    result = self.preprocess_agent.run(state["validation_data"])
    state["preprocessed_data"] = result.get("cleaned_data")
    state["preprocess_result"] = result
    return state

def _analyze_node(self, state: AgentState) -> AgentState:
    """分析节点"""
    visualizations = state["preprocess_result"]["visualizations"]
    result = self.analysis_agent.run(
        state["preprocessed_data"], 
        visualizations
    )
    state["analysis_result"] = result
    return state
```

### 完整执行流程

```
数据输入 → 预处理 → 分析 → 验证 → 预测 → 报告 → 输出
```

---

## 代码示例与最佳实践

### 示例 1: 自定义一个新的 Agent

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class CustomAgent:
    def __init__(self, model: str = "gpt-4o", config: dict = None):
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        self.config = config or {}
        
    def run(self, data: pd.DataFrame) -> dict:
        # 你的自定义逻辑
        return {"result": "custom output"}
```

### 示例 2: 优化 LLM 提示词

好的提示词设计原则:
1. 明确角色
2. 提供上下文
3. 清晰指令
4. 示例引导

---

## 升级指南

### 从旧版本升级

#### 1. 依赖更新
```bash
pip install langchain==1.2.0
pip install langchain-core==0.3.27
pip install langchain-openai==0.3.2
pip install langgraph==1.0.5
```

#### 2. 导入语句更新
```python
# ❌ 旧版本
from langchain.schema import HumanMessage, SystemMessage

# ✅ 新版本
from langchain_core.messages import HumanMessage, SystemMessage
```

#### 3. StateGraph 使用更新
```python
# ❌ 旧版本
workflow = StateGraph(dict)

# ✅ 新版本
class AgentState(TypedDict):
    data: str

workflow = StateGraph(AgentState)
```

---

## 常见问题

### Q1: 如何减少 API 成本？
- 使用更便宜的模型 (gpt-3.5-turbo)
- 减少切片数量
- 启用缓存

### Q2: LLM 格式不正确怎么办？
- 使用 JsonOutputParser
- 明确输出格式要求
- 添加示例

### Q3: 如何调试工作流？
- 启用 debug 模式
- 使用 LangSmith
- 添加日志记录

---

## 学习资源

- [LangChain 文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [OpenAI API 文档](https://platform.openai.com/docs/)

---

## 许可证

MIT License

---

**最后更新**: 2025-12-18  
**版本**: v1.0.0 (LangChain 1.2.0 + LangGraph 1.0.5)
