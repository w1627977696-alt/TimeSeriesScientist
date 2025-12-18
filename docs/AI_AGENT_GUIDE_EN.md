# TimeSeriesScientist AI Agent Architecture Guide

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Core Architecture](#core-architecture)
3. [LangChain & LangGraph Basics](#langchain--langgraph-basics)
4. [Agent Deep Dive](#agent-deep-dive)
5. [Workflow Execution](#workflow-execution)
6. [Code Examples & Best Practices](#code-examples--best-practices)
7. [Migration Guide](#migration-guide)
8. [FAQ](#faq)

---

## Project Overview

**TimeSeriesScientist** (TSci) is the first LLM-driven agentic framework for time series forecasting. This project uses the latest LangChain 1.2.0 and LangGraph 1.0.5 to build a multi-agent collaborative system.

### Why AI Agents?

Traditional time series forecasting requires:
- Manual selection of preprocessing methods
- Hand-picking suitable models
- Repeated hyperparameter tuning
- Writing complex analysis reports

The AI Agent framework provides:
- ✅ **Automated Decision-Making**: LLM automatically selects optimal strategies based on data characteristics
- ✅ **Intelligent Collaboration**: Multiple specialized agents work together
- ✅ **Explainability**: Every decision has clear reasoning
- ✅ **Extensibility**: Easy to add new models and features

---

## Core Architecture

### Technology Stack

```
┌─────────────────────────────────────────┐
│       Application Layer                  │
│      TimeSeriesScientist Main            │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│       Orchestration Layer                │
│   LangGraph 1.0.5 - Workflow Management  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Agent Layer                      │
│ PreprocessAgent | AnalysisAgent         │
│ ValidationAgent | ForecastAgent         │
│               ReportAgent               │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│        LLM Layer                        │
│   LangChain 1.2.0 + OpenAI API          │
└─────────────────────────────────────────┘
```

### Five Core Agents

1. **PreprocessAgent (Data Curator)**
   - Responsibilities: Data cleaning, outlier handling, missing value imputation
   - LLM Role: Analyzes data quality, recommends optimal preprocessing strategies
   - Output: Cleaned data, quality reports, visualizations

2. **AnalysisAgent (Data Analyst)**
   - Responsibilities: Trend analysis, seasonality detection, stationarity testing
   - LLM Role: Interprets data patterns, identifies key features
   - Output: Statistical analysis report, data insights

3. **ValidationAgent (Model Planner)**
   - Responsibilities: Model selection, hyperparameter optimization
   - LLM Role: Selects most suitable models based on data characteristics
   - Output: Selected models list, optimized hyperparameters

4. **ForecastAgent (Forecaster)**
   - Responsibilities: Model training, prediction, ensemble learning
   - LLM Role: Decides optimal ensemble strategy
   - Output: Prediction results, performance metrics, confidence intervals

5. **ReportAgent (Reporter)**
   - Responsibilities: Generates comprehensive analysis reports
   - LLM Role: Summarizes experimental results, provides recommendations
   - Output: Complete Markdown format report

---

## LangChain & LangGraph Basics

### What is LangChain?

LangChain is a framework for developing applications powered by large language models.

**Core Concepts:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,  # Control randomness of output
)

# 2. Use messages for conversation
messages = [
    SystemMessage(content="You are a time series forecasting expert"),
    HumanMessage(content="Please analyze the features of this time series data")
]

response = llm.invoke(messages)
print(response.content)
```

**Key Components:**

- **Messages**: System messages, user messages, AI messages
- **Prompts**: Structured prompt templates
- **Chains**: Link multiple components together
- **Output Parsers**: Parse LLM outputs

### What is LangGraph?

LangGraph is a state graph orchestration framework built on LangChain for building complex multi-step workflows.

**Core Concepts:**

```python
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# 1. Define state structure
class AgentState(TypedDict):
    data: str
    result: str

# 2. Define node function
def process_node(state: AgentState) -> AgentState:
    state["result"] = f"Processing complete: {state['data']}"
    return state

# 3. Build graph
workflow = StateGraph(AgentState)
workflow.add_node("process", process_node)
workflow.set_entry_point("process")
workflow.add_edge("process", END)

# 4. Compile and execute
app = workflow.compile()
result = app.invoke({"data": "test data"})
```

**Key Features:**

- **State Management**: Automatically manages state across nodes
- **Conditional Routing**: Dynamically selects paths based on state
- **Loop Support**: Supports iterative workflows
- **Checkpoints**: Can save and restore execution state

---

## Agent Deep Dive

### 1. PreprocessAgent Implementation

**File Location**: `time_series_agent/agents/preprocess_agent.py`

**Core Structure:**

```python
class PreprocessAgent:
    def __init__(self, model: str = "gpt-4o", config: dict = None):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
        )
        self.tools = PreprocessLLMTools(self.llm)
        self.config = config
        
    def run(self, data: pd.DataFrame) -> dict:
        """
        Execute preprocessing pipeline
        
        Steps:
        1. Use LLM to analyze data quality
        2. Get preprocessing recommendations
        3. Execute data cleaning
        4. Generate visualizations
        5. Return processing results
        """
        # 1. LLM analyzes data quality
        quality_analysis = self.tools.analyze_data_quality(data)
        
        # 2. Apply preprocessing based on LLM recommendations
        cleaned_data = self._apply_preprocessing(data, quality_analysis)
        
        # 3. Generate report
        report = self._generate_report(cleaned_data)
        
        return {
            "cleaned_data": cleaned_data,
            "analysis_report": report,
            "visualizations": {...}
        }
```

**LLM Prompt Design:**

```python
PREPROCESS_SYSTEM_PROMPT = """
You are the Data Preprocessing Chief Agent responsible for ensuring input data quality.

Background:
- You are an expert in time series data cleaning, anomaly detection, and ML preparation
- You understand the downstream impact of preprocessing choices on model performance

Responsibilities:
- Rigorously assess time series data quality
- Recommend the most appropriate handling strategy for each issue
- Explain your recommendations with clear reasoning
"""
```

### 2. AnalysisAgent Implementation

**File Location**: `time_series_agent/agents/analysis_agent.py`

**Key Functionality:**

```python
class AnalysisAgent:
    def run(self, data: pd.DataFrame, visualizations: dict) -> str:
        """
        Analyze time series data features
        
        Analysis includes:
        1. Trend Analysis
        2. Seasonality Analysis
        3. Stationarity Test
        4. Autocorrelation Analysis
        """
        
        # Build analysis prompt
        prompt = f"""
        Analyze the following time series data:
        Data statistics: {data.describe()}
        Visualizations: {visualizations}
        
        Return analysis results in JSON format:
        {{
          "trend_analysis": "trend description",
          "seasonality_analysis": "seasonality description",
          "stationarity": "stationarity assessment",
          "potential_issues": "potential issues",
          "summary": "summary"
        }}
        """
        
        response = self.llm.invoke([
            SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        
        return response.content
```

### 3. ValidationAgent Implementation

**File Location**: `time_series_agent/agents/validation_agent.py`

**Model Selection Process:**

```python
class ValidationAgent:
    def run(self, analysis: dict, available_models: list, 
            validation_data: pd.DataFrame) -> list:
        """
        Select best models and optimize hyperparameters
        
        Steps:
        1. LLM selects top-k models based on analysis
        2. Generate hyperparameter search space for each model
        3. Evaluate models on validation set
        4. Return sorted model list
        """
        
        # 1. LLM selects models
        model_selection = self._select_models_with_llm(
            analysis, available_models
        )
        
        # 2. Hyperparameter optimization
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

**Supported Models:**

| Type | Models | Features |
|------|--------|----------|
| Statistical | ARMA, ARIMA | Suitable for stationary time series |
| | Prophet | Excels at handling seasonality |
| | TBATS | Multi-seasonal modeling |
| Machine Learning | Random Forest | Non-linear patterns |
| | XGBoost | Gradient boosting |
| | LightGBM | Fast and efficient |
| Deep Learning | LSTM | Long short-term memory |
| | Transformer | Attention mechanism |

### 4. ForecastAgent Implementation

**File Location**: `time_series_agent/agents/forecast_agent.py`

**Ensemble Learning Strategy:**

```python
class ForecastAgent:
    def run(self, selected_models: list, 
            best_hyperparameters: dict, 
            test_data: pd.DataFrame) -> dict:
        """
        Generate prediction results
        
        Process:
        1. Train all selected models
        2. Generate single-model predictions
        3. LLM decides ensemble strategy
        4. Generate final predictions
        """
        
        # 1. Train models and predict
        individual_predictions = {}
        for model_name in selected_models:
            predictions = self._train_and_predict(
                model_name, 
                best_hyperparameters[model_name],
                test_data
            )
            individual_predictions[model_name] = predictions
        
        # 2. LLM decides ensemble strategy
        ensemble_strategy = self._get_ensemble_strategy(
            individual_predictions
        )
        
        # 3. Execute ensemble
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

**Ensemble Methods:**

- **Simple Average**: Simple averaging
- **Weighted Average**: Weighted averaging (weights decided by LLM)
- **Median**: Median (robust)
- **Trimmed Mean**: Trimmed mean
- **Best Model**: Select single best model

### 5. ReportAgent Implementation

**File Location**: `time_series_agent/agents/report_agent.py`

**Report Generation:**

```python
class ReportAgent:
    def run(self, experiment_summary: dict) -> str:
        """
        Generate comprehensive report
        
        Report includes:
        1. Executive summary
        2. Data quality assessment
        3. Model performance comparison
        4. Prediction result visualization
        5. Recommendations and conclusions
        """
        
        prompt = f"""
        Generate time series forecasting experiment report:
        {json.dumps(experiment_summary, indent=2)}
        
        Requirements:
        1. Concise executive summary
        2. Key findings and model performance
        3. Issues and limitations
        
        Output in Markdown format
        """
        
        response = self.llm.invoke([
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        
        return response.content
```

---

## Workflow Execution

### LangGraph Workflow Definition

**File Location**: `time_series_agent/graph/agent_graph.py`

**State Definition:**

```python
from typing_extensions import TypedDict

class AgentState(TypedDict, total=False):
    """Agent workflow state"""
    # Data related
    validation_data: pd.DataFrame
    test_data: pd.DataFrame
    preprocessed_data: pd.DataFrame
    
    # Agent results
    preprocess_result: Dict[str, Any]
    analysis_result: Any
    validation_result: List[Dict[str, Any]]
    forecast_result: Dict[str, Any]
    report: Any
    
    # Model related
    selected_models: List[str]
    best_hyperparameters: Dict[str, Dict[str, Any]]
    
    # Metadata
    slice_info: Dict[str, Any]
    config: Dict[str, Any]
```

**Graph Construction:**

```python
def _build_graph(self):
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("preprocess", self._preprocess_node)
    workflow.add_node("analyze", self._analyze_node)
    workflow.add_node("validate", self._validate_node)
    workflow.add_node("forecast", self._forecast_node)
    workflow.add_node("report", self._report_node)
    
    # Define edges (control flow)
    workflow.add_edge("preprocess", "analyze")
    workflow.add_edge("analyze", "validate")
    workflow.add_edge("validate", "forecast")
    workflow.add_edge("forecast", "report")
    workflow.add_edge("report", END)
    
    # Set entry point
    workflow.set_entry_point("preprocess")
    
    return workflow.compile()
```

**Node Function Example:**

```python
def _preprocess_node(self, state: AgentState) -> AgentState:
    """Preprocess node"""
    result = self.preprocess_agent.run(state["validation_data"])
    state["preprocessed_data"] = result.get("cleaned_data")
    state["preprocess_result"] = result
    return state

def _analyze_node(self, state: AgentState) -> AgentState:
    """Analysis node"""
    visualizations = state["preprocess_result"]["visualizations"]
    result = self.analysis_agent.run(
        state["preprocessed_data"], 
        visualizations
    )
    state["analysis_result"] = result
    return state
```

---

## Code Examples & Best Practices

### Example 1: Creating a Custom Agent

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class CustomAgent:
    def __init__(self, model: str = "gpt-4o", config: dict = None):
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        self.config = config or {}
        
    def run(self, data: pd.DataFrame) -> dict:
        # Your custom logic
        return {"result": "custom output"}
```

### Example 2: Optimizing LLM Prompts

Good prompt design principles:
1. Clear role definition
2. Provide context
3. Clear instructions
4. Example guidance

---

## Migration Guide

### Upgrading from Old Versions

#### 1. Dependency Updates
```bash
pip install langchain==1.2.0
pip install langchain-core==0.3.27
pip install langchain-openai==0.3.2
pip install langgraph==1.0.5
```

#### 2. Import Statement Updates
```python
# ❌ Old version
from langchain.schema import HumanMessage, SystemMessage

# ✅ New version
from langchain_core.messages import HumanMessage, SystemMessage
```

#### 3. StateGraph Usage Update
```python
# ❌ Old version
workflow = StateGraph(dict)

# ✅ New version
class AgentState(TypedDict):
    data: str

workflow = StateGraph(AgentState)
```

---

## FAQ

### Q1: How to reduce API costs?
- Use cheaper models (gpt-3.5-turbo)
- Reduce number of slices
- Enable caching

### Q2: What if LLM output format is incorrect?
- Use JsonOutputParser
- Specify output format requirements clearly
- Add examples

### Q3: How to debug workflow?
- Enable debug mode
- Use LangSmith
- Add logging

---

## Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

---

## License

MIT License

---

**Last Updated**: 2025-12-18  
**Version**: v1.0.0 (LangChain 1.2.0 + LangGraph 1.0.5)
