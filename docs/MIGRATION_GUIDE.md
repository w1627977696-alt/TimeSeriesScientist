# LangChain & LangGraph 升级迁移指南
# LangChain & LangGraph Migration Guide

[English](#english-version) | [中文](#中文版本)

---

## 中文版本

### 概述

本文档详细说明了如何从旧版本的 LangChain 和 LangGraph 迁移到最新版本：
- **LangChain**: 0.1.x → 1.2.0
- **LangGraph**: 0.1.x/0.4.x → 1.0.5

### 主要变更

#### 1. 包结构重组

**langchain-core 独立**

旧版本中，核心功能在 `langchain` 包中。新版本将核心功能拆分到 `langchain-core` 包。

```python
# ❌ 旧版本
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser

# ✅ 新版本
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

**langchain-openai 独立**

OpenAI 集成现在在独立的包中。

```python
# ❌ 旧版本
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# ✅ 新版本
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
```

#### 2. StateGraph API 变更

**类型安全的状态定义**

LangGraph 1.0 要求使用 TypedDict 定义状态，而不是普通的 dict。

```python
# ❌ 旧版本 (LangGraph 0.x)
from langgraph.graph import StateGraph, END

workflow = StateGraph(dict)

def my_node(state: dict) -> dict:
    state["result"] = "processed"
    return state

# ✅ 新版本 (LangGraph 1.0+)
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

class MyState(TypedDict):
    data: str
    result: str

workflow = StateGraph(MyState)

def my_node(state: MyState) -> MyState:
    state["result"] = "processed"
    return state
```

**为什么这样改？**
- 类型安全：编辑器可以提供更好的自动完成
- 更好的文档：状态结构一目了然
- 运行时检查：可以捕获更多错误

#### 3. ChatOpenAI 参数变更

```python
# ❌ 旧版本
llm = ChatOpenAI(
    model_name="gpt-4",      # 参数名已弃用
    max_tokens=None,
    request_timeout=60,
)

# ✅ 新版本
llm = ChatOpenAI(
    model="gpt-4o",          # 新参数名
    temperature=0.1,
    timeout=60,              # request_timeout → timeout
)
```

### 完整迁移步骤

#### 步骤 1: 更新依赖

**requirements.txt**:
```txt
# 旧版本
langchain>=0.1.0
langchain-openai>=0.1.0
langgraph>=0.1.0

# 新版本
langchain==1.2.0
langchain-core==0.3.27
langchain-openai==0.3.2
langgraph==1.0.5
```

安装：
```bash
pip install --upgrade langchain==1.2.0 langchain-core==0.3.27 langchain-openai==0.3.2 langgraph==1.0.5
```

#### 步骤 2: 更新导入语句

使用以下脚本批量替换：

```python
# replace_imports.py
import os
import re

replacements = {
    'from langchain.schema import': 'from langchain_core.messages import',
    'from langchain.chat_models import ChatOpenAI': 'from langchain_openai import ChatOpenAI',
    'from langchain.embeddings import OpenAIEmbeddings': 'from langchain_openai import OpenAIEmbeddings',
    'from langchain.prompts import': 'from langchain_core.prompts import',
    'from langchain.output_parsers import': 'from langchain_core.output_parsers import',
}

def replace_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# 遍历所有 Python 文件
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            replace_in_file(filepath)
            print(f"Updated: {filepath}")
```

#### 步骤 3: 定义状态类型

为每个 LangGraph 工作流创建 TypedDict：

```python
from typing_extensions import TypedDict
from typing import Dict, Any, List
import pandas as pd

class AgentState(TypedDict, total=False):
    """
    total=False 表示所有字段都是可选的
    这允许状态逐步构建
    """
    # 输入数据
    validation_data: pd.DataFrame
    test_data: pd.DataFrame
    
    # 处理结果
    preprocessed_data: pd.DataFrame
    preprocess_result: Dict[str, Any]
    analysis_result: str
    
    # 模型相关
    selected_models: List[str]
    best_hyperparameters: Dict[str, Dict[str, Any]]
    
    # 最终输出
    forecast_result: Dict[str, Any]
    report: str
    
    # 元数据
    slice_info: Dict[str, Any]
    config: Dict[str, Any]
```

#### 步骤 4: 更新 StateGraph 初始化

```python
# ❌ 旧版本
workflow = StateGraph(dict)

# ✅ 新版本
workflow = StateGraph(AgentState)
```

#### 步骤 5: 更新节点函数签名

```python
# ❌ 旧版本
def preprocess_node(state: dict) -> dict:
    result = preprocess(state["data"])
    state["result"] = result
    return state

# ✅ 新版本
def preprocess_node(state: AgentState) -> AgentState:
    result = preprocess(state["data"])
    state["result"] = result
    return state
```

#### 步骤 6: 测试

运行测试确保一切正常：

```bash
# 单元测试
pytest tests/

# 集成测试
python main.py --debug --num_slices 1
```

### 常见迁移问题

#### 问题 1: 导入错误

```
ImportError: cannot import name 'HumanMessage' from 'langchain.schema'
```

**解决方案**:
```python
# 更新导入
from langchain_core.messages import HumanMessage
```

#### 问题 2: StateGraph 类型错误

```
TypeError: StateGraph() missing required argument: 'schema'
```

**解决方案**:
```python
# 定义并使用 TypedDict
class MyState(TypedDict):
    data: str

workflow = StateGraph(MyState)
```

#### 问题 3: ChatOpenAI 参数错误

```
TypeError: __init__() got an unexpected keyword argument 'model_name'
```

**解决方案**:
```python
# 使用新参数名
llm = ChatOpenAI(model="gpt-4o")  # 不是 model_name
```

### 版本兼容性表

| 功能 | 旧版本 | 新版本 | 说明 |
|------|--------|--------|------|
| 核心消息 | `langchain.schema` | `langchain_core.messages` | 模块重组 |
| OpenAI 集成 | `langchain.chat_models` | `langchain_openai` | 独立包 |
| StateGraph | `StateGraph(dict)` | `StateGraph(TypedDict)` | 类型安全 |
| ChatOpenAI 参数 | `model_name` | `model` | 参数重命名 |
| 超时参数 | `request_timeout` | `timeout` | 参数简化 |

### 性能优化建议

#### 1. 使用缓存减少 API 调用

```python
from langchain.cache import InMemoryCache
import langchain

# 启用缓存
langchain.llm_cache = InMemoryCache()

# 现在相同的请求会被缓存
response1 = llm.invoke("What is 2+2?")
response2 = llm.invoke("What is 2+2?")  # 从缓存返回
```

#### 2. 批处理请求

```python
# ❌ 不推荐：逐个调用
results = []
for item in items:
    result = llm.invoke(item)
    results.append(result)

# ✅ 推荐：批处理
results = llm.batch(items)
```

#### 3. 异步调用

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

async def process_async():
    tasks = [llm.ainvoke(msg) for msg in messages]
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(process_async())
```

### 测试清单

- [ ] 所有依赖包已更新
- [ ] 导入语句已更新
- [ ] StateGraph 使用 TypedDict
- [ ] 节点函数使用正确类型注解
- [ ] ChatOpenAI 参数已更新
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 性能测试通过
- [ ] 文档已更新

---

## English Version

### Overview

This document details how to migrate from older versions of LangChain and LangGraph to the latest versions:
- **LangChain**: 0.1.x → 1.2.0
- **LangGraph**: 0.1.x/0.4.x → 1.0.5

### Major Changes

#### 1. Package Structure Reorganization

**langchain-core Separation**

In older versions, core functionality was in the `langchain` package. The new version splits core functionality into the `langchain-core` package.

```python
# ❌ Old version
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser

# ✅ New version
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

**langchain-openai Separation**

OpenAI integration is now in a separate package.

```python
# ❌ Old version
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# ✅ New version
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
```

#### 2. StateGraph API Changes

**Type-Safe State Definition**

LangGraph 1.0 requires using TypedDict for state definition instead of plain dict.

```python
# ❌ Old version (LangGraph 0.x)
from langgraph.graph import StateGraph, END

workflow = StateGraph(dict)

def my_node(state: dict) -> dict:
    state["result"] = "processed"
    return state

# ✅ New version (LangGraph 1.0+)
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

class MyState(TypedDict):
    data: str
    result: str

workflow = StateGraph(MyState)

def my_node(state: MyState) -> MyState:
    state["result"] = "processed"
    return state
```

**Why this change?**
- Type safety: Better IDE autocomplete
- Better documentation: State structure is clear at a glance
- Runtime checks: Can catch more errors

#### 3. ChatOpenAI Parameter Changes

```python
# ❌ Old version
llm = ChatOpenAI(
    model_name="gpt-4",      # Deprecated parameter name
    max_tokens=None,
    request_timeout=60,
)

# ✅ New version
llm = ChatOpenAI(
    model="gpt-4o",          # New parameter name
    temperature=0.1,
    timeout=60,              # request_timeout → timeout
)
```

### Complete Migration Steps

#### Step 1: Update Dependencies

**requirements.txt**:
```txt
# Old version
langchain>=0.1.0
langchain-openai>=0.1.0
langgraph>=0.1.0

# New version
langchain==1.2.0
langchain-core==0.3.27
langchain-openai==0.3.2
langgraph==1.0.5
```

Install:
```bash
pip install --upgrade langchain==1.2.0 langchain-core==0.3.27 langchain-openai==0.3.2 langgraph==1.0.5
```

#### Step 2: Update Import Statements

Use the following script for batch replacement:

```python
# replace_imports.py
import os
import re

replacements = {
    'from langchain.schema import': 'from langchain_core.messages import',
    'from langchain.chat_models import ChatOpenAI': 'from langchain_openai import ChatOpenAI',
    'from langchain.embeddings import OpenAIEmbeddings': 'from langchain_openai import OpenAIEmbeddings',
    'from langchain.prompts import': 'from langchain_core.prompts import',
    'from langchain.output_parsers import': 'from langchain_core.output_parsers import',
}

def replace_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# Traverse all Python files
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            replace_in_file(filepath)
            print(f"Updated: {filepath}")
```

#### Step 3: Define State Types

Create TypedDict for each LangGraph workflow:

```python
from typing_extensions import TypedDict
from typing import Dict, Any, List
import pandas as pd

class AgentState(TypedDict, total=False):
    """
    total=False means all fields are optional
    This allows state to be built incrementally
    """
    # Input data
    validation_data: pd.DataFrame
    test_data: pd.DataFrame
    
    # Processing results
    preprocessed_data: pd.DataFrame
    preprocess_result: Dict[str, Any]
    analysis_result: str
    
    # Model related
    selected_models: List[str]
    best_hyperparameters: Dict[str, Dict[str, Any]]
    
    # Final output
    forecast_result: Dict[str, Any]
    report: str
    
    # Metadata
    slice_info: Dict[str, Any]
    config: Dict[str, Any]
```

#### Step 4: Update StateGraph Initialization

```python
# ❌ Old version
workflow = StateGraph(dict)

# ✅ New version
workflow = StateGraph(AgentState)
```

#### Step 5: Update Node Function Signatures

```python
# ❌ Old version
def preprocess_node(state: dict) -> dict:
    result = preprocess(state["data"])
    state["result"] = result
    return state

# ✅ New version
def preprocess_node(state: AgentState) -> AgentState:
    result = preprocess(state["data"])
    state["result"] = result
    return state
```

#### Step 6: Test

Run tests to ensure everything works:

```bash
# Unit tests
pytest tests/

# Integration test
python main.py --debug --num_slices 1
```

### Common Migration Issues

#### Issue 1: Import Errors

```
ImportError: cannot import name 'HumanMessage' from 'langchain.schema'
```

**Solution**:
```python
# Update import
from langchain_core.messages import HumanMessage
```

#### Issue 2: StateGraph Type Error

```
TypeError: StateGraph() missing required argument: 'schema'
```

**Solution**:
```python
# Define and use TypedDict
class MyState(TypedDict):
    data: str

workflow = StateGraph(MyState)
```

#### Issue 3: ChatOpenAI Parameter Error

```
TypeError: __init__() got an unexpected keyword argument 'model_name'
```

**Solution**:
```python
# Use new parameter name
llm = ChatOpenAI(model="gpt-4o")  # Not model_name
```

### Version Compatibility Table

| Feature | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| Core Messages | `langchain.schema` | `langchain_core.messages` | Module reorganization |
| OpenAI Integration | `langchain.chat_models` | `langchain_openai` | Separate package |
| StateGraph | `StateGraph(dict)` | `StateGraph(TypedDict)` | Type safety |
| ChatOpenAI Parameter | `model_name` | `model` | Parameter rename |
| Timeout Parameter | `request_timeout` | `timeout` | Parameter simplification |

### Performance Optimization Tips

#### 1. Use Caching to Reduce API Calls

```python
from langchain.cache import InMemoryCache
import langchain

# Enable caching
langchain.llm_cache = InMemoryCache()

# Now identical requests will be cached
response1 = llm.invoke("What is 2+2?")
response2 = llm.invoke("What is 2+2?")  # Returns from cache
```

#### 2. Batch Requests

```python
# ❌ Not recommended: Individual calls
results = []
for item in items:
    result = llm.invoke(item)
    results.append(result)

# ✅ Recommended: Batch processing
results = llm.batch(items)
```

#### 3. Async Calls

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

async def process_async():
    tasks = [llm.ainvoke(msg) for msg in messages]
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(process_async())
```

### Testing Checklist

- [ ] All dependency packages updated
- [ ] Import statements updated
- [ ] StateGraph uses TypedDict
- [ ] Node functions use correct type annotations
- [ ] ChatOpenAI parameters updated
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass
- [ ] Documentation updated

---

**Last Updated**: 2025-12-18  
**Version**: LangChain 1.2.0 + LangGraph 1.0.5
