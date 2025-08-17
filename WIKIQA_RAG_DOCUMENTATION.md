# WikiQA RAG系统文档

## 系统概述

WikiQA RAG系统是一个基于WikiQA数据集构建的检索增强生成（RAG）问答系统，系统集成了数据清洗、向量数据库构建、问答系统、重排序优化、用户反馈收集等完整功能。

## 核心功能

### 1. 数据处理与清洗
- **数据加载**: 从WikiQA TSV格式文件加载问答数据
- **数据清洗**: 自动去除HTML标签、URL、邮箱、电话号码等噪声
- **负样本生成**: 基于原始数据生成负样本用于对比学习
- **文本标准化**: Unicode标准化、停用词过滤、拼写检查

### 2. 向量数据库构建
- **多种分块策略**:
  - 字符分块（Character-based）- 按固定字符长度分块
  - 句子分块（Sentence-based）- 保持语义完整性
  - 主题分块（Topic-based）- 基于主题聚类的智能分块
- **嵌入模型**: 使用BGE-small-en-v1.5或Sentence Transformers
- **向量存储**: 基于FAISS的高效相似度搜索

### 3. 问答系统
- **LLM集成**: 支持OpenAI API、DeepSeek、本地Ollama模型
- **检索增强**: 基于向量相似度的上下文检索
- **答案生成**: 结合检索结果生成准确回答

### 4. 重排序优化
- **简单重排序**: 基于关键词匹配和长度适应
- **对比学习重排序**: 利用正负样本信息优化排序
- **质量分析**: 自动评估检索质量并提供改进建议

### 5. 用户反馈系统
- **实时反馈**: 用户对每个回答进行评价
- **统计分析**: 收集和分析用户反馈数据
- **持续改进**: 基于反馈优化系统性能

## 文件结构

```
wikiqa_rag/
├── WikiQA/                    # WikiQA数据集目录
│   ├── WikiQA-train.tsv      # 训练数据
│   ├── WikiQA-test.tsv       # 测试数据
│   ├── WikiQA-dev.tsv        # 验证数据
│   └── eval.py               # 官方评估脚本
├── main.py                   # 在线版主程序
├── qa_system.py             # 问答系统核心
├── data_processing.py       # 数据处理和清洗
├── vector_db.py             # 向量数据库构建
├── evaluation.py            # 模型评估
├── reranking.py             # 重排序优化
├── feedback.py              # 用户反馈收集
├── model_manager.py         # 模型管理器
├── webapi.py                # Web API接口
└── requirements.txt         # 在线版依赖
```

## 核心模块详解

### 1. main.py - 在线版主程序

**功能**: 完整的在线RAG系统入口，集成所有功能模块

**运行命令**:
```bash
python main.py
```

**交互流程**:
1. 数据加载和清洗
2. 选择分块策略（1-4选项）
3. 构建向量数据库
4. 模型评估（可选）
5. 启动交互式问答

**参数配置**:
```python
# 在main.py中配置API
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

### 2. data_processing.py - 数据处理模块

**核心类**: `RAGDataCleaner`

**主要功能**:
- 数据清洗和标准化
- 负样本生成（三种策略）
- 对比学习数据构建
- 数据质量分析

**关键函数**:
```python
# 加载和清洗数据
load_and_clean_data(tsv_path, include_negative_samples=True, negative_ratio=0.3)

# 创建对比学习数据
create_contrastive_learning_data(qa_texts, sample_info)

# 数据质量评估
analyze_data_quality(qa_texts)
```

### 3. vector_db.py - 向量数据库模块

**分块策略**:
- **字符分块**: `chunking_strategy="character"`
- **句子分块**: `chunking_strategy="sentence"`
- **主题分块**: `chunking_strategy="topic"`

**使用示例**:
```python
# 构建向量数据库
db = build_vector_db(qa_texts, 
                    chunking_strategy="character",
                    chunk_size=200, 
                    chunk_overlap=20)

# 演示不同分块策略
demo_chunking_strategies(qa_texts)
```

### 4. qa_system.py - 问答系统

**核心函数**:
```python
# 创建问答链
qa = create_qa_chain(vector_db)

# 执行问答
answer = qa.invoke({"query": question})
```

### 5. evaluation.py - 模型评估

**评估指标**:
- 准确率（Accuracy）
- 召回率（Recall）
- F1分数
- 语义匹配度

**使用示例**:
```python
# 评估模型性能
evaluate_model(qa_chain, tsv_path="WikiQA/WikiQA-test.tsv", max_q=50)
```

### 6. reranking.py - 重排序模块

**重排序器类型**:
- **SimpleReranker**: 基于关键词和长度
- **ContrastiveReranker**: 利用对比学习数据

**使用示例**:
```python
# 创建重排序器
reranker = create_reranker("contrastive")

# 应用重排序
reranked_docs = show_retrieved_documents_with_rerank(db, question, contrastive_data, reranker)
```

### 7. feedback.py - 用户反馈

**功能**:
- 收集用户反馈（1=正确，0=错误）
- 统计分析反馈数据
- 会话总结报告

## 运行模式

### 在线模式

**特点**:
- 使用云端API（OpenAI、DeepSeek等）
- 响应速度快
- 需要API密钥和网络连接

**启动命令**:
```bash
# 安装依赖
pip install -r requirements.txt

# 启动系统
python main.py
```

**环境变量配置**:
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
```



## 系统配置

### 模型配置

**在线模型**:
- LLM: `deepseek-chat` 
- 嵌入模型: `BAAI/bge-small-en-v1.5`

**离线模型**:
- LLM: `llama3.2:3b` 或 `phi3:3.8b` (通过Ollama)
- 嵌入模型: `BAAI/bge-small-en-v1.5` (本地HuggingFace)

### 分块参数

| 策略 | 参数 | 说明 | 默认值 |
|------|------|------|--------|
| 字符分块 | chunk_size | 分块大小 | 200 |
| 字符分块 | chunk_overlap | 重叠大小 | 20 |
| 主题分块 | num_topics | 主题数量 | 5 |

### 重排序参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| keyword_weight | 关键词权重 | 0.7 |
| length_weight | 长度权重 | 0.3 |
| threshold | 语义匹配阈值 | 0.7 |

## 使用示例

### 基础使用

```python
# 1. 数据准备
from data_processing import load_and_clean_data
qa_texts, sample_info = load_and_clean_data("WikiQA/WikiQA-train.tsv")

# 2. 构建向量数据库
from vector_db import build_vector_db
db = build_vector_db(qa_texts)

# 3. 创建问答系统
from qa_system import create_qa_chain
qa = create_qa_chain(db)

# 4. 执行问答
answer = qa.invoke({"query": "What is machine learning?"})
print(answer["result"])
```

### 高级使用

```python
# 完整流程示例
import os
from data_processing import load_and_clean_data, create_contrastive_learning_data
from vector_db import build_vector_db
from qa_system import create_qa_chain
from reranking import create_reranker, show_retrieved_documents_with_rerank
from evaluation import evaluate_model

# 1. 加载数据并生成对比学习数据
qa_texts, sample_info = load_and_clean_data("WikiQA/WikiQA-train.tsv", 
                                           include_negative_samples=True,
                                           negative_ratio=0.3)
contrastive_data = create_contrastive_learning_data(qa_texts, sample_info)

# 2. 构建向量数据库
db = build_vector_db(qa_texts, chunking_strategy="topic", num_topics=5)

# 3. 创建问答系统
qa = create_qa_chain(db)

# 4. 评估模型
evaluate_model(qa, max_q=100)

# 5. 应用重排序
reranker = create_reranker("contrastive")
show_retrieved_documents_with_rerank(db, "What is AI?", contrastive_data, reranker)
```



## 故障排除

### 常见问题及解决方案

| 问题 | 解决方案 |
|------|----------|
| API连接失败 | 检查网络连接和API密钥 |
| Ollama连接失败 | 运行`ollama serve`启动服务 |
| 内存不足 | 使用更小的模型或减少批处理大小 |
| 回答质量差 | 调整模型参数或优化训练数据 |
| 检索结果不准确 | 尝试不同的分块策略或重排序方法 |



## API参考

### 主要函数接口

#### data_processing.py
- `load_and_clean_data(tsv_path, include_negative_samples, negative_ratio)`
- `create_contrastive_learning_data(qa_texts, sample_info)`
- `analyze_data_quality(qa_texts)`

#### vector_db.py
- `build_vector_db(qa_texts, chunking_strategy, chunk_size, chunk_overlap, num_topics)`
- `demo_chunking_strategies(qa_texts)`

#### qa_system.py
- `create_qa_chain(db)`

#### evaluation.py
- `evaluate_model(qa_chain, tsv_path, max_q)`

#### reranking.py
- `create_reranker(reranker_type, **kwargs)`
- `show_retrieved_documents_with_rerank(db, question, contrastive_data, reranker)`
