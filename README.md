
# PaperMate · 多智能体协作科研自助系统

<p align="center">
  <strong>基于 LangGraph 的 4 专家协作架构 · 动态重排检索 · PDF 多模态提取</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-0.2+-green.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Ollama-0.1+-orange.svg" alt="Ollama">
  <img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License">
</p>

---

## 📖 简介

**PaperMate** 是一个面向学术研究者的本地化论文阅读与知识问答系统。它基于 RAG（检索增强生成）架构，将用户本地的 PDF 论文构建为可检索的知识库，并通过 **4 专家协作架构** 实现从简单问答到文献综述的全流程自动化。

### ✨ 核心特性

| 功能 | 描述 | 技术实现 |
|------|------|----------|
| 📚 **论文知识库构建** | 解析 PDF 文字层 + OCR 提取图表文字，自动向量化 | PyMuPDF + Tesseract OCR + Ollama Embedding |
| 🎯 **动态重排检索** | 根据查询复杂度 + 分数分布自适应决定 K 值 | BGE-Reranker + 三重 K 决策策略 |
| 🤖 **多智能体问答** | Supervisor 调度 4 专家协作完成复杂任务 | LangGraph StateGraph + 4 专家角色 |
| 📊 **文献对比综述** | 跨论文方法对比、研究脉络梳理 | 结构化综述报告生成 |
| 💬 **Web 交互界面** | 多会话对话管理 + 流式输出 | Streamlit |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    前端层 (Presentation)                         │
│              Streamlit App (app.py)                              │
│  多会话管理 · 流式输出 · 快捷提问 · 历史切换                      │
└──────────────────────────┬──────────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────────┐
│                   编排层 (Orchestration)                          │
│              MultiAgentWrapper (react_agent.py)                   │
│         ↓                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         LangGraph StateGraph 工作流                        │   │
│  │                                                           │   │
│  │  [Supervisor] ────→ [Retriever]                           │   │
│  │       │                   │                               │   │
│  │       ├───────────→ [Analyst]                             │   │
│  │       │                   │                               │   │
│  │       └───────────→ [Reviewer]                            │   │
│  │       ↑___________↓ (loop)                                │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────────┐
│                   服务层 (Services)                               │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐       │
│  │ RagService   │  │ DataLoader   │  │ ImageExtractor   │       │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘       │
│         │                 │                    │                 │
│  ┌──────▼────────┐  ┌─────▼────────────────────▼──────────┐    │
│  │DynamicReranker│  │      VectorStoreService              │    │
│  └──────┬────────┘  └─────────────────────────────────────┘    │
│         │                                                       │
│  ┌──────▼────────┐                                              │
│  │RerankerService│                                              │
│  └───────────────┘                                              │
└──────────────────────────┬──────────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────────┐
│                   模型层 (Model)                                  │
│  ┌──────────────────┐    ┌───────────────────────┐              │
│  │  ChatModelFactory │    │ EmbeddingsFactory      │              │
│  └──────────────────┘    └───────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

### 专家角色

| 专家 | 职责 | 核心能力 |
|------|------|----------|
| 🧠 **Supervisor** | 意图识别 → 任务分解 → 调度分发 → 结果整合 | LLM 解析意图，子任务队列，路由决策 |
| 🔍 **Retriever** | 从知识库搜索论文，提取关键段落 | 向量检索 + 动态重排，格式化结果 |
| 📝 **Analyst** | RAG 问答、概念解释、数据提取 | 基于检索结果的深度分析 |
| 📊 **Reviewer** | 文献对比、研究脉络、综述撰写 | 多论文对比表格，结构化综述报告 |

---

## 🚀 快速开始

### 环境要求

| 组件 | 版本要求 | 用途 |
|------|----------|------|
| Python | ≥ 3.10 | 运行环境 |
| Ollama | latest | 本地 LLM 服务 |
| Tesseract OCR | ≥ 5.0 | 图片文字识别 |
| PyTorch | ≥ 2.0 | BGE-Reranker 推理 |
| 内存 | ≥ 8 GB | 向量存储 + 模型加载 |

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/lalakaka22/langchain-agent.git
cd langchain-agent

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 Tesseract OCR (Windows)
winget install UB-Mannheim.TesseractOCR
# 确认路径: C:\Program Files\Tesseract-OCR\tesseract.exe

# 5. 启动 Ollama 服务
ollama serve

# 6. 拉取模型
ollama pull qwen3:4b
ollama pull nomic-embed-text

# 7. 放置论文 PDF 到指定目录
# C:\Users\你的用户名\Desktop\论文\

# 8. 启动 Web 服务
streamlit run app.py
```

### 配置说明

创建 `.env` 文件配置模型服务：

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen3:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_TEMPERATURE=0.7
OLLAMA_NUM_PREDICT=2048
```

---

## 📁 项目结构

```
LangChain ReAct Agent/
│
├── agent/                          # 智能体模块
│   ├── react_agent.py              # 入口: MultiAgentWrapper
│   └── multi_agent/                # 多智能体子系统
│       ├── state.py                # 共享状态协议
│       ├── supervisor.py           # 主管智能体
│       ├── retriever.py            # 检索专家
│       ├── analyst.py              # 分析专家
│       ├── reviewer.py             # 综述专家
│       └── workflow.py             # LangGraph 工作流
│
├── rag/                            # RAG 检索模块
│   ├── vector_store.py             # 内存向量存储
│   ├── data_loader.py              # PDF 加载器
│   ├── image_extractor.py          # 图片提取 + OCR
│   ├── rag_service.py              # RAG 问答服务
│   ├── reranker.py                 # BGE-Reranker 重排序
│   └── dynamic_reranker.py         # 动态重排智能体
│
├── model/                          # 模型工厂
│   └── factory.py                  # ChatModelFactory + EmbeddingsFactory
│
├── utils/                          # 工具模块
│   ├── config_handler.py           # YAML 配置加载
│   ├── file_handler.py             # 文件工具
│   ├── logger_handler.py           # 日志管理
│   └── prompt_loader.py            # 提示模板加载
│
├── prompts/                        # 提示模板
├── config/                         # 配置文件
├── app.py                          # Streamlit 前端
├── requirements.txt                # 依赖清单
└── .env                            # 环境变量
```

---

## 🔧 使用示例

### Python API

```python
from agent.react_agent import MultiAgentWrapper

# 初始化智能体
agent = MultiAgentWrapper()

# 同步执行
answer = agent.execute("Transformer 的自注意力机制是什么？")
print(answer)

# 流式执行
for chunk in agent.execute_stream("比较 BERT 和 GPT 的预训练方法"):
    print(chunk, end="", flush=True)
```

### 快捷问题示例

- 💡 "什么是 Transformer 的自注意力机制？"
- 🔬 "比较 BERT 和 GPT 的预训练目标"
- 📚 "综述一下 2023 年大语言模型的发展"
- 📊 "这篇论文用了什么数据集？实验效果如何？"

---

## 🛠️ 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| LLM | Ollama (qwen3:4b) | 聊天生成/任务分解/结果整合 |
| Embedding | Ollama (nomic-embed-text) | 文档与查询向量化 |
| 重排序 | BGE-Reranker-base | 精准相关性评分 |
| 工作流 | LangGraph StateGraph | 多智能体任务编排 |
| OCR | Tesseract 5.x | PDF 图片文字识别 |
| PDF 提取 | PyMuPDF + PyPDF | 文字提取 + 图片提取 |
| 前端 | Streamlit | Web 对话界面 |
| 向量库 | 自定义内存存储 (Cosine Similarity) | 文档相似度检索 |

---

## 📝 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

- 作者：lalakaka22
- GitHub：[https://github.com/lalakaka22/langchain-agent](https://github.com/lalakaka22/langchain-agent)

---

<p align="center">
  <strong>PaperMate — 让 AI 学会像研究者一样协作思考。</strong>
</p>

---
