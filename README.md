
                                 PaperMate
                   多智能体协作科研自助系统 · 架构设计文档
            基于 LangGraph 的 4 专家协作架构 · 动态重排检索 · PDF 多模态提取

================================================================================

目录
  1. 系统概述
  2. 总体架构
  3. 核心模块设计
    3.1 多智能体协作引擎
    3.2 动态重排智能体 (Dynamic Reranker Agent)
    3.3 RAG 检索管道
    3.4 PDF 多模态提取
    3.5 向量存储服务
    3.6 模型工厂
  4. 多智能体工作流
  5. 接口规范
  6. 配置说明
  7. 部署指南
  8. 目录结构

================================================================================

1. 系统概述
-----------

1.1 系统定位
PaperMate 是一个面向学术研究者的本地化论文阅读与知识问答系统。它基于 RAG
（检索增强生成）架构，将用户本地的 PDF 论文构建为可检索的知识库，并通过
多智能体协作架构实现从简单问答到文献综述的全流程自动化。

1.2 核心功能

  功能                    描述                              涉及模块
  ─────────────────────────────────────────────────────────────────
  论文知识库构建          解析 PDF 文字层 + OCR 提取图表文字    data_loader,
                          自动向量化                           image_extractor
  动态重排检索            根据查询复杂度 + 分数分布自适应决定    dynamic_reranker,
                          K 值                                 reranker
  多智能体问答            Supervisor 调度 4 专家协作            supervisor,
                                                               workflow
  文献对比综述            跨论文方法对比、研究脉络梳理           reviewer
  Web 交互界面            Streamlit 多会话对话界面              app.py

1.3 技术选型

  类别            技术                                 用途
  ─────────────────────────────────────────────────────────────────
  LLM             Ollama (qwen3:4b)                    聊天生成/任务分解/结果整合
  Embedding       Ollama (nomic-embed-text)            文档与查询向量化
  重排序           BGE-Reranker-base (XLMRoBERTa)       精准相关性评分
  工作流           LangGraph StateGraph                 多智能体任务编排
  OCR             Tesseract 5.x                         PDF 图片文字识别
  PDF 提取        PyMuPDF (fitz) + PyPDF                文字提取 + 图片提取
  前端            Streamlit                             Web 对话界面
  向量库           自定义内存向量存储 (Cosine Similarity)  文档相似度检索

================================================================================

2. 总体架构
-----------

2.1 分层架构图

  ┌─────────────────────────────────────────────────────────────────┐
  │                    前端层 (Presentation)                         │
  │              Streamlit App (app.py)                              │
  │  多会话管理 · 流式输出 · 快捷提问 · 历史切换                      │
  └──────────────────────────┬──────────────────────────────────────┘
                              │ execute() / execute_stream()
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
  │  │ (RAG 问答)   │  │ (PDF 加载)   │  │ (OCR 图片识别)   │       │
  │  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘       │
  │         │                 │                    │                 │
  │  ┌──────▼────────┐  ┌─────▼────────────────────▼──────────┐    │
  │  │DynamicReranker│  │      VectorStoreService              │    │
  │  │(自适应K决策)  │  │ (内存向量存储 + 余弦相似度)           │    │
  │  └──────┬────────┘  └─────────────────────────────────────┘    │
  │         │                                                       │
  │  ┌──────▼────────┐                                              │
  │  │RerankerService│                                              │
  │  │(BGE-Reranker) │                                              │
  │  └───────────────┘                                              │
  └──────────────────────────┬──────────────────────────────────────┘
                              │
  ┌──────────────────────────▼──────────────────────────────────────┐
  │                   模型层 (Model)                                  │
  │                                                                  │
  │  ┌──────────────────┐    ┌───────────────────────┐              │
  │  │  ChatModelFactory │    │ EmbeddingsFactory      │              │
  │  │  (Ollama/Tongyi)  │    │ (Ollama/Tongyi)        │              │
  │  └──────────────────┘    └───────────────────────┘              │
  └─────────────────────────────────────────────────────────────────┘

2.2 数据流

  用户问题
      │
      ▼
  ┌──────────┐    意图识别    ┌──────────────┐
  │Supervisor│ ────────────→ │ 任务类型判定   │
  │  (主管)  │               │ + 子任务分解   │
  └────┬─────┘               └──────┬───────┘
       │                            │
       │   search_queries           │ 子任务队列
       ▼                            ▼
  ┌──────────┐  向量检索 Top10  ┌──────────┐  RAG 问答  ┌──────────┐
  │Retriever │ ───────────────→ │ Analyst  │ ←──────── │ Reviewer │
  │  (检索)  │                  │ (分析)   │  检索结果  │ (综述)   │
  └────┬─────┘                  └────┬─────┘           └────┬─────┘
       │                            │                      │
       │  动态重排 (自适应K)         │  分析结果             │  综述报告
       ▼                            ▼                      ▼
  ┌───────────────────────────────────────────────────────────────┐
  │                    Supervisor 结果整合                           │
  │                  LLM 润色 → 最终答案                            │
  └───────────────────────────────────────────────────────────────┘
       │
       ▼
  Streamlit 前端流式展示

================================================================================

3. 核心模块设计
---------------

3.1 多智能体协作引擎

设计理念：将复杂的研究任务分解为检索、分析、综述三个子任务，由 Supervisor
（主管）统一调度各专家智能体协作完成。

3.1.1 共享状态协议 (state.py)

系统基于 LangGraph 的 TypedDict 定义了统一的状态协议 MultiAgentState，
所有专家节点共享读写此状态：

  class MultiAgentState(TypedDict, total=False):
      # ── 用户输入 ──
      user_query: str                          # 用户原始问题

      # ── 主管决策 ──
      task_type: str                           # 任务类型
      complexity_score: float                  # 查询复杂度 [0, 1]
      plan: str                                # 任务规划描述

      # ── 任务队列 ──
      subtasks: List[Dict]                     # [{description, expert, status, result}]
      current_subtask_index: int

      # ── 结果池 (Annotated[List, add] 自动合并) ──
      retriever_results: Annotated[List[str], add]
      analyst_results: Annotated[List[str], add]
      reviewer_results: Annotated[List[str], add]

      # ── 执行控制 ──
      phase: str                               # idle → planning → ... → finished
      iteration: int                           # 当前迭代轮数 (上限: 8)
      supervisor_decision: str                 # continue | finish | retry

      # ── 最终输出 ──
      final_answer: str
      execution_log: Annotated[List[Dict], add]

关键设计点：
  - 使用 Annotated[List, add] 实现结果自动追加合并，无需手动拼接
  - total=False 允许节点只返回部分字段，LangGraph 自动处理状态合并

3.1.2 专家角色定义

  专家          文件              职责                              核心能力
  ─────────────────────────────────────────────────────────────────────────────
  Supervisor    supervisor.py     意图识别→任务分解→调度分发→结果整合   LLM 解析意图，子任务队列，路由决策
  Retriever     retriever.py      从知识库搜索论文，提取关键段落         向量检索+动态重排，格式化结果
  Analyst       analyst.py        RAG 问答、概念解释、数据提取           基于检索结果的深度分析，兜底直接RAG
  Reviewer      reviewer.py       文献对比、研究脉络、综述撰写           多论文对比表格，结构化综述报告

3.1.3 Supervisor 路由规则

  任务类型判定:
    纯闲聊/问候 ──────────────────────→ Analyst (直接回答)
    概念解释 (学术) ──────────────────→ Retriever → Analyst
    需要查论文 ──────────────────────→ Retriever → Analyst
    对比多个方法/模型 ────────────────→ Retriever → Analyst → Reviewer
    复杂多子问题/研究总结 ────────────→ Retriever → Analyst → Reviewer (多轮)
    文献综述/领域调研 ────────────────→ Retriever → Reviewer (多轮)

3.1.4 终止条件
  1. 最大迭代轮数: 8 轮强制终止
  2. 所有子任务完成: subtasks[*].status == "completed"
  3. 主管判定质量达标: supervisor_decision == "finish"
  4. 信息不足时重试: supervisor_decision == "retry" → 补充检索

--------------------------------------------------------------------------------

3.2 动态重排智能体 (Dynamic Reranker Agent)

设计理念：替代固定 Top-K 重排序，根据查询特点和 BGE-Reranker 分数分布动态
决定最终送入 LLM 的文档数量。

3.2.1 查询画像 (QueryProfile)

  @dataclass
  class QueryProfile:
      length: int              # 查询长度
      word_count: int          # 词数
      has_comparison: bool     # 含比较意图 (vs/区别/对比)
      has_causal: bool         # 含因果推理 (为什么/导致)
      has_multi_part: bool     # 含多子问题
      domain_specificity: float  # 领域特异性 [0, 1]
      complexity_score: float    # 综合复杂度 [0, 1]

基于中英文关键词词典进行意图特征提取，加权计算复杂度分数。

3.2.2 三重 K 决策策略

  最终 K = min(复杂度K, 缺口K, 阈值K)，且 K ∈ [1, 8]

  策略 1: 复杂度决定基准 K
    c < 0.15 → K=1 (极简单定义)        c < 0.30 → K=2 (单一概念)
    c < 0.50 → K=3 (中等)              c < 0.70 → K=4 (对比)
    c ≥ 0.70 → K=5 (复杂多子问题)
    对比意图 +1, 多子问题保底 4

  策略 2: 分数缺口检测
    计算相邻分数落差 gaps[i] = score[i] - score[i+1]
    阈值 = avg(gaps) × (2.0 - complexity × 0.8)
    在 gap > 阈值处截断 (至少保留第1个)

  策略 3: 分数阈值过滤
    归一化分数范围 → 根据复杂度决定保留比例 (30%~60%)
    反归一化得到实际截断阈值

3.2.3 RL 反馈追踪

每次决策记录为 DREpisode，支持后期质量评估和策略优化：

  @dataclass
  class DREpisode:
      query: str
      query_profile: QueryProfile
      all_scores: List[float]
      chosen_k: int
      score_threshold: float
      decision_reason: str
      output_quality_score: Optional[float] = None  # 反馈

--------------------------------------------------------------------------------

3.3 RAG 检索管道

管道流程：

  用户查询
      │
      ▼
  ┌──────────────────────┐
  │ 1. 向量召回 Top-10    │  ← VectorStoreService (cosine similarity)
  │    (nomic-embed-text) │
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ 2. BGE-Reranker 评分  │  ← RerankerService (XLMRoBERTa CrossEncoder)
  │    (所有候选)          │
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ 3. 动态 K 决策        │  ← DynamicRerankerAgent (三重策略)
  │    自适应截断          │
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ 4. LLM 生成回答       │  ← ChatModel (Ollama qwen3:4b)
  │    + 参考文献列表      │
  └──────────────────────┘

参考文献展示格式：

  ## 参考文献（动态重排智能体 · 自适应K）
  *动态重排智能体决策: 自适应 K=3 (复杂度=0.45, 原因: 复杂度K=3 | 阈值过滤K=3)*

  [1] Attention.pdf (页码: 3)
      BGE-Reranker 相关性分数: 5.234  |  ★★★★★ 极强相关
      > Self-attention computes attention weights across all positions...

  [2] Transformer.pdf (页码: 5)
      BGE-Reranker 相关性分数: 3.102  |  ★★★★ 高度相关
      > The Transformer uses multi-head attention...

--------------------------------------------------------------------------------

3.4 PDF 多模态提取

设计理念：学术论文中大量关键信息（公式、图表、实验结果）存在于图片中，
仅提取文字层会丢失 30%+ 的可检索内容。

3.4.1 双路提取架构

  PDF 文件
      │
      ├── 第 1 路: 文字提取 ──────────────────────┐
      │   PyPDFLoader → Document (type=text)      │
      │                                            ▼
      └── 第 2 路: 图片提取 ───────────────┐   ┌──────┴─────┐
          PyMuPDF (fitz) 提取图片           │   │ 文本分割器  │
          │                                 │   │(600/100)   │
          ├─ 嵌入图片 (get_images)          │   └──────┬─────┘
          └─ 整页渲染 (2x 缩放, 兜底)       │          │
              │                             │          ▼
              ▼                             │   ┌──────────────┐
          Tesseract OCR 识别                │   │ VectorStore  │
          │ (--psm 6 统一文本块)            │   │   向量存储    │
          │                                 │   └──────────────┘
          ├─ 灰度预处理                      │
          ├─ 质量过滤 (尺寸 100~3000px)      │
          └─ 文字清洗 (去噪声行)             │
              │                             │
              ▼                             │
          Document (type=图片OCR,           │
                    content_type=image_text)─┘

3.4.2 图片质量过滤

  条件              阈值           目的
  ─────────────────────────────────────────
  最小宽度           100px          过滤图标/装饰元素
  最小高度           100px          同上
  最大宽度           3000px         跳过超大渲染图
  最小文字长度        10 字符        跳过纯图表/无文字图片

3.4.3 文档元数据

每个 Document 携带完整来源信息：

  Document(
      page_content="OCR 识别文字内容",
      metadata={
          "source": "BERT.pdf",           # 源文件名
          "type": "图片OCR",               # 提取类型
          "domain": "NLP",                # 领域分类 (子目录名)
          "content_type": "image_text",   # 内容类型标记
          "page": 3,                      # 所在页码
          "image_index": 0,               # 图片序号
          "extraction_method": "tesseract_ocr",
      }
  )

--------------------------------------------------------------------------------

3.5 向量存储服务

设计：轻量级内存向量存储，使用 nomic-embed-text 生成嵌入，scikit-learn 计算
余弦相似度。

  class VectorStoreService:
      documents: List[Document]    # 原始文档
      embeddings: List[ndarray]    # 预计算嵌入向量

      def similarity_search(query, k):
          query_vec = embed_model.embed_query(query)
          similarities = cosine_similarity([query_vec], embeddings_matrix)
          top_indices = argsort(similarities)[-k:][::-1]
          return [documents[i] for i in top_indices]

为什么不直接用 ChromaDB 持久化？避免 Windows 环境下 SQLite 文件锁定问题，
且内存模式启动更快、更适合本地单机场景。

--------------------------------------------------------------------------------

3.6 模型工厂

基于工厂模式，支持多提供商切换：

  # .env 配置
  LLM_PROVIDER=ollama          # ollama | tongyi
  EMBEDDING_PROVIDER=ollama    # ollama | tongyi
  OLLAMA_CHAT_MODEL=qwen3:4b
  OLLAMA_EMBEDDING_MODEL=nomic-embed-text

  # 全局实例
  chat_model = ChatModelFactory().generator()     # ChatOllama / ChatTongyi
  embed_model = EmbeddingsFactory().generator()   # OllamaEmbeddings / DashScopeEmbeddings

================================================================================

4. 多智能体工作流
-----------------

4.1 LangGraph 状态图

                      ┌─────────────────────┐
                      │   supervisor_plan   │ ← Entry Point
                      │  (意图识别/决策)      │
                      └──────┬──────┬───────┘
                             │      │
                ┌────────────┤      ├────────────┐
                ▼            ▼      ▼            ▼
          ┌──────────┐ ┌──────────┐ ┌──────────┐
          │retriever │ │ analyst  │ │ reviewer │
          │  节点    │ │  节点    │ │  节点    │
          └────┬─────┘ └────┬─────┘ └────┬─────┘
               │            │            │
               └────────────┼────────────┘
                            │
                route_after_expert()
                            │
                      ┌─────▼─────┐
                      │ supervisor│ ← 循环检查
                      │    or     │
                      │   END     │ ← 终止
                      └───────────┘

4.2 路由函数

  def route_after_supervisor(state) → "retriever" | "analyst" | "reviewer" | "end":
      根据 supervisor_decision 和子任务状态路由到对应专家
      if decision == "finish": return "end"

  def route_after_expert(state) → "supervisor" | "end":
      专家执行完毕 → 检查是否继续
      if iteration >= 8: return "end"         # 轮数超限
      if all_done: return "supervisor"         # 所有子任务完成 → 最终决策

4.3 执行时序

  时间 →
  Supervisor: [规划] ────────────────────────────── [检查] ────── [整合]
                 │                                    ▲            ▲
  Retriever:     └──→ [向量检索] → [动态重排] ────────┘            │
                                                         ▲         │
  Analyst:      ┌──────────────────────── [RAG分析] ────┘         │
                │                                                  │
  Reviewer:     └───────────────────────────────── [综述撰写] ─────┘

================================================================================

5. 接口规范
-----------

5.1 对外 API

  from agent.react_agent import MultiAgentWrapper

  agent = MultiAgentWrapper()

  # 同步执行
  answer: str = agent.execute("Transformer 的自注意力机制是什么？")

  # 流式执行
  for chunk in agent.execute_stream("比较 BERT 和 GPT 的预训练方法"):
      print(chunk, end="", flush=True)

5.2 工厂函数

  from agent.react_agent import create_agent

  # 多智能体模式 (默认)
  agent = create_agent("multi")     # → MultiAgentWrapper

  # 单智能体模式
  agent = create_agent("simple")    # → SimpleReactAgent

5.3 环境变量 (.env)

  LLM_PROVIDER=ollama                      # 模型提供商: ollama | tongyi
  OLLAMA_BASE_URL=http://localhost:11434   # Ollama 服务地址
  OLLAMA_CHAT_MODEL=qwen3:4b              # 聊天模型
  OLLAMA_EMBEDDING_MODEL=nomic-embed-text  # 嵌入模型
  OLLAMA_TEMPERATURE=0.7                   # 生成温度
  OLLAMA_NUM_PREDICT=2048                  # 最大输出 token 数
  CHROMA_PERSIST=false                     # 向量库持久化 (false=内存模式)

================================================================================

6. 配置说明
-----------

  配置文件              用途            关键参数
  ─────────────────────────────────────────────────────────────────
  .env                  模型/服务配置     LLM_PROVIDER, OLLAMA_BASE_URL, 模型名
  config/rag.yml        RAG 服务配置     chat_model_name, embedding_model_name
  config/chroma.yml     向量库配置       collection_name, chunk_size(200),
                                         chunk_overlap(20)
  config/prompts.yml    提示模板路径     各 prompt 文件位置映射
  config/agent.yml      Agent 配置       external_data_path

知识库目录结构：

  C:\Users\22\Desktop\论文\
  ├── NLP\
  │   ├── Transformer.pdf
  │   ├── BERT.pdf
  │   └── GPT.pdf
  ├── CV\
  │   └── pix2pix.pdf
  └── ...

子目录名自动作为论文领域分类标记 (domain 元数据)。

================================================================================

7. 部署指南
-----------

7.1 环境要求

  组件                 版本要求        用途
  ──────────────────────────────────────────
  Python               ≥ 3.10         运行环境
  Ollama               latest         本地 LLM 服务
  Tesseract OCR        ≥ 5.0          图片文字识别
  PyTorch              ≥ 2.0          BGE-Reranker 推理
  内存                  ≥ 8 GB         向量存储 + 模型加载

7.2 安装步骤

  # 1. 克隆项目
  git clone <repo-url>
  cd "LangChain ReAct Agent"

  # 2. 创建虚拟环境
  python -m venv .venv
  .venv\Scripts\activate

  # 3. 安装依赖
  pip install -r requirements.txt

  # 4. 安装 Tesseract OCR (Windows)
  winget install UB-Mannheim.TesseractOCR
  # 确认路径: C:\Program Files\Tesseract-OCR\tesseract.exe

  # 5. 下载 BGE-Reranker 模型 (首次运行自动下载)
  # 或手动下载放到 models/bge-reranker-base/

  # 6. 启动 Ollama 服务
  ollama serve

  # 7. 拉取模型
  ollama pull qwen3:4b
  ollama pull nomic-embed-text

  # 8. 放置论文 PDF 到 C:\Users\22\Desktop\论文\

  # 9. 启动 Web 服务
  streamlit run app.py

7.3 首次运行流程

  1. 启动后自动加载 C:\Users\22\Desktop\论文\ 下所有 PDF
  2. 双路提取（文字层 + 图片 OCR）
  3. 文本分块（chunk_size=600, overlap=100）
  4. 向量化并存入内存
  5. 进入对话界面

================================================================================

8. 目录结构
-----------

  LangChain ReAct Agent/
  │
  ├── agent/                          # 智能体模块
  │   ├── react_agent.py              # 入口: MultiAgentWrapper, create_agent()
  │   ├── multi_agent/                # 多智能体子系统
  │   │   ├── __init__.py             # 公开 API
  │   │   ├── state.py                # 共享状态协议 (TypedDict + 枚举)
  │   │   ├── supervisor.py           # 主管智能体 (意图识别/调度/整合)
  │   │   ├── retriever.py            # 检索专家 (向量搜索/动态重排)
  │   │   ├── analyst.py              # 分析专家 (RAG 问答/概念解释)
  │   │   ├── reviewer.py             # 综述专家 (文献对比/综述撰写)
  │   │   └── workflow.py             # LangGraph 工作流定义 + 编排器
  │   └── tools/                      # 工具函数
  │       ├── agent_tools.py          # RAG 工具 + 运维模拟工具
  │       └── middleware.py
  │
  ├── rag/                            # RAG 检索模块
  │   ├── vector_store.py             # 内存向量存储 (cosine similarity)
  │   ├── data_loader.py              # PDF 加载器 (双路提取入口)
  │   ├── image_extractor.py          # PDF 图片提取 + Tesseract OCR
  │   ├── rag_service.py              # RAG 问答服务 (管道编排)
  │   ├── reranker.py                 # BGE-Reranker 重排序 (CrossEncoder)
  │   └── dynamic_reranker.py         # 动态重排智能体 (自适应K决策)
  │
  ├── model/                          # 模型工厂
  │   └── factory.py                  # ChatModelFactory + EmbeddingsFactory
  │
  ├── utils/                          # 工具模块
  │   ├── config_handler.py           # YAML 配置加载器
  │   ├── file_handler.py             # 文件 MD5 / PDF 加载器
  │   ├── logger_handler.py           # 日志管理器 (控制台 + 文件)
  │   ├── path_tool.py                # 路径工具
  │   └── prompt_loader.py            # 提示模板加载器
  │
  ├── prompts/                        # 提示模板
  │   ├── system_prompt.txt           # 系统角色定义
  │   ├── main_prompt.txt             # 主提示模板
  │   ├── rag_summarize.txt           # RAG 总结模板
  │   └── report_prompt.txt           # 报告生成模板
  │
  ├── config/                         # 配置文件
  │   ├── agent.yml                   # Agent 配置
  │   ├── chroma.yml                  # 向量库配置
  │   ├── prompts.yml                 # 提示模板路径映射
  │   └── rag.yml                     # RAG 模型配置
  │
  ├── logs/                           # 日志输出目录
  ├── models/                         # 本地模型缓存 (BGE-Reranker)
  │   └── bge-reranker-base/
  │
  ├── app.py                          # Streamlit Web 前端入口
  ├── requirements.txt                # 依赖清单
  ├── .env                            # 环境变量配置
  └── .gitignore

================================================================================

  PaperMate — 让 AI 学会像研究者一样协作思考。
