"""
多智能体共享状态定义

基于 LangGraph 的 TypedDict State，作为各专家节点间的统一通信协议。

状态结构:
  - 用户原始问题
  - 任务类型标识
  - 子任务队列
  - 各专家结果池
  - 当前执行状态
  - 最终答案
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict
from operator import add
from dataclasses import dataclass, field
from enum import Enum


# ================================================================
# 任务类型枚举
# ================================================================
class TaskType(str, Enum):
    """主管智能体识别的任务类型"""
    SIMPLE_QA = "simple_qa"           # 简单问答：直接分析
    RETRIEVAL_QA = "retrieval_qa"     # 需要检索的问答：检索→分析
    COMPARISON = "comparison"         # 文献对比：检索→分析→综述
    COMPLEX_RESEARCH = "complex"      # 复杂研究：检索→分析→综述（多轮）
    LITERATURE_REVIEW = "review"      # 文献综述：多轮检索→综述


# ================================================================
# 专家角色枚举
# ================================================================
class ExpertRole(str, Enum):
    SUPERVISOR = "supervisor"
    RETRIEVER = "retriever"
    ANALYST = "analyst"
    REVIEWER = "reviewer"


# ================================================================
# 执行状态
# ================================================================
class ExecutionPhase(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"         # 主管正在分解任务
    RETRIEVING = "retrieving"     # 检索专家工作中
    ANALYZING = "analyzing"       # 分析专家工作中
    REVIEWING = "reviewing"       # 综述专家工作中
    INTEGRATING = "integrating"   # 主管整合结果
    FINISHED = "finished"
    ERROR = "error"


# ================================================================
# 子任务定义
# ================================================================
@dataclass
class SubTask:
    description: str
    assigned_expert: ExpertRole
    status: str = "pending"       # pending | running | completed | failed
    result: Optional[str] = None


# ================================================================
# 共享状态 (LangGraph State Schema)
# ================================================================
class MultiAgentState(TypedDict, total=False):
    """
    多智能体协作共享状态

    所有专家节点共享读写此状态。LangGraph 自动合并各节点的返回。
    """

    # ---- 用户输入 ----
    user_query: str                          # 用户原始问题

    # ---- 主管决策 ----
    task_type: str                           # 任务类型 (TaskType)
    complexity_score: float                  # 查询复杂度 [0, 1]
    plan: str                                # 任务规划描述

    # ---- 任务队列 ----
    subtasks: List[Dict[str, Any]]           # 子任务列表 [{description, expert, status, result}]
    current_subtask_index: int               # 当前执行的子任务索引

    # ---- 结果池 (各专家产出) ----
    retriever_results: Annotated[List[str], add]   # 检索专家的产出 (叠加)
    analyst_results: Annotated[List[str], add]     # 分析专家的产出 (叠加)
    reviewer_results: Annotated[List[str], add]    # 综述专家的产出 (叠加)

    # ---- 检索数据 ----
    search_queries: List[str]                # 检索专家要执行的搜索词
    retrieved_documents: List[Dict]          # 检索到的文档摘要

    # ---- 执行控制 ----
    phase: str                               # 当前阶段 (ExecutionPhase)
    iteration: int                           # 当前迭代轮数
    max_iterations: int                      # 最大迭代轮数
    supervisor_decision: str                 # 主管的下一步决策: "continue" | "finish" | "retry"

    # ---- 最终输出 ----
    final_answer: str                        # 最终答案

    # ---- 日志记录 ----
    execution_log: Annotated[List[Dict[str, Any]], add]   # 执行日志 (追加)
    started_at: float                        # 开始时间戳
    quality_score: Optional[float]           # 输出质量评分


# ================================================================
# 初始状态工厂
# ================================================================
def create_initial_state(user_query: str, max_iterations: int = 8) -> MultiAgentState:
    """创建初始状态"""
    import time
    return MultiAgentState(
        user_query=user_query,
        task_type="",
        complexity_score=0.0,
        plan="",
        subtasks=[],
        current_subtask_index=0,
        retriever_results=[],
        analyst_results=[],
        reviewer_results=[],
        search_queries=[],
        retrieved_documents=[],
        phase=ExecutionPhase.IDLE.value,
        iteration=0,
        max_iterations=max_iterations,
        supervisor_decision="continue",
        final_answer="",
        execution_log=[],
        started_at=time.time(),
        quality_score=None,
    )
