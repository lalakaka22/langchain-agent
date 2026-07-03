"""
PaperMate 多智能体协作系统

支持 4 个专家角色:
  - Supervisor (主管): 意图识别、任务分解、调度分发、结果整合
  - Retriever  (检索): 搜索论文、提取关键内容
  - Analyst    (分析): RAG 问答、概念解释、数据提取
  - Reviewer   (综述): 文献对比、研究脉络、综述撰写

基于 LangGraph 构建的 StateGraph 工作流。
"""
from agent.multi_agent.workflow import MultiAgentOrchestrator, get_multi_agent

__all__ = ["MultiAgentOrchestrator", "get_multi_agent"]
