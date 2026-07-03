"""
综述专家 (Reviewer Expert)

职责:
  - 文献对比：对比多篇论文的方法、结果、贡献
  - 文献综述：梳理研究脉络，写综述段落
  - 报告撰写：整合分析结果，生成结构化报告
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from agent.multi_agent.state import MultiAgentState, ExpertRole
from model.factory import chat_model

REVIEWER_SYSTEM_PROMPT = """你是 PaperMate 多智能体系统的综述专家（Reviewer）。

你的核心职责：
1. 对比多篇论文的方法、实验结果、贡献和局限
2. 梳理研究脉络，识别技术演进趋势
3. 撰写结构化的文献综述或对比报告

## 输出要求
- 用中文撰写，学术风格
- 使用对比表格呈现方法差异
- 标注每篇论文的来源和年份（如果已知）
- 总结研究趋势和未来方向
- 结构：背景概述 → 方法对比 → 实验对比 → 总结展望

## 上下文
你会收到分析专家的初步分析和检索到的论文内容。
"""


class ReviewerExpert:
    """综述专家 — 文献对比与综述撰写"""

    def __init__(self):
        self.model = chat_model

    def execute(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        执行综述任务

        从共享状态读取：
        - user_query: 用户问题
        - analyst_results: 分析专家的初步分析
        - retriever_results: 检索到的论文内容
        """
        user_query = state.get("user_query", "")
        analyst_results = state.get("analyst_results", [])
        retriever_results = state.get("retriever_results", [])
        subtasks = state.get("subtasks", [])

        # 找到当前子任务
        current_task = ""
        for task in subtasks:
            if task.get("status") == "running" or task.get("expert") == ExpertRole.REVIEWER.value:
                current_task = task.get("description", "")
                break

        # 构建上下文
        analysis_context = "\n\n".join(analyst_results) if analyst_results else "（无初步分析）"
        retrieval_context = "\n\n".join(retriever_results) if retriever_results else "（无检索数据）"

        prompt = f"""任务：{current_task or '撰写文献综述报告'}

用户原始问题：{user_query}

## 初步分析结果
{analysis_context[:3000]}

## 检索到的论文内容
{retrieval_context[:3000]}

请基于以上信息，撰写一份结构化的文献综述/对比报告。"""

        try:
            response = self.model.invoke([
                SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            content = response.content if hasattr(response, 'content') else str(response)
            review = content.strip()
        except Exception as e:
            review = f"[Reviewer] 综述生成失败: {e}"

        log_entry = {
            "expert": ExpertRole.REVIEWER.value,
            "action": "综述完成",
            "task": (current_task or user_query)[:80],
            "result_length": len(review),
        }

        return {
            "reviewer_results": [review],
            "execution_log": [log_entry],
        }
