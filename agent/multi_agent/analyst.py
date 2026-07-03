"""
分析专家 (Analyst Expert)

职责:
  - RAG 问答：基于检索到的文献内容回答用户问题
  - 图表解析：解释论文中的图表和数据
  - 概念解释：对学术概念进行深入解释
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from agent.multi_agent.state import MultiAgentState, ExpertRole
from model.factory import chat_model

ANALYST_SYSTEM_PROMPT = """你是 PaperMate 多智能体系统的分析专家（Analyst）。

你的核心职责：
1. 基于检索到的论文内容，深入回答用户问题
2. 解释论文中的核心概念、方法、实验
3. 提供结构化的分析，包含引用来源

## 输出要求
- 使用中文回答
- 引用具体的论文和页码（例如："根据 BERT 论文（p.3）..."）
- 结构化呈现：概念解释 → 方法细节 → 关键发现
- 如果信息不足，明确指出

## 上下文
你会收到检索专家找到的论文内容，请基于这些内容进行分析。
"""


class AnalystExpert:
    """分析专家 — RAG 问答与概念解释"""

    def __init__(self):
        self.model = chat_model

    def execute(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        执行分析任务

        从共享状态读取：
        - user_query: 用户问题
        - retriever_results: 检索结果
        - current_task: 当前子任务描述
        """
        user_query = state.get("user_query", "")
        retriever_results = state.get("retriever_results", [])
        subtasks = state.get("subtasks", [])

        # 找到当前执行的子任务
        current_task = ""
        for task in subtasks:
            if task.get("status") == "running" or task.get("expert") == ExpertRole.ANALYST.value:
                current_task = task.get("description", "")
                break
        if not current_task:
            current_task = user_query

        # 构建上下文
        context = "\n\n".join(retriever_results) if retriever_results else ""

        # 如果检索结果为空，尝试直接 RAG 搜索
        if not context or "未检索到" in context or "未找到" in context:
            context = self._direct_rag_search(user_query)

        prompt = f"""当前子任务：{current_task}

用户原始问题：{user_query}

检索到的论文内容：
{context[:5000]}

请基于以上论文内容进行分析回答。"""

        try:
            response = self.model.invoke([
                SystemMessage(content=ANALYST_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            content = response.content if hasattr(response, 'content') else str(response)
            answer = content.strip()
        except Exception as e:
            answer = f"[Analyst] 分析失败: {e}"

        log_entry = {
            "expert": ExpertRole.ANALYST.value,
            "action": "分析完成",
            "task": current_task[:80],
            "result_length": len(answer),
        }

        return {
            "analyst_results": [answer],
            "execution_log": [log_entry],
        }

    def _direct_rag_search(self, query: str) -> str:
        """兜底：当检索专家无结果时，自行RAG搜索"""
        try:
            from rag.rag_service import RagSummarizeService
            rag = RagSummarizeService(recall_k=10)
            result = rag.rag_summarize_with_sources(query)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            if sources:
                refs = "\n\n参考文件:\n"
                for s in sources:
                    refs += f"- {s.get('metadata', {}).get('source', '未知')} (score={s.get('score', 0):.3f})\n"
                return answer + refs
            return answer or "未能检索到相关内容。"
        except Exception as e:
            return f"[Analyst Fallback] 直接 RAG 搜索失败: {e}"
