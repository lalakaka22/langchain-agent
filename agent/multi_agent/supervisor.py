"""
主管智能体 (Supervisor Agent)

职责:
  1. 意图识别 — 分析用户问题，判断任务类型
  2. 任务分解 — 将复杂问题拆分为子任务
  3. 调度分发 — 按路由规则分配专家执行
  4. 结果整合 — 汇总各专家输出，生成最终答案

路由决策规则:
  - 简单问答 → 分析专家
  - 需要查资料 → 检索专家 → 分析专家
  - 文献对比 → 检索 → 分析 → 综述
  - 复杂研究 → 检索 → 分析 → 综述（多轮循环）
  - 信息不足 → 触发循环，补充检索
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agent.multi_agent.state import (
    MultiAgentState,
    TaskType,
    ExpertRole,
    ExecutionPhase,
    SubTask,
)
from model.factory import chat_model

# ================================================================
# Supervisor 系统提示
# ================================================================
SUPERVISOR_SYSTEM_PROMPT = """你是 PaperMate 多智能体系统的主管智能体（Supervisor）。

你的核心职责是：
1. **意图识别**：分析用户问题，判断任务类型
2. **任务分解**：将复杂问题拆分为可执行的子任务
3. **调度决策**：决定下一步由哪个专家执行
4. **整合输出**：汇总各专家结果，生成最终答案

## 可调度的专家

| 专家 | 能力 |
|------|------|
| retriever | 从论文知识库搜索相关文献，提取关键段落 |
| analyst | RAG 问答、图表数据解析、概念解释 |
| reviewer | 文献对比分析、研究脉络梳理、综述撰写 |

## 任务类型判定规则（重要：所有论文相关问题必须先检索！）

| 用户问题特征 | 任务类型 | 专家调度顺序 |
|-------------|---------|-------------|
| 概念解释/定义（涉及学术论文的概念） | retrieval_qa | retriever → analyst |
| 需要查具体论文内容 | retrieval_qa | retriever → analyst |
| 对比多个方法/模型 | comparison | retriever → analyst → reviewer |
| 复杂多子问题/研究总结 | complex | retriever → analyst → reviewer（多轮） |
| 文献综述/领域调研 | review | retriever → reviewer（多轮） |
| 纯闲聊/问候/非学术 | simple_qa | analyst（直接回答） |

注意：只要涉及论文、学术概念、方法、模型等问题，都必须先用 retriever 检索文献！
只有明显的闲聊、问候、系统测试类问题才直接用 analyst。

## 输出格式

每次分析后，请严格按以下 JSON 格式输出：

```json
{
  "task_type": "simple_qa | retrieval_qa | comparison | complex | review",
  "decision": "continue | finish | retry",
  "subtasks": [
    {"description": "...", "expert": "retriever|analyst|reviewer", "status": "pending"}
  ],
  "next_expert": "retriever | analyst | reviewer | none",
  "reasoning": "决策理由（简短）"
}
```

如果是最终答案，在 decision="finish" 时，额外输出 final_answer 字段。
"""


class SupervisorAgent:
    """主管智能体"""

    def __init__(self):
        self.model = chat_model

    def analyze_intent(self, query: str) -> Dict[str, Any]:
        """
        第一步：意图识别 + 任务分解

        返回包含 task_type、subtasks、plan 的决策结果。
        """
        prompt = f"""分析以下用户问题，判断任务类型并分解为子任务。

用户问题：{query}

请按 JSON 格式输出：
```json
{{
  "task_type": "...",
  "complexity_score": 0.0-1.0,
  "subtasks": [
    {{"description": "...", "expert": "retriever|analyst|reviewer"}}
  ],
  "search_queries": ["关键词1", "关键词2"],
  "plan": "简要的任务规划说明"
}}
```"""

        try:
            response = self.model.invoke([
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            content = response.content if hasattr(response, 'content') else str(response)

            # 提取 JSON
            parsed = self._extract_json(content)
            return parsed
        except Exception as e:
            print(f"[Supervisor] 意图识别失败: {e}")
            return {
                "task_type": TaskType.SIMPLE_QA.value,
                "complexity_score": 0.2,
                "subtasks": [{"description": query, "expert": "analyst"}],
                "search_queries": [],
                "plan": "默认：直接分析",
            }

    def decide_next_step(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        第二步：根据当前状态判断下一步行动

        读取共享状态中各专家结果，决定继续/结束/重试。
        """
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 8)
        subtasks = state.get("subtasks", [])
        analyst_results = state.get("analyst_results", [])
        reviewer_results = state.get("reviewer_results", [])
        retriever_results = state.get("retriever_results", [])

        # 检查终止条件
        if iteration >= max_iter:
            return self._integrate_results(state, reason="达到最大迭代次数")

        # 检查所有子任务是否完成
        all_done = all(t.get("status") == "completed" for t in subtasks) if subtasks else False

        # 检查是否有足够信息
        has_content = bool(analyst_results or reviewer_results)

        if all_done and has_content:
            return self._integrate_results(state, reason="所有子任务完成")

        if all_done and not has_content:
            # 任务完成了但没有产出，尝试检索
            return self._request_retrieval(state)

        # 找到下一个待执行的子任务
        next_task = None
        for task in subtasks:
            if task.get("status") in ("pending", None):
                next_task = task
                break

        if not next_task:
            if has_content:
                return self._integrate_results(state, reason="无更多子任务")
            return self._integrate_results(state, reason="无法继续")

        return {
            "decision": "continue",
            "next_expert": next_task.get("expert", "analyst"),
            "current_task": next_task.get("description", ""),
            "reasoning": f"执行子任务: {next_task.get('description', '')[:50]}",
        }

    def _integrate_results(self, state: MultiAgentState, reason: str) -> Dict[str, Any]:
        """整合所有专家结果，生成最终答案"""
        analyst = state.get("analyst_results", [])
        reviewer = state.get("reviewer_results", [])

        # 优先使用 reviewer 结果（更完整的综述）
        if reviewer:
            final_answer = "\n\n".join(reviewer)
        elif analyst:
            final_answer = "\n\n".join(analyst)
        else:
            final_answer = "未能生成有效回答，请尝试重新提问。"

        # 如果结果较长，用 LLM 做最后润色
        combined = "\n\n".join(analyst + reviewer)
        if len(combined) > 500:
            try:
                summary_prompt = f"""请基于以下专家分析结果，生成一份结构化的最终回答。

用户问题：{state.get('user_query', '')}

专家分析结果：
{combined[:4000]}

要求：
- 用中文回答，结构清晰
- 引用论文来源
- 保持所有参考文件信息完整"""
                response = self.model.invoke([
                    SystemMessage(content="你是学术论文分析助手，负责整合多专家分析结果。保持参考文献完整。"),
                    HumanMessage(content=summary_prompt),
                ])
                content = response.content if hasattr(response, 'content') else str(response)
                final_answer = content.strip() if content.strip() else final_answer
            except Exception:
                pass  # 兜底使用原始结果

        return {
            "decision": "finish",
            "next_expert": "none",
            "final_answer": final_answer,
            "reasoning": reason,
        }

    def _request_retrieval(self, state: MultiAgentState) -> Dict[str, Any]:
        """信息不足时请求检索"""
        return {
            "decision": "retry",
            "next_expert": "retriever",
            "current_task": "补充检索相关文献",
            "reasoning": "当前信息不足，需要补充检索",
        }

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """从模型输出中提取 JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试从 ```json ... ``` 中提取
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 尝试从 { ... } 中提取
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # 兜底
        return {
            "task_type": TaskType.SIMPLE_QA.value,
            "complexity_score": 0.3,
            "subtasks": [{"description": "分析用户问题", "expert": "analyst"}],
            "search_queries": [],
            "plan": "解析失败，默认简单问答",
        }
