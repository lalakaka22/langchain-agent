"""
LangGraph 多智能体工作流

任务流程图:

  用户输入 → [SUPERVISOR 规划]
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
  [RETRIEVER] [ANALYST] [REVIEWER]
       │          │          │
       └──────────┼──────────┘
                  ▼
           [SUPERVISOR 检查]
                  │
          ┌───────┼───────┐
          ▼       ▼       ▼
       finish   retry   continue → 循环

核心逻辑:
  主管收到问题 → 判断任务类型 → 按规划调度专家 → 专家返回结果 → 主管判断是否继续或结束
"""
from __future__ import annotations

import time
from typing import Any, Dict, Literal

from langgraph.graph import StateGraph, END

from agent.multi_agent.state import (
    MultiAgentState,
    TaskType,
    ExpertRole,
    ExecutionPhase,
    create_initial_state,
)
from agent.multi_agent.supervisor import SupervisorAgent
from agent.multi_agent.retriever import RetrieverExpert
from agent.multi_agent.analyst import AnalystExpert
from agent.multi_agent.reviewer import ReviewerExpert


# ================================================================
# 节点函数 — 每个节点 = 一个专家的一次执行
# ================================================================

def supervisor_plan_node(state: MultiAgentState) -> Dict[str, Any]:
    """主管规划节点：意图识别 + 任务分解"""
    print(f"\n{'='*50}")
    print(f"[Workflow] SUPERVISOR 规划中...")

    supervisor = SupervisorAgent()

    # 首次规划
    if not state.get("subtasks"):
        plan = supervisor.analyze_intent(state["user_query"])
        task_type = plan.get("task_type", TaskType.SIMPLE_QA.value)
        subtasks = plan.get("subtasks", [])

        # 标记第一个子任务为 running
        if subtasks:
            subtasks[0]["status"] = "running"

        log_entry = {
            "expert": ExpertRole.SUPERVISOR.value,
            "action": "任务规划",
            "task_type": task_type,
            "complexity": plan.get("complexity_score", 0),
            "subtask_count": len(subtasks),
            "timestamp": time.time(),
        }

        print(f"[Supervisor] 任务类型: {task_type}, 子任务: {len(subtasks)} 个")

        return {
            "task_type": task_type,
            "complexity_score": plan.get("complexity_score", 0.0),
            "plan": plan.get("plan", ""),
            "subtasks": subtasks,
            "search_queries": plan.get("search_queries", []),
            "phase": ExecutionPhase.PLANNING.value,
            "execution_log": [log_entry],
        }

    # 后续决策
    decision = supervisor.decide_next_step(state)
    print(f"[Supervisor] 决策: {decision.get('decision')}, 下一步: {decision.get('next_expert')}")

    result = {
        "supervisor_decision": decision.get("decision", "continue"),
        "phase": ExecutionPhase.INTEGRATING.value if decision.get("decision") == "finish" else state.get("phase"),
    }

    if decision.get("decision") == "finish":
        result["final_answer"] = decision.get("final_answer", "")
        result["phase"] = ExecutionPhase.FINISHED.value
        print(f"[Supervisor] 任务完成!")

    return result

def retriever_node(state: MultiAgentState) -> Dict[str, Any]:
    """检索专家节点"""
    print(f"[Workflow] RETRIEVER 执行中...")
    retriever = RetrieverExpert()

    # 更新当前子任务状态
    subtasks = state.get("subtasks", [])
    for task in subtasks:
        if task.get("expert") == ExpertRole.RETRIEVER.value and task.get("status") == "running":
            task["status"] = "completed"
            break

    result = retriever.execute(state)

    # 标记下一个子任务
    next_found = False
    for task in subtasks:
        if task.get("status") in ("pending", None) and not next_found:
            task["status"] = "running"
            next_found = True

    result["subtasks"] = subtasks
    result["phase"] = ExecutionPhase.RETRIEVING.value
    result["iteration"] = state.get("iteration", 0) + 1

    return result

def analyst_node(state: MultiAgentState) -> Dict[str, Any]:
    """分析专家节点"""
    print(f"[Workflow] ANALYST 执行中...")
    analyst = AnalystExpert()

    # 更新当前子任务状态
    subtasks = state.get("subtasks", [])
    for task in subtasks:
        if task.get("expert") == ExpertRole.ANALYST.value and task.get("status") == "running":
            task["status"] = "completed"
            break

    result = analyst.execute(state)

    # 标记下一个子任务
    next_found = False
    for task in subtasks:
        if task.get("status") in ("pending", None) and not next_found:
            task["status"] = "running"
            next_found = True

    result["subtasks"] = subtasks
    result["phase"] = ExecutionPhase.ANALYZING.value
    result["iteration"] = state.get("iteration", 0) + 1

    return result

def reviewer_node(state: MultiAgentState) -> Dict[str, Any]:
    """综述专家节点"""
    print(f"[Workflow] REVIEWER 执行中...")
    reviewer = ReviewerExpert()

    # 更新当前子任务状态
    subtasks = state.get("subtasks", [])
    for task in subtasks:
        if task.get("expert") == ExpertRole.REVIEWER.value and task.get("status") == "running":
            task["status"] = "completed"
            break

    result = reviewer.execute(state)

    # 标记下一个子任务
    next_found = False
    for task in subtasks:
        if task.get("status") in ("pending", None) and not next_found:
            task["status"] = "running"
            next_found = True

    result["subtasks"] = subtasks
    result["phase"] = ExecutionPhase.REVIEWING.value
    result["iteration"] = state.get("iteration", 0) + 1

    return result


# ================================================================
# 路由函数 — 条件边
# ================================================================

def route_after_supervisor(state: MultiAgentState) -> Literal["retriever", "analyst", "reviewer", "end"]:
    """
    主管决策后的路由

    根据 supervisor_decision 决定下一个节点。
    """
    decision = state.get("supervisor_decision", "continue")

    if decision == "finish" or decision not in ("continue", "retry"):
        return "end"

    # 根据当前 running 状态的子任务路由
    subtasks = state.get("subtasks", [])
    for task in subtasks:
        if task.get("status") == "running":
            expert = task.get("expert", "")
            if expert == ExpertRole.RETRIEVER.value:
                return "retriever"
            elif expert == ExpertRole.ANALYST.value:
                return "analyst"
            elif expert == ExpertRole.REVIEWER.value:
                return "reviewer"

    # 兜底：检查任务类型决定下一步
    task_type = state.get("task_type", "")
    if task_type in (TaskType.COMPARISON.value, TaskType.COMPLEX_RESEARCH.value, TaskType.LITERATURE_REVIEW.value):
        return "reviewer"
    elif task_type == TaskType.RETRIEVAL_QA.value:
        # 如果已有检索结果 → 分析，否则 → 检索
        if state.get("retriever_results"):
            return "analyst"
        return "retriever"
    else:
        return "analyst"

def route_after_expert(state: MultiAgentState) -> Literal["supervisor", "end"]:
    """
    专家执行后的路由

    检查终止条件：轮数超限 → end，否则 → supervisor
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 8)

    if iteration >= max_iter:
        print(f"[Workflow] 达到最大迭代次数 {max_iter}，强制结束")
        return "end"

    # 检查是否所有子任务都已完成
    subtasks = state.get("subtasks", [])
    all_done = all(t.get("status") == "completed" for t in subtasks) if subtasks else False
    if all_done:
        print(f"[Workflow] 所有子任务完成")
        return "supervisor"

    return "supervisor"


# ================================================================
# 构建工作流
# ================================================================

def build_multi_agent_workflow() -> StateGraph:
    """
    构建多智能体协作 LangGraph 工作流

    图结构:
        supervisor_plan → [路由] → retriever / analyst / reviewer
                                         ↓
                                    supervisor_plan (循环或结束)
    """
    workflow = StateGraph(MultiAgentState)

    # 注册节点
    workflow.add_node("supervisor_plan", supervisor_plan_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("reviewer", reviewer_node)

    # 入口
    workflow.set_entry_point("supervisor_plan")

    # 主管 → 路由到各专家
    workflow.add_conditional_edges(
        "supervisor_plan",
        route_after_supervisor,
        {
            "retriever": "retriever",
            "analyst": "analyst",
            "reviewer": "reviewer",
            "end": END,
        },
    )

    # 检索专家 → 回到主管
    workflow.add_conditional_edges(
        "retriever",
        route_after_expert,
        {
            "supervisor": "supervisor_plan",
            "end": END,
        },
    )

    # 分析专家 → 回到主管
    workflow.add_conditional_edges(
        "analyst",
        route_after_expert,
        {
            "supervisor": "supervisor_plan",
            "end": END,
        },
    )

    # 综述专家 → 回到主管
    workflow.add_conditional_edges(
        "reviewer",
        route_after_expert,
        {
            "supervisor": "supervisor_plan",
            "end": END,
        },
    )

    compiled = workflow.compile()
    print("[OK] 多智能体工作流构建完成")
    return compiled


# ================================================================
# 工作流执行器
# ================================================================

class MultiAgentOrchestrator:
    """多智能体编排器 — 对外暴露 execute() 和 execute_stream()"""

    def __init__(self):
        self.workflow = build_multi_agent_workflow()

    def execute(self, query: str) -> str:
        """同步执行多智能体协作"""
        initial_state = create_initial_state(query)

        try:
            final_state = self.workflow.invoke(initial_state)
            return final_state.get("final_answer", "") or self._fallback_answer(final_state)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"[MultiAgent ERROR] {type(e).__name__}: {str(e)}"

    def execute_stream(self, query: str):
        """流式执行多智能体协作"""
        initial_state = create_initial_state(query)

        full_text = ""
        try:
            # LangGraph 不支持真正的 token 级流式, 用状态流式模拟
            for event in self.workflow.stream(initial_state):
                for node_name, node_state in event.items():
                    # 检查是否有新产出
                    phase = node_state.get("phase", "")
                    if phase == ExecutionPhase.FINISHED.value:
                        final = node_state.get("final_answer", "")
                        if final:
                            full_text = final
                            yield final
                            return

            # 兜底：最终状态
            final = self._get_final_answer(full_text)
            if not full_text:
                yield final
        except Exception as e:
            yield f"[MultiAgent ERROR] {type(e).__name__}: {str(e)}"

    def _fallback_answer(self, state: MultiAgentState) -> str:
        """退路答案：从状态中拼接"""
        parts = []
        analyst = state.get("analyst_results", [])
        reviewer = state.get("reviewer_results", [])
        if analyst:
            parts.append("\n\n".join(analyst))
        if reviewer:
            parts.append("\n\n".join(reviewer))
        return "\n\n".join(parts) if parts else "多智能体分析完成，但未生成有效回答。"

    def _get_final_answer(self, current: str) -> str:
        """获取最终答案"""
        if current:
            return current
        return "多智能体分析完成。"


# ================================================================
# 全局实例
# ================================================================

_multi_agent: MultiAgentOrchestrator | None = None

def get_multi_agent() -> MultiAgentOrchestrator:
    global _multi_agent
    if _multi_agent is None:
        _multi_agent = MultiAgentOrchestrator()
    return _multi_agent
