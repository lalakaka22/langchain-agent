# agent/react_agent.py
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from model.factory import chat_model
from agent.tools.agent_tools import (
    rag_summarize,
    get_target_service,
    get_time_range,
    fetch_alert_data,
    fetch_metric_data,
    fetch_log_summary,
    fetch_service_topology,
    fetch_report_data,
    fill_context_for_report,
)
from utils.prompt_loader import load_system_prompts
import time
from typing import Dict, Any


class ReactAgent:
    def __init__(self):
        """初始化 ReAct Agent"""

        # 准备工具列表
        self.tools = [
            rag_summarize,
            get_target_service,
            get_time_range,
            fetch_alert_data,
            fetch_metric_data,
            fetch_log_summary,
            fetch_service_topology,
            fetch_report_data,
            fill_context_for_report,
        ]

        # 加载系统提示
        system_prompt = load_system_prompts()

        # 创建 Agent（使用 LangChain 1.x 新 API）
        self.agent = create_agent(
            model=chat_model,
            tools=self.tools,
            system_prompt=system_prompt,
        )

        print(f"[OK] ReAct Agent 初始化成功，工具数量: {len(self.tools)}")

    def _extract_text(self, result: Dict[str, Any]) -> str:
        """提取文本结果"""
        if isinstance(result, dict):
            if "output" in result:
                return str(result["output"])

            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                content = getattr(last, "content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("text"):
                            texts.append(block["text"])
                    return "".join(texts)

        return str(result)

    def execute_stream(self, query: str):
        """流式执行"""
        try:
            print(f"📝 执行查询: {query[:50]}...")

            result = self.agent.invoke({
                "messages": [HumanMessage(content=query)],
            })

            final_text = self._extract_text(result)

            if not final_text:
                final_text = "抱歉，我无法处理这个请求，请稍后重试。"

            # 流式输出
            chunk_size = 15
            for i in range(0, len(final_text), chunk_size):
                yield final_text[i:i + chunk_size]
                time.sleep(0.02)

        except Exception as e:
            error_msg = f"[ERROR] 执行失败: {str(e)}"
            print(error_msg)
            yield error_msg

    def execute(self, query: str) -> str:
        """非流式执行"""
        try:
            result = self.agent.invoke({
                "messages": [HumanMessage(content=query)],
            })
            return self._extract_text(result)
        except Exception as e:
            return f"[ERROR] 执行失败: {str(e)}"


# 轻量级智能体 - 自定义工具调用循环，避免 ReAct 严格格式问题
class SimpleReactAgent:
    """自定义工具调用循环，适配 qwen3:4b 等小模型"""

    MAX_ITERATIONS = 10

    def __init__(self):
        self.tools = [
            rag_summarize,
            get_target_service,
            get_time_range,
            fetch_alert_data,
            fetch_metric_data,
            fetch_log_summary,
            fetch_service_topology,
            fetch_report_data,
            fill_context_for_report,
        ]

        # 构建工具映射
        self.tool_map = {tool.name: tool for tool in self.tools}
        system_prompt = load_system_prompts()

        # 工具描述
        tool_lines = []
        for tool in self.tools:
            desc = tool.description[:80] if tool.description else ""
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema
                try:
                    from pydantic import BaseModel
                    if issubclass(schema, BaseModel):
                        fields = schema.model_fields
                        arg_names = list(fields.keys())
                        tool_lines.append(f"- {tool.name}({', '.join(arg_names)}): {desc}")
                        continue
                except:
                    pass
            tool_lines.append(f"- {tool.name}: {desc}")
        tools_desc = "\n".join(tool_lines)
        tool_names = ", ".join(self.tool_map.keys())

        # 简化提示 - 去掉严格的 ReAct 格式约束
        self.base_prompt = f"""你是学术论文阅读和分析助手，通过检索论文知识库帮助研究者理解和对比学术论文。

{system_prompt}

## 可用工具 (只能用这些名称)
{tools_desc}

## 回答格式要求
当需要检索论文内容时，用以下格式调用工具：
```
Action: 工具名称
Action Input: {{"参数名": "参数值"}}
```
工具执行后你会收到检索结果，然后可以继续检索或给出最终分析。

简单的问题可以直接回答。
回答用中文，结构化、有条理，引用论文来源。

【关键规则】当你使用 rag_summarize 工具后，必须在 Final Answer 中完整包含检索结果里的 "## 参考文件（动态重排智能体 · 自适应K）" 部分，包括论文文件名、BGE-Reranker 相关性分数、星级评价和动态K决策信息，不得省略。"""

        print(f"[OK] Custom Agent 初始化成功，工具数量: {len(self.tools)}")

    def execute(self, query: str) -> str:
        result = self._run_agent_loop(query)
        if not result or not result.strip():
            return "分析超时，请尝试更具体地描述问题。"
        return result

    def execute_stream(self, query: str):
        result = self._run_agent_loop(query)
        if not result or not result.strip():
            yield "分析超时，请尝试更具体地描述问题。"
            return
        for i in range(0, len(result), 15):
            yield result[i:i + 15]
            time.sleep(0.02)

    def _run_agent_loop(self, query: str) -> str:
        """自定义工具调用循环 - 灵活解析模型输出"""
        import re
        import json

        messages = [
            SystemMessage(content=self.base_prompt),
            HumanMessage(content=f"用户问题：{query}\n\n请按格式调用工具或直接回答。")
        ]

        for i in range(self.MAX_ITERATIONS):
            try:
                response = chat_model.invoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                if not content:
                    content = ""
            except Exception as e:
                return f"[ERROR] 模型调用失败: {str(e)}"

            # 1) 检查是否包含 Final Answer
            fa_match = re.search(r'Final\s*Answer\s*[:：]\s*(.*)', content, re.DOTALL | re.IGNORECASE)
            if fa_match:
                return fa_match.group(1).strip()

            # 2) 检查是否包含 Action 调用
            action_match = re.search(
                r'Action\s*[:：]\s*(\S+)\s*\n\s*Action\s*Input\s*[:：]\s*(\{.*?\})\s*$',
                content, re.DOTALL | re.IGNORECASE
            )
            if not action_match:
                # 宽松匹配：允许参数写在同一行或任意位置
                action_match = re.search(
                    r'Action\s*[:：]\s*(\w+)\s*\n?\s*(?:Action\s*Input\s*[:：]\s*)?(\{.*?\})',
                    content, re.DOTALL
                )

            if action_match:
                tool_name = action_match.group(1).strip()
                args_str = action_match.group(2).strip()

                if tool_name not in self.tool_map:
                    messages.append(AIMessage(content=content))
                    messages.append(HumanMessage(content=f"工具 '{tool_name}' 不存在。可用工具：{', '.join(self.tool_map.keys())}。请用正确的工具名重试。"))
                    continue

                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    messages.append(AIMessage(content=content))
                    messages.append(HumanMessage(content=f"参数 JSON 格式错误: '{args_str}'。请修正为合法 JSON 后重试。"))
                    continue

                tool = self.tool_map[tool_name]
                try:
                    tool_result = tool.invoke(args)
                    result_str = str(tool_result)
                except Exception as e:
                    result_str = f"工具执行错误: {str(e)}"

                messages.append(AIMessage(content=content))
                messages.append(HumanMessage(content=f"Observation: {result_str}\n\n请继续分析，或给出 Final Answer。"))
                continue

            # 3) 不包含 Action 也不包含 Final Answer → 当做最终回答
            result = content.strip()
            if result:
                return result
            # 内容为空，让模型再试一次
            messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content="你的回复内容为空，请重新回答用户的问题。"))

        # 迭代上限 → 让模型基于上下文总结
        try:
            messages.append(HumanMessage(content="你已收集了足够信息。请基于以上所有对话内容，给出最终分析结论。"))
            final_response = chat_model.invoke(messages)
            content = final_response.content if hasattr(final_response, 'content') else str(final_response)
            return content.strip() if content and content.strip() else "分析超时，请简化问题重试。"
        except Exception:
            return "分析超时，请尝试更具体地描述问题。"


# 工厂函数
def create_agent(agent_type: str = "multi"):
    """
    创建 Agent 实例

    Args:
        agent_type: "react" | "simple" | "multi" (默认 multi, 多智能体协作)
    """
    if agent_type == "react":
        return ReactAgent()
    elif agent_type == "simple":
        return SimpleReactAgent()
    else:
        return MultiAgentWrapper()


# ================================================================
# 多智能体包装器 (向下兼容 execute/execute_stream 接口)
# ================================================================

class MultiAgentWrapper:
    """
    多智能体协作系统包装器

    保持与 SimpleReactAgent 相同的 execute() / execute_stream() 接口。
    内部使用 LangGraph MultiAgentOrchestrator 编排 4 个专家协作。

    专家角色:
      - Supervisor (主管): 意图识别、任务分解、调度分发、结果整合
      - Retriever  (检索): 搜索论文、提取关键内容
      - Analyst    (分析): RAG 问答、概念解释、数据提取
      - Reviewer   (综述): 文献对比、研究脉络、综述撰写

    终止条件:
      - 最大迭代轮数: 8
      - 主管判断答案质量达标
      - 所有子任务完成
    """

    MAX_ITERATIONS = 8

    def __init__(self):
        from agent.multi_agent.workflow import get_multi_agent
        self.orchestrator = get_multi_agent()
        print(f"[OK] 多智能体协作系统初始化成功，最大迭代: {self.MAX_ITERATIONS}")

    def execute(self, query: str) -> str:
        """非流式执行"""
        try:
            return self.orchestrator.execute(query)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"[MultiAgent ERROR] {type(e).__name__}: {str(e)}"

    def execute_stream(self, query: str):
        """流式执行"""
        try:
            yield from self.orchestrator.execute_stream(query)
        except Exception as e:
            yield f"[MultiAgent ERROR] {type(e).__name__}: {str(e)}"


if __name__ == '__main__':
    agent = MultiAgentWrapper()
    result = agent.execute("Transformer 的自注意力机制是什么？")
    print(result)