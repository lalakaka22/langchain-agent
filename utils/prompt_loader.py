# utils/prompt_loader.py
import os


def load_system_prompts() -> str:
    """
    加载系统提示

    Returns:
        系统提示字符串
    """
    default_prompt = """你是一位学术论文阅读与分析助手，帮助研究者高效理解和交叉对比学术论文。

核心能力：
1. 用 RAG 工具检索论文知识库中的相关内容
2. 解释核心概念、方法论和创新点
3. 对比不同论文的观点、方法和结果
4. 梳理研究脉络和演进趋势

回答风格：
- 学术严谨但易于理解，结构化呈现
- 引用具体论文作为来源
- 专业术语保留英文原名
- 指出论文间的关联与差异"""

    # 尝试从文件加载
    prompt_file = os.getenv("SYSTEM_PROMPT_FILE", "prompts/system_prompt.txt")
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[WARN] 加载系统提示文件失败: {e}")

    return default_prompt


def load_rag_prompts() -> str:
    """
    加载 RAG 总结提示

    Returns:
        RAG 提示字符串
    """
    default_prompt = """根据以下论文内容回答问题。

参考资料：
{context}

用户问题：{input}

请基于参考资料给出回答："""

    # 从配置文件读取提示文件路径
    try:
        from utils.config_handler import load_prompts_config
        prompts_conf = load_prompts_config()
        rag_prompt_path = prompts_conf.get("rag_summarize_prompt_path", "prompts/rag_summarize.txt")
    except Exception:
        rag_prompt_path = os.getenv("RAG_PROMPT_FILE", "prompts/rag_summarize.txt")

    if os.path.exists(rag_prompt_path):
        try:
            with open(rag_prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[WARN] 加载 RAG 提示文件失败: {e}")

    return default_prompt


def load_report_prompts() -> str:
    """
    加载报告生成提示

    Returns:
        报告提示字符串
    """
    default_prompt = """你是一个专业的运维报告生成助手，请根据收集到的数据生成详细的运行报告。

报告应包含：
1. 报告摘要：概述整体情况
2. 详细数据：展示关键指标
3. 问题分析：识别存在的问题
4. 改进建议：提出具体的改进方案

请生成专业、清晰的运维报告。"""

    report_file = os.getenv("REPORT_PROMPT_FILE", "prompts/report_prompt.txt")
    if os.path.exists(report_file):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[WARN] 加载报告提示文件失败: {e}")

    return default_prompt