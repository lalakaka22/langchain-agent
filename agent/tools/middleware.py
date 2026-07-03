# agent/tools/middleware.py
from typing import Callable, Any, Dict, Optional
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from utils.logger_handler import logger
import time
import functools


def monitor_tool(func):
    """
    监控工具执行时间的装饰器中间件

    Args:
        func: 被装饰的工具函数

    Returns:
        包装后的函数
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 获取工具名称
        tool_name = getattr(func, "__name__", "unknown_tool")

        # 记录开始时间
        start_time = time.time()
        logger.info(f"[tool monitor] 开始执行工具: {tool_name}")
        logger.info(f"[tool monitor] 参数: args={args}, kwargs={kwargs}")

        try:
            # 执行工具
            result = func(*args, **kwargs)

            # 计算执行时间
            elapsed_time = time.time() - start_time
            logger.info(f"[tool monitor] 工具 {tool_name} 执行成功，耗时: {elapsed_time:.2f}秒")

            # 如果是 fill_context_for_report，设置上下文标记
            if tool_name == "fill_context_for_report":
                # 这里可以通过全局变量或其他方式设置上下文
                logger.info("[tool monitor] 检测到报告生成工具调用")
                # 可以在线程本地存储中设置标志
                import threading
                threading.current_thread().report_mode = True

            return result

        except Exception as e:
            logger.error(f"[tool monitor] 工具 {tool_name} 执行失败: {str(e)}", exc_info=True)
            raise

    return wrapper


def log_before_model(model_func):
    """
    模型调用前记录日志的装饰器中间件

    Args:
        model_func: 模型调用函数

    Returns:
        包装后的函数
    """

    @functools.wraps(model_func)
    def wrapper(*args, **kwargs):
        logger.info("[log_before_model] 即将调用模型")

        # 尝试记录输入信息
        try:
            if args:
                input_data = args[0] if args else None
                if isinstance(input_data, dict):
                    messages = input_data.get("messages", [])
                    logger.info(f"[log_before_model] 输入消息数量: {len(messages)}")
                    if messages:
                        last_msg = messages[-1]
                        content = getattr(last_msg, "content", str(last_msg))
                        logger.debug(f"[log_before_model] 最后消息: {content[:200]}...")
                else:
                    logger.debug(f"[log_before_model] 输入类型: {type(input_data)}")
        except Exception as e:
            logger.debug(f"[log_before_model] 记录输入信息失败: {e}")

        # 执行模型调用
        result = model_func(*args, **kwargs)

        # 记录输出信息
        try:
            if hasattr(result, "content"):
                logger.info(f"[log_before_model] 模型返回内容长度: {len(result.content)}")
            elif isinstance(result, dict) and "output" in result:
                logger.info(f"[log_before_model] 模型返回输出长度: {len(result['output'])}")
        except Exception as e:
            logger.debug(f"[log_before_model] 记录输出信息失败: {e}")

        return result

    return wrapper


def report_prompt_switch(prompt_func):
    """
    根据上下文切换提示词的装饰器中间件

    Args:
        prompt_func: 提示词生成函数

    Returns:
        包装后的函数
    """

    @functools.wraps(prompt_func)
    def wrapper(*args, **kwargs):
        # 检查是否为报告模式
        is_report = False

        # 从线程本地存储获取状态
        import threading
        if hasattr(threading.current_thread(), 'report_mode'):
            is_report = threading.current_thread().report_mode

        # 也可以通过全局变量
        # is_report = getattr(globals(), '_report_mode', False)

        if is_report:
            logger.info("[report_prompt_switch] 切换到报告生成提示词")
            # 这里可以加载报告提示词
            try:
                from utils.prompt_loader import load_report_prompts
                return load_report_prompts()
            except Exception as e:
                logger.error(f"[report_prompt_switch] 加载报告提示词失败: {e}")
                return prompt_func(*args, **kwargs)
        else:
            logger.info("[report_prompt_switch] 使用默认系统提示词")
            return prompt_func(*args, **kwargs)

    return wrapper


# 装饰器类版本（如果需要更复杂的中间件）
class ToolMiddleware:
    """工具中间件类"""

    def __init__(self, tool: BaseTool):
        self.tool = tool
        self.start_time = None

    def before(self, *args, **kwargs):
        """执行前的钩子"""
        self.start_time = time.time()
        logger.info(f"[ToolMiddleware] 执行工具: {self.tool.name}")
        logger.info(f"[ToolMiddleware] 参数: {args}, {kwargs}")

    def after(self, result):
        """执行后的钩子"""
        elapsed = time.time() - self.start_time
        logger.info(f"[ToolMiddleware] 工具 {self.tool.name} 执行完成，耗时: {elapsed:.2f}秒")
        return result

    def on_error(self, error):
        """错误处理钩子"""
        logger.error(f"[ToolMiddleware] 工具 {self.tool.name} 执行失败: {error}")
        raise error

    def __call__(self, *args, **kwargs):
        """调用工具"""
        try:
            self.before(*args, **kwargs)
            result = self.tool._run(*args, **kwargs)
            return self.after(result)
        except Exception as e:
            return self.on_error(e)


# 简单的上下文管理器
class ReportContext:
    """报告上下文管理器"""

    def __enter__(self):
        """进入报告模式"""
        import threading
        threading.current_thread().report_mode = True
        logger.info("[ReportContext] 进入报告模式")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出报告模式"""
        import threading
        threading.current_thread().report_mode = False
        logger.info("[ReportContext] 退出报告模式")


# 工具包装函数
def wrap_tool(tool_func):
    """
    通用工具包装器

    Args:
        tool_func: 工具函数

    Returns:
        包装后的工具函数
    """

    @monitor_tool
    @functools.wraps(tool_func)
    def wrapped(*args, **kwargs):
        return tool_func(*args, **kwargs)

    return wrapped