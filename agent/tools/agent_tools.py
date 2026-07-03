# agent/tools/agent_tools.py
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional, Type
import datetime
import json


# 定义参数模型
class ServiceTimeParams(BaseModel):
    """服务和时间的参数模型"""
    service: str = Field(description="服务名称，如 order-service, payment-service")
    time_range: str = Field(description="时间范围，如 最近1小时, 今天, 本月")


class ServiceMetricParams(BaseModel):
    """服务、指标和时间的参数模型"""
    service: str = Field(description="服务名称，如 order-service, payment-service")
    metric: str = Field(description="指标名称，如 CPU, Memory, ResponseTime")
    time_range: str = Field(description="时间范围，如 最近1小时, 今天, 本月")


class ServiceParams(BaseModel):
    """服务参数模型"""
    service_name: str = Field(description="服务名称，如 order-service, payment-service")


class QueryParams(BaseModel):
    """查询参数模型"""
    query: str = Field(description="查询内容")


class ContextParams(BaseModel):
    """上下文参数模型"""
    context: str = Field(description="上下文内容")


class TimeRangeParams(BaseModel):
    """时间范围参数模型"""
    time_range: str = Field(description="时间范围，如 最近1小时, 今天, 本月")


# 定义工具函数
def rag_summarize_func(query: str) -> str:
    """使用RAG从知识库中检索并总结相关信息，返回回答及自适应K参考文件+动态重排相关性评分"""
    try:
        from rag.rag_service import RagSummarizeService
        rag_service = RagSummarizeService(recall_k=10, final_k=3)
        result = rag_service.rag_summarize_with_sources(query)

        answer = result.get("answer", "未能生成回复。")
        sources = result.get("sources", [])

        # 参考文件 section 已由 _build_reference_section 生成在 answer 末尾
        # 这里显式标注动态重排信息
        if sources:
            ref_lines = ["\n\n---", "## 参考文件（动态重排智能体 · 自适应K）", f"共 {len(sources)} 篇（动态决策），按 BGE-Reranker 相关性排序", ""]
            for s in sources:
                rank = s.get("rank", "?")
                score = s.get("score", 0)
                src = s.get("metadata", {}).get("source", "未知来源")
                page = s.get("metadata", {}).get("page", "")

                if score >= 5:
                    level = "★★★★★ 极强相关"
                elif score >= 2:
                    level = "★★★★ 高度相关"
                elif score >= 0:
                    level = "★★★ 中度相关"
                elif score >= -3:
                    level = "★★ 低度相关"
                else:
                    level = "★ 弱相关"

                ref_lines.append(f"**[{rank}] {src}**" + (f" (页码: {page})" if page else ""))
                ref_lines.append(f"    BGE-Reranker 相关性分数: {score:.3f}  |  {level}")
                ref_lines.append("")

            return answer + "\n".join(ref_lines)

        return answer
    except Exception as e:
        return f"RAG 总结失败: {str(e)}"


def get_target_service_func(service_name: str) -> str:
    """获取目标服务的详细信息"""
    services = {
        "order-service": "订单服务，处理订单创建、查询、更新等操作",
        "payment-service": "支付服务，处理支付请求、退款等操作",
        "inventory-service": "库存服务，管理商品库存",
        "user-service": "用户服务，管理用户信息和认证"
    }
    return services.get(service_name, f"未找到服务: {service_name}")


def get_time_range_func(time_range: str) -> str:
    """解析并返回时间范围"""
    now = datetime.datetime.now()

    if "1小时" in time_range:
        start = now - datetime.timedelta(hours=1)
        return f"从 {start.strftime('%Y-%m-%d %H:%M:%S')} 到 {now.strftime('%Y-%m-%d %H:%M:%S')}"
    elif "今天" in time_range:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return f"从 {start.strftime('%Y-%m-%d %H:%M:%S')} 到 {now.strftime('%Y-%m-%d %H:%M:%S')}"
    elif "本月" in time_range:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return f"从 {start.strftime('%Y-%m-%d %H:%M:%S')} 到 {now.strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        return f"时间范围: {time_range}"


def fetch_alert_data_func(service: str, time_range: str = "最近1小时") -> str:
    """获取告警数据"""
    alerts = [
        f"告警1: {service} CPU使用率超过90%",
        f"告警2: {service} 内存使用率超过85%",
        f"告警3: {service} 响应时间超过3秒",
        f"告警4: {service} 错误率超过5%",
    ]
    return f"服务 {service} 在 {time_range} 的告警数据:\n" + "\n".join(alerts)


def fetch_metric_data_func(service: str, metric: str = "CPU", time_range: str = "最近1小时") -> str:
    """获取监控指标数据"""
    metrics = {
        "CPU": "75% - 95%",
        "Memory": "70% - 88%",
        "ResponseTime": "200ms - 800ms",
        "ErrorRate": "0.5% - 2.3%",
        "Throughput": "1000 - 5000 req/s"
    }
    value = metrics.get(metric, "未知指标")
    return f"服务 {service} 的 {metric} 指标在 {time_range}: {value}"


def fetch_log_summary_func(service: str, time_range: str = "最近1小时") -> str:
    """获取日志摘要"""
    logs = [
        f"ERROR: {service} 连接数据库超时",
        f"WARN: {service} 缓存命中率较低",
        f"INFO: {service} 处理请求正常",
        f"ERROR: {service} 调用外部API失败",
        f"INFO: {service} 服务启动成功",
    ]
    return f"服务 {service} 在 {time_range} 的日志摘要:\n" + "\n".join(logs)


def fetch_service_topology_func(service_name: str) -> str:
    """获取服务的拓扑信息（依赖关系）"""
    topology = {
        "order-service": "依赖: user-service, payment-service, inventory-service",
        "payment-service": "依赖: user-service, 第三方支付网关",
        "inventory-service": "依赖: database-service, cache-service",
        "user-service": "依赖: database-service",
    }
    return topology.get(service_name, f"服务 {service_name} 的拓扑信息: 无依赖")


def fetch_report_data_func(service: str, time_range: str = "本月") -> str:
    """获取服务的运行报告数据"""
    return f"正在生成 {service} 在 {time_range} 的运行报告..."


def fill_context_for_report_func(context: str) -> str:
    """填充报告上下文信息"""
    return f"报告上下文已填充: {context}"


# 创建 StructuredTool
rag_summarize = StructuredTool.from_function(
    func=rag_summarize_func,
    name="rag_summarize",
    description="从知识库中检索相关信息并总结",
    args_schema=QueryParams,
)

get_target_service = StructuredTool.from_function(
    func=get_target_service_func,
    name="get_target_service",
    description="获取目标服务的详细信息",
    args_schema=ServiceParams,
)

get_time_range = StructuredTool.from_function(
    func=get_time_range_func,
    name="get_time_range",
    description="解析用户指定的时间范围",
    args_schema=TimeRangeParams,
)

fetch_alert_data = StructuredTool.from_function(
    func=fetch_alert_data_func,
    name="fetch_alert_data",
    description="获取指定服务的告警数据",
    args_schema=ServiceTimeParams,
)

fetch_metric_data = StructuredTool.from_function(
    func=fetch_metric_data_func,
    name="fetch_metric_data",
    description="获取指定服务的监控指标数据",
    args_schema=ServiceMetricParams,
)

fetch_log_summary = StructuredTool.from_function(
    func=fetch_log_summary_func,
    name="fetch_log_summary",
    description="获取指定服务的日志摘要",
    args_schema=ServiceTimeParams,
)

fetch_service_topology = StructuredTool.from_function(
    func=fetch_service_topology_func,
    name="fetch_service_topology",
    description="获取服务的拓扑和依赖关系",
    args_schema=ServiceParams,
)

fetch_report_data = StructuredTool.from_function(
    func=fetch_report_data_func,
    name="fetch_report_data",
    description="获取服务的运行报告数据",
    args_schema=ServiceTimeParams,
)

fill_context_for_report = StructuredTool.from_function(
    func=fill_context_for_report_func,
    name="fill_context_for_report",
    description="填充报告上下文信息",
    args_schema=ContextParams,
)