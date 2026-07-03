# rag/rag_service.py
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复

检索管道（v3）：向量召回 Top10 → 动态重排智能体（自适应K） → 送入 LLM

新增: 动态重排智能体（DynamicRerankerAgent）根据查询复杂度 + 分数分布
      动态决定最终送入 LLM 的文档数量 K，避免固定 K 的静态缺陷。
"""
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag.vector_store import VectorStoreService, get_vector_store
from rag.data_loader import init_vector_store as load_knowledge_base
from rag.dynamic_reranker import get_dynamic_reranker, DynamicRerankerAgent, DREpisode
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model


class RagSummarizeService:
    """RAG 总结服务类

    检索管道: 向量召回 Top10 → 动态重排智能体（自适应K） → 送入 LLM
    """

    def __init__(self, recall_k: int = 10, final_k: int = 3):
        """
        初始化 RAG 总结服务

        Args:
            recall_k: 向量召回候选文档数量 (默认 10, 动态重排需要更多候选)
            final_k: 固定K模式下的兜底值 (动态重排时忽略此参数)
        """
        self.recall_k = recall_k
        self.final_k = final_k

        # 动态重排智能体（替代固定K重排序器）
        self.dynamic_reranker = get_dynamic_reranker()

        # 使用单例向量存储，避免重复加载文档
        self.vector_store = get_vector_store()

        # 自动加载知识库数据到向量存储（首次加载，后续跳过）
        print("[INFO] 正在加载知识库数据到 RAG 系统...")
        from rag.data_loader import init_vector_store as load_knowledge_base
        load_knowledge_base(self.vector_store)
        print(f"[OK] 知识库加载完成，共 {self.vector_store.get_document_count()} 个文档块")

        # 获取检索器
        self.retriever = self._get_retriever()

        # 加载提示模板
        self.prompt_text = self._load_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)

        # 初始化模型
        self.model = chat_model

        # 最近一次决策记录（用于RL反馈）
        self.last_episode: Optional[DREpisode] = None

        # 初始化链
        self.chain = self._init_chain()

    def _load_prompt(self) -> str:
        """加载提示模板"""
        try:
            return load_rag_prompts()
        except Exception as e:
            print(f"[WARN] 加载提示模板失败: {e}，使用默认模板")
            return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """获取默认提示模板"""
        return """你是一个专业的智能助手，请根据提供的参考资料来回答用户的问题。

【任务要求】
1. 基于参考资料中的信息进行回答
2. 如果参考资料中没有相关信息，请明确告知用户
3. 回答要准确、简洁、有条理
4. 在回答中引用具体的参考资料编号

【参考资料】
{context}

【用户问题】
{input}

【回答】
请基于以上参考资料，给出详细的回答："""

    def _get_retriever(self):
        """获取配置好的检索器（用于向量召回 step）"""
        return self.vector_store.get_retriever(
            search_kwargs={"k": self.recall_k}
        )

    def _retrieve_with_rerank(self, query: str) -> List[Document]:
        """
        两阶段检索: 向量召回 TopN → 动态重排智能体（自适应K）

        动态重排智能体会根据查询复杂度 + BGE-Reranker 分数分布
        自动决定最终送入 LLM 的文档数量 K。

        Args:
            query: 用户查询

        Returns:
            动态重排序后的 Top-K 文档（含 rerank_score 和 rerank_rank）
        """
        # Step 1: 向量快速召回
        candidate_docs = self.retriever.invoke(query)
        if not candidate_docs:
            return []

        # Step 2: 动态重排智能体（自适应K）
        docs, episode = self.dynamic_reranker.rerank_with_dynamic_k(
            query=query,
            documents=candidate_docs,
        )
        self.last_episode = episode

        return docs

    def _format_docs(self, docs: List[Document]) -> str:
        """格式化文档为上下文（含重排序相关性分数）"""
        if not docs:
            return "未找到相关参考资料。"

        context_parts = []
        for idx, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            score = doc.metadata.get("rerank_score", None)

            ref_info = f"[参考资料{idx}]"
            if score is not None:
                ref_info += f" (相关性: {score:.3f})"
            if page:
                ref_info += f" 来源: {source}, 页码: {page}"
            else:
                ref_info += f" 来源: {source}"

            context_parts.append(
                f"{ref_info}\n{doc.page_content}\n"
            )

        return "\n---\n".join(context_parts)

    def _init_chain(self):
        """初始化处理链（保留兼容）"""
        def format_docs(docs: List[Document]) -> str:
            return self._format_docs(docs)

        # 使用 LCEL 构建链（使用重排序检索）
        chain = (
            {
                "context": lambda x: format_docs(self._retrieve_with_rerank(x)),
                "input": RunnablePassthrough()
            }
            | self.prompt_template
            | self.model
            | StrOutputParser()
        )

        return chain

    def retriever_docs(self, query: str) -> List[Document]:
        """两阶段检索: 向量召回 + 重排序"""
        return self._retrieve_with_rerank(query)

    def add_documents(self, documents: List[Document]):
        """添加文档到向量存储"""
        self.vector_store.add_documents(documents)

    def rag_summarize(self, query: str) -> str:
        """
        执行 RAG 总结（含重排序管道）

        Args:
            query: 用户查询

        Returns:
            总结后的回复（末尾附参考文献+相关性参数）
        """
        try:
            print(f"[INFO] 处理查询: {query[:50]}...")
            # 两阶段检索: 向量召回 + 重排序
            docs = self._retrieve_with_rerank(query)
            # 格式化上下文
            context = self._format_docs(docs)
            # 构建提示
            prompt_text = self.prompt_text.format(context=context, input=query)
            # 直接调用模型
            from langchain_core.messages import HumanMessage
            response = self.model.invoke([HumanMessage(content=prompt_text)])
            answer = response.content if response.content else "未能生成回复，请重试。"

            # 附加参考文献+相关性参数
            ref_section = self._build_reference_section(docs)
            return answer + ref_section
        except Exception as e:
            error_msg = f"RAG 总结失败: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg

    def _build_reference_section(self, docs: List[Document]) -> str:
        """构建参考文献+动态重排相关性参数 section"""
        if not docs:
            return ""

        # 获取动态K决策信息
        k_info = ""
        if self.last_episode:
            k_info = (
                f"*动态重排智能体决策: 自适应 K={self.last_episode.chosen_k} "
                f"(复杂度={self.last_episode.query_profile.complexity_score:.2f}, "
                f"原因: {self.last_episode.decision_reason})*\n"
            )

        lines = ["\n\n---", "## 参考文献（动态重排智能体 · 自适应K）", "", k_info]

        for doc in docs:
            rank = doc.metadata.get("rerank_rank", "?")
            score = doc.metadata.get("rerank_score", 0)
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            content_preview = doc.page_content[:120].replace("\n", " ") + "..."

            # 相关性评级
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

            lines.append(f"**[{rank}] {source}**" + (f" (页码: {page})" if page else ""))
            lines.append(f"    BGE-Reranker 相关性分数: {score:.3f}  |  {level}")
            lines.append(f"    > {content_preview}")
            lines.append("")

        return "\n".join(lines)

    def rag_summarize_with_sources(self, query: str) -> Dict[str, Any]:
        """
        执行 RAG 总结并返回来源（含重排序管道+相关性分数）

        Args:
            query: 用户查询

        Returns:
            包含总结、来源和相关性分数的字典
        """
        try:
            # 两阶段检索: 向量召回 + 重排序
            docs = self._retrieve_with_rerank(query)

            # 生成总结
            context = self._format_docs(docs)
            prompt_text = self.prompt_text.format(context=context, input=query)
            from langchain_core.messages import HumanMessage
            response = self.model.invoke([HumanMessage(content=prompt_text)])
            answer = response.content if response.content else "未能生成回复。"

            # 附加参考文献 section
            answer += self._build_reference_section(docs)

            # 提取来源信息（含相关性分数）
            sources = []
            for doc in docs:
                sources.append({
                    "rank": doc.metadata.get("rerank_rank", "?"),
                    "score": doc.metadata.get("rerank_score", 0),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": {
                        "source": doc.metadata.get("source", "未知来源"),
                        "page": doc.metadata.get("page", ""),
                    }
                })

            return {
                "answer": answer,
                "sources": sources,
                "source_count": len(sources)
            }
        except Exception as e:
            return {
                "answer": f"RAG 总结失败: {str(e)}",
                "sources": [],
                "source_count": 0
            }


if __name__ == '__main__':
    print("测试 RAG 总结服务 (含重排序)\n")

    rag = RagSummarizeService(recall_k=5, final_k=3)

    query = "Transformer 的自注意力机制是什么"
    print(f"问题: {query}\n")

    result = rag.rag_summarize(query)
    print("总结结果:")
    print("-" * 50)
    print(result)
    print("-" * 50)