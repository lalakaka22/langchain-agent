"""
检索专家 (Retriever Expert)

职责:
  - 根据搜索关键词从知识库检索相关论文
  - 提取关键段落的文本和图片OCR内容
  - 返回检索到的文献摘要和来源信息
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document

from agent.multi_agent.state import MultiAgentState, ExpertRole
from rag.vector_store import get_vector_store
from rag.dynamic_reranker import get_dynamic_reranker


class RetrieverExpert:
    """检索专家 — 负责从知识库搜索论文"""

    def __init__(self):
        self.vector_store = get_vector_store()
        self.reranker = get_dynamic_reranker()

    def execute(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        执行检索任务

        从共享状态中读取 search_queries，执行多轮检索。
        """
        queries = state.get("search_queries", [])
        user_query = state.get("user_query", "")

        # 如果没有指定搜索词，用原始问题搜索
        if not queries:
            queries = [user_query]

        all_docs: List[Document] = []
        seen_content = set()

        for query in queries:
            # 向量检索
            retriever = self.vector_store.get_retriever(search_kwargs={"k": 10})
            candidates = retriever.invoke(query)

            # 动态重排
            if candidates:
                docs, episode = self.reranker.rerank_with_dynamic_k(query, candidates)
                for doc in docs:
                    key = (doc.metadata.get("source", ""), doc.page_content[:100])
                    if key not in seen_content:
                        seen_content.add(key)
                        all_docs.append(doc)

        if not all_docs:
            return {
                "retriever_results": [f"[检索专家] 未找到关于 '{user_query}' 的相关论文。"],
                "retrieved_documents": [],
            }

        # 格式化检索结果
        formatted = self._format_results(all_docs)
        doc_summaries = self._summarize_docs(all_docs)

        log_entry = {
            "expert": ExpertRole.RETRIEVER.value,
            "action": "检索完成",
            "queries": queries,
            "docs_found": len(all_docs),
            "detail": formatted[:200] + "..." if len(formatted) > 200 else formatted,
        }

        return {
            "retriever_results": [formatted],
            "retrieved_documents": doc_summaries,
            "execution_log": [log_entry],
        }

    def _format_results(self, docs: List[Document]) -> str:
        """格式化检索结果"""
        lines = ["## 检索结果\n"]
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "")
            score = doc.metadata.get("rerank_score", 0)
            content_type = doc.metadata.get("content_type", "text")
            section = doc.metadata.get("section", "")
            content = doc.page_content[:200].replace("\n", " ")

            type_label = "[图片OCR]" if content_type == "image_text" else ""
            lines.append(
                f"**[{i}] {source}** {type_label}"
                + (f" (p.{page})" if page else "")
                + (f" [{section}]" if section else "")
                + f" — 相关性: {score:.3f}"
            )
            lines.append(f"  > {content}...")
            lines.append("")

        return "\n".join(lines)

    def _summarize_docs(self, docs: List[Document]) -> List[Dict]:
        """生成文档摘要列表"""
        return [
            {
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page", ""),
                "score": doc.metadata.get("rerank_score", 0),
                "content_type": doc.metadata.get("content_type", "text"),
                "preview": doc.page_content[:150],
            }
            for doc in docs
        ]
