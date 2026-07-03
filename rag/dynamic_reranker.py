"""
动态重排智能体（Dynamic Reranker Agent）—— 动态决定 K 值

基于 DynamicRAG 框架思路，将重排序器建模为智能体，根据查询特点动态决定
重排序后应保留的文档数量（K 值），而非使用固定的 Top-K。

核心策略:
1. 查询复杂度分析: 评估问题的复杂度、领域特异性、歧义程度
2. 分数分布分析: 基于 BGE-Reranker 分数分布动态确定截断阈值
3. 自适应 K 值: 简单问题少取 (K=1~2)，复杂问题多取 (K=4~5)
4. 分数缺口检测: 在显著分数落差处自然截断
5. RL 反馈追踪: 记录每次决策与输出质量，支持持续优化
"""
import re
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from langchain_core.documents import Document
from rag.reranker import get_reranker


@dataclass
class QueryProfile:
    """查询画像 —— 描述查询的多维特征"""
    raw_query: str
    length: int                          # 查询长度（字符数）
    word_count: int                      # 词数
    has_comparison: bool                 # 含比较/对比意图
    has_causal: bool                     # 含因果推理意图
    has_definition: bool                 # 含定义/解释意图
    has_multi_part: bool                 # 含多子问题
    domain_specificity: float            # 领域特异性 [0, 1]
    complexity_score: float              # 综合复杂度 [0, 1]

    # 静态预计算词表
    COMPARISON_KEYWORDS: Tuple = (
        "compare", "comparison", "difference", "diff", "vs", "versus",
        "区别", "对比", "比较", "不同", "差异", "相比", "哪个更好", "优缺点",
    )
    CAUSAL_KEYWORDS: Tuple = (
        "why", "because", "cause", "reason", "impact", "effect",
        "为什么", "原因", "导致", "影响", "如何", "怎么",
    )
    DEFINITION_KEYWORDS: Tuple = (
        "what is", "define", "definition", "meaning", "explain",
        "是什么", "定义", "解释", "含义", "概念",
    )
    DOMAIN_KEYWORDS: Tuple = (
        "transformer", "attention", "bert", "gpt", "embedding", "lstm",
        "neural", "deep learning", "reinforcement", "gradient", "optimization",
        "卷积", "循环神经网络", "生成对抗", "自监督", "对比学习",
        "cross-modal", "multimodal", "多模态", "recommendation", "推荐",
        "神经网络", "注意力", "预训练", "微调", "fine-tune",
    )

    @classmethod
    def from_query(cls, query: str) -> "QueryProfile":
        """从原始查询构建查询画像"""
        q_lower = query.lower()
        word_count = len(query.split())
        length = len(query)

        has_comparison = any(kw in q_lower for kw in cls.COMPARISON_KEYWORDS)
        has_causal = any(kw in q_lower for kw in cls.CAUSAL_KEYWORDS)
        has_definition = any(kw in q_lower for kw in cls.DEFINITION_KEYWORDS)

        # 多子问题: 检查多个问号或显式编号
        has_multi_part = (
            query.count("?") >= 2
            or query.count("？") >= 2
            or bool(re.search(r"(\d+[\.\、\)）])\s*\w", query))
            or any(kw in q_lower for kw in ["first", "second", "firstly", "also", "additionally", "同时", "另外", "此外", "第一", "第二", "首先", "其次"])
        )

        # 领域特异性: 领域关键词命中率
        domain_hits = sum(1 for kw in cls.DOMAIN_KEYWORDS if kw in q_lower)
        domain_specificity = min(domain_hits / max(word_count * 0.3, 1), 1.0)

        # 综合复杂度
        complexity = cls._compute_complexity(
            length, word_count, has_comparison, has_causal,
            has_multi_part, domain_specificity
        )

        return cls(
            raw_query=query,
            length=length,
            word_count=word_count,
            has_comparison=has_comparison,
            has_causal=has_causal,
            has_definition=has_definition,
            has_multi_part=has_multi_part,
            domain_specificity=round(domain_specificity, 3),
            complexity_score=round(complexity, 3),
        )

    @staticmethod
    def _compute_complexity(
        length: int, word_count: int,
        has_comparison: bool, has_causal: bool,
        has_multi_part: bool, domain_specificity: float,
    ) -> float:
        """综合计算查询复杂度 [0, 1]"""
        # 长度因子: 短查询简单, 长查询复杂
        length_factor = min(length / 300, 1.0) * 0.15

        # 词数因子
        word_factor = min(word_count / 40, 1.0) * 0.10

        # 意图因子
        intent_score = 0.0
        if has_comparison:
            intent_score += 0.15
        if has_causal:
            intent_score += 0.10
        if has_multi_part:
            intent_score += 0.20

        # 领域因子: 领域性越高, 可能需要更多文献支撑
        domain_factor = domain_specificity * 0.15

        # 综合
        total = length_factor + word_factor + intent_score + domain_factor
        return min(total, 1.0)


@dataclass
class DREpisode:
    """动态重排决策记录 —— 用于 RL 反馈"""
    query: str
    query_profile: QueryProfile
    candidate_count: int
    all_scores: List[float]
    chosen_k: int
    score_threshold: float
    decision_reason: str
    # 可选反馈
    output_quality_score: Optional[float] = None


class DynamicRerankerAgent:
    """动态重排智能体（Dynamic Reranker Agent）

    根据查询特点 + BGE-Reranker 分数分布，动态决定最终送入 LLM 的 Top-K。

    决策流程:
    1. 分析查询画像 → 获取查询复杂度
    2. 对候选文档进行 BGE-Reranker 评分
    3. 基于分数分布 + 复杂度 → 动态确定 K 值
    4. 返回重排序后的 Top-K 文档
    """

    MIN_K = 1
    MAX_K = 8
    DEFAULT_RECALL_K = 10  # 向量召回数量

    def __init__(self):
        self.reranker = get_reranker()
        self.episodes: List[DREpisode] = []  # RL 经验回放缓存

    def rerank_with_dynamic_k(
        self,
        query: str,
        documents: List[Document],
        max_candidates: Optional[int] = None,
    ) -> Tuple[List[Document], DREpisode]:
        """
        动态重排序: 分析查询 + 分数分布 → 自适应 K → Top-K 文档

        Args:
            query: 用户查询
            documents: 向量召回的候选文档
            max_candidates: 最大重排文档数 (None=全部)

        Returns:
            (重排序后的动态 Top-K 文档, 决策记录)
        """
        if not documents:
            return [], self._empty_episode(query, documents)

        # Step 0: 查询画像分析
        profile = QueryProfile.from_query(query)
        print(f"[DynamicReranker] 查询画像: 复杂度={profile.complexity_score:.2f}, "
              f"对比={profile.has_comparison}, 因果={profile.has_causal}, "
              f"多子问题={profile.has_multi_part}, 领域={profile.domain_specificity:.2f}")

        # Step 1: BGE-Reranker 评分（对所有候选）
        docs_for_score = documents[:max_candidates] if max_candidates else documents
        scored = self.reranker.rerank_with_scores(query, docs_for_score, top_k=len(docs_for_score))
        # scored: List[Tuple[float, Document]], 已按分数降序

        all_scores = [s for s, _ in scored]

        if not all_scores:
            return [], self._empty_episode(query, documents)

        # Step 2: 动态决定 K 值
        chosen_k, threshold, reason = self._decide_k(profile, all_scores, scored)

        # Step 3: 取 Top-K 文档
        top_docs = [doc for _, doc in scored[:chosen_k]]

        # Step 4: 附上 metadata
        for rank, (score, doc) in enumerate(scored[:chosen_k], 1):
            doc.metadata["rerank_score"] = round(score, 3)
            doc.metadata["rerank_rank"] = rank

        # 记录决策
        episode = DREpisode(
            query=query,
            query_profile=profile,
            candidate_count=len(docs_for_score),
            all_scores=all_scores,
            chosen_k=chosen_k,
            score_threshold=threshold,
            decision_reason=reason,
        )
        self.episodes.append(episode)

        print(f"[DynamicReranker] 决策: K={chosen_k} (阈值={threshold:.3f}) | {reason}")
        print(f"[DynamicReranker] 分数分布: {[f'{s:.3f}' for s in all_scores[:chosen_k+2]]}")

        return top_docs, episode

    def _decide_k(
        self,
        profile: QueryProfile,
        all_scores: List[float],
        scored_docs: List[Tuple[float, Document]],
    ) -> Tuple[int, float, str]:
        """
        核心决策: 基于查询复杂度 + 分数分布 → 动态 K

        策略:
        1. 复杂度决定 K 的取值范围
        2. 分数缺口检测: 在显著落差处截断
        3. 分数阈值过滤: 低于相对阈值的不取
        4. 最终 K = min(复杂度K, 缺口K, 阈值K)
        """
        max_score = all_scores[0]
        n = len(all_scores)

        # ---- 策略 1: 复杂度决定基准 K ----
        base_k = self._complexity_to_base_k(profile)

        # ---- 策略 2: 分数缺口检测 ----
        # 计算相邻分数落差, 在 gap > 均值的 2.5 倍处截断
        gap_k = n
        if len(all_scores) >= 2:
            gaps = [all_scores[i] - all_scores[i + 1] for i in range(len(all_scores) - 1)]
            avg_gap = sum(gaps) / len(gaps) if gaps else 0
            # 动态阈值: 复杂度低时更敏感 (更早截断)
            gap_multiplier = 2.0 - profile.complexity_score * 0.8
            gap_threshold = avg_gap * gap_multiplier
            gap_threshold = max(gap_threshold, 0.5)  # 最小阈值，避免过于敏感

            for i, gap in enumerate(gaps):
                if i >= 1 and gap > gap_threshold:  # 至少保留第一个文档
                    gap_k = i + 1
                    break

        # ---- 策略 3: 分数阈值过滤 ----
        # 基于分数范围动态计算阈值（normalize 到 [0, 1] 范围后再截断）
        min_score = all_scores[-1]
        score_range = max_score - min_score

        if score_range > 0:
            # 归一化后根据复杂度决定保留比例
            # 复杂度越高保留越多: low=top 30%, high=top 60%
            keep_ratio = 0.30 + profile.complexity_score * 0.30
            normalized_threshold = 1.0 - keep_ratio

            # 反归一化得到实际阈值
            score_threshold = min_score + score_range * (1.0 - keep_ratio)
            # 确保阈值不高于最高分
            score_threshold = min(score_threshold, max_score)
        else:
            # 分数都一样，取全部
            score_threshold = min_score - 1.0
            threshold_k = n
            return base_k, score_threshold, f"复杂度K={base_k} | 分数均匀, 保留全部"

        threshold_k = 0
        for s in all_scores:
            if s >= score_threshold:
                threshold_k += 1
            else:
                break

        # ---- 最终决策 ----
        # 取三种策略的最小 K, 确保在 [MIN_K, MAX_K] 范围内
        raw_k = min(base_k, gap_k, threshold_k)
        chosen_k = max(self.MIN_K, min(raw_k, self.MAX_K))

        # 构建决策原因
        reasons = [f"复杂度K={base_k}"]
        if gap_k < n:
            reasons.append(f"缺口截断K={gap_k}")
        if threshold_k < n:
            reasons.append(f"阈值过滤K={threshold_k}")

        return chosen_k, score_threshold, " | ".join(reasons)

    def _complexity_to_base_k(self, profile: QueryProfile) -> int:
        """将查询复杂度映射为基础 K 值"""
        c = profile.complexity_score

        if c < 0.15:
            k = 1  # 极简单: 定义类, 一句话回答
        elif c < 0.30:
            k = 2  # 简单: 单一概念解释
        elif c < 0.50:
            k = 3  # 中等: 标准问题
        elif c < 0.70:
            k = 4  # 较复杂: 含对比或多方面
        else:
            k = 5  # 复杂: 多子问题, 深度推理

        # 额外调节
        if profile.has_comparison:
            k = max(k, 3)  # 对比至少需要2-3个来源
        if profile.has_multi_part:
            k = max(k, 4)  # 多子问题至少需要4个
        if profile.has_causal and profile.domain_specificity > 0.3:
            k += 1  # 领域因果推理要多一些参考

        return min(k, self.MAX_K)

    def _empty_episode(self, query: str, documents: List[Document]) -> DREpisode:
        """空结果决策记录"""
        return DREpisode(
            query=query,
            query_profile=QueryProfile.from_query(query),
            candidate_count=len(documents),
            all_scores=[],
            chosen_k=0,
            score_threshold=0.0,
            decision_reason="空候选集",
        )

    def feedback(self, episode: DREpisode, quality_score: float):
        """
        RL 反馈: 记录本次决策的输出质量

        Args:
            episode: 之前的决策记录
            quality_score: 输出质量评分 [0, 1]
        """
        episode.output_quality_score = quality_score
        # 如果是低质量且取了过多文档, 可以降低后续的 base_k
        if quality_score < 0.4 and episode.chosen_k > 3:
            print(f"[DynamicReranker RL] 低质量反馈 (K={episode.chosen_k}, score={quality_score:.2f})")
            print(f"  → 提示: 过多文档可能引入噪声, 建议降低 K")

    def get_statistics(self) -> Dict[str, Any]:
        """获取决策统计信息"""
        if not self.episodes:
            return {"total_episodes": 0}

        k_values = [e.chosen_k for e in self.episodes]
        complexities = [e.query_profile.complexity_score for e in self.episodes]
        quality_scores = [e.output_quality_score for e in self.episodes if e.output_quality_score is not None]

        return {
            "total_episodes": len(self.episodes),
            "avg_k": sum(k_values) / len(k_values),
            "k_distribution": {
                "min": min(k_values),
                "max": max(k_values),
                "mode": max(set(k_values), key=k_values.count),
            },
            "avg_complexity": sum(complexities) / len(complexities),
            "avg_quality": sum(quality_scores) / len(quality_scores) if quality_scores else None,
        }


# 全局单例
_dynamic_reranker: Optional[DynamicRerankerAgent] = None


def get_dynamic_reranker() -> DynamicRerankerAgent:
    """获取动态重排智能体单例"""
    global _dynamic_reranker
    if _dynamic_reranker is None:
        _dynamic_reranker = DynamicRerankerAgent()
    return _dynamic_reranker


if __name__ == "__main__":
    print("=== 动态重排智能体测试 ===\n")

    agent = DynamicRerankerAgent()

    # 模拟候选文档
    from langchain_core.documents import Document
    docs = [
        Document(page_content="Self-attention computes attention weights across all positions.", metadata={"source": "Transformer.pdf"}),
        Document(page_content="Multi-head attention allows attending to different representation subspaces.", metadata={"source": "Transformer.pdf"}),
        Document(page_content="LSTM uses gates to control information flow in sequential data.", metadata={"source": "lstm.pdf"}),
        Document(page_content="Word2Vec learns word embeddings via skip-gram and CBOW.", metadata={"source": "Word2Vec.pdf"}),
        Document(page_content="GloVe combines global matrix factorization with local context windows.", metadata={"source": "GloVe.pdf"}),
        Document(page_content="BERT uses masked language modeling for bidirectional pretraining.", metadata={"source": "BERT.pdf"}),
        Document(page_content="The pix2pix model uses conditional GANs for image-to-image translation.", metadata={"source": "pix2pix.pdf"}),
        Document(page_content="Deep Q-Networks combine Q-learning with deep neural networks.", metadata={"source": "DQN.pdf"}),
        Document(page_content="Multimodal recommenders fuse visual and textual features.", metadata={"source": "RecSys.pdf"}),
        Document(page_content="TRPO constrains policy updates with trust regions.", metadata={"source": "TRPO.pdf"}),
    ]

    test_queries = [
        "什么是 Self-Attention？",                                    # 简单定义
        "Transformer 的注意力机制是如何工作的？",                      # 中等
        "比较 BERT 和 GPT 的预训练方法有什么区别？",                  # 对比
        "深度学习中的注意力机制有哪些类型？各自的优缺点是什么？",       # 复杂多子问题
        "为什么 transformer 比 LSTM 更适合处理长序列？请详细解释原因", # 因果推理
    ]

    for q in test_queries:
        print(f"\n查询: {q}")
        print("-" * 50)
        results, episode = agent.rerank_with_dynamic_k(q, docs)
        print(f"  → K={episode.chosen_k}, 原因: {episode.decision_reason}")
        for i, doc in enumerate(results, 1):
            score = doc.metadata.get("rerank_score", "?")
            print(f"    [{i}] score={score:.3f} | {doc.metadata['source']}")
        print()

    print("\n=== 统计 ===")
    print(agent.get_statistics())
