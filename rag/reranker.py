# rag/reranker.py
"""
交叉编码器重排序模块
BGE-Reranker-base 使用 XLMRoBERTa 架构进行精准重排序
流水线: 向量召回 Top5 → CrossEncoder 重排序 → 取 Top3
"""
import os
import json
from typing import List, Tuple
import torch
from langchain_core.documents import Document

# 模型缓存路径
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bge-reranker-base")


def _download_and_patch_model():
    """下载 BGE-Reranker 模型并修复 config.json（补充 model_type）"""
    if not os.path.exists(_MODEL_DIR):
        os.makedirs(_MODEL_DIR, exist_ok=True)

    config_path = os.path.join(_MODEL_DIR, "config.json")

    # 使用 huggingface_hub 下载全部文件
    from huggingface_hub import snapshot_download
    snapshot_download(
        "BAAI/bge-reranker-base",
        local_dir=_MODEL_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.onnx", "onnx/*", "*.msgpack", "*.h5"],
    )

    # 修补 config.json: 添加 model_type（transformers 4.x 要求）
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "model_type" not in cfg:
        # bge-reranker-base 基于 XLMRoBERTa
        cfg["model_type"] = "xlm-roberta"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 已修补 config.json: model_type=xlm-roberta")


def _patch_config(model_dir: str):
    """修补 config.json: 添加 model_type 字段（transformers 4.x 要求）"""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "model_type" not in cfg:
        cfg["model_type"] = "xlm-roberta"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 已修补 config.json: model_type=xlm-roberta")


class RerankerService:
    """交叉编码器重排序服务 - 单例模式，延迟加载

    模型路径优先级:
    1. 环境变量 BGE_RERANKER_PATH
    2. 项目 ./models/bge-reranker-base/
    3. HuggingFace 自动下载 (BAAI/bge-reranker-base)
    """

    _instance = None
    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        print(f"[INFO] 加载 BGE-Reranker 模型 ...")

        try:
            from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification

            # 确定模型路径
            model_path = os.environ.get("BGE_RERANKER_PATH")
            if not model_path or not os.path.isdir(model_path):
                if os.path.isdir(_MODEL_DIR):
                    model_path = _MODEL_DIR
                else:
                    model_path = "BAAI/bge-reranker-base"

            if os.path.isdir(model_path):
                _patch_config(model_path)

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = XLMRobertaTokenizerFast.from_pretrained(
                model_path,
                trust_remote_code=True if not os.path.isdir(model_path) else False
            )
            self._model = XLMRobertaForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=True if not os.path.isdir(model_path) else False
            )
            self._model.to(self._device)
            self._model.eval()

            print(f"[OK] BGE-Reranker 加载完成 (device={self._device})")
        except Exception as e:
            print(f"[ERROR] BGE-Reranker 加载失败: {e}")
            print(f"[HINT] 请按以下步骤下载模型:")
            print(f"  pip install modelscope  # 如未安装")
            print(f"  python -c \"from modelscope import snapshot_download; snapshot_download('BAAI/bge-reranker-base', local_dir='models/bge-reranker-base')\"")
            raise

    def _compute_scores(self, pairs: List[List[str]]) -> List[float]:
        """批量计算相关性分数"""
        with torch.no_grad():
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self._device)
            scores = self._model(**inputs, return_dict=True).logits.view(-1).float()
            return scores.cpu().tolist()

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Document]:
        """
        向量召回 Top5 → CrossEncoder 精确重排序 → 取 TopK

        Args:
            query: 用户查询
            documents: 向量召回的候选文档列表 (建议 5 个)
            top_k: 最终返回的文档数量 (建议 3 个)

        Returns:
            重排序后的 TopK 文档列表
        """
        if not documents:
            return []
        if len(documents) <= top_k:
            return documents

        self._ensure_model()

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._compute_scores(pairs)

        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        top_docs = [doc for _, doc in scored_docs[:top_k]]
        top_scores_str = [f"{s:.3f}" for s, _ in scored_docs[:top_k]]

        print(f"[Rerank] {len(documents)}候选 -> Top{top_k} (分数: {top_scores_str})")
        return top_docs

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Tuple[float, Document]]:
        """重排序并返回带分数的结果"""
        if not documents:
            return []
        if len(documents) <= top_k:
            self._ensure_model()
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self._compute_scores(pairs)
            scored_docs = list(zip(scores, documents))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            return scored_docs

        self._ensure_model()
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._compute_scores(pairs)
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return scored_docs[:top_k]


def get_reranker() -> RerankerService:
    """获取重排序服务单例"""
    return RerankerService()


if __name__ == '__main__':
    print("=== 测试 BGE-Reranker 重排序 ===\n")

    docs = [
        Document(
            page_content="LSTM networks use input, forget, and output gates to control information flow.",
            metadata={"source": "lstm.pdf", "page": 1}
        ),
        Document(
            page_content="Self-attention is an attention mechanism relating different positions of a single sequence.",
            metadata={"source": "Attention.pdf", "page": 3}
        ),
        Document(
            page_content="The Transformer uses multi-head attention to attend to different subspaces.",
            metadata={"source": "Attention.pdf", "page": 5}
        ),
        Document(
            page_content="Word2Vec uses skip-gram and CBOW to learn word embeddings.",
            metadata={"source": "Word2Vec.pdf", "page": 2}
        ),
        Document(
            page_content="GloVe combines matrix factorization and context window for word vectors.",
            metadata={"source": "GloVe.pdf", "page": 1}
        ),
    ]

    query = "How does attention mechanism work in Transformer models?"
    print(f"Query: {query}\n")
    print("--- 向量召回 Top5 (排序前) ---")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.metadata['source']}: {doc.page_content[:60]}...")

    print()
    reranker = get_reranker()
    results = reranker.rerank(query, docs, top_k=3)

    print(f"\n--- 重排序后 Top3 ---")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.metadata['source']}: {doc.page_content[:60]}...")

    print(f"\n--- 分数详情 ---")
    scored = reranker.rerank_with_scores(query, docs, top_k=5)
    for score, doc in scored:
        bar = "#" * max(1, int(score * 30))
        print(f"  {score:.3f} {bar} {doc.metadata['source']}")
