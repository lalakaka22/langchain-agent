"""
论文知识库数据加载模块
从论文 PDF 目录加载学术论文到向量存储中

支持两路提取:
  1. 文本提取: PyPDF 提取 PDF 文字层
  2. 图片提取: PyMuPDF + Tesseract OCR 识别图表/截图中的文字
"""
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from rag.vector_store import VectorStoreService

# 论文目录
PAPERS_DIR = r"C:\Users\22\Desktop\论文"


def load_knowledge_files(papers_dir: Optional[str] = None,
                        enable_ocr: bool = True) -> List[Document]:
    """
    加载论文目录下的所有 PDF（递归扫描子目录）

    两路提取:
      1. PyPDF 提取文字层 → 生成文本 Document
      2. PyMuPDF + Tesseract OCR 识别图片文字 → 生成图片 OCR Document

    Args:
        papers_dir: 论文目录路径，默认为桌面论文目录
        enable_ocr: 是否启用图片 OCR 提取 (默认 True)

    Returns:
        文档列表
    """
    if papers_dir is None:
        papers_dir = PAPERS_DIR

    if not os.path.isdir(papers_dir):
        logger.error(f"[ERROR] 论文目录不存在: {papers_dir}")
        return []

    all_docs = []
    ocr_docs_count = 0

    # 递归扫描所有子目录
    for root, dirs, files in os.walk(papers_dir):
        for filename in files:
            if not filename.lower().endswith(".pdf"):
                continue

            filepath = os.path.join(root, filename)
            # 提取领域分类（相对于论文根目录的子目录名）
            rel_dir = os.path.relpath(root, papers_dir)
            domain = rel_dir if rel_dir != "." else "未分类"

            try:
                # --- 第 1 路: 文本提取 ---
                loader = PyPDFLoader(filepath)
                docs = loader.load()

                # 添加来源元数据
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["type"] = "论文"
                    doc.metadata["domain"] = domain
                    doc.metadata["content_type"] = "text"

                all_docs.extend(docs)

                # --- 第 2 路: 图片 OCR 提取 ---
                if enable_ocr:
                    from rag.image_extractor import extract_images_from_pdf
                    img_docs = list(extract_images_from_pdf(filepath))

                    for doc in img_docs:
                        doc.metadata["domain"] = domain
                        doc.metadata.setdefault("source", filename)
                        doc.metadata.setdefault("type", "图片OCR")

                    all_docs.extend(img_docs)
                    ocr_docs_count += len(img_docs)

                logger.info(
                    f"[OK] 加载论文 [{domain}]: {filename} "
                    f"({len(docs)} 页文字 + {len(img_docs) if enable_ocr else 0} 张图片)"
                )
            except Exception as e:
                logger.error(f"[ERROR] 加载论文失败 {filename}: {e}")

    logger.info(
        f"共加载 {len(all_docs)} 个文档 "
        f"(含 {ocr_docs_count} 个图片OCR)"
    )
    return all_docs


def split_documents(documents: List[Document],
                    chunk_size: int = 600,
                    chunk_overlap: int = 100) -> List[Document]:
    """
    将论文切分为语义块（学术论文需要更大的上下文窗口）

    Args:
        documents: 原始论文文档列表
        chunk_size: 每个块的大小（字符数），论文需要更大以保留完整段落
        chunk_overlap: 块之间的重叠字符数

    Returns:
        切分后的文档块列表
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    split_docs = text_splitter.split_documents(documents)
    logger.info(f"论文切分: {len(documents)} 页 -> {len(split_docs)} 个语义块 (size={chunk_size}, overlap={chunk_overlap})")
    return split_docs


def init_vector_store(vector_store: Optional[VectorStoreService] = None) -> VectorStoreService:
    """
    初始化向量存储并加载论文知识库

    Args:
        vector_store: 已有的向量存储实例，为 None 则创建新实例

    Returns:
        已加载数据的 VectorStoreService 实例
    """
    if vector_store is None:
        vector_store = VectorStoreService()

    # 检查是否已有数据
    if vector_store.get_document_count() > 0:
        logger.info(f"向量存储已有 {vector_store.get_document_count()} 个文档，跳过加载")
        return vector_store

    # 加载论文
    logger.info("开始加载论文 PDF...")
    raw_docs = load_knowledge_files()

    if not raw_docs:
        logger.warning("未找到任何论文文件")
        return vector_store

    # 切分文档
    chunked_docs = split_documents(raw_docs)

    if not chunked_docs:
        logger.warning("论文切分后为空")
        return vector_store

    # 添加到向量存储
    vector_store.add_documents(chunked_docs)
    logger.info(f"向量存储初始化完成，共 {vector_store.get_document_count()} 个文档块")
    return vector_store


if __name__ == '__main__':
    print("测试论文数据加载...\n")
    vs = init_vector_store()
    print(f"\n向量存储文档数量: {vs.get_document_count()}")

    if vs.get_document_count() > 0:
        retriever = vs.get_retriever(search_kwargs={"k": 3})
        results = retriever.invoke("Transformer 的注意力机制是什么")
        print(f"\n检索结果 ({len(results)} 条):")
        for i, doc in enumerate(results, 1):
            print(f"\n--- 结果 {i} ---")
            print(f"来源: {doc.metadata.get('source', 'unknown')}")
            print(f"页码: {doc.metadata.get('page', 'unknown')}")
            print(f"内容: {doc.page_content[:200]}...")