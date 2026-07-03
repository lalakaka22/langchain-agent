"""
PDF 图片提取与 OCR 文字识别模块

流程:
  PDF → PyMuPDF 提取图片 → Tesseract OCR 识别文字 → LangChain Document 块

支持两种模式:
  1. Tesseract OCR (默认, 本地离线, 需安装 Tesseract)
  2. 嵌入式图片文本提取 (PyMuPDF 内置)

输出: 每个识别出文字内容的图片生成一个 Document，包含图片元数据。
"""
from __future__ import annotations

import io
import os
import re
from typing import List, Iterator, Optional, Tuple

import fitz  # PyMuPDF
import pytesseract
from langchain_core.documents import Document
from PIL import Image

from utils.logger_handler import logger

# ================================================================
# Tesseract 配置
# ================================================================
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_LANG = "eng"  # 学术论文以英文为主

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    logger.info(f"[OK] Tesseract OCR 已配置: {TESSERACT_PATH}")
else:
    logger.warning(f"[WARN] Tesseract 未找到 {TESSERACT_PATH}，OCR 将不可用")

# 最小图片尺寸 (像素), 过滤掉太小的图标/装饰图片
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
MIN_IMAGE_AREA = MIN_IMAGE_WIDTH * MIN_IMAGE_HEIGHT  # 10,000 px

# 最大图片尺寸 (像素), 跳过整页超大渲染图避免 OCR 过慢
MAX_IMAGE_WIDTH = 3000
MAX_IMAGE_HEIGHT = 3000


def _is_image_meaningful(img_bytes: bytes, ext: str = "png") -> bool:
    """检测图片是否足够大/有意义（过滤图标、装饰元素）"""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
            return False
        if w > MAX_IMAGE_WIDTH or h > MAX_IMAGE_HEIGHT:
            return False
        return True
    except Exception:
        return False


def _ocr_image(img_bytes: bytes) -> str:
    """
    对单张图片执行 OCR

    返回识别的文字内容，如果识别为空或失败返回空字符串。
    """
    try:
        img = Image.open(io.BytesIO(img_bytes))

        # 转换为灰度以提高 OCR 准确率
        img = img.convert("L")

        # --psm 6: Assume a uniform block of text (适合论文图表中的文字)
        text = pytesseract.image_to_string(
            img,
            lang=TESSERACT_LANG,
            config="--psm 6 --oem 3",
        )
        return _clean_ocr_text(text)
    except Exception as e:
        logger.error(f"[OCR] 识别失败: {e}")
        return ""


def _clean_ocr_text(text: str) -> str:
    """清理 OCR 输出: 去空行、合并断裂行、去噪"""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # 保留段落之间的空行（最多一行）
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        # 过滤纯符号/噪声行（超过80%是非字母数字）
        alpha_ratio = sum(1 for c in stripped if c.isalnum() or c in ".,;:!?()[]{}") / max(len(stripped), 1)
        if alpha_ratio < 0.2:
            continue
        cleaned.append(stripped)

    return "\n".join(cleaned)


def _extract_images_from_page(
    pdf_path: str,
    page_num: int,
    page: fitz.Page,
) -> List[Tuple[int, bytes, str]]:
    """
    从单页 PDF 中提取所有有意义的图片

    Returns:
        List of (image_index, image_bytes, extension)
    """
    images = []

    # 方法 1: 提取页面中嵌入的图片对象
    img_list = page.get_images(full=True)
    for img_idx, img_info in enumerate(img_list):
        xref = img_info[0]
        try:
            base_image = pdf_path  # dummy, we need the doc object
        except Exception:
            pass

    # 使用 get_images 返回的 xref 提取
    for img_idx, img_info in enumerate(img_list):
        xref = img_info[0]
        try:
            pix = fitz.Pixmap(page.parent, xref)
            # 跳过透明通道或非RGB图片（但保留灰度）
            if pix.n < 4:
                img_bytes = pix.tobytes("png") if pix.colorspace else pix.tobytes("png")
            else:
                # CMYK → RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")

            if not _is_image_meaningful(img_bytes):
                pix = None
                continue

            images.append((img_idx, img_bytes, "png"))
            pix = None
        except Exception as e:
            logger.warning(f"[IMG] 提取图片失败 p{page_num+1}_img{img_idx}: {e}")

    # 方法 2: 如果页面没有嵌入图片，尝试将整个页面渲染为图片
    # （用于扫描版 PDF 或图表为主的页面）
    if not images:
        try:
            mat = fitz.Matrix(2.0, 2.0)  # 2x 缩放以提高 OCR 精度
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            if _is_image_meaningful(img_bytes):
                images.append((0, img_bytes, "png"))
            pix = None
        except Exception as e:
            logger.warning(f"[IMG] 页面渲染失败 p{page_num+1}: {e}")

    return images


def extract_images_from_pdf(
    pdf_path: str,
    max_pages: Optional[int] = None,
) -> Iterator[Document]:
    """
    从 PDF 中提取图片并 OCR 识别，生成 Document 对象

    每个识别出文字内容的图片 → 一个 Document 块。
    跳过无文字内容的纯图表图片。

    Args:
        pdf_path: PDF 文件路径
        max_pages: 最大处理页数 (None = 全部)

    Yields:
        Document 对象，包含 OCR 文字内容和图片元数据
    """
    if not os.path.exists(TESSERACT_PATH):
        logger.warning(f"[IMG] Tesseract 未安装，跳过图片提取: {pdf_path}")
        return

    filename = os.path.basename(pdf_path)
    extracted_count = 0
    skipped_count = 0

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages

        for page_num in range(pages_to_process):
            page = doc[page_num]
            images = _extract_images_from_page(pdf_path, page_num, page)

            for img_idx, img_bytes, ext in images:
                ocr_text = _ocr_image(img_bytes)

                if not ocr_text or len(ocr_text.strip()) < 10:
                    skipped_count += 1
                    continue

                extracted_count += 1

                yield Document(
                    page_content=ocr_text,
                    metadata={
                        "source": filename,
                        "type": "图片OCR",
                        "page": page_num + 1,
                        "image_index": img_idx,
                        "extraction_method": "tesseract_ocr",
                        "content_type": "image_text",
                    },
                )

        doc.close()
    except Exception as e:
        logger.error(f"[IMG] 处理失败 {filename}: {e}")

    if extracted_count > 0 or skipped_count > 0:
        logger.info(
            f"[IMG] {filename}: 提取 {extracted_count} 张图片文字, "
            f"跳过 {skipped_count} 张 (无文字/过小)"
        )


def extract_images_batch(
    pdf_files: List[str],
    max_pages_per_pdf: Optional[int] = None,
) -> List[Document]:
    """批量处理多个 PDF 文件的图片提取"""
    all_docs = []
    for pdf_path in pdf_files:
        docs = list(extract_images_from_pdf(pdf_path, max_pages=max_pages_per_pdf))
        all_docs.extend(docs)
    return all_docs


# ================================================================
# 测试
# ================================================================
if __name__ == "__main__":
    import glob

    pdf_dir = r"C:\Users\22\Desktop\论文"
    pdf_files = glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True)[:2]

    print(f"测试 PDF 图片提取: {len(pdf_files)} 个文件\n")
    for pdf_path in pdf_files:
        print(f"处理: {os.path.basename(pdf_path)}")
        count = 0
        for doc in extract_images_from_pdf(pdf_path):
            count += 1
            preview = doc.page_content[:100].replace("\n", " ").encode("ascii", errors="replace").decode("ascii")
            p = doc.metadata.get("page", "?")
            img = doc.metadata.get("image_index", "?")
            print(f"  p{p} img{img}: {preview}...")
        print(f"  共提取 {count} 张图片文字\n")
