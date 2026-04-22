import os
import base64
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI


def _is_garbled(text: str) -> bool:
    """텍스트가 깨진 인코딩인지 판단 (null바이트 비율 기준)."""
    if not text:
        return True
    null_ratio = text.count("\x00") / len(text)
    cid_ratio = text.count("(cid:") / len(text) * 10
    return null_ratio > 0.05 or cid_ratio > 0.05


def _page_to_base64(page: fitz.Page, dpi: int = 150) -> str:
    """PDF 페이지를 PNG base64 문자열로 변환."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("png")).decode()


def _ocr_page_with_vision(b64_image: str, page_num: int, api_key: str) -> str:
    """GPT-4o Vision으로 페이지 이미지에서 텍스트 추출."""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "이 PDF 페이지의 텍스트를 빠짐없이 그대로 추출해줘. 서식이나 레이아웃 설명 없이 순수 텍스트만 출력해.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}", "detail": "high"},
                },
            ],
        }],
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""


def extract_text_from_path(file_path: str, api_key: str = "") -> tuple[list[dict], bool]:
    """
    PDF에서 텍스트 추출. 인코딩이 깨진 경우 GPT-4o Vision OCR로 대체.
    반환: (pages, used_ocr)
    """
    filename = os.path.basename(file_path)
    doc = fitz.open(file_path)
    pages = []
    used_ocr = False

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()

        if _is_garbled(text):
            if not api_key:
                raise ValueError(
                    f"p.{page_num} 텍스트가 깨져 있습니다. OCR을 위해 OpenAI API Key가 필요합니다."
                )
            used_ocr = True
            b64 = _page_to_base64(page)
            text = _ocr_page_with_vision(b64, page_num, api_key)

        if text.strip():
            pages.append({
                "text": text,
                "page": page_num,
                "source": filename,
            })

    doc.close()
    return pages, used_ocr


def extract_text_from_pdf(file_bytes: bytes, filename: str, api_key: str = "") -> tuple[list[dict], bool]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    used_ocr = False

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()

        if _is_garbled(text):
            if not api_key:
                raise ValueError(f"p.{page_num} 텍스트가 깨져 있어 OCR이 필요합니다.")
            used_ocr = True
            b64 = _page_to_base64(page)
            text = _ocr_page_with_vision(b64, page_num, api_key)

        if text.strip():
            pages.append({
                "text": text,
                "page": page_num,
                "source": filename,
            })

    doc.close()
    return pages, used_ocr


def chunk_pages(pages: list[dict], chunk_size: int = 800, chunk_overlap: int = 100) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "page": page["page"],
                "source": page["source"],
                "chunk_index": i,
            })
    return chunks
