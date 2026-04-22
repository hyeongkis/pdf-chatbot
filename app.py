import os
import streamlit as st
from dotenv import load_dotenv

from pdf_processor import extract_text_from_pdf, extract_text_from_path, chunk_pages
from vector_store import add_chunks, search, list_sources, remove_source, reset_collection
from chat import stream_answer

load_dotenv()

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(page_title="PDF 챗봇", page_icon="📄", layout="wide")
st.title("📄 PDF 기반 AI 챗봇")

# ── 세션 초기화 ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── API Key 로드 (secrets → .env → 사용자 입력 순) ───────────
_server_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
_server_key = _server_key or os.getenv("OPENAI_API_KEY", "")

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    if _server_key:
        openai_key = _server_key
        st.success("OpenAI API Key 로드됨", icon="🔑")
    else:
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="임베딩 및 답변 생성에 사용됩니다.",
        )

    st.divider()
    st.header("📂 PDF 불러오기")

    # 파일 업로드
    uploaded_files = st.file_uploader(
        "PDF 파일 선택 (복수 가능)",
        type="pdf",
        accept_multiple_files=True,
    )

    # 로컬 경로 입력 (로컬 실행 시)
    with st.expander("📁 로컬 파일 경로로 불러오기"):
        pdf_path_input = st.text_area(
            "파일 경로 (줄바꿈으로 여러 개 입력)",
            placeholder="C:\\Users\\경로\\파일.pdf",
            height=80,
        )

    if st.button("📥 PDF 로드 및 인덱싱", use_container_width=True, type="primary"):
        if not openai_key:
            st.error("OpenAI API Key를 입력하세요.")
        elif not uploaded_files and not pdf_path_input.strip():
            st.warning("PDF 파일을 선택하거나 경로를 입력하세요.")
        else:
            # 업로드된 파일 처리
            for f in uploaded_files:
                with st.spinner(f"{f.name} 처리 중..."):
                    pages, used_ocr = extract_text_from_pdf(f.read(), f.name, api_key=openai_key)
                    chunks = chunk_pages(pages)
                    added = add_chunks(chunks, openai_key)
                    if added == 0:
                        st.info(f"{f.name} — 이미 인덱싱된 파일입니다.")
                    else:
                        label = " (OCR 적용)" if used_ocr else ""
                        st.success(f"{f.name}{label} — {added}개 청크 인덱싱 완료")

            # 로컬 경로 처리
            if pdf_path_input.strip():
                paths = [p.strip().strip('"').strip("'") for p in pdf_path_input.strip().splitlines() if p.strip()]
                for path in paths:
                    if not os.path.exists(path):
                        st.error(f"파일을 찾을 수 없습니다: {path}")
                        continue
                    name = os.path.basename(path)
                    with st.spinner(f"{name} 처리 중..."):
                        pages, used_ocr = extract_text_from_path(path, api_key=openai_key)
                        chunks = chunk_pages(pages)
                        added = add_chunks(chunks, openai_key)
                        if added == 0:
                            st.info(f"{name} — 이미 인덱싱된 파일입니다.")
                        else:
                            label = " (OCR 적용)" if used_ocr else ""
                            st.success(f"{name}{label} — {added}개 청크 인덱싱 완료")

    st.divider()
    st.header("📚 로드된 문서")

    sources = list_sources()
    if sources:
        for src in sources:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"- `{src}`")
            if col2.button("🗑️", key=f"del_{src}", help=f"{src} 삭제"):
                remove_source(src)
                st.rerun()

        if st.button("🗑️ 전체 초기화", use_container_width=True):
            reset_collection()
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("아직 로드된 문서가 없습니다.")

    st.divider()
    n_results = st.slider("검색할 청크 수", min_value=2, max_value=10, value=5)

# ── 채팅 영역 ────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 입력창 ───────────────────────────────────────────────────
if prompt := st.chat_input("PDF 내용에 대해 질문하세요..."):
    if not openai_key:
        st.error("사이드바에서 OpenAI API Key를 입력해주세요.")
        st.stop()

    if not list_sources():
        st.warning("먼저 PDF를 업로드하고 인덱싱해주세요.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    hits = search(prompt, openai_key, n_results=n_results)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in stream_answer(prompt, hits, st.session_state.chat_history, openai_key):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    if hits:
        with st.expander("📎 참조된 문서 청크 보기"):
            for i, hit in enumerate(hits, 1):
                st.markdown(f"**[{i}] {hit['source']} — p.{hit['page']} (유사도: {hit['score']:.2f})**")
                st.text(hit["text"][:400] + ("..." if len(hit["text"]) > 400 else ""))
                st.divider()
