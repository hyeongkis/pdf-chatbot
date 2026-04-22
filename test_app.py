"""
전체 파이프라인 테스트: PDF 생성 → 청킹 → 임베딩 → 검색 → Claude 답변
"""
import os
import sys
import fitz
from dotenv import load_dotenv

# Windows 콘솔 UTF-8 출력
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


# ── 1. 샘플 PDF 생성 ──────────────────────────────────────────
def create_sample_pdf() -> bytes:
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((50, 80), "인공지능과 머신러닝 입문", fontsize=18)
    page1.insert_text((50, 130), """
머신러닝(Machine Learning)은 컴퓨터가 데이터로부터 스스로 학습하는 기술입니다.
지도 학습(Supervised Learning)은 레이블이 있는 데이터를 이용해 모델을 훈련합니다.
비지도 학습(Unsupervised Learning)은 레이블 없이 데이터의 패턴을 찾습니다.
강화 학습(Reinforcement Learning)은 에이전트가 환경과 상호작용하며 보상을 최대화합니다.
딥러닝은 여러 층의 신경망을 활용하여 복잡한 패턴을 학습합니다.
    """.strip(), fontsize=11)

    page2 = doc.new_page()
    page2.insert_text((50, 80), "자연어 처리(NLP)", fontsize=18)
    page2.insert_text((50, 130), """
자연어 처리는 컴퓨터가 인간의 언어를 이해하고 생성하는 기술입니다.
트랜스포머(Transformer) 모델은 2017년 구글이 발표한 혁신적인 아키텍처입니다.
BERT는 양방향 트랜스포머로 문맥을 이해하는 데 탁월한 성능을 보입니다.
GPT는 자동회귀 방식으로 텍스트를 생성하는 대형 언어 모델입니다.
임베딩(Embedding)은 텍스트를 벡터 공간에 표현하는 방법입니다.
코사인 유사도를 이용해 두 텍스트의 의미적 유사성을 측정합니다.
    """.strip(), fontsize=11)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ── 테스트 실행 ───────────────────────────────────────────────
errors = []

print("\n" + "="*55)
print("  PDF 챗봇 파이프라인 테스트")
print("="*55)

# STEP 1: PDF 생성
print("\n[1] 샘플 PDF 생성")
try:
    pdf_bytes = create_sample_pdf()
    assert len(pdf_bytes) > 1000
    print(f"{PASS} PDF 생성 완료 ({len(pdf_bytes):,} bytes)")
except Exception as e:
    print(f"{FAIL} {e}")
    errors.append(e); sys.exit(1)

# STEP 2: 텍스트 추출
print("\n[2] 텍스트 추출 (PyMuPDF)")
try:
    from pdf_processor import extract_text_from_pdf
    pages = extract_text_from_pdf(pdf_bytes, "test_ai.pdf")
    assert len(pages) == 2
    print(f"{PASS} {len(pages)}페이지 추출 완료")
    for p in pages:
        preview = p['text'].replace('\n', ' ')[:60]
        print(f"  {INFO} p.{p['page']}: {preview}...")
except Exception as e:
    print(f"{FAIL} {e}")
    errors.append(e); sys.exit(1)

# STEP 3: 청킹
print("\n[3] 텍스트 청킹 (RecursiveCharacterTextSplitter)")
try:
    from pdf_processor import chunk_pages
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 0
    print(f"{PASS} {len(chunks)}개 청크 생성")
    for c in chunks[:3]:
        print(f"  {INFO} [{c['source']} p.{c['page']} chunk#{c['chunk_index']}] {c['text'][:50]}...")
except Exception as e:
    print(f"{FAIL} {e}")
    errors.append(e); sys.exit(1)

# STEP 4: OpenAI 임베딩 + ChromaDB 저장
print("\n[4] 임베딩 생성 및 ChromaDB 저장 (OpenAI)")
if not OPENAI_KEY:
    print(f"{SKIP} OPENAI_API_KEY 없음 - .env 파일에 키를 설정하면 테스트 가능")
else:
    try:
        from vector_store import add_chunks, list_sources
        added = add_chunks(chunks, OPENAI_KEY)
        print(f"{PASS} {added}개 청크 인덱싱 완료")
        srcs = list_sources()
        print(f"  {INFO} 저장된 소스: {srcs}")
    except Exception as e:
        print(f"{FAIL} {e}")
        errors.append(e)

# STEP 5: 벡터 검색
print("\n[5] 벡터 검색 테스트")
if not OPENAI_KEY:
    print(f"{SKIP} OPENAI_API_KEY 없음")
else:
    try:
        from vector_store import search
        query = "트랜스포머 모델이란 무엇인가요?"
        hits = search(query, OPENAI_KEY, n_results=3)
        assert len(hits) > 0
        print(f"{PASS} '{query}' → {len(hits)}개 결과")
        for i, h in enumerate(hits, 1):
            print(f"  {INFO} [{i}] score={h['score']:.3f} | {h['source']} p.{h['page']}: {h['text'][:50]}...")
    except Exception as e:
        print(f"{FAIL} {e}")
        errors.append(e)

# STEP 6: OpenAI 스트리밍 답변
print("\n[6] OpenAI 답변 생성 테스트 (gpt-4o)")
if not OPENAI_KEY:
    print(f"{SKIP} OPENAI_API_KEY 없음")
else:
    try:
        from vector_store import search
        from chat import stream_answer
        hits = search("임베딩이란?", OPENAI_KEY, n_results=3)
        print(f"  {INFO} 스트리밍 응답:")
        print("  " + "-"*40)
        full = ""
        for token in stream_answer("임베딩이란 무엇인가요?", hits, [], OPENAI_KEY):
            print(token, end="", flush=True)
            full += token
        print("\n  " + "-"*40)
        assert len(full) > 50
        print(f"{PASS} 응답 생성 완료 ({len(full)}자)")
    except Exception as e:
        print(f"{FAIL} {e}")
        errors.append(e)

# ── 결과 요약 ─────────────────────────────────────────────────
print("\n" + "="*55)
if errors:
    print(f"  결과: {len(errors)}개 오류 발생")
else:
    print("  결과: 모든 테스트 통과")
print("="*55 + "\n")
