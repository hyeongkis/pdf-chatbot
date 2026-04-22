from openai import OpenAI

MODEL = "gpt-4o"

SYSTEM_PROMPT = """당신은 업로드된 PDF 문서를 기반으로 질문에 답하는 전문 AI 어시스턴트입니다.

규칙:
1. 반드시 제공된 문서 컨텍스트를 근거로 답변하세요.
2. 컨텍스트에 없는 내용은 "제공된 문서에서 해당 내용을 찾을 수 없습니다."라고 명확히 밝히세요.
3. 출처(파일명, 페이지)를 답변 끝에 표기하세요.
4. 한국어로 답변하되, 원문이 영어라면 번역 후 답변하세요."""


def build_context(hits: list[dict]) -> str:
    if not hits:
        return "관련 문서 내용이 없습니다."
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(
            f"[참조 {i}] 파일: {hit['source']} | 페이지: {hit['page']} | 유사도: {hit['score']:.2f}\n{hit['text']}"
        )
    return "\n\n---\n\n".join(parts)


def stream_answer(
    query: str,
    hits: list[dict],
    history: list[dict],
    api_key: str,
):
    context = build_context(hits)
    user_message = f"[문서 컨텍스트]\n{context}\n\n[질문]\n{query}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    client = OpenAI(api_key=api_key)
    with client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=2048,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
