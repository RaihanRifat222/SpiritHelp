import os
import gradio as gr

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma


# -------------------------
# Config (Spaces-safe)
# -------------------------
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "example_collection"
TOP_K = 12

# A helper query to always pull worldview/teachings context
PHILO_QUERY = (
    "core principles worldview teachings healing transformation "
    "self identity community ritual meaning"
)


def require_env(name: str) -> str:
    """Fail fast if a required environment variable is missing."""
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Add it in Hugging Face Space → Settings → Secrets."
        )
    return val


# -------------------------
# Secrets / API key
# -------------------------
# Hugging Face Spaces: put OPENAI_API_KEY in Settings → Secrets.
os.environ["OPENAI_API_KEY"] = require_env("OPENAI_API_KEY")

# -------------------------
# Models / Vector store
# -------------------------
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(temperature=0.6, model="gpt-4o-mini")

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Note: avoid printing keys; logging Chroma count is fine.
try:
    print("CHROMA COUNT:", vector_store._collection.count())
except Exception as e:
    print("Could not read Chroma count:", e)

retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})


def stream_response(message, history):
    """
    Gradio ChatInterface streaming function.
    message: str
    history: list of [user, assistant] pairs (unused here, but available)
    """
    if not message or not message.strip():
        yield "Type a question and I’ll help."
        return

    # 1) Retrieve: specific + philosophy (merged)
    docs_specific = retriever.invoke(message)
    docs_philo = retriever.invoke(PHILO_QUERY)

    seen = set()
    docs = []
    for d in (docs_specific + docs_philo):
        key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:120])
        if key not in seen:
            seen.add(key)
            docs.append(d)

    knowledge = "\n\n".join([d.page_content for d in docs]).strip()

    # Keep sources internal to grounding (do not show user unless you want to)
    sources = "\n".join(
        [
            f"- {d.metadata.get('source','unknown')} | page {d.metadata.get('page','?')}"
            for d in docs
        ]
    ).strip()

    if not knowledge:
        yield "I couldn’t find that in the provided documents. Try rephrasing your question."
        return

    # 2) Prompt (kept very close to your original)
    rag_prompt = f"""
You are a calm, grounded, compassionate guide.
Someone has come to you to be understood — not judged, not “fixed”.

IMPORTANT: The wisdom you share must come from <knowledge>.
Treat <knowledge> as your spiritual/philosophical foundation for this conversation.
Do not introduce new teachings that are not supported by <knowledge>.

How to respond:
1) Start naturally: reflect what the person is feeling in plain human language.
2) Ask one gentle question (only if it helps the person open up).
3) Then offer guidance that is shaped by <knowledge> — not as quotes or references,
   but as insight that feels integrated and lived-in.
4) Keep the tone conversational, warm, and emotionally present.

Grounding rules (keep invisible to the user):
- If <knowledge> contains relevant ideas, weave them in clearly.
- If <knowledge> does NOT contain relevant ideas, say so briefly and stay supportive.
  Example: “I don’t have enough from the material to guide this directly, but I can sit with you in it…”
- Never mention documents, sources, or chapters.
- Do not make up spiritual teachings. Stay anchored in <knowledge>.

Style:
- Warm, human, simple language.
- Short to medium paragraphs.
- No bullet points.
- No clinical tone.
- Use metaphors only if they naturally fit the situation.
- Keep advice practical, gentle, and doable.

User message:
{message}

<knowledge>
{knowledge}
</knowledge>
""".strip()


    # 3) Stream back
    partial = ""
    try:
        for chunk in llm.stream(rag_prompt):
            if chunk and getattr(chunk, "content", None):
                partial += chunk.content
                yield partial
    except Exception as e:
        # Don’t leak internals; keep it user-friendly
        yield (
            "Something went wrong while generating a response. "
            "Please try again in a moment, or rephrase your question."
        )
        print("LLM streaming error:", repr(e))


# -------------------------
# UI
# -------------------------
demo = gr.ChatInterface(
    fn=stream_response,
    textbox=gr.Textbox(
        placeholder="Ask me anything (I will answer using the provided documents)...",
        container=False,
    ),
    title="SpiritHelp",
    description="Grounded answers based on your uploaded documents.",
)

# -------------------------
# Launch (Hugging Face Spaces)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        ssr_mode=False  # 🚫 disables HF auth redirect
    )