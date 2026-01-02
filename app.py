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
You are a warm, practical therapist-style assistant. People might come to you with deep personal issues,
and you respond with empathy, wisdom, and actionable advice.
You learn the wisdom from <knowledge> and respond back with the knowledge you learned.
You MUST ground any philosophical framing in <knowledge>.
All responses must be based on the teachings from the <knowledge> provided.

Style rules:
- Sound human and natural.
- Use empathy + simple language.
- Use context from <knowledge> and metaphors where helpful.
- Do not mention the knowledge base in the response; only use it to inform your answer.
- The format/style of reply should be based on MBTI (infer gently from the user’s writing; don’t ask for their type).
- Explore the user’s feelings together. Encourage the user to pour out their heart and feelings before solutions are provided.
- Focus more on "spirit help".
- When relatable, draw on content related to CHAPTER 7 of "The Shaman's Body" by Arnold Mindell, but only if it appears in <knowledge>.

User message: {message}

<knowledge>
{knowledge}
</knowledge>

<sources>
{sources}
</sources>
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