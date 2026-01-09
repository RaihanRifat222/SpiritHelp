from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

from pathlib import Path
from dotenv import load_dotenv
import os
import math

# ---------- Load env (DON'T print your key) ----------
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)
api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key:
    raise ValueError("OPENAI_API_KEY is missing. Put it in .env or environment variables.")
os.environ["OPENAI_API_KEY"] = api_key

# ---------- Paths ----------
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

# ---------- Embeddings ----------
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------- Vector store ----------
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# (Optional) Clear existing collection to avoid duplicates on re-runs
# Comment this out if you want to append.
try:
    vector_store._collection.delete(where={})
    print("Cleared existing collection.")
except Exception as e:
    print("Could not clear collection (ok on first run):", e)

# ---------- Load PDFs ----------
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()
print("Loaded docs:", len(raw_documents))

# ---------- Split ----------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(raw_documents)

# Remove empty/whitespace-only chunks (optional but helps)
chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
print("Chunks:", len(chunks))

# ---------- Add to Chroma in batches ----------
MAX_BATCH = 5000  # must be <= your max (5461). Keep headroom.
total = len(chunks)
num_batches = math.ceil(total / MAX_BATCH)

for b in range(num_batches):
    start = b * MAX_BATCH
    end = min(start + MAX_BATCH, total)
    batch = chunks[start:end]
    uuids = [str(uuid4()) for _ in range(len(batch))]

    vector_store.add_documents(documents=batch, ids=uuids)
    print(f"Upserted batch {b+1}/{num_batches}: {end-start} docs (total {end}/{total})")

# ---------- Verify ----------
print("Chroma count:", vector_store._collection.count())
print("Done.")
