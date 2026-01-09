---
title: SpiritHelp
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
---

# SpiritHelp — Retrieval-Grounded Spiritual Support Chatbot

🔗 **Live demo:**  
https://huggingface.co/spaces/RaihanRifat222/SpiritHelp

SpiritHelp is a retrieval-augmented conversational system designed to explore how **LLMs can support people spiritually** when grounded in carefully curated, human-written source material.

The project focuses on **responsible spiritual guidance**, where the model responds naturally and empathetically **without inventing teachings or advice** outside its knowledge base.

---

## What this project does

- Ingests a curated dataset of books and academic literature related to:
  - spirituality and shamanic traditions  
  - self-reflection, healing, and meaning-making  
  - psychology, consciousness, and human belief systems
- Stores this material in a vector database (Chroma)
- Uses semantic retrieval at runtime to ground responses
- Generates calm, conversational replies that:
  - start naturally (no over-preaching on simple inputs)
  - gradually deepen only when the user opens up
  - stay faithful to the retrieved source material

The goal is not therapy, but **gentle spiritual support informed by real texts**.

---

## Why this matters

LLMs often sound wise even when they are hallucinating.  
This project explores how **dataset choice, retrieval design, and prompt constraints** can reduce that risk — especially in sensitive domains like spirituality and inner reflection.

SpiritHelp prioritizes:
- grounding over eloquence
- pacing over verbosity
- honesty over confident fabrication

---

## Tech stack

- **LLM:** OpenAI (API)
- **Framework:** LangChain
- **Vector DB:** ChromaDB
- **UI:** Gradio
- **Hosting:** Hugging Face Spaces
- **Storage:** Git LFS for large vector assets

---

## Notes

- The system is retrieval-based, not fine-tuned
- All spiritual framing is derived from the embedded dataset
- Deployed publicly with real production constraints

---

Built by **Raihan Rifat**
