# Pure RAG — Welcome & Architecture Reference Guide

## 1. Overview of Pure RAG

**Pure RAG** is a universal, local-first Retrieval-Augmented Generation (RAG) pipeline and AI chatbot interface designed for researchers, engineers, clinicians, legal analysts, students, and everyday users across **macOS, Windows, and Linux**.

Simply place your PDFs, Markdown files, or text documents into the designated `pdfs/` folder (or drag and drop them directly into the web sidebar). Pure RAG automatically parses their layout, enriches every chunk with contextual breadcrumbs, indexes them for both semantic vector similarity and exact keyword matching, and serves a modern AI chat application locally at `http://127.0.0.1:8000`.

---

## 2. Why SOTA Hybrid RAG Outperforms Naive RAG

Traditional "Naive RAG" systems split text into fixed 500-character blocks and run a single cosine similarity vector search. Research between 2024 and 2026 demonstrated three major failure modes in Naive RAG:
1. **Context Loss at Chunk Boundaries**: Isolated chunks lose track of which document, section, or table they belong to.
2. **Lexical Blindness**: Dense embeddings frequently miss exact part numbers, legal statutes, gene names, acronyms, and numerical figures.
3. **Bi-Encoder Noise ("Lost in the Middle")**: Top-K bi-encoder vectors often surface superficially similar passages while burying the exact answer.

### The 6-Stage Pure RAG Pipeline
Pure RAG resolves all three failure modes using a researched 6-stage architecture:
- **Stage 1: Layout-Aware Extraction (`PyMuPDF4LLM` + `pypdf`)**: Extracts clean Markdown while preserving tables, headings, and multi-column reading order.
- **Stage 2: Hierarchical Structure-Aware Chunking**: Splits documents along natural section headers (`#`, `##`, `###`) and paragraph boundaries with sliding sentence overlap.
- **Stage 3: Contextual Breadcrumb Enrichment**: Inspired by Anthropic's Contextual Retrieval research (which reduces retrieval failure rates by 49% to 67%), every chunk is prepended with `[Document: filename | Page X of Y | Section: H1 > H2]` prior to indexing.
- **Stage 4 & 5: Parallel Hybrid Retrieval + Reciprocal Rank Fusion (RRF)**: Combines **Dense Vector Search** (`BAAI/bge-*` via FastEmbed / Ollama) and **Sparse Keyword Search** (`BM25Okapi`) using Reciprocal Rank Fusion ($k=60$, dense weight $0.55$, BM25 weight $0.45$).
- **Stage 6: Cross-Encoder Second-Stage Reranking**: Passes top RRF candidates through a Cross-Encoder (`ms-marco-MiniLM-L-12-v2` via FlashRank / Cross-Proximity Scorer) so query tokens attend directly to passage tokens before selecting the final top-$K$ sources.

---

## 3. Promptability, Voice-to-Speech, and Hardware Tiers

### Promptability Controls
Above the prompt box, users can dynamically control how Pure RAG synthesizes answers:
- **Length Modifiers**: `Shorter` (<120 words, crisp summary), `Balanced` (150–300 words), or `Longer` (comprehensive, in-depth analysis).
- **Style & Format Modifiers**: `Standard`, `Precise` (technical figures & definitions), `Bulleted` (structured bullet points), `Executive` (Bottom-Line-Up-Front brief), `1-2-3 Steps` (sequential walkthrough), `Simple (ELI5)`, or `Strict Quotes` (verbatim evidence & citations only).
- **Domain Persona**: Customize system instructions for Academic Research, Legal & Compliance, Software Engineering, or Biomedical review.

### Voice-to-Speech (STT) & Read-Aloud (TTS)
- Click the **Voice** microphone button in the prompt box to dictate questions hands-free.
- Click **Speak** on any response (or enable **Auto-Read Voice** in the top bar) to hear cited answers read aloud.

### Permissioned Hardware Profiling
On first launch, Pure RAG inspects your OS, CPU cores, available RAM, and GPU acceleration (Apple Silicon Metal, NVIDIA CUDA, AMD ROCm, or CPU) and recommends one of four tiers (`ultra_light`, `balanced`, `performance`, or `workstation`). No models or heavy packages are downloaded without your explicit permission.
