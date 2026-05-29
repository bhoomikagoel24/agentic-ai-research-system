<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=7C6AF7&height=200&section=header&text=Agentic%20Research%20Synthesis%20System&fontSize=32&fontColor=ffffff&fontAlignY=38&desc=Semi-Agentic%20%C2%B7%20Multi-Stage%20Reasoning%20%C2%B7%20Reliability-First%20Engineering&descAlignY=58&descSize=14&animation=fadeIn" />

<br/>

[![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/gemini)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![arXiv](https://img.shields.io/badge/arXiv_API-B31B1B?style=for-the-badge&logoColor=white)](https://arxiv.org)
[![Semantic Scholar](https://img.shields.io/badge/Semantic_Scholar-097AE9?style=for-the-badge&logoColor=white)](https://api.semanticscholar.org)

[![LangGraph](https://img.shields.io/badge/LangGraph-planned-6B46C1?style=flat-square)](https://langchain-ai.github.io/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG_planned-22C55E?style=flat-square)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-UI_planned-009688?style=flat-square&logo=fastapi&logoColor=white)]()
[![Status](https://img.shields.io/badge/Status-Active_Development-F59E0B?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-7C6AF7?style=flat-square)](LICENSE)

<br/>

> **Beyond RAG. Beyond summarization. Toward grounded, cross-paper analytical reasoning.**

<br/>

**[→ Full Visual Documentation & Architecture](https://bhoomikagoel24.github.io/agentic-ai-research-system)**

</div>

---

## The Problem

Literature review is not an information retrieval problem. It requires understanding **relationships across papers** — comparing methodologies, detecting contradictions, identifying trends, surfacing research gaps, and synthesizing findings into a coherent whole.

Traditional LLM systems can summarize individual papers. They struggle with everything else. Single-prompt pipelines lose context across documents, overgeneralize from incomplete evidence, and produce outputs that are shallow and difficult to trust.

This system was built to address that.

---

## System Architecture

<p align="center">
  <img src="assets/architecture_preview.png" width="920" alt="System Architecture"/>
</p>

<p align="center">
  <a href="assets/architecture_full.png"><b>View Full Architecture Diagram →</b></a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://bhoomikagoel24.github.io/agentic-ai-research-system"><b>Interactive Documentation →</b></a>
</p>

---

## What Each Agent Does

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  01  Planner Agent       Query decomposition · Self-evaluation loop         │
│  02  Research Agent      arXiv + Semantic Scholar · Adaptive expansion      │
│  03  Filter + Rank       Relevance scoring · Recency weighting · Top-K      │
│  04  Summarizer Agent    Structured extraction · Confidence calibration      │
│  05  Synthesis Agent ★   Cross-paper reasoning · Gaps · Trends · Tradeoffs  │
│  06  Critic Agent        Hallucination detection · Grounding evaluation      │
│  07  Formatter Agent     Professional literature review generation           │
└─────────────────────────────────────────────────────────────────────────────┘
```

The **Synthesis Agent** is the core intelligence layer — where the system transitions from information extraction to genuine knowledge synthesis. It reasons across all papers simultaneously, not independently.

---

## Sample Output

### Comparative Method Analysis
> Transformer-based models demonstrate stronger sequential reasoning for temporal forecasting, while GAN-based approaches address data scarcity but introduce distributional shift risks. Neither has been robustly benchmarked across market regimes.

### Synthesized Research Insight
> Financial AI is shifting toward multimodal, context-aware forecasting systems — but robustness, scalability, and interpretability remain major unresolved challenges. Most benchmarks are single-institution, limiting generalizability.

### Research Gaps Identified
> - No standardized evaluation benchmarks for LLM-based financial reasoning
> - Underexplored multimodal fusion of price signals, text, and macroeconomic context
> - Limited real-time deployment work under volatile market conditions

---

## Reliability Engineering

> In multi-stage agentic pipelines, infrastructure reliability is as important as model quality.

| Mechanism | What it does |
|---|---|
| **File-level caching** | Summaries persist to disk — re-runs skip computed stages entirely |
| **Exponential backoff** | Randomized jitter on every retry — no API hammering under rate limits |
| **Adaptive retrieval** | Queries expand dynamically when paper quality falls below threshold |
| **State persistence** | Shared state accumulates across agents — any stage reruns independently |
| **JSON validation** | Structured output validated at every stage before passing downstream |
| **DEV_MODE** | Lightweight execution path for fast, low-cost development iteration |

---

## Tech Stack

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)]()
[![Google Gemini](https://img.shields.io/badge/Gemini_API-4285F4?style=flat-square&logo=google&logoColor=white)]()
[![arXiv](https://img.shields.io/badge/arXiv_API-B31B1B?style=flat-square)]()
[![Semantic Scholar](https://img.shields.io/badge/Semantic_Scholar-097AE9?style=flat-square)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-6B46C1?style=flat-square)]()
[![ChromaDB](https://img.shields.io/badge/ChromaDB-22C55E?style=flat-square)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)]()

</div>

---

## Known Limitations

The system currently reasons over **abstracts**, not full papers — limiting deep experimental understanding. Retrieval is **lexical**, not semantic — conceptually similar papers with different terminology are under-recalled. The Critic Agent evaluates but does not yet **autonomously trigger refinement**. Evaluation metrics are heuristic rather than learned.

These are known constraints — each maps directly to the next development phase.

---

## What's Next

```
→  Semantic retrieval via embedding-based search
→  Critic-driven refinement loop (Critic triggers Synthesis re-runs)
→  Persistent knowledge memory — JSON/SQLite before vector DB
→  Full-paper reasoning beyond abstract-only analysis
→  LangGraph orchestration with conditional routing and retry nodes
→  Quantitative evaluation framework for synthesis quality
→  Citation-grounded synthesis with academic formatting
```

---

## Design Philosophy

The architecture was built without heavy orchestration frameworks intentionally — to develop a deep understanding of state flow, retry logic, evaluation patterns, and agent interaction **before** abstracting them away.

One realization: useful AI systems are not primarily about prompting. As systems scale, the hard problems become **orchestration**, **reliability**, **state management**, and **grounded reasoning** — as critical as model capability itself.

---

<div align="center">

## Author

[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://bhoomika-ai-portfolio.vercel.app/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bhoomikagoel111)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/bhoomikagoel24)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:bhoomikagoel24@gmail.com)

**Bhoomika Goel** · AI/ML Engineer · Agentic Systems · Research Automation

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=7C6AF7&height=100&section=footer" />

</div>