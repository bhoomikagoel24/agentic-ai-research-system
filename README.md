# Agentic AI Research System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=flat&logo=chainlink&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-planned-6B46C1?style=flat)
![Gemini](https://img.shields.io/badge/Gemini-API-0052CC?style=flat&logo=gemini&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Development-F59E0B?style=flat)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat)

> An agentic AI system that mimics how a human researcher thinks, retrieves, analyzes, and synthesizes knowledge from scientific literature.

---

## What this is

Most research tools stop at summarization. This system goes further — it reasons across multiple papers, compares methodologies, identifies research gaps, and produces a structured report that reflects actual analytical thinking.

The architecture is built around the idea that **complex research tasks require coordinated agent workflows**, not a single LLM prompt. Each agent in the pipeline has a specific responsibility, and the system can critique and refine its own outputs before finalizing a result.

---

## 🚀 What Makes This Different

Unlike traditional RAG or summarization pipelines, this system:

- Uses a Planner Agent for structured reasoning
- Performs cross-paper synthesis (not isolated summaries)
- Includes a Critic Agent for self-refinement
- Mimics real research workflows end-to-end

---

## How it works

A user submits a research query. The system runs it through a coordinated pipeline of specialized agents:

### 🔁 System Flow

**User Query**  
↓  
**Planner → Research → Filter → Summarize → Synthesize → Critic → Formatter**  
↓  
**Final Report**

---

### 🛠 Agents and Roles

| Agent | Role |
|---|---|
| **Planner** | Breaks the query into sub-questions and maps a retrieval strategy |
| **Research** | Fetches papers from arXiv and Semantic Scholar, deduplicates results |
| **Filter** | Scores relevance using LLM, weighted by citations & recency |
| **Summarizer** | Extracts methodology, contributions, and limitations |
| **Synthesis** | Performs cross-paper comparison and identifies research gaps |
| **Critic** | Validates output quality, triggers targeted re-runs if needed |
| **Formatter** | Produces final report in structured JSON and Markdown |

---

## Example output

```json
{
  "topic": "Pneumonia Detection using Deep Learning",
  "methods": ["CNN-based classification", "Vision Transformer (ViT)", "Hybrid CNN-Transformer"],
  "comparison": [
    "CNNs offer faster inference, suitable for real-time deployment",
    "Transformers achieve higher accuracy but require larger datasets"
  ],
  "research_gaps": [
    "Limited work on multimodal approaches combining imaging with clinical notes",
    "Most benchmarks use single-institution datasets — generalizability is unclear"
  ],
  "future_work": [
    "Real-time edge deployment for low-resource clinical settings",
    "Federated learning to address data privacy constraints"
  ],
  "summary": "Transformer-based models demonstrate strong diagnostic performance but face deployment barriers. Hybrid architectures show promise but remain underexplored."
}
```

---

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| LLM | OpenAI API (GPT-4o-mini / GPT-4o) |
| Agent Framework | LangChain, LangGraph (planned) |
| Research APIs | arXiv API, Semantic Scholar API |
| Memory / RAG | ChromaDB, FAISS (planned) |
| API Layer | FastAPI (planned) |
| Output | Structured JSON, Markdown |

---

## Skills Demonstrated

- Multi-agent system design
- LLM orchestration
- Prompt engineering
- Research automation
- System architecture thinking

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| Core pipeline (Planner → Formatter) | 🟡 In progress | Notebook-first, then modularized |
| Modular Python codebase | ⬜ Planned | Clean agent classes, config, logging |
| Synthesis + Critic agents | ⬜ Planned | Cross-paper reasoning, self-refinement |
| LangGraph orchestration | ⬜ Planned | Conditional routing, retry logic |
| Memory layer (RAG) | ⬜ Planned | ChromaDB vector store for past queries |
| Evaluation framework | ⬜ Planned | Metrics for summary quality and relevance |
| REST API + UI | ⬜ Planned | FastAPI backend, Streamlit frontend |

---

## Setup

```bash
git clone https://github.com/bhoomikagoel24/agentic-ai-research-system.git
cd agentic-ai-research-system
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```env
GOOGLE_API_KEY=your_key_here
```

Run the pipeline:

```bash
python main.py
```

---

## Why I built this

I'm interested in AI systems that do more than wrap a model — specifically in how multi-agent architectures can handle tasks that require planning, retrieval, reasoning, and self-correction. This project is my attempt to build that end-to-end, grounded in real research retrieval rather than synthetic data.

It reflects the kind of systems I want to work on: agentic, reasoning-first, and production-minded.

---

## Development approach

Each agent is implemented and tested independently in a Jupyter notebook before being converted to a modular Python class. The pipeline is assembled only after individual components are verified — this keeps debugging tractable and makes each component independently reusable.

---

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

---

## Disclaimer

This system is intended for research assistance only. It does not replace expert judgment, peer review, or academic validation.

---

## Author

**Bhoomika Goel**
[bhoomikagoel24@gmail.com](mailto:bhoomikagoel24@gmail.com) · [GitHub](https://github.com/bhoomikagoel24) · [LinkedIn](https://www.linkedin.com/in/bhoomikagoel111/)

---

## License

This project is licensed under the [MIT License](LICENSE).
