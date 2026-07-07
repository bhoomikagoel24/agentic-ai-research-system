# -*- coding: utf-8 -*-
import warnings
import os
import sys

warnings.filterwarnings("ignore", message=".*Accessing __path__.*")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
import json

# ─────────────────────────────────────────────
# PAGE CONFIG — must be first st call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ARIA — Autonomous Research Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# IMPORTS — after set_page_config
# ─────────────────────────────────────────────
try:
    from research_agent_system.config.config import Config
    from research_agent_system.schemas import PipelineState, SynthesisOutput, ResearchPlan
    from research_agent_system.workflows.research_pipeline import evaluate_pipeline
    from research_agent_system.workflows.langgraph_workflow import build_graph
    from research_agent_system.utils.validators import extract_json
    from research_agent_system.tools.llm_tool import call_llm
    from pydantic import ValidationError
    import hashlib
    BACKEND_AVAILABLE = True
except Exception as e:
    BACKEND_AVAILABLE = False
    BACKEND_ERROR = str(e)

# ─────────────────────────────────────────────
# PDF EXPORT HELPER
# ─────────────────────────────────────────────
def markdown_to_pdf_bytes(markdown_text: str, title: str = "ARIA Research Report") -> bytes:
    """Convert the formatter's markdown report into a downloadable PDF.
    Imports are lazy: if markdown/fpdf2 aren't installed yet, this raises
    inside the function instead of crashing the whole app at import time —
    callers wrap this in their own try/except and just hide the button."""
    import markdown as md_lib
    from fpdf import FPDF

    html = md_lib.markdown(markdown_text, extensions=["extra", "sane_lists"])
    pdf = FPDF()
    pdf.set_title(title)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.write_html(html)
    return bytes(pdf.output())

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050508 !important;
    color: #e8e6f0 !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 0%, rgba(99,102,241,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(6,182,212,0.06) 0%, transparent 60%),
        #050508 !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 2px; }

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(8, 8, 16, 0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
    min-width: 280px !important;
    max-width: 280px !important;
    transform: none !important;
    visibility: visible !important;
}

[data-testid="collapsedControl"] {
    display: none !important;
}
            
[data-testid="stSidebarContent"] {
    padding: 1.5rem 1rem !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #a0a0b8 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 8px !important;
    color: #e8e6f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #4b4b6a !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.25) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.4) !important;
}

/* ── Slider ── */
.stSlider > div > div > div {
    background: rgba(99,102,241,0.3) !important;
}
.stSlider [data-testid="stTickBar"] { display: none !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 8px !important;
    color: #e8e6f0 !important;
}

/* ── Toggle ── */
.stToggle > label { color: #a0a0b8 !important; font-size: 0.82rem !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 8px !important;
    color: #c4c2d4 !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(99,102,241,0.08) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1, #06b6d4) !important;
    border-radius: 4px !important;
}
.stProgress > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 4px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    padding: 4px !important;
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.5rem 1.2rem !important;
    min-width: fit-content !important;
    white-space: nowrap !important;
    color: #7070a0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem !important; }

hr { border-color: rgba(99,102,241,0.12) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def set_agent(name: str, state: str):
    st.session_state.agent_states[name.lower()] = state


def render_hero():
    st.markdown("""
    <div style="padding:3rem 0 2rem 0;text-align:center;">
        <div style="display:inline-block;background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(6,182,212,0.1));border:1px solid rgba(99,102,241,0.25);border-radius:100px;padding:0.3rem 1rem;font-family:monospace;font-size:0.7rem;color:#818cf8;letter-spacing:0.12em;margin-bottom:1.5rem;">
            AUTONOMOUS RESEARCH INTELLIGENCE SYSTEM v2.0 · LANGGRAPH
        </div>
        <br><br>
        <div style="font-family:'Syne',sans-serif;font-size:3.8rem;font-weight:800;background:linear-gradient(135deg,#e8e6f0 30%,#818cf8 70%,#06b6d4 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;letter-spacing:-0.02em;margin-bottom:1rem;">
            ARIA
        </div>
        <div style="font-family:'Inter',sans-serif;font-size:1rem;color:#6b6b8a;max-width:560px;margin:0 auto 0.5rem;line-height:1.6;">
            Multi-agent autonomous system for planning, retrieval, synthesis, critique and grounded research reasoning
        </div>
        <div style="display:flex;justify-content:center;gap:1.5rem;margin-top:1.2rem;flex-wrap:wrap;">
            <span style="font-family:monospace;font-size:0.7rem;color:#4b5563;"><span style="color:#10b981">&#9679;</span> Planner Agent</span>
            <span style="font-family:monospace;font-size:0.7rem;color:#4b5563;"><span style="color:#6366f1">&#9679;</span> Research Agent</span>
            <span style="font-family:monospace;font-size:0.7rem;color:#4b5563;"><span style="color:#06b6d4">&#9679;</span> Synthesis Agent</span>
            <span style="font-family:monospace;font-size:0.7rem;color:#4b5563;"><span style="color:#f59e0b">&#9679;</span> Critic Agent</span>
        </div>
    </div>
        """, unsafe_allow_html=True)


def render_pipeline_flow(agent_states: dict):
    agents = [
        ("Planner",    "#6366f1"),
        ("Research",   "#06b6d4"),
        ("Filter",     "#8b5cf6"),
        ("Summarizer", "#10b981"),
        ("Synthesis",  "#f59e0b"),
        ("Critic",     "#ef4444"),
        ("Formatter",  "#c4c2d4"),
    ]
    icons = {"Planner":"⬡","Research":"◈","Filter":"◎","Summarizer":"◇",
             "Synthesis":"◆","Critic":"◉","Formatter":"✦"}

    cols = st.columns(len(agents))
    for col, (name, color) in zip(cols, agents):
        icon  = icons[name]
        state = agent_states.get(name.lower(), "waiting")

        if state == "running":
            bg = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)"
            border = color; sc = color; si = "..."
        elif state == "done":
            bg = "rgba(16,185,129,0.08)"; border = "#10b981"; sc = "#10b981"; si = "done"
        elif state == "error":
            bg = "rgba(239,68,68,0.08)"; border = "#ef4444"; sc = "#ef4444"; si = "error"
        else:
            bg = "rgba(255,255,255,0.02)"; border = "rgba(255,255,255,0.06)"; sc = "#3a3a5c"; si = "waiting"

        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:10px;
                padding:0.8rem 0.3rem;text-align:center;transition:all 0.3s ease;">
                <div style="font-size:1.2rem;color:{color};margin-bottom:0.3rem">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:600;
                    color:#c4c2d4;letter-spacing:0.05em">{name.upper()}</div>
                <div style="font-family:monospace;font-size:0.6rem;color:{sc};
                    margin-top:0.3rem">{si}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin:0.5rem 0 0.3rem;">
        <div style="width:100%;height:1px;background:linear-gradient(90deg,transparent,
            rgba(99,102,241,0.3),rgba(6,182,212,0.3),transparent);"></div>
    </div>""", unsafe_allow_html=True)


def render_paper_card(paper: dict, idx: int):
    title       = paper.get("title", "Untitled")[:90]
    year        = paper.get("year", "")
    source      = paper.get("source", "unknown").replace("_", " ").title()
    relevance   = float(paper.get("relevance_score", 0))
    recency     = float(paper.get("recency_score", 0))
    authors     = paper.get("authors", [])
    authors_str = ", ".join(str(a) for a in authors[:2]) + (" et al." if len(authors) > 2 else "")
    url         = paper.get("url", "")
    abstract    = paper.get("abstract", "")[:400]
    rel_pct     = int(relevance * 100)
    rel_color   = "#10b981" if rel_pct > 70 else "#f59e0b" if rel_pct > 50 else "#6b6b8a"
    src_color   = "#06b6d4" if "arxiv" in source.lower() else "#8b5cf6"

    with st.expander(f"{'⬡' if idx % 2 == 0 else '◈'}  {title}", expanded=False):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            st.markdown(f"<p style='font-size:0.75rem;color:#6b6b8a;margin:0'>{authors_str}</p>",
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='background:rgba(6,182,212,0.1);border:1px solid {src_color}44;"
                        f"border-radius:4px;padding:0.15rem 0.5rem;font-family:monospace;"
                        f"font-size:0.62rem;color:{src_color};display:inline-block'>{source}</div>",
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f"<p style='font-family:monospace;font-size:0.72rem;color:#818cf8;"
                        f"text-align:center'>{year}</p>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<p style='font-family:monospace;font-size:0.72rem;color:{rel_color};"
                        f"text-align:right'>rel {rel_pct}%</p>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            st.markdown("<p style='font-size:0.7rem;color:#6b6b8a;margin-bottom:2px'>Relevance</p>",
                        unsafe_allow_html=True)
            st.progress(min(max(relevance, 0.0), 1.0))
        with cb:
            st.markdown("<p style='font-size:0.7rem;color:#6b6b8a;margin-bottom:2px'>Recency</p>",
                        unsafe_allow_html=True)
            st.progress(min(recency * 2, 1.0))

        if abstract:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.02);border-left:2px solid rgba(99,102,241,0.3);
                border-radius:0 6px 6px 0;padding:0.7rem 1rem;margin-top:0.7rem;
                font-size:0.78rem;color:#9090b0;line-height:1.6;'>
                {abstract}{"..." if len(paper.get("abstract","")) > 400 else ""}
            </div>""", unsafe_allow_html=True)

        if url:
            st.markdown(f"<a href='{url}' target='_blank' style='font-family:monospace;"
                        f"font-size:0.68rem;color:#6366f1;text-decoration:none;'>View Paper</a>",
                        unsafe_allow_html=True)


def render_summary_card(summary, idx: int):
    s          = summary.model_dump() if hasattr(summary, "model_dump") else (summary or {})
    title      = s.get("source_title", "Unknown")[:70]
    confidence = s.get("confidence", "medium")
    quality    = s.get("quality_score", 5)
    findings   = s.get("key_findings", "")[:300]
    method     = s.get("method", "")[:150]
    limitations= s.get("limitations", "")[:150]
    tradeoff   = s.get("core_tradeoff", "")[:100]
    cc = {"high": "#10b981", "medium": "#f59e0b", "low": "#ef4444"}.get(confidence, "#6b6b8a")

    with st.expander(f"◇  {title}", expanded=False):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.markdown(f"<p style='font-family:monospace;font-size:0.68rem;color:{cc}'>"
                        f"&#9679; {confidence.upper()} CONFIDENCE</p>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<p style='font-family:monospace;font-size:0.68rem;color:#818cf8;"
                        f"text-align:center'>Quality {quality}/10</p>", unsafe_allow_html=True)
        with c3:
            st.progress(quality / 10)

        if method:
            st.markdown(f"""
            <div style='margin-top:0.8rem;'>
                <p style='font-family:monospace;font-size:0.62rem;color:#6366f1;letter-spacing:0.1em;margin-bottom:4px'>METHOD</p>
                <p style='font-size:0.78rem;color:#c4c2d4;line-height:1.5'>{method}</p>
            </div>""", unsafe_allow_html=True)

        if findings:
            st.markdown(f"""
            <div style='margin-top:0.8rem;background:rgba(16,185,129,0.05);border-left:2px solid #10b98140;
                border-radius:0 6px 6px 0;padding:0.7rem 1rem;'>
                <p style='font-family:monospace;font-size:0.62rem;color:#10b981;letter-spacing:0.1em;margin-bottom:4px'>KEY FINDINGS</p>
                <p style='font-size:0.78rem;color:#9090b0;line-height:1.5'>{findings}</p>
            </div>""", unsafe_allow_html=True)

        if limitations:
            st.markdown(f"""
            <div style='margin-top:0.6rem;background:rgba(239,68,68,0.04);border-left:2px solid #ef444430;
                border-radius:0 6px 6px 0;padding:0.7rem 1rem;'>
                <p style='font-family:monospace;font-size:0.62rem;color:#ef4444;letter-spacing:0.1em;margin-bottom:4px'>LIMITATIONS</p>
                <p style='font-size:0.78rem;color:#9090b0;line-height:1.5'>{limitations}</p>
            </div>""", unsafe_allow_html=True)

        if tradeoff:
            st.markdown(f"""
            <div style='margin-top:0.6rem;'>
                <p style='font-family:monospace;font-size:0.62rem;color:#f59e0b;letter-spacing:0.1em;margin-bottom:4px'>CORE TRADEOFF</p>
                <p style='font-size:0.78rem;color:#9090b0;font-style:italic'>{tradeoff}</p>
            </div>""", unsafe_allow_html=True)


def render_synthesis_panel(synthesis):
    s    = synthesis.model_dump() if hasattr(synthesis, "model_dump") else (synthesis or {})
    tabs = st.tabs(["Methods", "Agreements", "Contradictions", "Gaps", "Trends", "Future"])

    with tabs[0]:
        st.markdown("<p style='font-size:1rem;font-weight:700;color:#818cf8;"
                    "margin-bottom:1rem'>Method Comparisons</p>", unsafe_allow_html=True)
        
        comparisons = s.get("method_comparisons", [])
        if comparisons:
            for cmp in comparisons:
                # Sanitize — HTML characters escape karo
                comparison     = str(cmp.get("comparison","")).replace("<","&lt;").replace(">","&gt;")
                evidence       = str(cmp.get("evidence_reasoning",""))[:300].replace("<","&lt;").replace(">","&gt;")
                tradeoff_text  = str(cmp.get("core_tradeoff",""))[:80].replace("<","&lt;").replace(">","&gt;")
                
                st.markdown(f"""
                <div style='background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.2);
                    border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem;'>
                    <p style='font-weight:600;color:#c4c2d4;margin-bottom:0.5rem'>{comparison}</p>
                    <p style='font-size:0.76rem;color:#7070a0;line-height:1.5;margin-bottom:0.5rem'>{evidence}</p>
                    <div style='display:inline-block;background:rgba(245,158,11,0.1);
                        border:1px solid rgba(245,158,11,0.3);border-radius:4px;
                        padding:0.2rem 0.6rem;font-family:monospace;font-size:0.62rem;color:#f59e0b;'>
                        {tradeoff_text}
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#3a3a5c;font-size:0.8rem'>No comparisons extracted.</p>",
                        unsafe_allow_html=True)

        st.markdown("<p style='font-size:1rem;font-weight:700;color:#818cf8;"
                    "margin:1.2rem 0 0.8rem'>Common Methods</p>", unsafe_allow_html=True)
        methods = s.get("common_methods", [])
        if methods:
            cols = st.columns(min(len(methods), 3))
            for i, m in enumerate(methods):
                with cols[i % 3]:
                    method_name = str(m.get("method","")).replace("<","&lt;").replace(">","&gt;")
                    papers      = m.get("source_papers", [])
                    pstr        = ", ".join([
                        str(p.get("title","") if isinstance(p,dict) else p)[:25]
                        for p in papers[:2]
                    ]).replace("<","&lt;").replace(">","&gt;")
                    st.markdown(f"""
                    <div style='background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                        border-radius:8px;padding:0.8rem;margin-bottom:0.6rem;'>
                        <p style='font-size:0.82rem;font-weight:600;color:#a5b4fc'>{method_name}</p>
                        <p style='font-size:0.65rem;color:#505070;margin-top:0.3rem'>{pstr}</p>
                    </div>""", unsafe_allow_html=True)

    with tabs[1]:
        agreements = s.get("agreements", [])
        if agreements:
            for ag in agreements:
                insight   = str(ag.get("insight","")).replace("<","&lt;").replace(">","&gt;")
                why       = str(ag.get("why_supported",""))[:200].replace("<","&lt;").replace(">","&gt;")
                st.markdown(f"""
                <div style='background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.2);
                    border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem;'>
                    <p style='font-family:monospace;font-size:0.62rem;color:#10b981;
                        letter-spacing:0.1em;margin-bottom:0.4rem'>AGREEMENT</p>
                    <p style='font-weight:600;color:#c4c2d4;margin-bottom:0.4rem'>{insight}</p>
                    <p style='font-size:0.74rem;color:#10b98190;line-height:1.5'>{why}</p>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#3a3a5c;font-size:0.8rem'>No consensus patterns found.</p>",
                        unsafe_allow_html=True)

    with tabs[2]:
        contradictions = s.get("contradictions", [])
        if contradictions:
            for ct in contradictions:
                issue   = str(ct.get("issue","")).replace("<","&lt;").replace(">","&gt;")
                reason  = str(ct.get("reason_for_conflict",""))[:200].replace("<","&lt;").replace(">","&gt;")
                st.markdown(f"""
                <div style='background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.2);
                    border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem;'>
                    <p style='font-family:monospace;font-size:0.62rem;color:#ef4444;
                        letter-spacing:0.1em;margin-bottom:0.4rem'>CONTRADICTION</p>
                    <p style='font-weight:600;color:#c4c2d4;margin-bottom:0.4rem'>{issue}</p>
                    <p style='font-size:0.74rem;color:#ef444490;line-height:1.5'>{reason}</p>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#3a3a5c;font-size:0.8rem'>No contradictions detected.</p>",
                        unsafe_allow_html=True)

    with tabs[3]:
        gaps = s.get("research_gaps", [])
        if gaps:
            for gap in gaps:
                gap_text  = str(gap.get("gap","")).replace("<","&lt;").replace(">","&gt;")
                why_imp   = str(gap.get("why_important",""))[:200].replace("<","&lt;").replace(">","&gt;")
                st.markdown(f"""
                <div style='background:rgba(245,158,11,0.05);border:1px solid rgba(245,158,11,0.2);
                    border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem;'>
                    <p style='font-family:monospace;font-size:0.62rem;color:#f59e0b;
                        letter-spacing:0.1em;margin-bottom:0.4rem'>GAP IDENTIFIED</p>
                    <p style='font-weight:600;color:#c4c2d4;margin-bottom:0.4rem'>{gap_text}</p>
                    <p style='font-size:0.74rem;color:#7070a0;line-height:1.5'>{why_imp}</p>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#3a3a5c;font-size:0.8rem'>No gaps identified.</p>",
                        unsafe_allow_html=True)

    with tabs[4]:
        trends = s.get("emerging_trends", [])
        if trends:
            for tr in trends:
                trend_text = str(tr.get("trend","")).replace("<","&lt;").replace(">","&gt;")
                why_em     = str(tr.get("why_emerging",""))[:200].replace("<","&lt;").replace(">","&gt;")
                st.markdown(f"""
                <div style='background:rgba(6,182,212,0.05);border:1px solid rgba(6,182,212,0.2);
                    border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem;'>
                    <p style='font-family:monospace;font-size:0.62rem;color:#06b6d4;
                        letter-spacing:0.1em;margin-bottom:0.4rem'>EMERGING TREND</p>
                    <p style='font-weight:600;color:#c4c2d4;margin-bottom:0.4rem'>{trend_text}</p>
                    <p style='font-size:0.74rem;color:#7070a0;line-height:1.5'>{why_em}</p>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#3a3a5c;font-size:0.8rem'>No trends extracted.</p>",
                        unsafe_allow_html=True)

    with tabs[5]:
        directions = s.get("future_directions", [])
        if directions:
            for i, fd in enumerate(directions):
                direction  = str(fd.get("direction","")).replace("<","&lt;").replace(">","&gt;")
                motivation = str(fd.get("motivation",""))[:200].replace("<","&lt;").replace(">","&gt;")
                st.markdown(f"""
                <div style='display:flex;gap:1rem;align-items:flex-start;
                    background:rgba(139,92,246,0.05);border:1px solid rgba(139,92,246,0.15);
                    border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem;'>
                    <div style='flex-shrink:0;width:26px;height:26px;background:rgba(139,92,246,0.2);
                        border-radius:50%;display:flex;align-items:center;justify-content:center;
                        font-size:0.72rem;font-weight:700;color:#8b5cf6;'>{i+1:02d}</div>
                    <div>
                        <p style='font-weight:600;color:#c4c2d4;margin-bottom:0.3rem'>{direction}</p>
                        <p style='font-size:0.74rem;color:#7070a0;line-height:1.5'>{motivation}</p>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#3a3a5c;font-size:0.8rem'>No future directions identified.</p>",
                        unsafe_allow_html=True)

    final = s.get("final_insight", "")
    if final:
        final_safe = str(final).replace("<","&lt;").replace(">","&gt;")
        st.markdown(f"""
                <div style='background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(6,182,212,0.05));
                    border:1px solid rgba(99,102,241,0.25);border-radius:12px;
                    padding:1.2rem 1.5rem;margin-top:1rem;'>
                    <p style='font-family:monospace;font-size:0.62rem;color:#818cf8;
                        letter-spacing:0.12em;margin-bottom:0.6rem'>SYNTHESIS CONCLUSION</p>
                    <p style='font-size:0.88rem;color:#c4c2d4;line-height:1.7;
                        font-style:italic'>{final_safe}</p>
                </div>""", unsafe_allow_html=True)


def render_critic_panel(critique: dict):
    if not critique:
        st.markdown("<p style='color:#3a3a5c;font-size:0.8rem'>Critic not run.</p>",
                    unsafe_allow_html=True)
        return

    c1, c2, c3, c4 = st.columns(4)
    for col, (label, score, color) in zip([c1,c2,c3,c4], [
        ("Reasoning Depth", critique.get("reasoning_depth_score", 0), "#6366f1"),
        ("Grounding",       critique.get("grounding_score", 0),       "#10b981"),
        ("Trend Quality",   critique.get("trend_quality_score", 0),   "#06b6d4"),
        ("Gap Analysis",    critique.get("gap_analysis_score", 0),    "#8b5cf6"),
    ]):
        with col:
            st.markdown(f"""
<div style='background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
    border-radius:10px;padding:1rem;text-align:center;'>
    <div style='font-size:1.8rem;font-weight:800;color:{color};line-height:1;'>
        {score}<span style='font-size:0.85rem;color:#3a3a5c'>/10</span>
    </div>
    <div style='font-size:0.65rem;color:#6b6b8a;margin-top:0.4rem;letter-spacing:0.06em'>
        {label.upper()}
    </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    risk     = critique.get("hallucination_risk", "medium")
    quality  = critique.get("overall_quality", "medium")
    rc = {"low":"#10b981","medium":"#f59e0b","high":"#ef4444"}.get(risk,"#6b6b8a")
    qc = {"high":"#10b981","medium":"#f59e0b","low":"#ef4444"}.get(quality,"#6b6b8a")

    cr, cq = st.columns(2)
    with cr:
        st.markdown(f"""
<div style='background:rgba(255,255,255,0.02);border:1px solid {rc}30;border-radius:8px;padding:0.8rem 1rem;'>
    <p style='font-family:monospace;font-size:0.62rem;color:#6b6b8a;letter-spacing:0.1em'>HALLUCINATION RISK</p>
    <p style='font-size:1.1rem;font-weight:700;color:{rc};margin-top:0.2rem'>{risk.upper()}</p>
</div>""", unsafe_allow_html=True)
    with cq:
        st.markdown(f"""
<div style='background:rgba(255,255,255,0.02);border:1px solid {qc}30;border-radius:8px;padding:0.8rem 1rem;'>
    <p style='font-family:monospace;font-size:0.62rem;color:#6b6b8a;letter-spacing:0.1em'>OVERALL QUALITY</p>
    <p style='font-size:1.1rem;font-weight:700;color:{qc};margin-top:0.2rem'>{quality.upper()}</p>
</div>""", unsafe_allow_html=True)

    for items, label, color, icon in [
        (critique.get("major_weaknesses",[]),       "Major Weaknesses",       "#ef4444", "&#10007;"),
        (critique.get("improvement_suggestions",[]),"Improvement Suggestions","#10b981", "&#8594;"),
    ]:
        if items:
            st.markdown(f"<p style='font-size:0.9rem;font-weight:600;color:{color};"
                        f"margin:1rem 0 0.5rem'>{label}</p>", unsafe_allow_html=True)
            for item in items:
                st.markdown(f"""
<div style='display:flex;gap:0.6rem;align-items:flex-start;margin-bottom:0.4rem;'>
    <span style='color:{color};font-size:0.8rem;flex-shrink:0;margin-top:2px'>{icon}</span>
    <p style='font-size:0.78rem;color:#9090b0;line-height:1.5;margin:0'>{item}</p>
</div>""", unsafe_allow_html=True)

    verdict = critique.get("final_verdict", "")
    if verdict:
        st.markdown(f"""
<div style='background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.2);
    border-radius:10px;padding:1rem 1.2rem;margin-top:1rem;'>
    <p style='font-family:monospace;font-size:0.62rem;color:#818cf8;letter-spacing:0.12em;margin-bottom:0.5rem'>
        FINAL VERDICT</p>
    <p style='font-size:0.8rem;color:#c4c2d4;line-height:1.6'>{verdict}</p>
</div>""", unsafe_allow_html=True)


def render_metrics_bar(metrics: dict):

    items = [
        ("Papers Retrieved", metrics.get("total_papers", 0), "◈", "#06b6d4"),
        ("Summaries", metrics.get("total_summaries", 0), "◇", "#10b981"),
        ("Avg Relevance", f"{metrics.get('avg_relevance_score', 0):.2f}", "◎", "#818cf8"),
        ("Avg Quality", f"{metrics.get('avg_summary_quality', 0):.1f}/10", "◆", "#f59e0b"),
        ("High Confidence", metrics.get("high_confidence_summaries", 0), "●", "#10b981"),
        ("Status", "Complete", "✦", "#6366f1"),
    ]
    cards_html = "".join([f"""
<div style='text-align:center;background:rgba(255,255,255,0.02);
    border:1px solid rgba(99,102,241,0.1);border-radius:10px;padding:0.8rem 0.3rem;'>
    <div style='font-size:1.3rem;font-weight:700;color:{color}'>{value}</div>
    <div style='font-size:0.6rem;color:#4b4b6a;letter-spacing:0.08em;
        text-transform:uppercase;margin-top:0.2rem'>{label}</div>
</div>""" for label, value, icon, color in items])

    st.markdown(
        "<div style='display:grid;grid-template-columns:repeat(6,1fr);gap:0.8rem;"
        "margin-bottom:1.5rem;'>" + cards_html + "</div>",
        unsafe_allow_html=True
    )

    # cols = st.columns(len(items))

    # for col, (label, value, icon, color) in zip(cols, items):

    #     with col:

    #         st.markdown(f"""
    #         <div style='text-align:center;'>
    #             <div style='font-family:Syne;font-size:1.3rem;
    #                 font-weight:700;color:{color}'>
    #                 {icon} {value}
    #             </div>

    #             <div style='font-family:Inter;
    #                 font-size:0.65rem;
    #                 color:#4b4b6a;
    #                 letter-spacing:0.08em;
    #                 text-transform:uppercase;
    #                 margin-top:0.1rem'>
    #                 {label}
    #             </div>
    #         </div>
    #         """, unsafe_allow_html=True)

def section_header(icon: str, title: str, subtitle: str = ""):
    sub = (f"<p style='font-size:0.75rem;color:#4b4b6a;margin:0.3rem 0 0 1.7rem'>{subtitle}</p>"
           if subtitle else "")
    st.markdown(f"""
<div style='margin:2rem 0 1rem;'>
    <div style='display:flex;align-items:center;gap:0.6rem;'>
        <span style='font-size:1.1rem;color:#6366f1'>{icon}</span>
        <span style='font-size:1.1rem;font-weight:700;color:#e8e6f0'>{title}</span>
    </div>
    {sub}
    <div style='height:1px;background:rgba(99,102,241,0.15);margin-top:0.8rem;'></div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style='padding:0.5rem 0 1.5rem;border-bottom:1px solid rgba(99,102,241,0.15);margin-bottom:1.2rem;'>
    <div style='font-family:"Syne",sans-serif;font-size:1.4rem;font-weight:800;color:#818cf8;'>ARIA</div>
    <div style='font-family:monospace;font-size:0.6rem;color:#3a3a5c;letter-spacing:0.1em;margin-top:2px;'>
        RESEARCH INTELLIGENCE
    </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<p style='font-family:monospace;font-size:0.62rem;color:#4b4b6a;"
                "letter-spacing:0.1em;margin-bottom:0.4rem;'>RESEARCH TOPIC</p>",
                unsafe_allow_html=True)
    topic = st.text_input("topic", placeholder="e.g. GenAI for stock prediction",
                           label_visibility="collapsed")

    st.markdown("<p style='font-family:monospace;font-size:0.62rem;color:#4b4b6a;"
                "letter-spacing:0.1em;margin:1rem 0 0.4rem;'>RETRIEVAL</p>",
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        top_k      = st.slider("Top-K", 3, 40, 12, label_visibility="visible",
                                help="Final number of papers kept for the report. "
                                     "Was capped at 10 by a hardcoded limit elsewhere "
                                     "in the pipeline — that's fixed now.")
    with col2:
        min_papers = st.slider("Min", 3, 15, 8, label_visibility="visible", disabled=True,
                                help="Not yet wired into the v2 graph — has no effect "
                                     "right now. This used to do something in v1; ask "
                                     "to have it rebuilt as a real 'keep searching "
                                     "until N papers are found' loop.")

    st.markdown("<p style='font-family:monospace;font-size:0.62rem;color:#4b4b6a;"
                "letter-spacing:0.1em;margin:1rem 0 0.4rem;'>MODEL</p>",
                unsafe_allow_html=True)
    primary_model = st.selectbox("model", ["gemini-2.5-flash", "gemini-2.0-flash"],
                                  label_visibility="collapsed")

    st.markdown("<p style='font-family:monospace;font-size:0.62rem;color:#4b4b6a;"
                "letter-spacing:0.1em;margin:1rem 0 0.4rem;'>AGENTS</p>",
                unsafe_allow_html=True)
    enable_critic    = st.toggle("Critic Agent",    value=True, disabled=True,
                                  help="v2 graph: critic now drives a real refine-loop "
                                       "(critic -> synthesis, capped retries), so it's "
                                       "load-bearing rather than optional.")
    enable_formatter = st.toggle("Formatter Agent", value=True, disabled=True,
                                  help="v2 graph: formatter is the graph's terminal node "
                                       "(-> END), always runs.")
    dev_mode         = st.toggle("Dev Mode",        value=False)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    run_btn = st.button("Launch Pipeline", use_container_width=True)

    st.markdown("""
<div style='margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(99,102,241,0.1);'>
    <p style='font-family:monospace;font-size:0.58rem;color:#2a2a4a;line-height:1.8;'>
        ARIA v2.0 · LangGraph StateGraph<br>
        Gemini + Groq Fallback<br>
        arXiv + Semantic Scholar<br>
        Pydantic · Semantic Embeddings
    </p>
</div>""", unsafe_allow_html=True)

    if not BACKEND_AVAILABLE:
        st.markdown(f"""
<div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);
    border-radius:6px;padding:0.6rem;margin-top:1rem;'>
    <p style='font-family:monospace;font-size:0.6rem;color:#ef4444;'>
    Backend error:<br>{BACKEND_ERROR[:120]}</p>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "pipeline_done"  not in st.session_state: st.session_state.pipeline_done  = False
if "agent_states"   not in st.session_state: st.session_state.agent_states   = {}
if "pipeline_state" not in st.session_state: st.session_state.pipeline_state = None
if "metrics"        not in st.session_state: st.session_state.metrics        = {}

if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False

if "pipeline_topic" not in st.session_state:
    st.session_state.pipeline_topic = ""

if "pipeline_cfg" not in st.session_state:
    st.session_state.pipeline_cfg = {}

if "pipeline_step" not in st.session_state:
    st.session_state.pipeline_step = ""
# ─────────────────────────────────────────────
# HERO + FLOW
# ─────────────────────────────────────────────
render_hero()
render_pipeline_flow(st.session_state.agent_states)
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────
if run_btn and topic.strip() and BACKEND_AVAILABLE:
    st.session_state.pipeline_running = True
    st.session_state.pipeline_step    = "planner"
    st.session_state.pipeline_topic   = topic
    st.session_state.pipeline_cfg     = {
        "dev_mode":   dev_mode,
        "topk":       top_k,
        "min_papers": min_papers,
        "enable_critic": enable_critic,
        "enable_formatter": enable_formatter,
    }
    st.rerun()

if st.session_state.get("pipeline_running"):
    topic_run = st.session_state.pipeline_topic
    cfg_dict  = st.session_state.pipeline_cfg

    status_box = st.empty()

    def show_status(msg):
        status_box.markdown(f"""
<div style='background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);
    border-radius:8px;padding:0.6rem 1rem;'>
    <p style='font-family:monospace;font-size:0.75rem;color:#818cf8;margin:0'>{msg}</p>
</div>""", unsafe_allow_html=True)

    try:
        cfg = Config(
            dev_mode=cfg_dict["dev_mode"],
            topk=cfg_dict["topk"],
            min_papers=cfg_dict["min_papers"],
        )

        # v2: build_graph(cfg) bakes cfg into the research/summarizer node
        # closures the same way v1 passed cfg into each Agent's __init__.
        graph = build_graph(cfg)

        # thread_id ties this run to a LangGraph checkpoint. Same topic ->
        # same id, so if the process dies mid-run, clicking "Launch
        # Pipeline" again on the same topic resumes from the last
        # completed node instead of starting over. (MemorySaver only
        # lives as long as this process does — swap it for
        # langgraph.checkpoint.sqlite.SqliteSaver in langgraph_workflow.py
        # if you want that resumability to survive a full app restart.)
        thread_id = hashlib.md5(topic_run.encode("utf-8")).hexdigest()
        run_config = {"configurable": {"thread_id": thread_id}}

        # Sequential hand-off map, used only to *optimistically* light up
        # the next box as "running" in the UI. stream_mode="updates" only
        # tells you when a node FINISHED, not when the next one started —
        # LangGraph doesn't push a separate "node started" event, so this
        # is a cheap approximation, not something the graph guarantees.
        NEXT_NODE = {
            "planner": "research", "research": "summarizer",
            "summarizer": "synthesis", "synthesis": "critic",
        }

        accumulated_state: dict = {"topic": topic_run}
        set_agent("planner", "running")

        # stream_mode="updates" yields {node_name: partial_state_dict}
        # right after each node returns — this is what main_langgraph.py's
        # CLI version prints to console; here the same stream drives the
        # Streamlit boxes instead. v1's on_step was hand-rolled to fake
        # this; LangGraph gives it natively.
        for update in graph.stream({"topic": topic_run}, config=run_config, stream_mode="updates"):
            for node_name, partial in update.items():
                accumulated_state.update(partial)
                set_agent(node_name, "done")

                if node_name == "research":
                    # v1 had a standalone "Filter" box; v2 does
                    # dedup/filter/rank inline inside research_node, so
                    # there's no separate node to wait on — mark it done
                    # alongside research instead of leaving it stuck on
                    # "waiting" forever.
                    set_agent("filter", "done")

                retries = accumulated_state.get("critique_retries", 0)
                if node_name == "synthesis" and retries:
                    label = f"SYNTHESIS — refining (pass {retries + 1}, addressing critic feedback)"
                elif node_name == "critic" and accumulated_state.get("critique", {}).get("major_weaknesses"):
                    label = f"CRITIC — found weaknesses, looping back to synthesis"
                else:
                    label = f"{node_name.upper()} — done"
                show_status(label)

                next_node = NEXT_NODE.get(node_name)
                if next_node:
                    set_agent(next_node, "running")

        # The rest of this file (tabs, render_paper_card, evaluate_pipeline,
        # etc.) was all written against PipelineState's attribute access
        # (state.papers, state.synthesis, ...). graph_state.ResearchState
        # uses the identical field names/types as a dict instead — so
        # rebuilding a PipelineState from the final dict means none of
        # that rendering code below needs to change at all.
        state = PipelineState(
            topic=accumulated_state.get("topic", topic_run),
            plan=accumulated_state.get("plan"),
            papers=accumulated_state.get("papers", []),
            summaries=accumulated_state.get("summaries", []),
            synthesis=accumulated_state.get("synthesis"),
            critique=accumulated_state.get("critique", {}),
            final_report=accumulated_state.get("final_report", ""),
        )

        for a in ["planner","research","filter","summarizer","synthesis","critic","formatter"]:
            set_agent(a, "done")

        st.session_state.pipeline_state   = state
        st.session_state.metrics          = evaluate_pipeline(state)
        st.session_state.pipeline_done    = True
        st.session_state.pipeline_running = False
        status_box.empty()
        st.rerun()

    except Exception as e:
        status_box.empty()
        st.session_state.pipeline_running = False
        for a in ["planner","research","filter","summarizer","synthesis","critic","formatter"]:
            if st.session_state.agent_states.get(a) == "running":
                set_agent(a, "error")
        st.error(f"Pipeline error: {str(e)[:400]}")

# ─────────────────────────────────────────────
# AUTO-LOAD state.json
# ─────────────────────────────────────────────
# if not st.session_state.pipeline_done and os.path.exists("state.json"):
#     try:
#         with open("state.json", "r", encoding="utf-8") as f:
#             raw = json.load(f)
#         if BACKEND_AVAILABLE:
#             loaded = PipelineState(**raw)
#             st.session_state.pipeline_state = loaded
#             st.session_state.metrics        = evaluate_pipeline(loaded)
#             for a in ["planner","research","filter","summarizer","synthesis","critic","formatter"]:
#                 set_agent(a, "done")
#             st.session_state.pipeline_done = True
#             st.rerun()
#     except Exception:
#         pass

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if st.session_state.pipeline_done and st.session_state.pipeline_state:
    try:
        state = st.session_state.pipeline_state

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        render_metrics_bar(st.session_state.metrics)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        tabs = st.tabs(["Papers", "Summaries", "Synthesis", "Critic", "Report", "Plan"])

        with tabs[0]:
            section_header("◈", "Retrieved Papers",
                           f"{len(state.papers)} papers ranked by semantic relevance + recency")
            for i, p in enumerate(state.papers):
                render_paper_card(p, i)

        with tabs[1]:
            section_header("◇", "Structured Summaries",
                           "AI-extracted findings, methods, limitations per paper")
            for i, s in enumerate(state.summaries):
                render_summary_card(s, i)

        with tabs[2]:
            section_header("◆", "Synthesis Intelligence",
                           "Cross-paper reasoning — patterns, gaps, trends, contradictions")
            render_synthesis_panel(state.synthesis)

        with tabs[3]:
            section_header("◉", "Reliability Observatory",
                           "AI self-critique — hallucination risk, grounding, quality scores")
            render_critic_panel(state.critique if isinstance(state.critique, dict) else {})

        with tabs[4]:
            section_header("✦", "Final Research Report",
                           "Formatted literature review generated from synthesis")
            report = state.final_report if isinstance(state.final_report, str) else ""
            if report:
                st.markdown(f"""
<div style='background:rgba(255,255,255,0.02);border:1px solid rgba(99,102,241,0.1);
    border-radius:12px;padding:2rem 2.5rem;font-size:0.85rem;color:#c4c2d4;line-height:1.8;'>
    {report.replace(chr(10),"<br>")}
</div>""", unsafe_allow_html=True)
                st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
                st.download_button(
                    "Download Report",
                    data=report,
                    file_name=f"aria_{state.topic[:25].replace(' ','_')}.md",
                    mime="text/markdown"
                )
                try:
                    pdf_bytes = markdown_to_pdf_bytes(report, title=f"ARIA Report — {state.topic[:60]}")
                    st.download_button(
                        "Download as PDF",
                        data=pdf_bytes,
                        file_name=f"aria_{state.topic[:25].replace(' ','_')}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.caption(f"PDF export unavailable ({e}) — run `pip install markdown fpdf2`.")
            else:
                st.markdown("<p style='color:#3a3a5c;font-size:0.8rem'>Report not generated.</p>",
                            unsafe_allow_html=True)

        with tabs[5]:
            section_header("⬡", "Research Plan", "Generated sub-questions and search queries")
            plan_dict = state.plan.model_dump() if hasattr(state.plan,"model_dump") else {}

            sub_qs  = plan_dict.get("sub_questions", [])
            queries = plan_dict.get("search_queries", [])

            if sub_qs:
                st.markdown("<p style='font-size:0.9rem;font-weight:600;color:#818cf8;"
                            "margin-bottom:0.6rem'>Sub-Questions</p>", unsafe_allow_html=True)
                for i, q in enumerate(sub_qs):
                    st.markdown(f"""
<div style='display:flex;gap:0.8rem;align-items:flex-start;margin-bottom:0.5rem;'>
    <span style='font-family:monospace;font-size:0.68rem;color:#3a3a5c;
        flex-shrink:0;margin-top:3px'>Q{i+1:02d}</span>
    <p style='font-size:0.82rem;color:#9090b0;line-height:1.5;margin:0'>{q}</p>
</div>""", unsafe_allow_html=True)

            if queries:
                st.markdown("<p style='font-size:0.9rem;font-weight:600;color:#06b6d4;"
                            "margin:1rem 0 0.6rem'>Search Queries</p>", unsafe_allow_html=True)
                for q in queries:
                    st.markdown(f"""
<div style='background:rgba(6,182,212,0.05);border:1px solid rgba(6,182,212,0.15);
    border-radius:6px;padding:0.5rem 0.8rem;margin-bottom:0.4rem;
    font-family:monospace;font-size:0.7rem;color:#06b6d4;'>{q}</div>""",
                                unsafe_allow_html=True)

    except Exception as e:
        st.error(
            f"Display error: {e}\n\n"
            "The AI pipeline itself finished successfully — this error is only in "
            "how results are shown. Your report is recovered below."
        )
        fallback_state = st.session_state.get("pipeline_state")
        if fallback_state is not None:
            raw_report = fallback_state.final_report if isinstance(
                getattr(fallback_state, "final_report", None), str
            ) else ""
            if raw_report:
                st.markdown("### Final Report (raw fallback view)")
                st.text(raw_report)
                st.download_button(
                    "Download Report (.md)",
                    data=raw_report,
                    file_name="aria_report_fallback.md",
                    mime="text/markdown",
                    key="fallback_md_dl"
                )
                try:
                    pdf_bytes = markdown_to_pdf_bytes(raw_report, title="ARIA Report")
                    st.download_button(
                        "Download Report (.pdf)",
                        data=pdf_bytes,
                        file_name="aria_report_fallback.pdf",
                        mime="application/pdf",
                        key="fallback_pdf_dl"
                    )
                except Exception:
                    pass
            else:
                st.warning("No report was recovered. Check the terminal logs for the underlying error.")
else:
    st.markdown("""
<div style='text-align:center;padding:4rem 2rem;border:1px dashed rgba(99,102,241,0.15);
    border-radius:16px;margin:2rem 0;'>
    <div style='font-size:3rem;margin-bottom:1rem;opacity:0.2;'>&#11201;</div>
    <p style='font-size:1rem;font-weight:600;color:#3a3a5c;margin-bottom:0.5rem;'>
        No pipeline run yet</p>
    <p style='font-size:0.78rem;color:#2a2a4a;line-height:1.6;max-width:400px;margin:0 auto;'>
        Enter a research topic in the sidebar and click Launch Pipeline.<br>
        Launch the pipeline to begin autonomous research analysis.
    </p>
</div>""", unsafe_allow_html=True)