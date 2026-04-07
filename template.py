import os

project_name = "research_agent_system"

list_of_files = [

    # ---------------- CORE PACKAGE ---------------- #
    f"{project_name}/__init__.py",

    # ---------------- CONFIG ---------------- #
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/config.py",
    f"{project_name}/config/settings.py",

    # ---------------- AGENTS ---------------- #
    f"{project_name}/agents/__init__.py",
    f"{project_name}/agents/base_agent.py",

    # Core agents
    f"{project_name}/agents/planner_agent.py",         
    f"{project_name}/agents/research_agent.py",
    f"{project_name}/agents/filtering_agent.py",
    f"{project_name}/agents/summarizer_agent.py",
    f"{project_name}/agents/synthesis_agent.py",        
    f"{project_name}/agents/critic_agent.py",           
    f"{project_name}/agents/formatter_agent.py",

    # ---------------- TOOLS ---------------- #
    f"{project_name}/tools/__init__.py",
    f"{project_name}/tools/arxiv_tool.py",
    f"{project_name}/tools/semantic_scholar_tool.py",
    f"{project_name}/tools/search_utils.py",
    f"{project_name}/tools/llm_tool.py",

    # ---------------- WORKFLOWS ---------------- #
    f"{project_name}/workflows/__init__.py",
    f"{project_name}/workflows/research_pipeline.py",
    f"{project_name}/workflows/langgraph_workflow.py",  

    # ---------------- MEMORY ---------------- #
    f"{project_name}/memory/__init__.py",
    f"{project_name}/memory/vector_store.py",
    f"{project_name}/memory/memory_manager.py",

    # ---------------- PROMPTS ---------------- #
    f"{project_name}/prompts/__init__.py",

    f"{project_name}/prompts/planner_prompt.txt",     
    f"{project_name}/prompts/research_prompt.txt",
    f"{project_name}/prompts/filter_prompt.txt",
    f"{project_name}/prompts/summary_prompt.txt",
    f"{project_name}/prompts/synthesis_prompt.txt",     
    f"{project_name}/prompts/critic_prompt.txt",        
    f"{project_name}/prompts/format_prompt.txt",

    # ---------------- SCHEMAS (VERY IMPORTANT) ---------------- #
    f"{project_name}/schemas/__init__.py",
    f"{project_name}/schemas/output_schema.py",         # JSON structure
    f"{project_name}/schemas/agent_schema.py",

    # ---------------- UTILS ---------------- #
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/helpers.py",
    f"{project_name}/utils/logger.py",
    f"{project_name}/utils/validators.py",

    # ---------------- DATA ---------------- #
    f"{project_name}/data/raw_papers.json",
    f"{project_name}/data/filtered_papers.json",
    f"{project_name}/data/final_report.md",

    # ---------------- EVALUATION ---------------- #
    f"{project_name}/evaluation/__init__.py",
    f"{project_name}/evaluation/metrics.py",            
    f"{project_name}/evaluation/benchmark.py",         

    # ---------------- TESTS ---------------- #
    f"{project_name}/tests/__init__.py",
    f"{project_name}/tests/test_pipeline.py",
    f"{project_name}/tests/test_agents.py",            

    # ---------------- API (OPTIONAL BUT STRONG) ---------------- #
    f"{project_name}/api/__init__.py",
    f"{project_name}/api/app.py",                       # FastAPI app

    # ---------------- NOTEBOOKS ---------------- #
    "notebooks/test_llm.ipynb",
    "notebooks/agent_testing.ipynb",                    

    # ---------------- ROOT ---------------- #
    "main.py",
    "requirements.txt",
    ".env",
    ".gitignore",
    "README.md",
]


# ---------------- CREATE FILES ---------------- #

for file_path in list_of_files:
    directory = os.path.dirname(file_path)

    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            pass

print("✅ Project structure created successfully!")