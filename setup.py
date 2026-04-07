from setuptools import setup, find_packages
from pathlib import Path

# ---------------- BASE DIR ---------------- #

BASE_DIR = Path(__file__).resolve().parent

# ---------------- READ FILES SAFELY ---------------- #

def read_file(file_name, default=""):
    try:
        return (BASE_DIR / file_name).read_text(encoding="utf-8")
    except FileNotFoundError:
        return default

long_description = read_file("README.md", "Agentic AI Research System")
requirements = read_file("requirements.txt").splitlines()

# ---------------- METADATA ---------------- #

REPO_NAME = "agentic-ai-research-system"
PACKAGE_NAME = "research_agent_system"

setup(
    name=REPO_NAME,
    version="0.1.0",

    author="Bhoomika Goel",
    author_email="bhoomikagoel24@gmail.com",

    description="Agentic AI multi-agent system for research paper discovery, filtering, synthesis, and structured report generation",

    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/bhoomikagoel24/agentic-ai-research-system",

    project_urls={
        "Source": "https://github.com/bhoomikagoel24/agentic-ai-research-system",
        "Bug Tracker": "https://github.com/bhoomikagoel24/agentic-ai-research-system/issues",
    },

    packages=find_packages(include=[PACKAGE_NAME, f"{PACKAGE_NAME}.*"]),

    install_requires=[req for req in requirements if req and not req.startswith("#")],

    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "ipykernel"
        ],
    },

    python_requires=">=3.8",

    license="MIT",

    keywords=[
        "AI Agents",
        "Agentic AI",
        "Multi-Agent Systems",
        "Research Automation",
        "LangGraph",
        "RAG",
        "LLM",
        "Autonomous Agents",
    ],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],

    include_package_data=True,
    zip_safe=False,

    # ---------------- CLI ENTRY ---------------- #
    entry_points={
        "console_scripts": [
            "research-agent=main:main",
        ],
    },
)