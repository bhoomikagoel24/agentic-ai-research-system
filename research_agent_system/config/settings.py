import os
from dotenv import load_dotenv


# LOAD ENV
load_dotenv(override=True)

# API KEYS
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY","")
GROQ_API_KEY = os.getenv("GROQ_API_KEY","")

# DIRECTORIES
CACHE_DIR = "cache"
DATA_DIR = "data"

# VALIDATION
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found")