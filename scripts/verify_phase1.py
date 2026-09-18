"""
Phase 1 Verification Script.

This script runs automated sanity checks to verify:
1. All critical libraries (FastAPI, LangChain, LangGraph, Google GenAI, psycopg, pgvector) are installed.
2. Configuration loads correctly via Pydantic Settings.
3. Gemini model names match current Google API specifications.
4. Security files (.gitignore, .env.example) are properly configured.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def check_dependencies() -> bool:
    print("\n--- 1. Testing Dependency Imports ---")
    dependencies = [
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("Pydantic", "pydantic"),
        ("Pydantic Settings", "pydantic_settings"),
        ("Python Dotenv", "dotenv"),
        ("LangChain", "langchain"),
        ("LangChain Core", "langchain_core"),
        ("LangChain Google GenAI", "langchain_google_genai"),
        ("LangGraph", "langgraph"),
        ("Psycopg 3", "psycopg"),
        ("pgvector", "pgvector"),
        ("HTTPX", "httpx"),
        ("Pytest", "pytest"),
    ]

    all_passed = True
    for name, module_name in dependencies:
        try:
            __import__(module_name)
            print(f"  [OK] {name} ({module_name}) imported successfully.")
        except ImportError as e:
            print(f"  [FAILED] {name} ({module_name}): {e}")
            all_passed = False

    return all_passed


def check_configuration() -> bool:
    print("\n--- 2. Testing Configuration & Model Verification ---")
    try:
        from app.core.config import settings

        print(f"  [OK] Settings class loaded successfully.")
        print(f"       Active Gemini Model:           {settings.gemini_model}")
        print(f"       Active Gemini Embedding Model: {settings.gemini_embedding_model}")
        print(f"       Environment:                  {settings.environment}")

        # Validate that deprecated models are NOT being used
        deprecated_models = ["gemini-2.0-flash", "gemini-1.0-pro", "gemini-pro"]
        deprecated_embeddings = ["text-embedding-004", "embedding-001"]

        if settings.gemini_model in deprecated_models:
            print(f"  [FAILED] Configured gemini_model '{settings.gemini_model}' is deprecated by Google!")
            return False
        else:
            print(f"  [OK] Generation model '{settings.gemini_model}' is currently supported.")

        if settings.gemini_embedding_model in deprecated_embeddings:
            print(f"  [FAILED] Configured gemini_embedding_model '{settings.gemini_embedding_model}' is retired by Google!")
            return False
        else:
            print(f"  [OK] Embedding model '{settings.gemini_embedding_model}' is currently supported.")

        # Check credentials status
        if settings.is_gemini_configured:
            print("  [OK] GEMINI_API_KEY is detected.")
        else:
            print("  [INFO] GEMINI_API_KEY is not set yet (expected before user populates .env).")

        if settings.is_database_configured:
            print("  [OK] DATABASE_URL is detected.")
        else:
            print("  [INFO] DATABASE_URL is not set yet (expected before user populates .env).")

        return True
    except Exception as e:
        print(f"  [FAILED] Error loading settings: {e}")
        return False


def check_security_files() -> bool:
    print("\n--- 3. Testing Project Security Configuration ---")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    gitignore_path = os.path.join(project_root, ".gitignore")
    env_example_path = os.path.join(project_root, ".env.example")

    all_passed = True
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            content = f.read()
        if ".env" in content and ".venv" in content:
            print("  [OK] .gitignore exists and protects .env and .venv.")
        else:
            print("  [FAILED] .gitignore missing .env or .venv rules.")
            all_passed = False
    else:
        print("  [FAILED] .gitignore not found.")
        all_passed = False

    if os.path.exists(env_example_path):
        print("  [OK] .env.example template exists.")
    else:
        print("  [FAILED] .env.example not found.")
        all_passed = False

    return all_passed


def main():
    print("==================================================")
    print("Darukaa.Earth Biodiversity Chatbot - Phase 1 Check")
    print("==================================================")

    step1 = check_dependencies()
    step2 = check_configuration()
    step3 = check_security_files()

    print("\n----------------- Summary -----------------")
    if step1 and step2 and step3:
        print("[SUCCESS] All Phase 1 verification checks passed!")
        sys.exit(0)
    else:
        print("[ERROR] One or more checks failed. Review the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
