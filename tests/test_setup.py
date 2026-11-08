"""Step 1 basic checks: are all libraries installed and config loading?

Run:  python -m tests.test_setup
"""
import importlib

LIBS = [
    "googleapiclient", "google_auth_oauthlib", "dotenv",
    "pypdf", "docx", "ollama", "sentence_transformers",
]

def main():
    ok = True
    for lib in LIBS:
        try:
            importlib.import_module(lib)
            print(f"[ok]   {lib}")
        except ImportError:
            print(f"[FAIL] {lib}  -> run: pip install -r requirements.txt")
            ok = False

    from src import config
    print(f"[ok]   config loaded (model = {config.OLLAMA_MODEL})")
    print("\nAll good!" if ok else "\nSome libraries are missing.")

if __name__ == "__main__":
    main()
