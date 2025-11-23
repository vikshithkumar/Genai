# test_imports.py
import importlib
import sys

packages = [
    "fastapi","uvicorn","pydantic","pandas","numpy","sklearn",
    "sentence_transformers","matplotlib","seaborn"
]

print("Python:", sys.version.replace('\\n',' '))
ok = True

for p in packages:
    try:
        m = importlib.import_module(p)
        ver = getattr(m, "__version__", None) or getattr(m, "VERSION", None) or "version-info-unavailable"
        print(f"{p:20s} OK -> {ver}")
    except Exception as e:
        print(f"{p:20s} FAILED -> {e.__class__.__name__}: {e}")
        ok = False

# faiss
try:
    import faiss
    print("faiss                OK ->", faiss.__version__)
except Exception as e:
    print("faiss                FAILED ->", type(e).__name__, e)
    ok = False

# torch
try:
    import torch
    print("torch                OK ->", torch.__version__)
except Exception as e:
    print("torch                FAILED ->", type(e).__name__, e)
    ok = False

print()
print("ALL OK status:", ok)
