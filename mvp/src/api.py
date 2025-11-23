"""
mvp/src/api.py

Fixed + improved:
- UI mounted at /ui (no root static mount conflict)
- Debug endpoint /_debug_ui
- load_taxonomy: robust, returns categories with id/name/aliases, metadata
- TxIndexer.load_documents_from_csv: auto-labels rows using taxonomy aliases
- /predict returns the JSON shape the UI expects
- Other endpoints preserved
"""

import os
import io
import json
import csv
import time
import copy
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, APIRouter, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import continuous learning and cost analysis
try:
    from .continuous_learning import auto_retrain_from_corrections, get_retraining_stats
except ImportError:
    auto_retrain_from_corrections = None
    get_retraining_stats = None

try:
    from .cost_analysis import get_cost_analysis, DEFAULT_SCENARIOS
except ImportError:
    get_cost_analysis = None
    DEFAULT_SCENARIOS = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("txcat")

# -------------------- Paths --------------------
# file is mvp/src/api.py -> BASE_DIR should be mvp (project root)
BASE_DIR = Path(__file__).resolve().parent.parent  # mvp/
PROJECT_ROOT = BASE_DIR.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
UI_DIR = BASE_DIR / "ui"  # mvp/ui/

TAX_PATH = CONFIG_DIR / "taxonomy.json"
_TRANSACTION_CANDIDATES = [
    DATA_DIR / "transactions.csv",
    PROJECT_ROOT / "data" / "transactions.csv",
]
CORRECTIONS_PATH = DATA_DIR / "corrections_buffer.jsonl"
INDEX_META_PATH = DATA_DIR / "index_meta.json"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

_TAXONOMY_CACHE: Optional[Dict[str, Any]] = None
_TAXONOMY_CACHE_MTIME: Optional[float] = None


def _resolve_transactions_csv() -> Path:
    for candidate in _TRANSACTION_CANDIDATES:
        if candidate.exists():
            return candidate
    # ensure the first path exists so we can write to it later if needed
    fallback = _TRANSACTION_CANDIDATES[0]
    if not fallback.parent.exists():
        fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


def _invalidate_taxonomy_cache():
    global _TAXONOMY_CACHE, _TAXONOMY_CACHE_MTIME
    _TAXONOMY_CACHE = None
    _TAXONOMY_CACHE_MTIME = None


DEFAULT_CATEGORIES = [
    {"id": "GROCERIES", "name": "Groceries", "aliases": ["grocery", "supermarket", "aldi", "kroger", "zepto", "blinkit", "bigbasket"]},
    {"id": "RESTAURANTS", "name": "Restaurants & Cafes", "aliases": ["starbucks", "mcdonald", "restaurant", "cafe", "swiggy", "zomato"]},
    {"id": "TRANSPORT", "name": "Transport", "aliases": ["uber", "ola", "taxi", "fuel", "shell", "petrol"]},
    {"id": "SHOPPING", "name": "Shopping", "aliases": ["amazon", "flipkart", "myntra", "shopping"]},
    {"id": "UTILITIES", "name": "Utilities", "aliases": ["electricity", "water", "gas", "utility"]},
    {"id": "ENTERTAINMENT", "name": "Entertainment", "aliases": ["netflix", "spotify", "movie", "concert"]},
    {"id": "TRAVEL", "name": "Travel", "aliases": ["indigo", "air", "flight", "train", "reservation"]},
    {"id": "INVESTMENT", "name": "Investment & Trading", "aliases": ["zerodha", "upstox", "groww", "angelone", "investment", "trading", "stock", "mutual fund", "sip"]},
]


def load_taxonomy(force: bool = False) -> Dict[str, Any]:
    """
    Returns taxonomy dict with shape:
    {
      "model": "...",
      "index_count": N,
      "low_confidence_threshold": 0.5,
      "categories": [ {id,name,aliases}, ... ]
    }
    """
    global _TAXONOMY_CACHE, _TAXONOMY_CACHE_MTIME
    disk_mtime = TAX_PATH.stat().st_mtime if TAX_PATH.exists() else None
    if not force and _TAXONOMY_CACHE is not None:
        # cache hit if file untouched or missing just like before
        if disk_mtime == _TAXONOMY_CACHE_MTIME:
            return copy.deepcopy(_TAXONOMY_CACHE)
        if disk_mtime is None and _TAXONOMY_CACHE_MTIME is None:
            return copy.deepcopy(_TAXONOMY_CACHE)

    t = None
    if TAX_PATH.exists():
        try:
            t = json.loads(TAX_PATH.read_text(encoding="utf8"))
        except Exception as e:
            logger.warning("Failed to parse taxonomy.json: %s", e)
            t = None

    if not t:
        t = {
            "model": INDEXER.model_name if 'INDEXER' in globals() else "tfidf-nearestneighbors",
            "index_count": len(INDEXER.docs) if 'INDEXER' in globals() else 0,
            "low_confidence_threshold": 0.5,
            "categories": DEFAULT_CATEGORIES,
        }

    t.setdefault("low_confidence_threshold", 0.5)
    t.setdefault("model", INDEXER.model_name if 'INDEXER' in globals() else "tfidf-nearestneighbors")
    t.setdefault("index_count", len(INDEXER.docs) if 'INDEXER' in globals() else 0)
    # normalize categories
    cats = []
    for c in t.get("categories", []):
        cid = c.get("id") or (c.get("name") or "").upper().replace(" ", "_")
        cats.append({
            "id": cid,
            "name": c.get("name", cid),
            "aliases": c.get("aliases", []) or []
        })
    t["categories"] = cats
    _TAXONOMY_CACHE = t
    _TAXONOMY_CACHE_MTIME = disk_mtime
    return copy.deepcopy(t)


def save_taxonomy_from_bytes(b: bytes):
    try:
        data = json.loads(b.decode("utf8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid json: {e}")
    with open(TAX_PATH, "w", encoding="utf8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    _invalidate_taxonomy_cache()
    return data

# -------------------- App --------------------
app = FastAPI(title="AI Tx Categorization MVP")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# -------------------- Static UI mount --------------------
if UI_DIR.exists():
    # mount at /ui to avoid API root conflicts
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
    logger.info("Mounted UI at /ui -> %s", UI_DIR)
else:
    logger.warning("UI directory not found at %s", UI_DIR)

# -------------------- Debug endpoint --------------------
debug_router = APIRouter()

@debug_router.get("/_debug_ui")
def _debug_ui_list():
    exists = UI_DIR.exists()
    files = []
    if exists:
        try:
            files = [p.name for p in sorted(UI_DIR.iterdir()) if p.is_file()]
        except Exception as e:
            files = [f"error_listing: {e}"]
    return {"ui_dir": str(UI_DIR), "exists": exists, "files": files}

app.include_router(debug_router)

# -------------------- Optional ML libs --------------------
USE_FAISS = False
USE_SENT_TRANS = False
try:
    from sentence_transformers import SentenceTransformer
    USE_SENT_TRANS = True
except Exception as e:
    logger.warning("sentence-transformers unavailable: %s", e)

try:
    import faiss
    USE_FAISS = True
except Exception as e:
    logger.warning("faiss unavailable: %s", e)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# -------------------- Indexer --------------------
class TxIndexer:
    def __init__(self):
        self.docs: List[Dict[str, Any]] = []
        self.embeddings = None
        self.index = None
        self.embedding_dim = None
        self.use_sentence_transformer = False
        self.use_faiss = False
        self.tfidf_vectorizer = None
        self.nn = None
        self.model_name = "fallback-tfidf"
        self._init_pipeline()

    def _init_pipeline(self):
        if USE_SENT_TRANS:
            try:
                logger.info("Loading SentenceTransformer all-MiniLM-L6-v2")
                self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_dim = self.st_model.get_sentence_embedding_dimension()
                self.use_sentence_transformer = True
                self.model_name = "all-MiniLM-L6-v2"
                if USE_FAISS:
                    self.use_faiss = True
                logger.info("ST loaded dim=%s use_faiss=%s", self.embedding_dim, self.use_faiss)
                return
            except Exception as e:
                logger.warning("Failed to init ST: %s", e)
        # fallback
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=4000)
        self.model_name = "tfidf-nearestneighbors"

    def load_documents_from_csv(self, csv_path: Optional[Path] = None):
        """
        Improved loader:
        - uses CSV fields 'description' or 'transaction' or 'text'
        - if a row has no label, attempt alias-based auto-label using taxonomy
        - assigned label will be category id or 'UNKNOWN'
        """
        self.docs = []
        csv_path = csv_path or _resolve_transactions_csv()
        if not csv_path.exists():
            logger.warning("transactions CSV not found at %s", csv_path)
            return

        taxonomy = load_taxonomy()
        # build alias -> category_id map
        alias_map = {}
        for cat in taxonomy.get("categories", []):
            cid = cat.get("id")
            for a in cat.get("aliases", []) or []:
                alias_map[a.lower()] = cid

        with open(csv_path, newline='', encoding='utf8') as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                text = (r.get("description") or r.get("transaction") or r.get("text") or "").strip()
                if not text:
                    continue
                label = (r.get("category") or r.get("label") or r.get("category_id") or "").strip()
                if not label:
                    # attempt simple alias match
                    low = text.lower()
                    found = None
                    for alias, cid in alias_map.items():
                        if alias in low:
                            found = cid
                            break
                    label = found or "UNKNOWN"
                entry = {"text": text, "label": label}
                source = r.get("source") or r.get("institution")
                if source:
                    entry["meta"] = {"source": source}
                self.docs.append(entry)
        logger.info("Loaded %d documents from %s", len(self.docs), csv_path)

    def build_index(self):
        if not self.docs:
            logger.warning("No documents to index")
            self.index = None
            self.embeddings = None
            return
        texts = [d["text"] for d in self.docs]
        if self.use_sentence_transformer:
            vectors = np.array(self.st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
            self.embeddings = vectors
            self.embedding_dim = vectors.shape[1]
            if self.use_faiss:
                faiss.normalize_L2(vectors)
                idx = faiss.IndexFlatIP(self.embedding_dim)
                idx.add(vectors)
                self.index = idx
            else:
                self.nn = NearestNeighbors(n_neighbors=8, metric="cosine")
                self.nn.fit(vectors)
                self.index = self.nn
        else:
            X = self.tfidf_vectorizer.fit_transform(texts)
            self.embeddings = X
            self.embedding_dim = X.shape[1]
            self.nn = NearestNeighbors(n_neighbors=8, metric="cosine")
            self.nn.fit(X)
            self.index = self.nn

        meta = {"model": self.model_name, "index_count": len(self.docs)}
        try:
            with open(INDEX_META_PATH, "w", encoding="utf8") as fh:
                json.dump(meta, fh)
        except Exception:
            pass
        logger.info("Built index (model=%s, docs=%d)", self.model_name, len(self.docs))

    def embed_text(self, text: str):
        if self.use_sentence_transformer:
            v = np.array(self.st_model.encode([text], convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
            if self.use_faiss:
                faiss.normalize_L2(v)
            return v
        else:
            return self.tfidf_vectorizer.transform([text])

    def query(self, text: str, k=5):
        if self.index is None:
            return []
        vec = self.embed_text(text)
        if self.use_faiss:
            D, I = self.index.search(vec, k)
            sims = D[0].tolist()
            idxs = I[0].tolist()
        else:
            dists, idxs = self.index.kneighbors(vec, n_neighbors=k, return_distance=True)
            # convert cosine distances to similarity
            sims = (1 - dists[0]).tolist()
            idxs = idxs[0].tolist()
        results = []
        for sim, idx in zip(sims, idxs):
            if idx < 0 or idx >= len(self.docs):
                continue
            results.append({"index": int(idx), "similarity": float(sim), "doc": self.docs[idx]})
        return results

    def add_document(self, text: str, label: str, meta: Optional[dict] = None, rebuild=True):
        self.docs.append({"text": text, "label": label, "meta": meta or {}})
        if rebuild:
            self.build_index()

# create global indexer
INDEXER = TxIndexer()
INDEXER.load_documents_from_csv()
try:
    INDEXER.build_index()
except Exception as e:
    logger.exception("Failed to build index: %s", e)

# -------------------- Helpers --------------------
def choose_category_from_neighbors(neighbors: List[Dict[str, Any]]):
    votes: Dict[str, float] = {}
    for nb in neighbors:
        lbl = (nb["doc"].get("label") or "").strip() or None
        if not lbl:
            continue
        score = float(nb.get("similarity") or 0.0)
        votes[lbl] = votes.get(lbl, 0.0) + score
    if not votes:
        return None, 0.0
    best = max(votes.items(), key=lambda kv: kv[1])
    total = sum(votes.values()) or 1.0
    confidence = best[1] / total
    return best[0], float(confidence)

# -------------------- Endpoints --------------------
@app.get("/")
def root():
    # redirect to UI index if available
    if UI_DIR.exists():
        return RedirectResponse(url="/ui/index.html")
    return {"status": "txcat backend running", "model": INDEXER.model_name}

@app.get("/taxonomy")
def get_taxonomy():
    t = load_taxonomy()
    t["model"] = INDEXER.model_name
    t["index_count"] = len(INDEXER.docs)
    return t

@app.post("/upload_taxonomy")
async def upload_taxonomy(file: UploadFile = File(...)):
    content = await file.read()
    data = save_taxonomy_from_bytes(content)
    logger.info("Uploaded taxonomy with %d categories", len(data.get("categories", [])))
    return {"ok": True, "categories": len(data.get("categories", []))}

@app.post("/rebuild_index")
def rebuild_index_endpoint():
    INDEXER.load_documents_from_csv()
    try:
        INDEXER.build_index()
        return {"status": "index_rebuilt", "count": len(INDEXER.docs)}
    except Exception as e:
        logger.exception("index rebuild failed: %s", e)
        raise HTTPException(status_code=500, detail="index rebuild failed")

@app.post("/add_to_index")
async def add_to_index(transaction: str = Form(...), correct_label: str = Form(...)):
    INDEXER.add_document(transaction, correct_label, meta={"source": "user_added"} if False else None, rebuild=True)
    return {"added": True, "new_index_count": len(INDEXER.docs)}

@app.post("/correct")
async def correct(
    transaction: str = Form(...), 
    correct_label: str = Form(...),
    auto_retrain: bool = Form(False)
):
    """
    Submit a correction. Optionally trigger automatic retraining.
    
    Args:
        transaction: The transaction text
        correct_label: The correct category label
        auto_retrain: If True, automatically retrain from all new corrections
    """
    buf = {"transaction": transaction, "correct_label": correct_label, "ts": time.time()}
    with open(CORRECTIONS_PATH, "a", encoding="utf8") as fh:
        fh.write(json.dumps(buf, ensure_ascii=False) + "\n")
    logger.info("Saved correction for '%s' -> %s", transaction, correct_label)
    
    result = {"saved": True}
    
    # Auto-retrain if requested and module is available
    if auto_retrain and auto_retrain_from_corrections:
        try:
            retrain_result = auto_retrain_from_corrections(
                INDEXER,
                CORRECTIONS_PATH,
                min_corrections=1,
                max_corrections_per_batch=100
            )
            result["auto_retrain"] = retrain_result
            logger.info("Auto-retraining completed: %s", retrain_result)
        except Exception as e:
            logger.error("Auto-retraining failed: %s", e)
            result["auto_retrain"] = {"status": "error", "error": str(e)}
    
    return result

@app.post("/predict")
async def predict(request: Request):
    tx = None
    try:
        form = await request.form()
        tx = (form.get("transaction") or form.get("description") or form.get("text") or "").strip()
    except Exception:
        tx = None

    if not tx:
        try:
            body = await request.json()
            tx = (body.get("transaction") or body.get("description") or body.get("text") or "").strip()
        except Exception:
            tx = None

    if not tx:
        raise HTTPException(status_code=422, detail="No transaction provided (transaction/description/text)")

    taxonomy = load_taxonomy()
    neighbors = INDEXER.query(tx, k=6)
    predicted_label, confidence = choose_category_from_neighbors(neighbors)
    if predicted_label is None:
        predicted_label = "UNKNOWN"
        confidence = 0.0

    explanations = []
    for nb in neighbors:
        doc = nb.get("doc", {})
        explanations.append({
            "description": doc.get("text"),
            "category": doc.get("label"),
            "similarity": round(float(nb.get("similarity", 0.0)), 4),
            "meta": doc.get("meta", {})
        })

    keyword_matches = []
    keyword_importance = {}  # Track importance scores for keywords
    lower = tx.lower()
    for c in taxonomy.get("categories", []):
        for alias in c.get("aliases", []) or []:
            if alias and alias.lower() in lower:
                keyword_matches.append(alias)
                # Calculate importance: how many times it appears, normalized
                count = lower.count(alias.lower())
                keyword_importance[alias] = {
                    "count": count,
                    "importance": min(1.0, count * 0.3),  # Normalized importance score
                    "category": c.get("id")
                }
    if keyword_matches:
        confidence = min(1.0, confidence + 0.15)
    
    # Calculate feature importance from neighbors
    neighbor_importance = {}
    for i, nb in enumerate(neighbors[:5]):  # Top 5 neighbors
        sim = float(nb.get("similarity", 0.0))
        neighbor_importance[f"neighbor_{i+1}"] = {
            "similarity": sim,
            "weight": sim * 0.2,  # Weight contribution
            "description": nb.get("doc", {}).get("text", "")[:50]  # Truncated
        }

    low_thr = taxonomy.get("low_confidence_threshold", 0.5)
    is_low_confidence = confidence < low_thr

    resp = {
        "description": tx,
        "predicted_category_id": predicted_label,
        "predicted_category_name": next((c["name"] for c in taxonomy.get("categories", []) if c["id"] == predicted_label), predicted_label),
        "category": predicted_label,
        "confidence": round(float(confidence), 4),
        "is_low_confidence": bool(is_low_confidence),
        "explanations": explanations,
        "keyword_matches": keyword_matches,
        "rationale": f"Prediction {predicted_label} (confidence {confidence:.2f}) based on nearest neighbors",
        "feature_importance": {
            "keywords": keyword_importance,
            "neighbors": neighbor_importance,
            "confidence_boost": 0.15 if keyword_matches else 0.0
        }
    }
    return JSONResponse(resp)

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    results = []
    for r in rows:
        tx = (r.get("transaction") or r.get("description") or r.get("text") or "").strip()
        if not tx:
            results.append({"input": r, "error": "no transaction found"})
            continue
        neighbors = INDEXER.query(tx, k=5)
        predicted_label, confidence = choose_category_from_neighbors(neighbors)
        exps = [{"description": nb["doc"]["text"], "category": nb["doc"]["label"], "similarity": nb["similarity"]} for nb in neighbors]
        results.append({
            "description": tx,
            "predicted_category": predicted_label,
            "confidence": round(float(confidence), 4),
            "explanations": exps
        })
    return {"predictions": results, "count": len(results)}

@app.get("/corrections")
def get_corrections(limit: int = 50):
    if not os.path.exists(CORRECTIONS_PATH):
        return {"corrections": []}
    out = []
    with open(CORRECTIONS_PATH, "r", encoding="utf8") as fh:
        for ln in fh:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return {"corrections": out[-limit:]}

@app.get("/index_meta")
def index_meta():
    meta = {"model": INDEXER.model_name, "count": len(INDEXER.docs)}
    if os.path.exists(INDEX_META_PATH):
        try:
            with open(INDEX_META_PATH, "r", encoding="utf8") as fh:
                meta2 = json.load(fh)
                meta.update(meta2)
        except Exception:
            pass
    return meta

@app.get("/download_taxonomy")
def download_taxonomy():
    if TAX_PATH.exists():
        return FileResponse(str(TAX_PATH), media_type="application/json", filename="taxonomy.json")
    tmp = json.dumps(load_taxonomy(), indent=2)
    return HTMLResponse(content=tmp, media_type="application/json")

@app.post("/auto_retrain")
def auto_retrain_endpoint(min_corrections: int = Query(1, ge=1), max_batch: int = Query(100, ge=1)):
    """
    Manually trigger automatic retraining from corrections.
    
    Args:
        min_corrections: Minimum number of new corrections required
        max_batch: Maximum corrections to process in one batch
    """
    if not auto_retrain_from_corrections:
        raise HTTPException(status_code=501, detail="Continuous learning module not available")
    
    try:
        result = auto_retrain_from_corrections(
            INDEXER,
            CORRECTIONS_PATH,
            min_corrections=min_corrections,
            max_corrections_per_batch=max_batch
        )
        return result
    except Exception as e:
        logger.exception("Auto-retraining failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Auto-retraining failed: {str(e)}")

@app.get("/retraining_stats")
def retraining_stats():
    """Get statistics about corrections and retraining status."""
    if not get_retraining_stats:
        raise HTTPException(status_code=501, detail="Continuous learning module not available")
    
    try:
        stats = get_retraining_stats(CORRECTIONS_PATH)
        return stats
    except Exception as e:
        logger.exception("Failed to get retraining stats: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/cost_analysis")
def cost_analysis_endpoint(scenario: str = Query("medium", regex="^(small|medium|large)$")):
    """
    Get cost savings analysis compared to third-party APIs.
    
    Args:
        scenario: One of "small", "medium", "large" (default: medium)
    """
    if get_cost_analysis is None:
        # Fallback if module not available
        return {
            "error": "Cost analysis module not available",
            "message": "Please ensure cost_analysis.py is properly imported"
        }
    
    try:
        analysis = get_cost_analysis(scenario)
        return analysis
    except Exception as e:
        logger.exception("Failed to calculate cost analysis: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to calculate cost analysis: {str(e)}")
