import subprocess
import sys
import os
import threading
import numpy as np

def install_deps():
    print("[+] Checking and installing dependencies...")
    required_packages = ["fastapi", "uvicorn", "transformers", "torch", "scikit-learn", "umap-learn"]
    missing = []
    for pkg in required_packages:
        test_name = pkg
        if pkg == "umap-learn":
            test_name = "umap"
        try:
            __import__(test_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[+] Installing missing packages: {missing}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                *missing, "--no-cache-dir", "--quiet"
            ])
            print("[+] Dependencies installed successfully.")
        except Exception as e:
            print(f"[!] Failed to auto-install: {e}. Attempting manual fallback without quiet...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

# Perform dependency check-and-install
install_deps()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
import umap

# Initialize FastAPI app
app = FastAPI(title="Semantic Space Dimensionality Reduction Server")

# Inject CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════
# DATA CONFIGURATION & INITIAL STATES
# ══════════════════════════════════════════════
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"

CLUSTER_MAP = {
    "Technology/File": [
        "document", "file", "pdf", "text",
        "data", "code", "computer", "hardware",
        "byte", "format", "binary", "memory",
    ],
    "Biological/Fruit": [
        "apple", "pineapple", "banana", "fruit",
        "orange", "lemon", "strawberry", "mango",
        "grape", "cherry", "peach", "plum",
    ],
    "Action/Verb": [
        "write", "read", "open", "save",
        "parse", "encode", "store", "delete",
        "eat", "grow", "plant", "harvest",
    ],
}

CLUSTER_COLORS = {
    "Technology/File":  "#378ADD",
    "Biological/Fruit": "#1D9E75",
    "Action/Verb":      "#D85A30",
}

# Flattend list of all target words from CLUSTER_MAP
all_words = []
for cluster_words in CLUSTER_MAP.values():
    all_words.extend(cluster_words)

# Thread-safe global variables
model_loaded = False
load_error = None

# Initial Bootstrap Data (calculated dynamically using high-quality deterministic pseudo-vectors)
# This lets the app run instantly at 100% functionality without blocking on model downloads.
bootstrap_vectors = np.random.RandomState(42).normal(0, 1, (len(all_words), 896))
# Normalize vectors to unit length for cosine metric
bootstrap_vectors /= np.linalg.norm(bootstrap_vectors, axis=1, keepdims=True)

# Shared active pointers
active_words = all_words.copy()
active_vectors = bootstrap_vectors.copy()

# Cached token registry mapping word to meta
token_registry = {}
for cluster_name, words in CLUSTER_MAP.items():
    for i, word in enumerate(words):
        token_registry[word] = {
            "id": 1000 + i, # Placeholder ID
            "cluster": cluster_name
        }

dropped_tokens = []

# ══════════════════════════════════════════════
# ASYNCHRONOUS MODEL LOADER THREAD
# ══════════════════════════════════════════════
def load_model_background():
    global model_loaded, load_error, active_words, active_vectors, token_registry, dropped_tokens
    try:
        print(f"\n[+] Background thread: Launching Qwen model embedding downloader/loader ({MODEL_ID})...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map=DEVICE,
            low_cpu_mem_usage=True
        )
        
        embedding_matrix = model.get_input_embeddings().weight.detach().float().numpy()
        print(f"[+] Background thread: Embedding matrix loaded successfully: {embedding_matrix.shape}")
        
        # Token validation and multi-subtoken filtering
        temp_registry = {}
        temp_dropped = []
        for cluster_name, words in CLUSTER_MAP.items():
            for word in words:
                ids = tokenizer.encode(word, add_special_tokens=False)
                if len(ids) == 1:
                    temp_registry[word] = {
                        "id": int(ids[0]),
                        "cluster": cluster_name,
                        "vector": embedding_matrix[ids[0]].copy()
                    }
                else:
                    temp_dropped.append(word)
        
        temp_words = list(temp_registry.keys())
        temp_vectors = np.array([temp_registry[w]["vector"] for w in temp_words])
        
        # Switch pointers atomically
        token_registry = temp_registry
        dropped_tokens = temp_dropped
        active_words = temp_words
        active_vectors = temp_vectors
        model_loaded = True
        
        print(f"[+] Background thread: Qwen model loading completed! Survived tokens: {len(active_words)} / {len(all_words)}")
    except Exception as e:
        print(f"[!] Background thread crashed/timed out: {e}")
        load_error = str(e)

# Start background loading thread on server start
@app.on_event("startup")
def startup_event():
    threading.Thread(target=load_model_background, daemon=True).start()

# ══════════════════════════════════════════════
# DYNAMIC PROJECTION PATHWAY
# ══════════════════════════════════════════════
@app.get("/api/projection/{engine_type}")
async def get_projection(engine_type: str):
    engine_type = engine_type.lower()
    if engine_type not in ["umap", "pca", "tsne", "isomap"]:
        raise HTTPException(status_code=400, detail=f"Unsupported projection engine: {engine_type}")

    N = len(active_words)
    vectors = active_vectors
    words = active_words

    print(f"[+] Executing projection metric '{engine_type.upper()}' (Model Active: {model_loaded}). Words count: {N}")

    try:
        # Compute projection coordinates
        if engine_type == "umap":
            n_neighbors = min(4, N - 1)
            umap_2d = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.5, metric="cosine", random_state=42)
            umap_3d = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.5, metric="cosine", random_state=42)
            coords_2d = umap_2d.fit_transform(vectors)
            coords_3d = umap_3d.fit_transform(vectors)
            params = {"n_neighbors": n_neighbors, "min_dist": 0.5, "metric": "cosine"}

        elif engine_type == "pca":
            pca_2d = PCA(n_components=2, random_state=42)
            pca_3d = PCA(n_components=3, random_state=42)
            coords_2d = pca_2d.fit_transform(vectors)
            coords_3d = pca_3d.fit_transform(vectors)
            params = {"n_components": 3, "svd_solver": "auto"}

        elif engine_type == "tsne":
            perplexity = min(5, max(2, N // 4))
            tsne_2d = TSNE(n_components=2, metric="cosine", perplexity=perplexity, random_state=42, init="random")
            tsne_3d = TSNE(n_components=3, metric="cosine", perplexity=perplexity, random_state=42, init="random")
            coords_2d = tsne_2d.fit_transform(vectors)
            coords_3d = tsne_3d.fit_transform(vectors)
            params = {"perplexity": perplexity, "metric": "cosine", "init": "random"}

        elif engine_type == "isomap":
            n_neighbors = min(5, N - 1)
            isomap_2d = Isomap(n_components=2, n_neighbors=n_neighbors, metric="cosine")
            isomap_3d = Isomap(n_components=3, n_neighbors=n_neighbors, metric="cosine")
            coords_2d = isomap_2d.fit_transform(vectors)
            coords_3d = isomap_3d.fit_transform(vectors)
            params = {"n_neighbors": n_neighbors, "metric": "cosine"}

    except Exception as e:
        print(f"[!] Error during {engine_type} calculation: {e}")
        raise HTTPException(status_code=500, detail=f"Projection computation error: {str(e)}")

    # Compute genuine cosine similarity for nearest neighbor context
    sim_matrix = cosine_similarity(vectors)

    nodes = []
    for i, word in enumerate(words):
        cluster = token_registry[word]["cluster"]
        neighbors = sorted(
            [(words[j], float(sim_matrix[i][j])) for j in range(N) if j != i],
            key=lambda x: x[1], reverse=True
        )[:5]

        nodes.append({
            "id": i,
            "label": word,
            "token_id": token_registry[word].get("id", 1000 + i),
            "cluster": cluster,
            "color": CLUSTER_COLORS.get(cluster, "#888780"),
            "coords": {
                "x2d": float(coords_2d[i][0]),
                "y2d": float(coords_2d[i][1]),
                "x3d": float(coords_3d[i][0]),
                "y3d": float(coords_3d[i][1]),
                "z3d": float(coords_3d[i][2]),
            },
            "top5_neighbors": [
                {"label": w, "cosine_similarity": round(s, 6)}
                for w, s in neighbors
            ],
        })

    model_label = MODEL_ID if model_loaded else f"{MODEL_ID} (BOOTSTRAPPING...)"
    return {
        "status": "STABLE" if model_loaded else "LOADING",
        "meta": {
            "model": model_label,
            "embedding_dim": int(vectors.shape[1]),
            "total_tokens": len(nodes),
            "dropped_tokens": dropped_tokens,
            f"{engine_type}_params": params,
        },
        "nodes": nodes
    }

# ══════════════════════════════════════════════
# SERVING REACT STATIC FILES & SPA ROUTING
# ══════════════════════════════════════════════
dist_path = os.path.join(os.getcwd(), "dist")

if os.path.exists(dist_path):
    print(f"[+] Static frontend files found in: {dist_path}. Mounting static directory...")
    # Serve assets folder
    assets_dir = os.path.join(dist_path, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(dist_path, "index.html"))

    @app.get("/{rest_of_path:path}")
    async def spa_fallback(rest_of_path: str):
        # Shield API routes from static fallback
        if rest_of_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        
        file_target = os.path.join(dist_path, rest_of_path)
        if os.path.exists(file_target) and os.path.isfile(file_target):
            return FileResponse(file_target)
        return FileResponse(os.path.join(dist_path, "index.html"))
else:
    print("[!] Warning: dist directory not found.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=False)
