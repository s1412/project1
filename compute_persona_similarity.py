"""
Compute pairwise cosine similarity of persona summaries across 7 datasets:
  LaMP4, LaMP5, LaMP8, LaMP9, LaMP10, LaMP-2, ultrachat

Pipeline per user:
  1. Load 400 summaries from JSON file.
  2. Encode via local embedding API (http://127.0.0.1:8400/v1/embeddings).
  3. Compute all C(400,2)=79,800 pairwise cosine similarities.
  4. Report: mean, SEM, max, min of pairwise similarities.

Aggregate per dataset (over all users):
  - Mean and SEM of per-user means.
  - Overall max and min across all users in dataset.

Results are printed to stdout and saved to similarity_results.json.
"""

import os, re, json, time, argparse
import asyncio
import aiohttp
import numpy as np
from glob import glob
from collections import defaultdict

# ── Default config ─────────────────────────────────────────────────────────
DATA_DIR        = "./rephase_data_400"
EMBED_URL       = "http://127.0.0.1:8400/v1/embeddings"
MODEL_NAME      = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_DIM       = 1024        # truncate embeddings to this dimension
REQUEST_TIMEOUT = 60          # seconds per request
MAX_RETRIES     = 5
RETRY_DELAY     = 2.0         # base retry delay (seconds)
RETRY_BACKOFF   = 2.0         # exponential backoff multiplier
OUTPUT_JSON     = "./rephase_data_400/similarity_results.json"
# ───────────────────────────────────────────────────────────────────────────

# Canonical display order for datasets
DATASET_ORDER = ["LaMP4", "LaMP5", "LaMP8", "LaMP9", "LaMP10", "LaMP-2", "ultrachat"]


# ── Embedding API (mirrors EmbeddingClient in ID_TAP.py) ───────────────────

def normalize_l2(x: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D vector (safe against zero-norm)."""
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x


async def _get_single_embedding(
    session: aiohttp.ClientSession,
    text: str,
) -> np.ndarray:
    """Async: fetch one embedding, truncate to EMBED_DIM, L2-normalize."""
    payload = {"model": MODEL_NAME, "input": [text]}
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(EMBED_URL, json=payload, timeout=timeout) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    raw = np.array(result["data"][0]["embedding"], dtype=np.float32)
                    raw = raw[:EMBED_DIM] if len(raw) > EMBED_DIM else raw
                    return normalize_l2(raw)
                else:
                    error_msg = f"HTTP {resp.status}"
                    if attempt < MAX_RETRIES - 1:
                        wait = RETRY_DELAY * (RETRY_BACKOFF ** attempt)
                        print(f"⚠️  Embedding attempt {attempt+1}/{MAX_RETRIES}: {error_msg}. "
                              f"Retrying in {wait:.1f}s...", flush=True)
                        await asyncio.sleep(wait)
                    else:
                        raise RuntimeError(f"Max retries reached: {error_msg}")
        except asyncio.TimeoutError as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (RETRY_BACKOFF ** attempt)
                print(f"⚠️  Embedding attempt {attempt+1}/{MAX_RETRIES}: Timeout. "
                      f"Retrying in {wait:.1f}s...", flush=True)
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(f"Max retries reached (timeout): {e}")
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (RETRY_BACKOFF ** attempt)
                print(f"⚠️  Embedding attempt {attempt+1}/{MAX_RETRIES}: {type(e).__name__}: {e}. "
                      f"Retrying in {wait:.1f}s...", flush=True)
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(f"Max retries reached [{type(e).__name__}]: {e}")

    raise RuntimeError("Embedding failed: exceeded max retries")


async def _encode_texts_async(texts: list) -> np.ndarray:
    """Async: encode all texts sequentially (one request per text)."""
    async with aiohttp.ClientSession() as session:
        embs = []
        for i, text in enumerate(texts):
            emb = await _get_single_embedding(session, text)
            embs.append(emb)
        return np.stack(embs)  # (N, EMBED_DIM)


def get_embeddings(texts: list) -> np.ndarray:
    """Synchronous wrapper: return (N, EMBED_DIM) float32 array."""
    return asyncio.run(_encode_texts_async(texts))


# ── Similarity helpers ──────────────────────────────────────────────────────

def pairwise_cosine(embs: np.ndarray) -> np.ndarray:
    """Compute upper-triangle pairwise cosine similarities; return 1-D array."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    normed = embs / norms
    sim_mat = normed @ normed.T          # (N, N)
    idx = np.triu_indices(sim_mat.shape[0], k=1)
    return sim_mat[idx]                  # C(N,2) values


def sem(values: np.ndarray) -> float:
    """Standard error of the mean (unbiased std)."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


# ── File processing ─────────────────────────────────────────────────────────

def parse_dataset(filename: str) -> str:
    """Extract dataset prefix before '_times<N>'."""
    base = os.path.basename(filename)
    m = re.match(r"^(.+?)_times\d+", base)
    return m.group(1) if m else base


def process_file(path: str) -> dict:
    """Load one user file, embed summaries, compute pairwise cosine stats."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summaries = data.get("summaries", [])
    if len(summaries) < 2:
        raise ValueError(f"Too few summaries ({len(summaries)}) in {os.path.basename(path)}")

    user_id = data.get("input_id", os.path.basename(path))
    dataset = parse_dataset(path)

    embs  = get_embeddings(summaries)       # (400, dim)
    pairs = pairwise_cosine(embs)           # (79800,)

    return {
        "dataset": dataset,
        "user_id": user_id,
        "n_summaries": len(summaries),
        "n_pairs":     int(len(pairs)),
        "mean":        float(np.mean(pairs)),
        "sem":         float(sem(pairs)),
        "max":         float(np.max(pairs)),
        "min":         float(np.min(pairs)),
    }


# ── Printing helpers ────────────────────────────────────────────────────────

W = 90   # table width

def sep(char="─"):
    print(char * W)

def section(title: str):
    sep("═")
    print(f"  {title}")
    sep("═")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    global EMBED_URL
    parser = argparse.ArgumentParser(description="Persona pairwise cosine similarity")
    parser.add_argument("--data_dir",   default=DATA_DIR,   help="Directory with JSON files")
    parser.add_argument("--embed_url",  default=EMBED_URL,  help="Embedding API endpoint")
    parser.add_argument("--output",     default=OUTPUT_JSON, help="Path for JSON results")
    args = parser.parse_args()

    EMBED_URL  = args.embed_url

    files = sorted(glob(os.path.join(args.data_dir, "*.json")))
    print(f"Files found : {len(files)}")
    print(f"Embed API   : {EMBED_URL}")
    print(f"Output      : {args.output}")
    sep("═")

    user_results   = []                        # list[dict]
    dataset_groups = defaultdict(list)         # dataset -> list[dict]
    failed         = []                        # [(filename, error_str)]

    # ── Process every user file ─────────────────────────────────────────────
    t_global = time.time()
    for idx, path in enumerate(files, 1):
        fname   = os.path.basename(path)
        t0      = time.time()
        try:
            r       = process_file(path)
            elapsed = time.time() - t0
            eta_s   = (time.time() - t_global) / idx * (len(files) - idx)
            print(f"[{idx:3d}/{len(files)}] ✓  {fname}  ({elapsed:.1f}s, ETA {eta_s/60:.1f}min)", flush=True)
            user_results.append(r)
            dataset_groups[r["dataset"]].append(r)
        except Exception as e:
            print(f"[{idx:3d}/{len(files)}] ✗  {fname}  ERROR: {e}", flush=True)
            failed.append((fname, str(e)))

    total_elapsed = time.time() - t_global
    sep("═")
    print(f"Finished in {total_elapsed:.1f}s  "
          f"(success={len(user_results)}, failed={len(failed)})\n")

    # Determine display order (canonical + any unexpected datasets at the end)
    ordered_ds = [d for d in DATASET_ORDER if d in dataset_groups]
    ordered_ds += [d for d in sorted(dataset_groups) if d not in DATASET_ORDER]

    # ── Per-user table ──────────────────────────────────────────────────────
    section("PER-USER RESULTS")
    col = f"{'Dataset':<12} {'UserID':>8}  {'Mean':>8}  {'SEM':>10}  {'Max':>8}  {'Min':>8}  {'N_pairs':>8}"
    print(col)
    sep()
    for ds in ordered_ds:
        for r in sorted(dataset_groups[ds], key=lambda x: x["user_id"]):
            print(f"{r['dataset']:<12} {str(r['user_id']):>8}  "
                  f"{r['mean']:>8.4f}  {r['sem']:>10.6f}  "
                  f"{r['max']:>8.4f}  {r['min']:>8.4f}  {r['n_pairs']:>8,}")
        sep("·")

    # ── Per-dataset summary ─────────────────────────────────────────────────
    section("PER-DATASET SUMMARY  (aggregated over all users)")
    col2 = (f"{'Dataset':<12} {'Users':>5}  "
            f"{'Mean(means)':>12}  {'SEM(means)':>12}  "
            f"{'Max(overall)':>13}  {'Min(overall)':>13}")
    print(col2)
    sep()

    dataset_summary = {}
    for ds in ordered_ds:
        rs      = dataset_groups[ds]
        means   = np.array([r["mean"] for r in rs])
        ds_mean = float(np.mean(means))
        ds_sem  = sem(means)
        ds_max  = float(max(r["max"] for r in rs))
        ds_min  = float(min(r["min"] for r in rs))
        n       = len(rs)
        dataset_summary[ds] = {
            "n_users": n, "mean": ds_mean, "sem": ds_sem,
            "max": ds_max, "min": ds_min,
        }
        print(f"{ds:<12} {n:>5}  "
              f"{ds_mean:>12.4f}  {ds_sem:>12.6f}  "
              f"{ds_max:>13.4f}  {ds_min:>13.4f}")

    sep("═")

    # ── Save JSON ───────────────────────────────────────────────────────────
    output = {
        "meta": {
            "data_dir":  args.data_dir,
            "embed_url": EMBED_URL,
            "n_files":   len(files),
            "n_success":  len(user_results),
            "n_failed":   len(failed),
            "elapsed_s":  round(total_elapsed, 1),
        },
        "per_user":    user_results,
        "per_dataset": dataset_summary,
        "failed":      failed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved → {args.output}")

    if failed:
        print(f"\n⚠  {len(failed)} file(s) failed:")
        for fname, err in failed:
            print(f"   {fname}: {err}")


if __name__ == "__main__":
    main()
