"""
Detailed distribution analysis of pairwise persona similarity.
Batches 10 texts per embedding request (~10x faster than original).
Outputs: percentiles, histogram bins, high/low similarity proportions.
"""

import os, re, json, time, asyncio, aiohttp
import numpy as np
from glob import glob
from collections import defaultdict

DATA_DIR   = "./rephase_data_400"
EMBED_URL  = "http://127.0.0.1:8400/v1/embeddings"
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_DIM  = 1024
BATCH_SIZE = 10          # texts per request
OUT_JSON   = "./rephase_data_400/similarity_dist_results.json"

DATASET_ORDER = ["LaMP4", "LaMP5", "LaMP8", "LaMP9", "LaMP10", "LaMP-2", "ultrachat"]

# Similarity range thresholds for proportion reporting
BINS = [
    ("sim=1.00 (exact duplicate)",    1.00 - 1e-5, 1.01),
    ("0.99 ≤ sim < 1.00 (near-dup)", 0.99,        1.00 - 1e-5),
    ("0.95 ≤ sim < 0.99 (very high)", 0.95,        0.99),
    ("0.90 ≤ sim < 0.95 (high)",      0.90,        0.95),
    ("0.80 ≤ sim < 0.90 (med-high)",  0.80,        0.90),
    ("0.70 ≤ sim < 0.80 (medium)",    0.70,        0.80),
    ("0.50 ≤ sim < 0.70 (low)",       0.50,        0.70),
    ("sim < 0.50 (very low)",        -1.01,        0.50),
]


def normalize_l2(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


async def embed_batch(session, texts):
    payload = {"model": MODEL_NAME, "input": texts}
    timeout = aiohttp.ClientTimeout(total=120)
    for attempt in range(5):
        try:
            async with session.post(EMBED_URL, json=payload, timeout=timeout) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return [normalize_l2(
                        np.array(item["embedding"], dtype=np.float32)[:EMBED_DIM]
                    ) for item in result["data"]]
                await asyncio.sleep(2.0 * (2 ** attempt))
        except Exception:
            await asyncio.sleep(2.0 * (2 ** attempt))
    raise RuntimeError("Embedding failed after 5 retries")


async def get_embeddings_async(texts):
    async with aiohttp.ClientSession() as session:
        embs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            embs.extend(await embed_batch(session, batch))
        return np.stack(embs)


def pairwise_cosine(embs):
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    n = embs / norms
    mat = n @ n.T
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx].astype(np.float32)


def compute_dist_stats(pairs):
    pcts = np.percentile(pairs, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    std  = float(np.std(pairs, ddof=1))
    props = {}
    for label, lo, hi in BINS:
        mask = (pairs >= lo) & (pairs < hi)
        props[label] = float(mask.sum() / len(pairs))
    return {
        "mean":   float(np.mean(pairs)),
        "std":    std,
        "p01":    float(pcts[0]),  "p05": float(pcts[1]),
        "p10":    float(pcts[2]),  "p25": float(pcts[3]),
        "p50":    float(pcts[4]),  "p75": float(pcts[5]),
        "p90":    float(pcts[6]),  "p95": float(pcts[7]),
        "p99":    float(pcts[8]),
        "proportions": props,
    }


def parse_dataset(filename):
    m = re.match(r"^(.+?)_times\d+", os.path.basename(filename))
    return m.group(1) if m else filename


def process_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    summaries = data.get("summaries", [])
    user_id   = data.get("input_id", os.path.basename(path))
    dataset   = parse_dataset(path)
    embs  = asyncio.run(get_embeddings_async(summaries))
    pairs = pairwise_cosine(embs)
    stats = compute_dist_stats(pairs)
    return {"dataset": dataset, "user_id": user_id, "n_pairs": int(len(pairs)), **stats}


# ── Printing ────────────────────────────────────────────────────────────────

W = 100
def sep(c="─"): print(c * W)
def section(t): sep("═"); print(f"  {t}"); sep("═")


def print_dataset_summary(ds, users):
    section(f"DATASET: {ds}  ({len(users)} users)")
    # Percentile table
    pct_keys = ["p01","p05","p10","p25","p50","p75","p90","p95","p99"]
    print(f"  {'User':<30} {'mean':>6} {'std':>6} " +
          "  ".join(f"{k:>5}" for k in pct_keys))
    sep()
    for u in users:
        uid = str(u["user_id"])[:28]
        row = f"  {uid:<30} {u['mean']:>6.4f} {u['std']:>6.4f} "
        row += "  ".join(f"{u[k]:>5.3f}" for k in pct_keys)
        print(row)
    sep()
    # Proportion table (averaged over users)
    print(f"\n  Proportion of pairs by similarity range (mean over {len(users)} users):\n")
    all_props = defaultdict(list)
    for u in users:
        for label, v in u["proportions"].items():
            all_props[label].append(v)
    for label, lo, hi in BINS:
        vals  = all_props[label]
        avg   = np.mean(vals) * 100
        mn    = np.min(vals) * 100
        mx    = np.max(vals) * 100
        print(f"  {label:<45}  avg={avg:6.2f}%  min={mn:5.2f}%  max={mx:5.2f}%")
    sep()


def main():
    files = sorted(glob(os.path.join(DATA_DIR, "*.json")))
    # Exclude output files
    files = [f for f in files if "similarity" not in os.path.basename(f)]
    print(f"Files: {len(files)}   Batch size: {BATCH_SIZE}   Output: {OUT_JSON}\n")
    sep("═")

    user_results   = []
    dataset_groups = defaultdict(list)
    t_global = time.time()

    for idx, path in enumerate(files, 1):
        fname = os.path.basename(path)
        t0    = time.time()
        try:
            r       = process_file(path)
            elapsed = time.time() - t0
            eta     = (time.time() - t_global) / idx * (len(files) - idx)
            print(f"[{idx:3d}/{len(files)}] ✓  {fname}  ({elapsed:.1f}s, ETA {eta/60:.1f}min)", flush=True)
            user_results.append(r)
            dataset_groups[r["dataset"]].append(r)
        except Exception as e:
            print(f"[{idx:3d}/{len(files)}] ✗  {fname}  ERROR: {e}", flush=True)

    sep("═")
    print(f"Done in {time.time()-t_global:.0f}s\n")

    # Print detailed per-dataset summaries
    ordered = [d for d in DATASET_ORDER if d in dataset_groups]
    ordered += [d for d in sorted(dataset_groups) if d not in ordered]
    for ds in ordered:
        print_dataset_summary(ds, sorted(dataset_groups[ds], key=lambda x: x["user_id"]))

    # Global proportion summary
    section("GLOBAL PROPORTION SUMMARY (all datasets)")
    print(f"  {'Dataset':<12} " + "  ".join(f"{'≥.99':>6} {'≥.95':>6} {'≥.90':>6} {'≥.80':>6} {'<.50':>6}"))
    sep()
    for ds in ordered:
        users = dataset_groups[ds]
        def avg_prop(lo, hi):
            return np.mean([sum(v for lbl, v in u["proportions"].items()
                               if any(lbl.startswith(f"{t:.2f}") or
                                      (lo <= float(lbl.split("≤")[0].strip().replace("sim=","").replace("sim<","").strip()[:4]) < hi
                                       if lbl[0].isdigit() or lbl.startswith("0") else False)
                                      for t in [lo])) for u in users]) * 100

        def prop_range(lo, hi):
            vals = []
            for u in users:
                s = sum(v for lbl, v in u["proportions"].items()
                        for bl, blo, bhi in BINS if lbl == bl and blo >= lo and bhi <= hi)
                vals.append(s)
            return np.mean(vals) * 100

        ge99 = np.mean([u["proportions"].get("sim=1.00 (exact duplicate)", 0) +
                        u["proportions"].get("0.99 ≤ sim < 1.00 (near-dup)", 0) for u in users]) * 100
        ge95 = np.mean([u["proportions"].get("0.95 ≤ sim < 0.99 (very high)", 0) for u in users]) * 100
        ge90 = np.mean([u["proportions"].get("0.90 ≤ sim < 0.95 (high)", 0) for u in users]) * 100
        ge80 = np.mean([u["proportions"].get("0.80 ≤ sim < 0.90 (med-high)", 0) for u in users]) * 100
        lt50 = np.mean([u["proportions"].get("sim < 0.50 (very low)", 0) for u in users]) * 100
        print(f"  {ds:<12}  {ge99:>6.2f}%  {ge95:>6.2f}%  {ge90:>6.2f}%  {ge80:>6.2f}%  {lt50:>6.2f}%")
    sep("═")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"per_user": user_results}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved → {OUT_JSON}")


if __name__ == "__main__":
    main()
