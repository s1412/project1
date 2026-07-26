"""
Full Persona Space Analysis — covers:
  A. Distribution (mean/std/percentiles/proportions per user)
  B. Effective arm count (greedy dedup at sim>0.95)
  C. Cluster analysis (K-means K=5,10,20; silhouette)
  D. Cross-user similarity (per-dataset centroid pairwise similarity)

Runs ~40-60 min in background; writes incremental JSON after each user.
Usage: nohup python full_persona_analysis.py > full_analysis.log 2>&1 &
"""

import os, re, json, time, asyncio, aiohttp, glob, collections
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = "./rephase_data_400"
EMBED_URL  = "http://127.0.0.1:8400/v1/embeddings"
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_DIM  = 1024
BATCH_SZ   = 10
OUT_JSON   = "./rephase_data_400/full_analysis_results.json"
KMEANS_KS  = [5, 10, 20]
DUP_THRESH = 0.95   # sim > this → near-duplicate

DATASET_ORDER = ["LaMP4","LaMP5","LaMP8","LaMP9","LaMP10","LaMP-2","ultrachat"]
SIM_BINS = [
    ("sim=1.00",          1.0-1e-5, 1.01),
    ("0.99≤sim<1.00",    0.99,      1.0-1e-5),
    ("0.95≤sim<0.99",    0.95,      0.99),
    ("0.90≤sim<0.95",    0.90,      0.95),
    ("0.80≤sim<0.90",    0.80,      0.90),
    ("0.70≤sim<0.80",    0.70,      0.80),
    ("0.50≤sim<0.70",    0.50,      0.70),
    ("sim<0.50",         -1.01,     0.50),
]

# ── Logging ──────────────────────────────────────────────────────────────────
def log(msg): print(msg, flush=True)

# ── Embedding ─────────────────────────────────────────────────────────────────
def normalize(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x

async def embed_batch(session, texts):
    payload = {"model": MODEL_NAME, "input": texts}
    timeout = aiohttp.ClientTimeout(total=120)
    for attempt in range(5):
        try:
            async with session.post(EMBED_URL, json=payload, timeout=timeout) as r:
                if r.status == 200:
                    res = await r.json()
                    return [normalize(np.array(item["embedding"], dtype=np.float32)[:EMBED_DIM])
                            for item in res["data"]]
                await asyncio.sleep(2.0 * 2**attempt)
        except Exception:
            await asyncio.sleep(2.0 * 2**attempt)
    raise RuntimeError("Embed failed after 5 retries")

async def get_embeddings(texts):
    async with aiohttp.ClientSession() as session:
        embs = []
        for i in range(0, len(texts), BATCH_SZ):
            embs.extend(await embed_batch(session, texts[i:i+BATCH_SZ]))
        return np.stack(embs)

# ── Similarity ────────────────────────────────────────────────────────────────
def pairwise_cosine(embs):
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    n = embs / norms
    mat = (n @ n.T).astype(np.float32)
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx], mat

# ── A. Distribution stats ─────────────────────────────────────────────────────
def dist_stats(pairs):
    pcts = np.percentile(pairs, [1,5,10,25,50,75,90,95,99]).tolist()
    props = {}
    for label, lo, hi in SIM_BINS:
        props[label] = float(((pairs >= lo) & (pairs < hi)).sum() / len(pairs))
    return {
        "mean": float(np.mean(pairs)), "std": float(np.std(pairs, ddof=1)),
        "p01":pcts[0],"p05":pcts[1],"p10":pcts[2],"p25":pcts[3],"p50":pcts[4],
        "p75":pcts[5],"p90":pcts[6],"p95":pcts[7],"p99":pcts[8],
        "proportions": props,
    }

# ── B. Effective arm count ────────────────────────────────────────────────────
def effective_arms(sim_mat, threshold=DUP_THRESH):
    n = sim_mat.shape[0]
    visited = np.zeros(n, dtype=bool)
    count = 0
    for i in range(n):
        if not visited[i]:
            count += 1
            dups = np.where(sim_mat[i] >= threshold)[0]
            visited[dups] = True
    return int(count)

# ── C. Cluster analysis ───────────────────────────────────────────────────────
def cluster_analysis(embs, sim_mat):
    results = {}
    for k in KMEANS_KS:
        km = KMeans(n_clusters=k, n_init=5, max_iter=200, random_state=42)
        labels = km.fit_predict(embs)
        try:
            sil = float(silhouette_score(embs, labels, sample_size=min(2000, len(embs))))
        except Exception:
            sil = float("nan")
        # intra-cluster avg similarity
        intra_sims = []
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx) < 2:
                continue
            sub = sim_mat[np.ix_(idx, idx)]
            tri = sub[np.triu_indices(len(idx), k=1)]
            if len(tri) > 0:
                intra_sims.append(float(np.mean(tri)))
        results[str(k)] = {
            "inertia": float(km.inertia_),
            "silhouette": sil,
            "avg_intra_sim": float(np.mean(intra_sims)) if intra_sims else float("nan"),
            "cluster_sizes": [int((labels==c).sum()) for c in range(k)],
        }
    return results

# ── Utilities ─────────────────────────────────────────────────────────────────
def parse_dataset(filename):
    m = re.match(r"^(.+?)_times\d+", os.path.basename(filename))
    return m.group(1) if m else filename

def process_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    summaries = data.get("summaries", [])
    user_id   = data.get("input_id", os.path.basename(path))
    dataset   = parse_dataset(path)

    embs            = asyncio.run(get_embeddings(summaries))
    pairs, sim_mat  = pairwise_cosine(embs)

    a = dist_stats(pairs)
    b = {"effective_arms": effective_arms(sim_mat), "total_arms": len(summaries),
         "dup_threshold": DUP_THRESH}
    c = cluster_analysis(embs, sim_mat)
    centroid = embs.mean(axis=0).tolist()

    return {
        "dataset": dataset, "user_id": user_id,
        "n_pairs": int(len(pairs)),
        "distribution": a, "dedup": b, "clustering": c,
        "centroid": centroid,
    }

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    files = sorted(f for f in glob.glob(os.path.join(DATA_DIR, "*.json"))
                   if "similarity" not in os.path.basename(f)
                   and "analysis" not in os.path.basename(f))
    log(f"{'='*80}")
    log(f"  Full Persona Space Analysis")
    log(f"  Files: {len(files)}  |  K-means K={KMEANS_KS}  |  DupThresh={DUP_THRESH}")
    log(f"{'='*80}\n")

    user_results = []
    t_global = time.time()

    for idx, path in enumerate(files, 1):
        fname = os.path.basename(path)
        t0    = time.time()
        try:
            r       = process_file(path)
            elapsed = time.time() - t0
            done    = idx / len(files)
            eta     = (time.time() - t_global) / idx * (len(files) - idx)

            # Compact per-user progress line
            d = r["distribution"]
            b = r["dedup"]
            log(f"[{idx:3d}/{len(files)}] ✓ {fname}")
            log(f"         mean={d['mean']:.4f} std={d['std']:.4f} "
                f"p50={d['p50']:.4f} p90={d['p90']:.4f} p99={d['p99']:.4f}")
            log(f"         props: sim≥.99={d['proportions'].get('sim=1.00',0)+d['proportions'].get('0.99≤sim<1.00',0):.3%}"
                f"  ≥.95={d['proportions'].get('0.95≤sim<0.99',0):.3%}"
                f"  <.50={d['proportions'].get('sim<0.50',0):.3%}")
            log(f"         eff_arms={b['effective_arms']}/{b['total_arms']}"
                f"  ({b['effective_arms']/b['total_arms']:.1%} unique)")
            sil5  = r['clustering'].get('5',{}).get('silhouette','?')
            sil10 = r['clustering'].get('10',{}).get('silhouette','?')
            sil20 = r['clustering'].get('20',{}).get('silhouette','?')
            log(f"         silhouette K5={sil5:.3f} K10={sil10:.3f} K20={sil20:.3f}"
                f"  |  {elapsed:.1f}s  ETA {eta/60:.1f}min")
            log("")

            # Drop centroid from saved result (large, only needed for cross-user)
            save_r = {k: v for k, v in r.items() if k != "centroid"}
            save_r["centroid_norm"] = float(np.linalg.norm(r["centroid"]))  # sanity check only
            user_results.append(r)

            # Incremental save every 10 users
            if idx % 10 == 0 or idx == len(files):
                _save(user_results, files, t_global)
                log(f"  💾  Incremental save at [{idx}/{len(files)}]\n")

        except Exception as e:
            log(f"[{idx:3d}/{len(files)}] ✗ {fname}  ERROR: {e}\n")

    # ── D. Cross-user similarity ───────────────────────────────────────────
    log("="*80)
    log("  Phase D: Cross-user similarity (centroid-based)")
    log("="*80)
    cross = cross_user_similarity(user_results)
    for ds, v in cross.items():
        if "msg" in v:
            log(f"  {ds:<12}: {v['msg']}")
        else:
            log(f"  {ds:<12}: n={v['n_users']}  mean={v['mean']:.4f}"
                f"  std={v['std']:.4f}  p50={v['p50']:.4f}  min={v['min']:.4f}  max={v['max']:.4f}")

    _save(user_results, files, t_global, cross_user=cross)
    log(f"\n✅  All done in {(time.time()-t_global)/60:.1f}min  →  {OUT_JSON}")


def _save(user_results, files, t0, cross_user=None):
    """Save current results to JSON (without large centroid arrays)."""
    per_ds = collections.defaultdict(list)
    for r in user_results:
        per_ds[r["dataset"]].append(r)

    ds_summary = {}
    for ds, users in per_ds.items():
        means    = [u["distribution"]["mean"] for u in users]
        stds     = [u["distribution"]["std"]  for u in users]
        eff_rats = [u["dedup"]["effective_arms"]/u["dedup"]["total_arms"] for u in users]
        props_ge99 = [u["distribution"]["proportions"].get("sim=1.00",0) +
                      u["distribution"]["proportions"].get("0.99≤sim<1.00",0) for u in users]
        props_lt50 = [u["distribution"]["proportions"].get("sim<0.50",0) for u in users]
        sil10_list = [u["clustering"].get("10",{}).get("silhouette",float("nan")) for u in users]
        ds_summary[ds] = {
            "n_users": len(users),
            "mean_of_means": float(np.mean(means)),
            "mean_std":      float(np.mean(stds)),
            "effective_arm_ratio": {"mean": float(np.mean(eff_rats)), "std": float(np.std(eff_rats))},
            "prop_ge99_mean": float(np.mean(props_ge99)),
            "prop_lt50_mean": float(np.mean(props_lt50)),
            "silhouette_K10_mean": float(np.nanmean(sil10_list)),
        }

    out = {
        "meta": {"n_files": len(files), "n_processed": len(user_results),
                 "elapsed_min": round((time.time()-t0)/60, 1)},
        "per_user": [{k:v for k,v in r.items() if k != "centroid"} for r in user_results],
        "per_dataset": ds_summary,
    }
    if cross_user:
        out["cross_user_similarity"] = cross_user

    os.makedirs(os.path.dirname(os.path.abspath(OUT_JSON)), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ── D. Cross-user similarity ──────────────────────────────────────────────────
def cross_user_similarity(user_results):
    by_ds = collections.defaultdict(list)
    for r in user_results:
        by_ds[r["dataset"]].append(r)
    out = {}
    for ds, users in by_ds.items():
        if len(users) < 2:
            out[ds] = {"n_users": len(users), "msg": "not enough users"}
            continue
        centroids = np.array([u["centroid"] for u in users], dtype=np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        cn = centroids / np.where(norms==0, 1e-8, norms)
        mat = cn @ cn.T
        idx = np.triu_indices(len(users), k=1)
        sims = mat[idx]
        out[ds] = {
            "n_users": len(users),
            "mean": float(np.mean(sims)), "std": float(np.std(sims, ddof=1)),
            "min": float(np.min(sims)),   "max": float(np.max(sims)),
            "p25": float(np.percentile(sims,25)), "p50": float(np.percentile(sims,50)),
            "p75": float(np.percentile(sims,75)),
        }
    return out


if __name__ == "__main__":
    main()
