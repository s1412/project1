import json, os, numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PROJECT_ROOT, 'rephase_data_400', 'full_analysis_results.json')) as f:
    d2 = json.load(f)

DS_ORDER = ['LaMP4','LaMP5','LaMP8','LaMP9','LaMP10','LaMP-2','ultrachat']
users2 = d2['per_user']
cu     = d2['cross_user_similarity']

rows = []
for ds in DS_ORDER:
    grp = [u for u in users2 if u['dataset']==ds]
    n   = len(grp)
    mean    = np.mean([u['distribution']['mean'] for u in grp])
    std     = np.mean([u['distribution']['std']  for u in grp])
    p10     = np.mean([u['distribution']['p10']  for u in grp])
    p90     = np.mean([u['distribution']['p90']  for u in grp])
    props   = lambda u, k: u['distribution']['proportions'].get(k, 0)
    dup_ge95= np.mean([(props(u,'sim=1.00')+
                        props(u,'0.99\u2264sim<1.00')+
                        props(u,'0.95\u2264sim<0.99'))*100 for u in grp])
    eff_ratio= np.mean([u['dedup']['effective_arms']/u['dedup']['total_arms'] for u in grp])*100
    sil20   = np.nanmean([u['clustering'].get('20',{}).get('silhouette', float('nan')) for u in grp])
    intra   = mean
    inter   = cu[ds]['mean'] if 'mean' in cu[ds] else float('nan')
    gap     = intra - inter
    rows.append((ds, n, mean, std, p10, p90, dup_ge95, eff_ratio, sil20, gap))

# ── Markdown table ──────────────────────────────────────────────────────────
print('=== Markdown Table ===')
print('| Dataset   | N  | mu_sim | sigma | p10    | p90    | NearDup>=0.95 | Eff.Arm | Sil(K20) | Pers.Gap |')
print('|-----------|----|----|----|----|----|----|----|----|-----|')
for ds,n,mean,std,p10,p90,dup,eff,sil,gap in rows:
    print(f'| {ds:<9} | {n:>3} | {mean:.4f} | {std:.4f} | {p10:.4f} | {p90:.4f} | {dup:.4f}%      | {eff:.1f}%   | {sil:.4f}   | {gap:+.4f}  |')

# ── LaTeX table ─────────────────────────────────────────────────────────────
print()
print('=== LaTeX Table ===')
print(r'\begin{table}[ht]')
print(r'\centering')
print(r'\caption{Persona Space Quality Analysis across 7 Datasets (180 Users, 400 Candidates each).}')
print(r'\label{tab:persona_space_quality}')
print(r'\resizebox{\textwidth}{!}{%')
print(r'\begin{tabular}{l r r r r r r r r r}')
print(r'\toprule')
print(r'Dataset & $N$ & $\bar{\mu}$ & $\bar{\sigma}$ & $p_{10}$ & $p_{90}$'
      r' & \makecell{Near-dup \\ $\geq$0.95 (\%)} & \makecell{Eff. Arm \\ (\%)}'
      r' & \makecell{Silhouette \\ ($K$=20)} & Pers. Gap \\')
print(r'\midrule')
for ds,n,mean,std,p10,p90,dup,eff,sil,gap in rows:
    gap_str = f'$+{gap:.3f}$' if gap >= 0 else f'$-{abs(gap):.3f}$'
    print(f'{ds} & {n} & {mean:.4f} & {std:.4f} & {p10:.4f} & {p90:.4f}'
          f' & {dup:.4f} & {eff:.1f} & {sil:.4f} & {gap_str} \\\\')
print(r'\bottomrule')
print(r'\end{tabular}%')
print(r'}')
print(r'\end{table}')
