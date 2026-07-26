#!/usr/bin/env python3
"""
分析日志文件中的 Query 相似度分布

功能:
1. 提取所有 cosine_sim 值并统计分布
2. 找出高相似度的 Query 对
3. 分析为什么某些 Query 之间相似度很高

用法:
    python analyze_query_similarity.py <log_file> [--threshold 0.8]
"""

import re
import sys
import argparse
from collections import defaultdict
import numpy as np


def parse_log_file(log_path):
    """解析日志文件，提取相似度信息和Query文本"""

    # 正则表达式
    pattern1 = r'历史Query (\d+): cosine_sim=([0-9.]+) → sigmoid_weight=([0-9.]+)'
    pattern2 = r'Query (\d+): cosine_sim=([0-9.]+) → weight=([0-9.]+)'
    pattern_current_query = r'\[概率矩阵继承\] Query (\d+): 聚合'

    # 提取 Query 文本的正则
    pattern_counter = r'counter=(\d+)'
    pattern_query_text = r'^\s+\[(\d+)\]:\s*(.+)$'

    similarities = []
    high_sim_pairs = []
    current_query_idx = None
    current_counter = None

    # 存储每个 (counter, query_idx) 对应的 Query 文本
    query_texts = {}  # {(counter, query_idx): text}

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 第一遍：提取 Query 文本
    in_query_section = False
    for i, line in enumerate(lines):
        # 检测 counter (格式: counter=14621)
        match_counter = re.search(pattern_counter, line)
        if match_counter:
            current_counter = int(match_counter.group(1))

        # 检测 query 列表开始 (格式: "[1] query (type=list):")
        if '[1] query' in line and 'type=' in line:
            in_query_section = True
            continue

        # 提取 query 文本 (格式: "       [0]:  A critical change...")
        if in_query_section and current_counter is not None:
            match_text = re.match(pattern_query_text, line)
            if match_text:
                q_idx = int(match_text.group(1))
                q_text = match_text.group(2).strip()
                query_texts[(current_counter, q_idx)] = q_text
            # 遇到新的 section 标记则结束 query 提取
            elif '[2]' in line or '[3]' in line or '======' in line:
                in_query_section = False

    # 第二遍：提取相似度信息
    current_counter = None
    for line in lines:
        match_counter = re.search(pattern_counter, line)
        if match_counter:
            current_counter = int(match_counter.group(1))

        match_current = re.search(pattern_current_query, line)
        if match_current:
            current_query_idx = int(match_current.group(1))

        match1 = re.search(pattern1, line)
        if match1:
            hist_query = int(match1.group(1))
            sim = float(match1.group(2))
            weight = float(match1.group(3))
            similarities.append(sim)
            if current_query_idx is not None:
                high_sim_pairs.append((current_counter, current_query_idx, hist_query, sim, weight))
            continue

        match2 = re.search(pattern2, line)
        if match2:
            hist_query = int(match2.group(1))
            sim = float(match2.group(2))
            weight = float(match2.group(3))
            similarities.append(sim)
            if current_query_idx is not None:
                high_sim_pairs.append((current_counter, current_query_idx, hist_query, sim, weight))

    return similarities, high_sim_pairs, query_texts


def analyze_distribution(similarities, threshold=0.8):
    """分析相似度分布"""
    if not similarities:
        print("❌ 未找到相似度数据")
        return
    
    sims = np.array(similarities)
    
    print("\n" + "="*60)
    print("📊 Query 相似度分布分析")
    print("="*60)
    
    print(f"\n📈 基础统计:")
    print(f"   样本数: {len(sims)}")
    print(f"   最小值: {sims.min():.4f}")
    print(f"   最大值: {sims.max():.4f}")
    print(f"   平均值: {sims.mean():.4f}")
    print(f"   标准差: {sims.std():.4f}")
    
    print(f"\n📊 分位数:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"   {p}%: {np.percentile(sims, p):.4f}")
    
    print(f"\n📉 分布区间:")
    bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for low, high in bins:
        count = np.sum((sims >= low) & (sims < high))
        pct = count / len(sims) * 100
        bar = "█" * int(pct / 2)
        print(f"   [{low:.1f}, {high:.1f}): {count:4d} ({pct:5.1f}%) {bar}")
    
    # 高相似度统计
    high_count = np.sum(sims >= threshold)
    print(f"\n⚠️  相似度 >= {threshold} 的样本: {high_count} ({high_count/len(sims)*100:.1f}%)")


def show_high_similarity_pairs(pairs, query_texts, threshold=0.8, top_n=10):
    """显示高相似度的 Query 对及其文本"""
    high_pairs = [(cnt, cq, hq, sim, w) for cnt, cq, hq, sim, w in pairs if sim >= threshold]

    # 去重：同一个 (counter, current_query, hist_query) 只保留一次
    seen = set()
    unique_pairs = []
    for cnt, cq, hq, sim, w in high_pairs:
        key = (cnt, cq, hq)
        if key not in seen:
            seen.add(key)
            unique_pairs.append((cnt, cq, hq, sim, w))

    unique_pairs.sort(key=lambda x: x[3], reverse=True)

    if not unique_pairs:
        print(f"\n✅ 没有发现相似度 >= {threshold} 的 Query 对")
        return

    print(f"\n" + "="*80)
    print(f"🔥 高相似度 Query 对 (top {min(top_n, len(unique_pairs))})")
    print("="*80)

    for i, (cnt, cq, hq, sim, w) in enumerate(unique_pairs[:top_n]):
        print(f"\n[{i+1}] Counter={cnt}, Query {cq} ↔ Query {hq}")
        print(f"    相似度: {sim:.4f}  权重: {w:.4f}")

        # 获取 Query 文本
        curr_text = query_texts.get((cnt, cq), "未找到")
        hist_text = query_texts.get((cnt, hq), "未找到")

        print(f"    📄 Query {cq}: {curr_text[:150]}..." if len(curr_text) > 150 else f"    📄 Query {cq}: {curr_text}")
        print(f"    📄 Query {hq}: {hist_text[:150]}..." if len(hist_text) > 150 else f"    📄 Query {hq}: {hist_text}")

    # 分析高相似度的原因
    print(f"\n" + "="*80)
    print(f"💡 高相似度分析")
    print("="*80)
    print(f"   高相似度对数量 (去重后): {len(unique_pairs)}")

    consecutive_count = sum(1 for cnt, cq, hq, sim, w in unique_pairs if cq - hq == 1)
    print(f"   连续 Query 对 (差值=1): {consecutive_count} ({consecutive_count/len(unique_pairs)*100:.1f}%)")

    # ⚠️ 神经网络区分度分析
    print(f"\n" + "="*80)
    print(f"⚠️ 神经网络区分度警告")
    print("="*80)

    very_high = [p for p in unique_pairs if p[3] >= 0.9]
    high = [p for p in unique_pairs if 0.8 <= p[3] < 0.9]

    print(f"\n   🔴 极高相似度 (≥0.9): {len(very_high)} 对")
    print(f"      → 神经网络可能完全无法区分这些 Query")
    print(f"      → 如果这些 Query 的偏好不同，会导致训练冲突")

    print(f"\n   🟡 高相似度 (0.8-0.9): {len(high)} 对")
    print(f"      → 神经网络区分能力受限")

    # 计算理论上的输入向量相似度
    print(f"\n   📐 理论分析 (假设 summary 完全随机):")
    for sim_val in [0.95, 0.90, 0.85, 0.80]:
        # 输入 = [query_emb | summary_emb]
        # 如果 query 相似度 = sim_val，summary 相似度 = 0.5（随机）
        # 完整输入相似度 ≈ (sim_val + 0.5) / 2 (简化估计)
        full_sim = (sim_val * 1024 + 0.5 * 1024) / 2048
        print(f"      query_sim={sim_val:.2f}, summary_sim=0.50 → full_input_sim≈{full_sim:.2f}")


def main():
    parser = argparse.ArgumentParser(description='分析日志中的 Query 相似度')
    parser.add_argument('log_file', help='日志文件路径')
    parser.add_argument('--threshold', type=float, default=0.8, help='高相似度阈值')
    parser.add_argument('--top', type=int, default=20, help='显示的高相似度对数量')
    args = parser.parse_args()
    
    print(f"📂 分析文件: {args.log_file}")

    similarities, pairs, query_texts = parse_log_file(args.log_file)
    analyze_distribution(similarities, args.threshold)
    show_high_similarity_pairs(pairs, query_texts, args.threshold, args.top)


if __name__ == "__main__":
    main()

