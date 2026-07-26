#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制消融实验结果 (Ablation Study)
处理 ablation 文件夹下的数据，格式与 persona_results_rougL 相同
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_dataset_results(dataset_name, data_dir="ablation"):
    """
    加载单个数据集的所有counter结果

    Args:
        dataset_name: 数据集名称 (如 'lamp4')
        data_dir: 数据目录

    Returns:
        normalized_data: {method: {query_idx: normalized_score}}
        query_counts: {query_idx: counter_count}
    """
    # 查找数据文件
    data_files = list(Path(data_dir).glob(f"{dataset_name}_counter_average_data_*.json"))

    if not data_files:
        print(f"⚠️ 未找到 {dataset_name} 的数据文件")
        return None, None

    data_file = data_files[0]
    print(f"\n📂 找到 {len(data_files)} 个 {dataset_name} 的结果文件")

    with open(data_file, 'r', encoding='utf-8') as f:
        all_counters = json.load(f)

    # 5个方法
    methods = ["ID-TAP", "POHF-Random", "POHF-RandomPair", "Random", "ID-TAP-NoHistory"]

    # 数据文件中的方法名到显示名称的映射
    method_mapping = {
        "POHF-InfoGain": "ID-TAP",
        "POHF-Random": "POHF-Random",
        "POHF-RandomPair": "POHF-RandomPair",
        "Random": "Random",
        "POHF-InfoGain-NoHistory": "ID-TAP-NoHistory"
    }

    # 存储每个query的所有counter的原始分数
    raw_scores = {method: {query_idx: [] for query_idx in range(10)} for method in methods}

    # 遍历所有counter
    for counter_name, counter_data in all_counters.items():
        if 'by_query' not in counter_data:
            continue

        by_query = counter_data['by_query']

        # 遍历每个query
        for query_key, query_data in by_query.items():
            query_idx = int(query_key.split('_')[1])

            # 遍历每个方法
            for data_method, display_method in method_mapping.items():
                if data_method in query_data:
                    # query_data[method] 是一个包含10个iteration分数的列表
                    # 我们取最后一个iteration的分数（iteration 10）
                    scores_list = query_data[data_method]
                    if scores_list and len(scores_list) > 0:
                        final_score = scores_list[-1]  # 取最后一个iteration
                        raw_scores[display_method][query_idx].append(final_score)

    # 归一化：同一个counter同一个query的5个方法的分数，找最大值进行归一化
    normalized_data = {method: {} for method in methods}
    query_counts = {}

    # 获取所有counter的数量（用于统计）
    num_counters = len(all_counters)

    for query_idx in range(10):
        # 收集当前query所有counter的所有方法的分数
        all_scores_this_query = []
        for method in methods:
            all_scores_this_query.extend(raw_scores[method][query_idx])

        if not all_scores_this_query:
            continue

        max_score = max(all_scores_this_query)

        # 归一化每个方法
        for method in methods:
            scores = raw_scores[method][query_idx]
            if scores:
                # 对每个counter的分数进行归一化，然后取平均
                normalized_scores = [s / max_score if max_score > 0 else 0 for s in scores]
                avg_normalized = np.mean(normalized_scores)
                normalized_data[method][query_idx] = avg_normalized

        query_counts[query_idx] = len(raw_scores[methods[0]][query_idx])

    # 打印统计信息
    print(f"\n  🔍 Query 0 统计信息:")
    for method in methods:
        if 0 in raw_scores[method] and raw_scores[method][0]:
            scores = raw_scores[method][0]
            print(f"     {method}: {len(scores)} 个归一化分数, 平均={np.mean(scores):.4f}, 最大={max(scores):.4f}, 最小={min(scores):.4f}")

    print(f"\n📈 统计信息:")
    for query_idx in sorted(query_counts.keys()):
        print(f"  Query {query_idx}: {query_counts[query_idx]} 个样本")

    return normalized_data, query_counts



def plot_normalized_results(dataset_name, normalized_data, save_path=None, show_plot=True):
    """
    绘制归一化后的结果对比图

    Args:
        dataset_name: 数据集名称
        normalized_data: {method: {query_idx: normalized_score}}
        save_path: 保存路径
        show_plot: 是否显示图表
    """
    methods = ["ID-TAP", "POHF-Random", "POHF-RandomPair", "Random", "ID-TAP-NoHistory"]
    colors = {
        "ID-TAP": "#22BDD2",      # 青色
        "POHF-Random": "#1B78B2",        # 蓝色
        "POHF-RandomPair": "#9368AB",    # 紫色
        "Random": "#F47F1E",             # 橙色
        "ID-TAP-NoHistory": "#E74C3C"  # 红色
    }
    markers = {
        "ID-TAP": "o",
        "POHF-Random": "s",
        "POHF-RandomPair": "^",
        "Random": "D",
        "ID-TAP-NoHistory": "v"
    }

    # 创建图表 - 10×7长方形
    plt.figure(figsize=(10, 7))

    # 绘制每个方法的折线
    for method in methods:
        query_indices = sorted(normalized_data[method].keys())
        scores = [normalized_data[method][idx] for idx in query_indices]

        # 过滤掉None值
        valid_data = [(idx, score) for idx, score in zip(query_indices, scores) if score is not None]
        if valid_data:
            indices, scores = zip(*valid_data)
            # 横坐标从1开始：将query_idx加1
            indices_display = [idx + 1 for idx in indices]
            plt.plot(indices_display, scores,
                    label=method,
                    color=colors[method],
                    marker=markers[method],
                    markersize=12,
                    linewidth=4.0,
                    alpha=0.9,
                    markeredgewidth=2.0)

    # 设置图表属性
    plt.xlabel("Query Index", fontsize=24, fontweight='bold')
    plt.ylabel("Normalized Average Score", fontsize=24, fontweight='bold')

    # 设置刻度字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(loc='best', fontsize=18, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=1.5)

    # 加粗边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"✅ 图片已保存到: {save_path}")

    # 显示图片
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_all_datasets(data_dir="ablation", save_dir="plots_ablation", show_plot=False):
    """
    可视化所有数据集的结果

    Args:
        data_dir: 数据目录
        save_dir: 保存目录
        show_plot: 是否显示图表

    Returns:
        all_results: {dataset_name: (normalized_data, query_counts)}
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 自动检测数据集
    data_files = list(Path(data_dir).glob("*_counter_average_data_*.json"))
    datasets = sorted(list(set([f.name.split('_counter_')[0] for f in data_files])))

    print(f"🔍 自动检测到 {len(datasets)} 个数据集: {datasets}\n")

    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"📊 开始处理数据集: {dataset_name.upper()}")
        print(f"{'='*60}")

        # 加载数据
        normalized_data, query_counts = load_dataset_results(dataset_name, data_dir)

        if normalized_data is None:
            continue

        # 保存结果
        all_results[dataset_name] = (normalized_data, query_counts)

        # 绘制图表
        save_path = os.path.join(save_dir, f"{dataset_name}_ablation_comparison.pdf")
        plot_normalized_results(dataset_name, normalized_data, save_path, show_plot)

        print(f"\n✅ {dataset_name.upper()} 处理完成！\n")

    print(f"\n{'='*60}")
    print(f"🎉 所有数据集处理完成！共处理 {len(all_results)} 个数据集")
    print(f"{'='*60}\n")

    return all_results


def generate_summary_table(all_results, save_path="plots_ablation/ablation_summary_table.csv"):
    """
    生成所有数据集的汇总表格

    Args:
        all_results: visualize_all_datasets的返回值
        save_path: 保存CSV文件的路径

    Returns:
        DataFrame: 汇总表格
    """
    rows = []

    for dataset_name, (normalized_data, query_counts) in all_results.items():
        methods = ["ID-TAP", "POHF-Random", "POHF-RandomPair", "Random", "ID-TAP-NoHistory"]

        # 计算每个方法的平均归一化分数
        for method in methods:
            scores = [s for s in normalized_data[method].values() if s is not None]
            if scores:
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                num_queries = len(scores)

                rows.append({
                    "Dataset": dataset_name.upper(),
                    "Method": method,
                    "Avg_Normalized_Score": f"{avg_score:.4f} ± {std_score:.4f}",
                    "Num_Queries": num_queries
                })

    df = pd.DataFrame(rows)

    # 保存到CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n📊 汇总表格已保存到: {save_path}")

    # 打印表格
    print("\n" + "="*80)
    print("📈 消融实验 - 所有数据集的性能汇总")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")

    return df


if __name__ == "__main__":
    # 可视化所有数据集
    all_results = visualize_all_datasets(data_dir="ablation", save_dir="plots_ablation", show_plot=False)

    # 生成汇总表格
    summary_df = generate_summary_table(all_results, save_path="plots_ablation/ablation_summary_table.csv")

    print("\n✅ 所有图表和表格已生成！")

