"""
可视化persona_results_rougL数据的脚本

功能：
1. 读取指定数据集（如lamp4）的所有counter文件
2. 对每个query计算4种方法的平均分数
3. 对同一个query进行最大值归一化
4. 绘制折线图比较4种方法
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import pandas as pd


def load_dataset_results(dataset_name, results_dir="persona_results_rougL"):
    """
    加载指定数据集的所有结果文件，并在每个counter的每个query内进行归一化

    Args:
        dataset_name: 数据集名称，如 "lamp4", "lamp5"
        results_dir: 结果文件夹路径

    Returns:
        dict: {
            "ID-TAP": {query_idx: [normalized_score1, normalized_score2, ...]},
            "POHF": {query_idx: [normalized_score1, normalized_score2, ...]},
            "DoubleTS": {query_idx: [normalized_score1, normalized_score2, ...]},
            "PersonaAgent": {query_idx: [normalized_score1, normalized_score2, ...]}
        }
    """
    methods = ["ID-TAP", "POHF", "DoubleTS", "PersonaAgent"]

    # 数据文件中的方法名到显示名称的映射
    method_mapping = {
        "POHF-InfoGain": "ID-TAP",
        "POHF": "POHF",
        "DoubleTS": "DoubleTS",
        "PersonaAgent": "PersonaAgent"
    }

    # 初始化数据结构
    data = {method: defaultdict(list) for method in methods}

    # 查找所有匹配的文件
    results_path = Path(results_dir)
    pattern = f"persona_{dataset_name}_counter*.json"
    files = list(results_path.glob(pattern))

    print(f"📂 找到 {len(files)} 个 {dataset_name} 的结果文件")

    # 读取每个文件
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                result = json.load(f)

            # 遍历所有query
            queries = result.get("queries", {})
            for query_key, query_data in queries.items():
                query_idx = query_data.get("query_index")
                algorithms = query_data.get("algorithms", {})

                # 🔧 [新逻辑] 先收集当前counter当前query的所有方法的分数
                current_scores = {}
                for data_method, display_method in method_mapping.items():
                    if data_method in algorithms:
                        score = algorithms[data_method].get("greedy_score")
                        if score is not None:
                            current_scores[display_method] = score

                # 找到当前counter当前query的最大分数
                if current_scores:
                    max_score = max(current_scores.values())

                    # 归一化并存储
                    for method, score in current_scores.items():
                        normalized_score = score / max_score if max_score > 0 else 0
                        data[method][query_idx].append(normalized_score)

        except Exception as e:
            print(f"⚠️  读取文件 {file_path.name} 时出错: {e}")

    return data


def compute_normalized_averages(data):
    """
    计算归一化后的平均分数

    注意：归一化已经在load_dataset_results中完成（每个counter的每个query内归一化）
    这里只需要计算每个query的平均值

    Args:
        data: load_dataset_results的返回值（已归一化的数据）

    Returns:
        dict: {
            "ID-TAP": {query_idx: avg_normalized_score},
            "POHF": {query_idx: avg_normalized_score},
            ...
        }
        dict: {query_idx: sample_count} 每个query的样本数量
    """
    methods = list(data.keys())

    # 获取所有query索引
    all_query_indices = set()
    for method_data in data.values():
        all_query_indices.update(method_data.keys())
    all_query_indices = sorted(all_query_indices)

    # 计算每个query的平均分数
    averaged_data = {method: {} for method in methods}
    query_counts = {}

    for query_idx in all_query_indices:
        # � [调试] 打印第一个query的信息
        if query_idx == 0:
            print(f"\n  🔍 Query {query_idx} 统计信息:")
            for method in methods:
                scores = data[method].get(query_idx, [])
                if scores:
                    print(f"     {method}: {len(scores)} 个归一化分数, 平均={np.mean(scores):.4f}, 最大={max(scores):.4f}, 最小={min(scores):.4f}")

        # 计算每个方法的平均值
        for method in methods:
            scores = data[method].get(query_idx, [])
            if scores:
                averaged_data[method][query_idx] = np.mean(scores)
            else:
                averaged_data[method][query_idx] = None

        # 记录样本数量
        for method in methods:
            if data[method].get(query_idx):
                query_counts[query_idx] = len(data[method][query_idx])
                break

    return averaged_data, query_counts


def plot_normalized_results(dataset_name, normalized_data, query_counts, 
                            save_path=None, show_plot=True):
    """
    绘制归一化后的折线图
    
    Args:
        dataset_name: 数据集名称
        normalized_data: compute_normalized_averages的返回值
        query_counts: 每个query的样本数量
        save_path: 保存图片的路径（可选）
        show_plot: 是否显示图片
    """
    methods = ["ID-TAP", "POHF", "DoubleTS", "PersonaAgent"]
    colors = {
        "ID-TAP": "#22BDD2",  # 青色
        "POHF": "#1B78B2",            # 蓝色
        "DoubleTS": "#9368AB",        # 紫色
        "PersonaAgent": "#F47F1E"     # 橙色
    }
    markers = {
        "ID-TAP": "o",
        "POHF": "s",
        "DoubleTS": "^",
        "PersonaAgent": "D"
    }
    
    # 创建图表
    # 🔧 设置为正方形：宽度 = 高度
    plt.figure(figsize=(10, 7))
    
    # 绘制每个方法的折线
    for method in methods:
        query_indices = sorted(normalized_data[method].keys())
        scores = [normalized_data[method][idx] for idx in query_indices]

        # 过滤掉None值
        valid_data = [(idx, score) for idx, score in zip(query_indices, scores) if score is not None]
        if valid_data:
            indices, scores = zip(*valid_data)
            # 🔧 横坐标从1开始：将query_idx加1
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
    # plt.title(f"Performance Comparison on {dataset_name.upper()}\n(Normalized by Max Score per Query)",
    #          fontsize=16, fontweight='bold', pad=20)

    # 🔧 设置y轴范围，缩小y轴长度
    #plt.ylim(0.5, 1.0)  # 可以根据需要调整这个范围

    # 设置刻度字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(loc='best', fontsize=18, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=1.5)

    # 加粗边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    
    # 添加样本数量标注
    # ax = plt.gca()
    # y_min, y_max = ax.get_ylim()
    # for query_idx, count in query_counts.items():
    #     plt.text(query_idx, y_min - 0.05 * (y_max - y_min), 
    #             f'n={count}', 
    #             ha='center', va='top', fontsize=9, color='gray')
    
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


def visualize_dataset(dataset_name, results_dir="persona_results_rougL", 
                     save_dir="plots", show_plot=True):
    """
    完整的可视化流程（主函数）
    
    Args:
        dataset_name: 数据集名称，如 "lamp4", "lamp5"
        results_dir: 结果文件夹路径
        save_dir: 保存图片的文件夹
        show_plot: 是否显示图片
        
    Returns:
        normalized_data: 归一化后的数据
        query_counts: 每个query的样本数量
    """
    print(f"\n{'='*60}")
    print(f"📊 开始处理数据集: {dataset_name.upper()}")
    print(f"{'='*60}\n")
    
    # 1. 加载数据
    data = load_dataset_results(dataset_name, results_dir)
    
    # 2. 计算归一化平均分数
    normalized_data, query_counts = compute_normalized_averages(data)
    
    # 3. 打印统计信息
    print(f"\n📈 统计信息:")
    for query_idx in sorted(query_counts.keys()):
        print(f"  Query {query_idx}: {query_counts[query_idx]} 个样本")
    
    # 4. 绘制图表
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_normalized_comparison.pdf")
    plot_normalized_results(dataset_name, normalized_data, query_counts, 
                           save_path=save_path, show_plot=show_plot)
    
    print(f"\n✅ {dataset_name.upper()} 处理完成！\n")
    
    return normalized_data, query_counts


def visualize_all_datasets(datasets=None, results_dir="persona_results_rougL",
                          save_dir="plots", show_plot=False):
    """
    批量可视化所有数据集

    Args:
        datasets: 数据集列表，如 ["lamp4", "lamp5", ...]。如果为None，自动检测
        results_dir: 结果文件夹路径
        save_dir: 保存图片的文件夹
        show_plot: 是否显示图片

    Returns:
        dict: {dataset_name: (normalized_data, query_counts)}
    """
    # 如果没有指定数据集，自动检测
    if datasets is None:
        results_path = Path(results_dir)
        all_files = list(results_path.glob("persona_*.json"))

        # 提取数据集名称
        dataset_names = set()
        for file in all_files:
            # 文件名格式: persona_lamp4_counter10108_all_algorithms.json
            parts = file.stem.split('_')
            if len(parts) >= 2:
                dataset_name = parts[1]  # lamp4, lamp5, etc.
                dataset_names.add(dataset_name)

        datasets = sorted(dataset_names)
        print(f"🔍 自动检测到 {len(datasets)} 个数据集: {datasets}\n")

    # 处理每个数据集
    all_results = {}
    for dataset in datasets:
        try:
            normalized_data, query_counts = visualize_dataset(
                dataset, results_dir, save_dir, show_plot
            )
            all_results[dataset] = (normalized_data, query_counts)
        except Exception as e:
            print(f"❌ 处理 {dataset} 时出错: {e}\n")

    print(f"\n{'='*60}")
    print(f"🎉 所有数据集处理完成！共处理 {len(all_results)} 个数据集")
    print(f"{'='*60}\n")

    return all_results


def plot_overall_average(all_results, save_path=None, show_plot=True):
    """
    绘制所有数据集的平均结果（Overall图）

    对每个query，计算所有数据集的平均分数

    Args:
        all_results: visualize_all_datasets的返回值
        save_path: 保存路径，如果为None则保存到plots/overall_average.pdf
        show_plot: 是否显示图表
    """
    methods = ["ID-TAP", "POHF", "DoubleTS", "PersonaAgent"]

    # 收集所有数据集的数据
    # overall_data[query_idx][method] = [dataset1_score, dataset2_score, ...]
    overall_data = {query_idx: {method: [] for method in methods} for query_idx in range(10)}

    for dataset_name, (normalized_data, query_counts) in all_results.items():
        for query_idx in range(10):
            for method in methods:
                if query_idx in normalized_data[method] and normalized_data[method][query_idx] is not None:
                    overall_data[query_idx][method].append(normalized_data[method][query_idx])

    # 计算每个query的平均分数
    overall_avg = {query_idx: {} for query_idx in range(10)}
    dataset_counts = {}

    for query_idx in range(10):
        for method in methods:
            if overall_data[query_idx][method]:
                overall_avg[query_idx][method] = np.mean(overall_data[query_idx][method])
                dataset_counts[query_idx] = len(overall_data[query_idx][method])

    # 打印统计信息
    print("\n" + "=" * 60)
    print("📊 Overall Average - 数据集统计")
    print("=" * 60)
    for query_idx in sorted(dataset_counts.keys()):
        print(f"  Query {query_idx}: {dataset_counts[query_idx]} 个数据集")
    print()

    # 绘图
    colors = {
        "ID-TAP": "#22BDD2",  # 青色
        "POHF": "#1B78B2",            # 蓝色
        "DoubleTS": "#9368AB",        # 紫色
        "PersonaAgent": "#F47F1E"     # 橙色
    }
    markers = {
        "ID-TAP": "o",
        "POHF": "s",
        "DoubleTS": "^",
        "PersonaAgent": "D"
    }

    # 创建图表 - 10×7长方形
    plt.figure(figsize=(10, 7))

    # 绘制每个方法的折线
    for method in methods:
        queries = []
        scores = []
        for query_idx in range(10):
            if method in overall_avg[query_idx]:
                queries.append(query_idx + 1)  # 从1开始
                scores.append(overall_avg[query_idx][method])

        if scores:
            plt.plot(queries, scores,
                    label=method,
                    color=colors[method],
                    marker=markers[method],
                    markersize=12,
                    linewidth=4.0,
                    alpha=0.9,
                    markeredgewidth=2.0)

    # 设置图表属性
    plt.xlabel("Query", fontsize=24, fontweight='bold')
    plt.ylabel("Normalized Average Score", fontsize=24, fontweight='bold')

    # 设置x轴刻度：只显示1, 5, 10
    plt.xticks([1, 5, 10], fontsize=20)
    plt.yticks(fontsize=20)

    # 设置y轴范围
    plt.ylim(0.5, 1.0)

    plt.legend(loc='best', fontsize=18, framealpha=0.9)

    # 只显示横向网格线（y方向）
    plt.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=1.5)

    # 加粗边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()

    # 保存图片
    if save_path is None:
        save_path = "plots/overall_average.pdf"

    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ Overall图片已保存到: {save_path}\n")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_lamp_only_average(all_results, save_path=None, show_plot=True):
    """
    绘制只包含LAMP数据集的平均结果（排除prefeval和ultrachat）

    只使用lamp4, lamp5, lamp8, lamp9, lamp10这5个数据集

    Args:
        all_results: visualize_all_datasets的返回值
        save_path: 保存路径，如果为None则保存到plots/lamp_only_average.pdf
        show_plot: 是否显示图表
    """
    methods = ["ID-TAP", "POHF", "DoubleTS", "PersonaAgent"]

    # 只选择LAMP数据集
    lamp_datasets = ['lamp4', 'lamp5', 'lamp8', 'lamp9', 'lamp10']

    # 收集LAMP数据集的数据
    # lamp_data[query_idx][method] = [dataset1_score, dataset2_score, ...]
    lamp_data = {query_idx: {method: [] for method in methods} for query_idx in range(10)}

    for dataset_name, (normalized_data, query_counts) in all_results.items():
        # 只处理LAMP数据集
        if dataset_name not in lamp_datasets:
            continue

        for query_idx in range(10):
            for method in methods:
                if query_idx in normalized_data[method] and normalized_data[method][query_idx] is not None:
                    lamp_data[query_idx][method].append(normalized_data[method][query_idx])

    # 计算每个query的平均分数
    lamp_avg = {query_idx: {} for query_idx in range(10)}

    for query_idx in range(10):
        for method in methods:
            if lamp_data[query_idx][method]:
                lamp_avg[query_idx][method] = np.mean(lamp_data[query_idx][method])

    # 打印统计信息
    print("\n" + "=" * 60)
    print("📊 LAMP Only Average - Query-level统计 (5个数据集)")
    print("=" * 60)
    for method in methods:
        scores = [lamp_avg[q][method] for q in range(10) if method in lamp_avg[q]]
        if scores:
            print(f"  {method}: {np.mean(scores):.4f}")
    print()

    # 绘图
    colors = {
        "ID-TAP": "#22BDD2",  # 青色
        "POHF": "#1B78B2",            # 蓝色
        "DoubleTS": "#9368AB",        # 紫色
        "PersonaAgent": "#F47F1E"     # 橙色
    }
    markers = {
        "ID-TAP": "o",
        "POHF": "s",
        "DoubleTS": "^",
        "PersonaAgent": "D"
    }

    # 创建图表 - 10×7长方形
    plt.figure(figsize=(10, 7))

    # 绘制每个方法的折线
    for method in methods:
        queries = []
        scores = []
        for query_idx in range(10):
            if method in lamp_avg[query_idx]:
                queries.append(query_idx + 1)  # 从1开始
                scores.append(lamp_avg[query_idx][method])

        if scores:
            plt.plot(queries, scores,
                    label=method,
                    color=colors[method],
                    marker=markers[method],
                    markersize=12,
                    linewidth=4.0,
                    alpha=0.9,
                    markeredgewidth=2.0)

    # 设置图表属性
    plt.xlabel("Query", fontsize=24, fontweight='bold')
    plt.ylabel("Normalized Average Score", fontsize=24, fontweight='bold')

    # 设置x轴刻度：只显示1, 5, 10
    plt.xticks([1, 5, 10], fontsize=20)
    plt.yticks(fontsize=20)

    # 设置y轴范围
    plt.ylim(0.5, 1.0)

    plt.legend(loc='best', fontsize=18, framealpha=0.9)

    # 只显示横向网格线（y方向）
    plt.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=1.5)

    # 加粗边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()

    # 保存图片
    if save_path is None:
        save_path = "plots/lamp_only_average.pdf"

    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ LAMP Only图片已保存到: {save_path}\n")

    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_summary_table(all_results, save_path="plots/summary_table.csv"):
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
        methods = ["ID-TAP", "POHF", "DoubleTS", "PersonaAgent"]

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
                    "Avg_Normalized_Score": f"{avg_score:.4f}",
                    "Std": f"{std_score:.4f}",
                    "Num_Queries": num_queries
                })

    df = pd.DataFrame(rows)

    # 保存到CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n📊 汇总表格已保存到: {save_path}")

    # 打印表格
    print("\n" + "="*80)
    print("📈 所有数据集的性能汇总")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")

    return df


def plot_method_comparison_across_datasets(all_results, save_path="plots/method_comparison.pdf"):
    """
    绘制不同方法在所有数据集上的性能对比柱状图

    Args:
        all_results: visualize_all_datasets的返回值
        save_path: 保存图片的路径
    """
    methods = ["ID-TAP", "POHF", "DoubleTS", "PersonaAgent"]
    datasets = sorted(all_results.keys())

    # 准备数据
    data_matrix = {method: [] for method in methods}

    for dataset in datasets:
        normalized_data, _ = all_results[dataset]
        for method in methods:
            scores = [s for s in normalized_data[method].values() if s is not None]
            avg_score = np.mean(scores) if scores else 0
            data_matrix[method].append(avg_score)

    # 绘制柱状图
    x = np.arange(len(datasets))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {
        "ID-TAP": "#22BDD2",  # 青色
        "POHF": "#1B78B2",            # 蓝色
        "DoubleTS": "#9368AB",        # 紫色
        "PersonaAgent": "#F47F1E"     # 橙色
    }

    for i, method in enumerate(methods):
        offset = width * (i - 1.5)
        ax.bar(x + offset, data_matrix[method], width,
               label=method, color=colors[method], alpha=0.9, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Dataset', fontsize=24, fontweight='bold')
    ax.set_ylabel('Average Normalized Score', fontsize=24, fontweight='bold')
    ax.set_title('Method Comparison Across All Datasets', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets], rotation=15, ha='right', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(loc='best', fontsize=18)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=1.5)

    # 加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ 方法对比图已保存到: {save_path}")
    plt.close()


if __name__ == "__main__":
    # 示例1：可视化单个数据集
    # visualize_dataset("lamp4", show_plot=False)

    # 示例2：批量可视化所有数据集
    all_results = visualize_all_datasets(show_plot=False)

    # 生成Overall平均图（所有7个数据集）
    plot_overall_average(all_results, show_plot=False)

    # 生成LAMP Only平均图（只有5个LAMP数据集）
    plot_lamp_only_average(all_results, show_plot=False)

    # 生成汇总表格
    generate_summary_table(all_results)

    # 生成方法对比图
    plot_method_comparison_across_datasets(all_results)

    # 示例3：可视化指定的数据集
    # visualize_all_datasets(datasets=["lamp4", "lamp5", "lamp8"], show_plot=False)

