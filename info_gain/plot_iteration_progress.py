"""
绘制优化过程中的iteration-reward图

数据来源：past_average/ 文件夹
每个数据集一个JSON文件，包含40个counter的数据
每个counter有多个query，每个query有10个iteration的分数

归一化和平均流程：
1. 对于同一个counter、同一个query、同一个iteration的3个方法，除以最大值归一化
2. 对每个counter，将所有query的同一个iteration求平均（跨query平均）
3. 对所有counter的同一个iteration求平均（跨counter平均）
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def load_personaagent_data(dataset_name, results_dir="persona_results_rougL"):
    """
    加载PersonaAgent的数据

    Args:
        dataset_name: 数据集名称，如 "lamp4"
        results_dir: persona结果文件夹路径

    Returns:
        dict: {counter_id: {query_id: score}}
    """
    results_path = Path(results_dir)
    pattern = f"persona_{dataset_name}_counter*_all_algorithms.json"
    files = list(results_path.glob(pattern))

    personaagent_data = {}

    for file in files:
        # 提取counter编号
        # 文件名格式: persona_lamp4_counter127_all_algorithms.json
        parts = file.stem.split('_')
        counter_str = [p for p in parts if p.startswith('counter')][0]
        counter_id = int(counter_str.replace('counter', ''))

        with open(file, 'r', encoding='utf-8') as f:
            file_data = json.load(f)

        personaagent_data[counter_id] = {}

        # 提取每个query的PersonaAgent分数
        queries = file_data.get('queries', {})
        for query_name, query_data in queries.items():
            if query_name.startswith('query_'):
                query_id = int(query_name.split('_')[1])
                algorithms = query_data.get('algorithms', {})
                if 'PersonaAgent' in algorithms:
                    score = algorithms['PersonaAgent'].get('greedy_score', 0)
                    personaagent_data[counter_id][query_id] = score

    return personaagent_data


def load_and_process_dataset(json_file, dataset_name):
    """
    加载并处理单个数据集的JSON文件

    Args:
        json_file: JSON文件路径
        dataset_name: 数据集名称，用于加载PersonaAgent数据

    Returns:
        dict: {method: [10个iteration的平均分数]}
        float: PersonaAgent的平均归一化分数
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 加载PersonaAgent数据
    personaagent_data = load_personaagent_data(dataset_name)

    methods = ["ID-TAP", "POHF", "DoubleTS"]

    # 数据文件中的方法名到显示名称的映射
    method_mapping = {
        "POHF-InfoGain": "ID-TAP",
        "POHF": "POHF",
        "DoubleTS": "DoubleTS"
    }

    # 存储每个counter的iteration分数（已经跨query平均）
    counter_iteration_scores = {method: [] for method in methods}

    # 存储PersonaAgent的归一化分数
    personaagent_normalized_scores = []

    # 遍历每个counter
    for counter_name, counter_data in data.items():
        if not counter_name.startswith("counter_"):
            continue

        # 提取counter编号
        counter_id = int(counter_name.split('_')[1])

        by_query = counter_data.get("by_query", {})
        if not by_query:
            continue

        # 收集这个counter的所有query的数据
        # query_iteration_data[iteration_idx][method] = [query0_score, query1_score, ...]
        query_iteration_data = [{method: [] for method in methods} for _ in range(10)]

        # 遍历每个query
        for query_name, query_data in by_query.items():
            if not query_name.startswith("query_"):
                continue

            query_id = int(query_name.split('_')[1])

            # 获取每个方法的10个iteration分数
            method_scores = {}
            for data_method, display_method in method_mapping.items():
                if data_method in query_data:
                    method_scores[display_method] = query_data[data_method]

            # 如果某个方法缺失，跳过这个query
            if len(method_scores) != 3:
                continue

            # 🔧 [修改] 对这个query的所有iteration的所有方法 + PersonaAgent进行归一化
            # 收集这个query的所有3×10=30个分数
            all_scores = []
            for method in methods:
                all_scores.extend(method_scores[method])

            # 添加PersonaAgent的分数（如果存在）
            personaagent_score = None
            if counter_id in personaagent_data and query_id in personaagent_data[counter_id]:
                personaagent_score = personaagent_data[counter_id][query_id]
                all_scores.append(personaagent_score)

            # 找到最大值（3×10+1=31个分数）
            max_score = max(all_scores) if all_scores else 0

            # 归一化并存储
            if max_score > 0:
                # 归一化iteration分数
                for iter_idx in range(10):
                    for method in methods:
                        normalized_score = method_scores[method][iter_idx] / max_score
                        query_iteration_data[iter_idx][method].append(normalized_score)

                # 归一化PersonaAgent分数
                if personaagent_score is not None:
                    personaagent_normalized_scores.append(personaagent_score / max_score)
        
        # 对这个counter，计算每个iteration的跨query平均
        for iter_idx in range(10):
            for method in methods:
                if query_iteration_data[iter_idx][method]:
                    avg_score = np.mean(query_iteration_data[iter_idx][method])
                    counter_iteration_scores[method].append(avg_score)
    
    # 转换为numpy数组，方便计算
    # counter_iteration_scores[method] = [counter0_iter0, counter0_iter1, ..., counter1_iter0, ...]
    # 需要重新组织为 [所有counter的iter0平均, 所有counter的iter1平均, ...]
    
    final_scores = {method: [] for method in methods}
    
    for method in methods:
        scores_array = np.array(counter_iteration_scores[method])
        # 重塑为 (num_counters, 10)
        num_counters = len(scores_array) // 10
        if num_counters > 0:
            scores_reshaped = scores_array[:num_counters * 10].reshape(num_counters, 10)
            # 对每个iteration（列）求平均
            final_scores[method] = np.mean(scores_reshaped, axis=0).tolist()

    # 计算PersonaAgent的平均归一化分数
    personaagent_avg = np.mean(personaagent_normalized_scores) if personaagent_normalized_scores else 0

    return final_scores, personaagent_avg


def plot_iteration_progress(dataset_name, iteration_scores, personaagent_avg, save_path=None, show_plot=True):
    """
    绘制单个数据集的iteration进度图

    Args:
        dataset_name: 数据集名称
        iteration_scores: {method: [10个iteration分数]}
        personaagent_avg: PersonaAgent的平均归一化分数
        save_path: 保存路径
        show_plot: 是否显示图表
    """
    methods = ["ID-TAP", "POHF", "DoubleTS"]
    colors = {
        "ID-TAP": "#22BDD2",  # 青色
        "POHF": "#1B78B2",            # 蓝色
        "DoubleTS": "#9368AB",        # 紫色
    }
    markers = {
        "ID-TAP": "o",
        "POHF": "s",
        "DoubleTS": "^",
    }
    
    # 创建图表 - 10×7长方形
    plt.figure(figsize=(10, 7))
    
    # 绘制每个方法的折线
    for method in methods:
        if method in iteration_scores and iteration_scores[method]:
            iterations = list(range(1, len(iteration_scores[method]) + 1))
            plt.plot(iterations, iteration_scores[method],
                    label=method,
                    color=colors[method],
                    marker=markers[method],
                    markersize=12,
                    linewidth=4.0,
                    alpha=0.9,
                    markeredgewidth=2.0)

    # 绘制PersonaAgent的横线
    if personaagent_avg > 0:
        plt.axhline(y=personaagent_avg, color='#F47F1E', linestyle='-',
                   linewidth=4.0, alpha=0.9, label='PersonaAgent')

    # 设置图表属性
    plt.xlabel("Iteration", fontsize=24, fontweight='bold')
    plt.ylabel("Normalized Average Score", fontsize=24, fontweight='bold')

    # 设置x轴刻度：只显示1, 5, 10
    plt.xticks([1, 5, 10], fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(loc='best', fontsize=18, framealpha=0.9)

    # 只显示横向网格线（y方向）
    plt.grid(True, alpha=0.3, linestyle='-', axis='y', linewidth=1.5)

    # 加粗边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"✅ 图片已保存到: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def process_all_datasets(data_dir="past_average", save_dir="plots_iteration", show_plot=False):
    """
    处理所有数据集并生成图表

    Args:
        data_dir: 数据文件夹路径
        save_dir: 保存图片的文件夹
        show_plot: 是否显示图片

    Returns:
        dict: {dataset_name: iteration_scores}
    """
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*_counter_average_data_*.json"))

    print(f"🔍 找到 {len(json_files)} 个数据集文件\n")

    os.makedirs(save_dir, exist_ok=True)

    all_results = {}

    for json_file in sorted(json_files):
        # 提取数据集名称
        # 文件名格式: lamp4_counter_average_data_20260122_034109.json
        dataset_name = json_file.stem.split('_counter_average_data_')[0]

        print("=" * 60)
        print(f"📊 处理数据集: {dataset_name.upper()}")
        print("=" * 60)

        # 加载并处理数据
        iteration_scores, personaagent_avg = load_and_process_dataset(json_file, dataset_name)
        all_results[dataset_name] = (iteration_scores, personaagent_avg)

        # 打印统计信息
        print(f"\n📈 Iteration分数统计:")
        for method in ["ID-TAP", "POHF", "DoubleTS"]:
            if method in iteration_scores and iteration_scores[method]:
                scores = iteration_scores[method]
                print(f"  {method}:")
                print(f"    Iteration 1: {scores[0]:.4f}")
                print(f"    Iteration 10: {scores[-1]:.4f}")
                print(f"    平均: {np.mean(scores):.4f}")
        print(f"  PersonaAgent:")
        print(f"    平均归一化分数: {personaagent_avg:.4f}")

        # 绘制图表
        save_path = os.path.join(save_dir, f"{dataset_name}_iteration_progress.pdf")
        plot_iteration_progress(dataset_name, iteration_scores, personaagent_avg, save_path, show_plot)

        print(f"\n✅ {dataset_name.upper()} 处理完成！\n")

    print("=" * 60)
    print(f"🎉 所有数据集处理完成！共处理 {len(json_files)} 个数据集")
    print("=" * 60)

    return all_results


def plot_overall_average(all_results, save_path=None, show_plot=True):
    """
    绘制所有数据集的平均iteration进度（Overall图）

    对每个iteration，计算所有数据集的平均分数

    Args:
        all_results: process_all_datasets的返回值
        save_path: 保存路径，如果为None则保存到plots_iteration/overall_iteration_average.pdf
        show_plot: 是否显示图表
    """
    methods = ["ID-TAP", "POHF", "DoubleTS"]

    # 收集所有数据集的数据
    # overall_data[method][iter_idx] = [dataset1_score, dataset2_score, ...]
    overall_data = {method: [[] for _ in range(10)] for method in methods}
    personaagent_scores = []

    for dataset_name, (iteration_scores, personaagent_avg) in all_results.items():
        for method in methods:
            if method in iteration_scores and iteration_scores[method]:
                for iter_idx, score in enumerate(iteration_scores[method]):
                    overall_data[method][iter_idx].append(score)

        # 收集PersonaAgent分数
        if personaagent_avg > 0:
            personaagent_scores.append(personaagent_avg)

    # 计算每个iteration的平均分数
    overall_avg = {method: [] for method in methods}

    for method in methods:
        for iter_idx in range(10):
            if overall_data[method][iter_idx]:
                avg_score = np.mean(overall_data[method][iter_idx])
                overall_avg[method].append(avg_score)

    # 计算PersonaAgent的overall平均
    overall_personaagent_avg = np.mean(personaagent_scores) if personaagent_scores else 0

    print("\n" + "=" * 60)
    print("📊 Overall Average - Iteration进度统计")
    print("=" * 60)
    for method in methods:
        if overall_avg[method]:
            print(f"  {method}:")
            print(f"    Iteration 1: {overall_avg[method][0]:.4f}")
            print(f"    Iteration 10: {overall_avg[method][-1]:.4f}")
            print(f"    平均: {np.mean(overall_avg[method]):.4f}")
    print(f"  PersonaAgent:")
    print(f"    平均归一化分数: {overall_personaagent_avg:.4f}")
    print()

    # 绘图
    colors = {
        "ID-TAP": "#22BDD2",  # 青色
        "POHF": "#1B78B2",            # 蓝色
        "DoubleTS": "#9368AB",        # 紫色
    }
    markers = {
        "ID-TAP": "o",
        "POHF": "s",
        "DoubleTS": "^",
    }

    # 创建图表 - 10×7长方形
    plt.figure(figsize=(10, 7))

    # 绘制每个方法的折线
    for method in methods:
        if overall_avg[method]:
            iterations = list(range(1, len(overall_avg[method]) + 1))
            plt.plot(iterations, overall_avg[method],
                    label=method,
                    color=colors[method],
                    marker=markers[method],
                    markersize=12,
                    linewidth=4.0,
                    alpha=0.9,
                    markeredgewidth=2.0)

    # 绘制PersonaAgent的横线
    if overall_personaagent_avg > 0:
        plt.axhline(y=overall_personaagent_avg, color='#F47F1E', linestyle='-',
                   linewidth=4.0, alpha=0.9, label='PersonaAgent')

    # 设置图表属性
    plt.xlabel("Iteration", fontsize=24, fontweight='bold')
    plt.ylabel("Normalized Average Score", fontsize=24, fontweight='bold')

    # 设置x轴刻度：只显示1, 5, 10
    plt.xticks([1, 5, 10], fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(loc='best', fontsize=18, framealpha=0.9)

    # 只显示横向网格线（y方向）
    plt.grid(True, alpha=0.3, linestyle='-', axis='y', linewidth=1.5)

    # 加粗边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()

    # 保存图片
    if save_path is None:
        save_path = "plots_iteration/overall_iteration_average.pdf"

    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ Overall图片已保存到: {save_path}\n")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_lamp_only_average(all_results, save_path=None, show_plot=True):
    """
    绘制只包含LAMP数据集的平均iteration进度（排除prefeval和ultrachat）

    只使用lamp4, lamp5, lamp8, lamp9, lamp10这5个数据集

    Args:
        all_results: process_all_datasets的返回值
        save_path: 保存路径，如果为None则保存到plots_iteration/lamp_only_iteration_average.pdf
        show_plot: 是否显示图表
    """
    methods = ["ID-TAP", "POHF", "DoubleTS"]

    # 只选择LAMP数据集
    lamp_datasets = ['lamp4', 'lamp5', 'lamp8', 'lamp9', 'lamp10']

    # 收集LAMP数据集的数据
    # lamp_data[method][iter_idx] = [dataset1_score, dataset2_score, ...]
    lamp_data = {method: [[] for _ in range(10)] for method in methods}
    personaagent_scores = []

    for dataset_name, (iteration_scores, personaagent_avg) in all_results.items():
        # 只处理LAMP数据集
        if dataset_name not in lamp_datasets:
            continue

        for method in methods:
            if method in iteration_scores and iteration_scores[method]:
                for iter_idx, score in enumerate(iteration_scores[method]):
                    lamp_data[method][iter_idx].append(score)

        # 收集PersonaAgent分数
        if personaagent_avg > 0:
            personaagent_scores.append(personaagent_avg)

    # 计算每个iteration的平均分数
    lamp_avg = {method: [] for method in methods}

    for method in methods:
        for iter_idx in range(10):
            if lamp_data[method][iter_idx]:
                avg_score = np.mean(lamp_data[method][iter_idx])
                lamp_avg[method].append(avg_score)

    # 计算PersonaAgent的平均
    lamp_personaagent_avg = np.mean(personaagent_scores) if personaagent_scores else 0

    print("\n" + "=" * 60)
    print("📊 LAMP Only Average - Iteration进度统计 (5个数据集)")
    print("=" * 60)
    for method in methods:
        if lamp_avg[method]:
            print(f"  {method}:")
            print(f"    Iteration 1: {lamp_avg[method][0]:.4f}")
            print(f"    Iteration 10: {lamp_avg[method][-1]:.4f}")
            print(f"    平均: {np.mean(lamp_avg[method]):.4f}")
    print(f"  PersonaAgent:")
    print(f"    平均归一化分数: {lamp_personaagent_avg:.4f}")
    print()

    # 绘图
    colors = {
        "ID-TAP": "#22BDD2",  # 青色
        "POHF": "#1B78B2",            # 蓝色
        "DoubleTS": "#9368AB",        # 紫色
    }
    markers = {
        "ID-TAP": "o",
        "POHF": "s",
        "DoubleTS": "^",
    }

    # 创建图表 - 10×7长方形
    plt.figure(figsize=(10, 7))

    # 绘制每个方法的折线
    for method in methods:
        if lamp_avg[method]:
            iterations = list(range(1, len(lamp_avg[method]) + 1))
            plt.plot(iterations, lamp_avg[method],
                    label=method,
                    color=colors[method],
                    marker=markers[method],
                    markersize=12,
                    linewidth=4.0,
                    alpha=0.9,
                    markeredgewidth=2.0)

    # 绘制PersonaAgent的横线
    if lamp_personaagent_avg > 0:
        plt.axhline(y=lamp_personaagent_avg, color='#F47F1E', linestyle='-',
                   linewidth=4.0, alpha=0.9, label='PersonaAgent')

    # 设置图表属性
    plt.xlabel("Iteration", fontsize=24, fontweight='bold')
    plt.ylabel("Normalized Average Score", fontsize=24, fontweight='bold')

    # 设置x轴刻度：只显示1, 5, 10
    plt.xticks([1, 5, 10], fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(loc='best', fontsize=18, framealpha=0.9)

    # 只显示横向网格线（y方向）
    plt.grid(True, alpha=0.3, linestyle='-', axis='y', linewidth=1.5)

    # 加粗边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()

    # 保存图片
    if save_path is None:
        save_path = "plots_iteration/lamp_only_iteration_average.pdf"

    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ LAMP Only图片已保存到: {save_path}\n")

    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # 处理所有数据集
    all_results = process_all_datasets(show_plot=False)

    # 生成Overall平均图（所有7个数据集）
    plot_overall_average(all_results, show_plot=False)

    # 生成LAMP Only平均图（只有5个LAMP数据集）
    plot_lamp_only_average(all_results, show_plot=False)

