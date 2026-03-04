"""
精简版 Evaluator - 仅支持 --persona 模式的 LLM-as-Judge 评估
使用 history_context 生成 naive response
"""
import os
import json
import glob
import math
import statistics
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI

from POHF_parameters import API_CONFIG


RESULTS_BASE_DIR = "./final_re"
PERSONA_RESULTS_DIR = "./persona_results"
FINAL_SU_DIR = "./final_su"


_history_context_cache: Dict[str, Dict[int, Dict[int, List[str]]]] = {}
_history_context_cache_lock = Lock()


_original_summary_cache: Dict[str, Dict[int, str]] = {}
_original_summary_cache_lock = Lock()


def _load_history_contexts(dataset_name: str) -> Dict[int, Dict[int, List[str]]]:
    filepath = os.path.join(FINAL_SU_DIR, f"{dataset_name}_greedy_prompts.json")
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: {filepath} not found")
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result = {}
        for item in data.get('counters', []):
            counter = item.get('counter')
            history_context = item.get('history_context', [])
            if counter is not None and history_context:
                counter_int = int(counter)
                result[counter_int] = {}

                if history_context and isinstance(history_context[0], list):

                    for query_idx, history_list in enumerate(history_context):
                        if isinstance(history_list, list):
                            result[counter_int][query_idx] = history_list
                else:
                    result[counter_int][0] = history_context
        return result
    except Exception as e:
        print(f"⚠️ Error loading {filepath}: {e}")
        return {}


def _load_original_summaries(dataset_name: str) -> Dict[int, str]:
    filepath = os.path.join(FINAL_SU_DIR, f"{dataset_name}_greedy_prompts.json")
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result = {}
        for item in data.get('counters', []):
            counter = item.get('counter')
            original_summary = item.get('original_summary', '')
            if counter is not None and original_summary:
                result[int(counter)] = original_summary
        return result
    except Exception as e:
        print(f"⚠️ Error loading original_summary from {filepath}: {e}")
        return {}


def _get_history_context(dataset_name: str, counter: int, query_index: int) -> List[str]:
    global _history_context_cache
    with _history_context_cache_lock:
        if dataset_name not in _history_context_cache:
            _history_context_cache[dataset_name] = _load_history_contexts(dataset_name)
        return _history_context_cache[dataset_name].get(counter, {}).get(query_index, [])


def _get_original_summary(dataset_name: str, counter: int) -> str:
    global _original_summary_cache
    with _original_summary_cache_lock:
        if dataset_name not in _original_summary_cache:
            _original_summary_cache[dataset_name] = _load_original_summaries(dataset_name)
        return _original_summary_cache[dataset_name].get(counter, '')


MAX_WORKERS = 20
NUM_REPEATS = 3
PRINT_NAIVE_RESPONSE = True


DATASET_ORDER = ["ultrachat", "prefeval", "lamp4", "lamp5", "lamp8", "lamp9", "lamp10"]

EVAL_PARALLEL_CONFIG = {"enabled": True, "max_workers": MAX_WORKERS, "show_progress": True}
EVAL_REPEAT_CONFIG = {"num_repeats": NUM_REPEATS}
ALGORITHM_EVAL_ORDER = ["IDS_TAP-InfoGain", "IDS_TAP", "DoubleTS", "PersonaAgent", "PersonaAgent_llm_as_judge", "PersonaAgent_rougeL"]
LAMP_INSTRUCTIONS = {
    "lamp4": "Generate a headline for the following article:",
    "lamp5": "Generate a title for the following abstract of a paper:",
    "prefeval": "Predict the user's preference based on the interaction history:",
}

SYSTEM_ROLE_GENERATION = "give response based on history context"
SYSTEM_ROLE_COMPARISON = "You are an expert evaluator. Your task is to compare two responses and determine which one is better based on the given ground truth. The PRIMARY consideration is: text similarity, content accuracy, structure alignment, and semantic coherence. The SECONDARY consideration is: language style and personality characteristics. You must respond with ONLY a single digit: 1 if Response 1 is better, or 0 if Response 2 is better. Do not include any other text, explanation, or formatting."


def get_llm_client():
    return OpenAI(api_key=API_CONFIG.get("openai_api_key"), base_url=API_CONFIG.get("openai_base_url"))


def call_llm(prompt: str, max_retries: int = 5, system_role: str = None) -> str:
    import time, random
    client = get_llm_client()
    model = API_CONFIG.get("openai_model", "deepseek/deepseek-v3.2")
    if system_role is None:
        system_role = SYSTEM_ROLE_COMPARISON
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_role}, {"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.uniform(0, 1))
            else:
                return ""
    return ""


def load_persona_results(data_dir: str) -> Dict[str, List[Dict]]:
    """加载 persona_results 目录中的 JSON 文件。"""
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    datasets = {}
    for filepath in sorted(json_files):
        try:
            filename = os.path.basename(filepath)
            parts = filename.replace('.json', '').split('_')
            if len(parts) >= 4 and parts[0] == 'persona':
                dataset_name = parts[1]
                counter = parts[2].replace('counter', '')
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if dataset_name not in datasets:
                    datasets[dataset_name] = []
                datasets[dataset_name].append({
                    'counter': counter,
                    'queries': data.get('queries', {}),
                    '_filepath': filepath
                })
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    for ds, records in datasets.items():
        print(f"Loaded {len(records)} files for dataset: {ds}")
    return datasets


def _sort_algorithms_by_priority(algorithm_names: set) -> List[str]:
    ordered = [alg for alg in ALGORITHM_EVAL_ORDER if alg in algorithm_names]
    ordered.extend(alg for alg in sorted(algorithm_names) if alg not in ordered)
    return ordered


def extract_algorithm_names_from_persona(datasets: Dict[str, List[Dict]]) -> List[str]:
    algorithm_names = set()
    for records in datasets.values():
        for record in records:
            for query_data in record.get('queries', {}).values():
                algorithm_names.update(query_data.get('algorithms', {}).keys())
    return _sort_algorithms_by_priority(algorithm_names)


def _get_ordered_datasets(datasets: Dict) -> List[str]:
    ordered = []
    for ds in DATASET_ORDER:
        if ds in datasets:
            ordered.append(ds)
    for ds in sorted(datasets.keys()):
        if ds not in ordered:
            ordered.append(ds)
    return ordered


def aggregate_repeated_results(all_run_results: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    if not all_run_results:
        return {}
    algorithm_names = list(all_run_results[0].keys())
    aggregated = {}
    for alg_name in algorithm_names:
        all_winrates = [r[alg_name]['winrate'] for r in all_run_results if alg_name in r]
        total_wins = sum(r[alg_name]['wins'] for r in all_run_results if alg_name in r)
        total_count = sum(r[alg_name]['total'] for r in all_run_results if alg_name in r)
        if all_winrates:
            std = statistics.stdev(all_winrates) if len(all_winrates) > 1 else 0.0
            se = std / math.sqrt(len(all_winrates)) if len(all_winrates) > 1 else 0.0
            aggregated[alg_name] = {
                'wins': total_wins, 'total': total_count,
                'winrate': statistics.mean(all_winrates),
                'winrate_se': se, 
                'all_winrates': all_winrates
            }
    return aggregated

def generate_naive_response_for_persona(query_text: str, dataset_name: str, counter: int, query_index: int) -> str:
    """生成带有 history_context 的 naive response。

    ultrachat/wildchat: 使用完整历史数据，预测用户下一个问题
    prefeval: 使用完整交互记录，根据 query 预测用户偏好
    其他数据集 (lamp4, lamp5, lamp8, lamp9, lamp10): 使用简化的 prompt
    """
    history_list = _get_history_context(dataset_name, counter, query_index)
    instruction = LAMP_INSTRUCTIONS.get(dataset_name, "")

    if dataset_name in ["ultrachat", "wildchat"]:
        if history_list:
            history_text = "\n\n".join(history_list)  
            prompt = f"""You are a personalization assistant. Based on the user's conversation history below, predict the next question that the user would ask.

=== User's Conversation History ===
{history_text}

=== Task ===
Based on the conversation history above, predict the next question that the user would ask. Your prediction should reflect the user's interests, communication style, and the natural flow of the conversation.

Provide only the predicted question, without any explanations."""
        else:
            prompt = """Based on the context, predict the next question that the user would ask.

Provide only the predicted question, without any explanations."""

    elif dataset_name == "prefeval":
        if history_list:
            history_text = "\n\n".join(history_list) 
            prompt = f"""You are a personalization assistant. Based on the user's interaction history below, predict the user's preference and provide a personalized response.

=== User's Interaction History ===
{history_text}

=== Current Query ===
{query_text}

=== Task ===
Based on the interaction history above, predict the user's preference and provide a response that aligns with their demonstrated preferences, interests, and communication style.

Provide only the response, without any explanations."""
        else:
            prompt = f"""{instruction}

{query_text}

Please provide a response based on the user's likely preferences."""

    elif dataset_name == "lamp8":
        original_summary = _get_original_summary(dataset_name, counter)
        if original_summary:
            prompt = f"""You are a personalization assistant. Based on the user's persona below, generate a response that matches their preferences and style.

=== User's Persona ===
{original_summary}

=== Task ===
Generate an abstract for the following paper title/content. Your response should reflect the user's interests and style shown in the persona above.

Query:
{query_text}

Provide only the abstract, without any explanations or meta-commentary."""
        else:
            prompt = f"""Generate an abstract for the following paper title/content.

{query_text}

Please provide only the abstract, without including any additional information or explanations."""

    elif dataset_name == "lamp9":
        original_summary = _get_original_summary(dataset_name, counter)
        if original_summary:
            prompt = f"""You are a personalization assistant. Based on the user's persona below, generate a review that matches their preferences and style.

=== User's Persona ===
{original_summary}

=== Task ===
Write a product review for the following product. Your review should reflect the user's interests and style shown in the persona above.

Query:
{query_text}

Provide only the review text, without any explanations or meta-commentary."""
        else:
            prompt = f"""Write a product review for the following product.

{query_text}

Please provide only the review text, without including any additional information or explanations."""

    elif dataset_name == "lamp10":
        original_summary = _get_original_summary(dataset_name, counter)
        if original_summary:
            prompt = f"""You are a personalization assistant. Based on the user's persona below, generate a post that matches their preferences and style.

=== User's Persona ===
{original_summary}

=== Task ===
Write a Reddit post for the following topic. Your post should reflect the user's interests and style shown in the persona above.

Query:
{query_text}

Provide only the post content, without any explanations or meta-commentary."""
        else:
            prompt = f"""Write a Reddit post for the following topic.

{query_text}

Please provide only the post content, without including any additional information or explanations."""

    else:
        if history_list:
            prompt = f"""You are a personalization assistant. You MUST carefully analyze the user's interest and generate a response that matches their content preferences and writing style.

Task: {instruction}

Query:
{query_text}

IMPORTANT: Your response MUST reflect the content preferences and stylistic patterns."""
        else:
            prompt = f"""{instruction}

{query_text}

Please provide only the response required, without including any additional information or explanations."""

    return call_llm(prompt, system_role=SYSTEM_ROLE_GENERATION)


def compare_response_with_ground_truth(ground_truth: str, algorithm_response: str, naive_response: str) -> int:
    """使用 LLM as Judge 比较 algorithm response 和 naive response。

    返回值：1 表示算法获胜，0 表示 naive 获胜
    注意：Response 1 = naive, Response 2 = algorithm，所以 judge 返回 0 表示算法获胜
    """
    prompt = f"""Ground Truth:
{ground_truth}

Response 1:
{naive_response}

Response 2:
{algorithm_response}

Which response is closer to Ground Truth? Reply with only 1 (Response 1 is better) or 0 (Response 2 is better).
If they are the same, return 0 by default.
Please provide the final result (1 or 0 only) below:"""

    import time
    for attempt in range(3):
        try:
            result = call_llm(prompt)
            judge_result = int(result.strip())
            return 1 if judge_result == 0 else 0
        except (ValueError, Exception):
            if attempt < 2:
                time.sleep(2 ** attempt)
    return 0  


def _evaluate_single_query(args: tuple) -> Dict:
    counter, query_key, query_data, dataset_name, algorithm_names = args
    ground_truth = query_data.get('ground_truth', '')
    query_text = query_data.get('query_text', '')
    algorithms = query_data.get('algorithms', {})
    results = {}

    try:
        counter_int = int(counter) if isinstance(counter, str) else counter
        query_index = int(query_key.split('_')[1]) if '_' in query_key else 0
        naive_response = generate_naive_response_for_persona(query_text, dataset_name, counter_int, query_index)
        if PRINT_NAIVE_RESPONSE:
            print(f"\n{'─'*60}")
            print(f"📌 [{dataset_name}] Counter={counter}, {query_key}")
            print(f"   Query: {query_text[:100]}..." if len(query_text) > 100 else f"   Query: {query_text}")
            print(f"   Ground Truth: {ground_truth}")
            print(f"   Naive Response: {naive_response}")
            print(f"{'─'*60}")

        for alg_name in algorithm_names:
            if alg_name in algorithms:
                alg_response = algorithms[alg_name].get('response', '')
                if alg_response and ground_truth:
                    results[alg_name] = compare_response_with_ground_truth(ground_truth, alg_response, naive_response)
    except Exception as e:
        print(f"❌ [Counter {counter}][{query_key}] Error: {e}")

    return {'counter': counter, 'query_key': query_key, 'results': results}

def run_persona_evaluation(datasets: Dict[str, List[Dict]], output_dir: str) -> Dict[str, Dict[str, Dict]]:
    """运行 persona_results 的 LLM as Judge 评估。"""
    print("\n" + "="*70)
    print("🎯 Persona Results Evaluation (LLM as Judge vs Naive Baseline)")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)

    all_dataset_results = {}
    all_algorithm_names = extract_algorithm_names_from_persona(datasets)
    print(f"\n📋 检测到的算法: {all_algorithm_names}")

    parallel_enabled = EVAL_PARALLEL_CONFIG.get("enabled", True)
    max_workers = EVAL_PARALLEL_CONFIG.get("max_workers", 10)
    show_progress = EVAL_PARALLEL_CONFIG.get("show_progress", True)

    for dataset_name in _get_ordered_datasets(datasets):
        records = datasets[dataset_name]
        print(f"\n{'='*60}\n📁 Processing dataset: {dataset_name} ({len(records)} files)\n{'='*60}")

        results = {name: {'wins': 0, 'total': 0} for name in all_algorithm_names}
        task_args = []
        for record in records:
            counter = record.get('counter', 'unknown')
            for query_key, query_data in record.get('queries', {}).items():
                task_args.append((counter, query_key, query_data, dataset_name, all_algorithm_names))

        print(f"   📝 总评估任务数: {len(task_args)}")

        if parallel_enabled and len(task_args) > 1:
            print(f"   🚀 并行模式: {max_workers} 线程")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_evaluate_single_query, args): args for args in task_args}
                pbar = tqdm(total=len(futures), desc=f"   📊 {dataset_name}", unit="query") if show_progress else None
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        for alg_name, pref in result.get('results', {}).items():
                            results[alg_name]['total'] += 1
                            if pref == 1:
                                results[alg_name]['wins'] += 1
                    except Exception as e:
                        print(f"   ❌ Task failed: {e}")
                    if pbar:
                        pbar.update(1)
                if pbar:
                    pbar.close()
        else:
            iterator = tqdm(task_args, desc=f"   📊 {dataset_name}", unit="query") if show_progress else task_args
            for args in iterator:
                result = _evaluate_single_query(args)
                for alg_name, pref in result.get('results', {}).items():
                    results[alg_name]['total'] += 1
                    if pref == 1:
                        results[alg_name]['wins'] += 1

        for alg_name, data in results.items():
            data['winrate'] = data['wins'] / data['total'] if data['total'] > 0 else 0.0

        all_dataset_results[dataset_name] = results

        print(f"\n   📊 {dataset_name} Results:")
        print(f"   {'Algorithm':<20} {'Wins':>8} {'Total':>8} {'Winrate':>10}")
        print(f"   {'-'*50}")
        for alg_name, data in sorted(results.items(), key=lambda x: x[1]['winrate'], reverse=True):
            if data['total'] > 0:
                print(f"   {alg_name:<20} {data['wins']:>8} {data['total']:>8} {data['winrate']:>10.2%}")

        # 保存结果
        output_data = {
            "dataset": dataset_name, "evaluation_mode": "llm_as_judge_vs_naive",
            "model": API_CONFIG.get("openai_model"), "timestamp": datetime.now().isoformat(),
            "results": [{"algorithm": a, "rank": r+1, "winrate": round(d['winrate'], 4), "wins": d['wins'], "total": d['total']}
                        for r, (a, d) in enumerate(sorted(results.items(), key=lambda x: x[1]['winrate'], reverse=True)) if d['total'] > 0]
        }
        output_path = os.path.join(output_dir, f"{dataset_name}_llm_judge_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"   💾 结果已保存到: {output_path}")

    return all_dataset_results


def main_persona_evaluation():
    """主函数 - 评估 persona_results，支持重复评估。"""
    import argparse
    parser = argparse.ArgumentParser(description='Persona Results Evaluator (LLM as Judge)')
    parser.add_argument('--data_dir', type=str, default=PERSONA_RESULTS_DIR)
    parser.add_argument('--output_dir', type=str, default=os.path.join(RESULTS_BASE_DIR, "persona_llm_judge"))
    args = parser.parse_args()

    num_repeats = EVAL_REPEAT_CONFIG.get("num_repeats", 3)
    print(f"\n{'#'*70}\n# Persona Results Evaluator - LLM as Judge")
    print(f"# Data: {args.data_dir}\n# Output: {args.output_dir}\n# Repeats: {num_repeats}\n{'#'*70}")

    datasets = load_persona_results(args.data_dir)
    if not datasets:
        print("❌ No data files found.")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    all_aggregated_results = {}
    for dataset_name in _get_ordered_datasets(datasets):
        print(f"\n{'='*70}\n📁 Processing dataset: {dataset_name} ({num_repeats} repeats)\n{'='*70}")

        all_run_results = []
        for run_idx in range(num_repeats):
            print(f"\n  🔄 Run {run_idx + 1}/{num_repeats}")
            single_dataset = {dataset_name: datasets[dataset_name]}
            run_results = run_persona_evaluation(single_dataset, args.output_dir)
            if dataset_name in run_results:
                all_run_results.append(run_results[dataset_name])

        if all_run_results:
            agg = aggregate_repeated_results(all_run_results)
            all_aggregated_results[dataset_name] = agg

            print(f"\n   📊 {dataset_name} Aggregated Results ({num_repeats} runs):")
            print(f"   {'Algorithm':<20} {'Mean WR':>10} {'SE':>8} {'Wins':>10} {'Total':>8}")
            print(f"   {'-'*60}")
            for alg, data in sorted(agg.items(), key=lambda x: x[1]['winrate'], reverse=True):
                if data['total'] > 0:
                    print(f"   {alg:<20} {data['winrate']:>10.2%} ±{data.get('winrate_se', 0):.2%} {data['wins']:>10} {data['total']:>8}")

            # 保存聚合结果
            output_data = {
                "dataset": dataset_name, "num_repeats": num_repeats, "timestamp": datetime.now().isoformat(),
                "results": [{"algorithm": a, "winrate": round(d['winrate'], 4), "winrate_se": round(d.get('winrate_se', 0), 4),
                             "wins": d['wins'], "total": d['total']} for a, d in sorted(agg.items(), key=lambda x: x[1]['winrate'], reverse=True) if d['total'] > 0]
            }
            with open(os.path.join(args.output_dir, f"{dataset_name}_aggregated_results.json"), 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 总汇总
    print("\n" + "#"*70 + "\n# 📊 FINAL SUMMARY\n" + "#"*70)
    overall = {}
    for ds, results in all_aggregated_results.items():
        print(f"\n  📁 {ds}:")
        for alg, data in sorted(results.items(), key=lambda x: x[1]['winrate'], reverse=True):
            if data['total'] > 0:
                print(f"      {alg:<20}: {data['winrate']:.2%} ±{data.get('winrate_se', 0):.2%}")
                if alg not in overall:
                    overall[alg] = {'wins': 0, 'total': 0, 'all_winrates': []}
                overall[alg]['wins'] += data['wins']
                overall[alg]['total'] += data['total']
                overall[alg]['all_winrates'].extend(data.get('all_winrates', []))

    for alg, data in overall.items():
        data['winrate'] = data['wins'] / data['total'] if data['total'] > 0 else 0.0
        std = statistics.stdev(data['all_winrates']) if len(data['all_winrates']) > 1 else 0.0
        data['winrate_se'] = std / math.sqrt(len(data['all_winrates'])) if len(data['all_winrates']) > 1 else 0.0

    print("\n" + "="*70 + "\n📊 OVERALL (All Datasets Combined)\n" + "="*70)
    print(f"{'Algorithm':<25} {'Mean WR':>12} {'SE':>10} {'Wins':>10} {'Total':>10}")
    for alg, data in sorted(overall.items(), key=lambda x: x[1]['winrate'], reverse=True):
        if data['total'] > 0:
            print(f"{alg:<25} {data['winrate']:>12.2%} ±{data['winrate_se']:.2%} {data['wins']:>10} {data['total']:>10}")

    # 保存总汇总
    with open(os.path.join(args.output_dir, "overall_aggregated_results.json"), 'w', encoding='utf-8') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "overall_results": [
            {"algorithm": a, "winrate": round(d['winrate'], 4), "winrate_se": round(d['winrate_se'], 4), "wins": d['wins'], "total": d['total']}
            for a, d in sorted(overall.items(), key=lambda x: x[1]['winrate'], reverse=True) if d['total'] > 0
        ]}, f, ensure_ascii=False, indent=2)

    print(f"\n{'#'*70}\n# ✅ Evaluation Complete!\n{'#'*70}")


if __name__ == "__main__":
    import sys
    if '--persona' in sys.argv:
        sys.argv.remove('--persona')
    main_persona_evaluation()

