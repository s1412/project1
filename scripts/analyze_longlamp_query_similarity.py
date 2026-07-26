#!/usr/bin/env python3
"""
分析 LongLaMP 8, 9, 10 数据集中 Query 的文本相似度

功能:
1. 随机选取 10 个 counter
2. 按照 load_data.py 中的方式构造 query
3. 计算每个 counter 内 query 之间的余弦相似度
4. 输出统计信息

使用方法:
    python analyze_longlamp_query_similarity.py
"""

import json
import random
import numpy as np
import asyncio
import sys
import os
from typing import List, Dict, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ========== 数据集配置 ==========
DATASET_CONFIG = {
    4: {
        "name": "LaMP-4 (Headline Generation)",
        "input_address": os.path.join(PROJECT_ROOT, "APOHF-main", "time", "LaMP_4", "train", "train_questions.json"),
        "output_address": os.path.join(PROJECT_ROOT, "APOHF-main", "time", "LaMP_4", "train", "train_outputs.json"),
        "format": "json",  # JSON 数组格式
    },
    5: {
        "name": "LaMP-5 (Title Generation)",
        "input_address": os.path.join(PROJECT_ROOT, "APOHF-main", "time", "LaMP_5", "train", "train_questions.json"),
        "output_address": os.path.join(PROJECT_ROOT, "APOHF-main", "time", "LaMP_5", "train", "train_outputs.json"),
        "format": "json",  # JSON 数组格式
    },
    8: {
        "name": "LongLaMP-8 (Abstract Generation)",
        "input_address": os.path.join(PROJECT_ROOT, "APOHF-main", "longLaMP", "abstract_generation", "user_train.json"),
        "format": "jsonl",  # JSONL 格式
    },
    9: {
        "name": "LongLaMP-9 (Product Review)",
        "input_address": os.path.join(PROJECT_ROOT, "APOHF-main", "longLaMP", "product_review", "user_train.json"),
        "format": "jsonl",
    },
    10: {
        "name": "LongLaMP-10 (Reddit Post)",
        "input_address": os.path.join(PROJECT_ROOT, "APOHF-main", "longLaMP", "topic_writing", "user_train.json"),
        "format": "jsonl",
    }
}

# Embedding API（与 POHF 一致）
EMBEDDING_API_URL = "http://127.0.0.1:7777/v1/embeddings"

# 配置参数
NUM_COUNTERS = 10  # 随机选取的 counter 数量
RANDOM_SEED = 62   # 随机种子

# 导入 POHF 的 EmbeddingClient
try:
    from POHF import EmbeddingClient
    print("✅ 成功导入 POHF.EmbeddingClient")
except ImportError as e:
    print(f"⚠️ 无法导入 POHF.EmbeddingClient: {e}")
    print("   将使用本地实现...")

    import aiohttp

    class EmbeddingClient:
        def __init__(self, api_url: str = "http://127.0.0.1:7777/v1/embeddings"):
            self.api_url = api_url

        def normalize_l2(self, x):
            x = np.array(x)
            if x.ndim == 1:
                norm = np.linalg.norm(x)
                return x / norm if norm > 0 else x
            else:
                norms = np.linalg.norm(x, axis=1, keepdims=True)
                return np.where(norms > 0, x / norms, x)

        async def get_embedding(self, text: str, max_retries=3, retry_delay=2.0, retry_backoff=2.0):
            payload = {"model": "bge-m3", "input": [text]}
            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(self.api_url, json=payload, timeout=30) as response:
                            if response.status == 200:
                                result = await response.json()
                                return result["data"][0]["embedding"]
                            else:
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(retry_delay * (retry_backoff ** attempt))
                                else:
                                    return None
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (retry_backoff ** attempt))
                    else:
                        return None
            return None

        async def encode_texts(self, texts: List[str], normalize: bool = True) -> List[np.ndarray]:
            embeddings = []
            for text in texts:
                embedding = await self.get_embedding(text)
                if embedding is not None:
                    embedding = np.array(embedding)
                    if normalize:
                        embedding = self.normalize_l2(embedding)
                    embeddings.append(embedding)
                else:
                    embeddings.append(np.zeros(1024))
            return embeddings

# ========== 数据加载函数 ==========
def load_lamp_data(input_address: str, counter: int, format_type: str = "jsonl") -> Dict:
    """加载 LaMP 数据（支持 JSON 和 JSONL 格式）"""
    if format_type == "json":
        return load_lamp_json_data(input_address, counter)
    else:
        return load_longlamp_data(input_address, counter)

def load_lamp_json_data(input_address: str, counter: int) -> Dict:
    """加载 LaMP 4/5 的 JSON 数组格式数据（使用流式解析）"""
    import ijson

    current_index = 0
    target_data = None

    with open(input_address, 'rb') as f:
        parser = ijson.items(f, 'item')
        for data in parser:
            if current_index == counter:
                target_data = data
                break
            current_index += 1

    return target_data

def load_longlamp_data(input_address: str, counter: int) -> Dict:
    """加载 LongLaMP 8/9/10 的 JSONL 格式数据"""
    current_index = 0
    target_data = None

    with open(input_address, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, list):
                    for item in data:
                        if current_index == counter:
                            target_data = item
                            break
                        current_index += 1
                    if target_data:
                        break
                else:
                    if current_index == counter:
                        target_data = data
                        break
                    current_index += 1
            except json.JSONDecodeError:
                continue
    
    return target_data

def construct_queries(data: Dict, lamp_type: int, strip_prefix: bool = False) -> List[str]:
    """按照 load_data.py 的方式构造 query 列表"""
    queries = []
    
    if "profile" not in data or not data["profile"]:
        return queries
    
    # 获取 profile 数量
    all_profiles = data["profile"]
    num_profiles = len(all_profiles)
    
    # 根据 profile 数量决定切分
    if num_profiles < 10:
        history_count = 5
        max_additional = min(9, num_profiles - history_count)
    elif num_profiles < 20:
        history_count = 10
        max_additional = min(9, num_profiles - history_count)
    else:
        history_count = 10
        max_additional = 9
    
    # 获取用于生成 query 的 profile
    additional_profiles = all_profiles[history_count:history_count + max_additional]
    
    # 根据 LaMP_type 构造 query
    for profile_item in additional_profiles:
        if lamp_type == 4:
            # LaMP 4: Headline Generation - query 是文章正文
            text = profile_item.get("text", "")
            query_str = text
        elif lamp_type == 5:
            # LaMP 5: Title Generation - query 是论文摘要
            abstract = profile_item.get("abstract", "")
            query_str = abstract
        elif lamp_type == 8:
            # LaMP 8: Abstract Generation
            title = profile_item.get("title", "")
            if strip_prefix:
                query_str = title  # 只用 title，不加前缀
            else:
                query_str = f"Generate an abstract for the title: {title}"
        elif lamp_type == 9:
            # LaMP 9: Product Review Generation
            description = profile_item.get("description", "")
            rating = profile_item.get("overall", "")
            if strip_prefix:
                query_str = f"{description} {rating}"  # 只用内容，不加前缀
            else:
                query_str = f"Generate the review text for a product with description: {description} and rating {rating}"
        elif lamp_type == 10:
            # LaMP 10: Reddit Post Generation
            summary = profile_item.get("summary", "")
            if strip_prefix:
                query_str = summary  # 只用 summary，不加前缀
            else:
                query_str = f"Generate the content for a reddit post: {summary}"
        else:
            continue

        if query_str and query_str.strip():  # 只添加非空 query
            queries.append(query_str)

    return queries

# ========== Embedding 计算 ==========
# 使用 EmbeddingClient（与 POHF 完全一致）
embedding_client = EmbeddingClient(api_url=EMBEDDING_API_URL)

async def get_embeddings(texts: List[str]) -> np.ndarray:
    """获取文本的 embedding（使用 POHF 的 EmbeddingClient）"""
    try:
        embeddings = await embedding_client.encode_texts(texts, normalize=True)
        return np.array(embeddings)
    except Exception as e:
        print(f"❌ Embedding API 错误: {e}")
        return None

def compute_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """计算余弦相似度矩阵"""
    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    # 计算相似度矩阵
    similarity_matrix = np.dot(normalized, normalized.T)
    return similarity_matrix

def get_dataset_size(input_address: str, format_type: str = "jsonl") -> int:
    """获取数据集大小"""
    if format_type == "json":
        return get_json_dataset_size(input_address)
    else:
        return get_jsonl_dataset_size(input_address)

def get_json_dataset_size(input_address: str) -> int:
    """获取 JSON 数组格式数据集大小（使用流式解析）"""
    import ijson
    count = 0
    with open(input_address, 'rb') as f:
        parser = ijson.items(f, 'item')
        for _ in parser:
            count += 1
    return count

def get_jsonl_dataset_size(input_address: str) -> int:
    """获取 JSONL 格式数据集大小"""
    total_items = 0
    with open(input_address, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if isinstance(data, list):
                    total_items += len(data)
                else:
                    total_items += 1
    return total_items

# ========== 分析函数 ==========
async def analyze_single_counter(lamp_type: int, counter: int, config: Dict, strip_prefix: bool = False) -> Dict:
    """分析单个 counter 的 query 相似度"""
    input_address = config["input_address"]
    format_type = config.get("format", "jsonl")

    # 加载数据
    data = load_lamp_data(input_address, counter, format_type)
    if not data:
        return {"error": f"无法加载 counter {counter}"}

    # 构造 query
    queries = construct_queries(data, lamp_type, strip_prefix=strip_prefix)
    if len(queries) < 2:
        return {"error": f"Query 数量不足 ({len(queries)})"}

    # 获取 embedding
    embeddings = await get_embeddings(queries)
    if embeddings is None:
        return {"error": "Embedding API 失败"}

    # 计算相似度矩阵
    sim_matrix = compute_cosine_similarity_matrix(embeddings)

    # 提取上三角（不含对角线）的相似度值
    upper_tri_indices = np.triu_indices(len(queries), k=1)
    similarities = sim_matrix[upper_tri_indices]

    # 🔍 检测相似度 >= 0.99 的 query 对并输出
    high_sim_pairs = []
    for idx, (i, j) in enumerate(zip(upper_tri_indices[0], upper_tri_indices[1])):
        if similarities[idx] >= 0.99:
            high_sim_pairs.append({
                "i": i, "j": j,
                "sim": similarities[idx],
                "query_i": queries[i],
                "query_j": queries[j]
            })

    if high_sim_pairs:
        print(f"\n   🚨 Counter {counter} 发现 {len(high_sim_pairs)} 对相似度 ≥ 0.99 的 Query:")
        for pair in high_sim_pairs:
            print(f"      ─────────────────────────────────────────")
            print(f"      Query {pair['i']} vs Query {pair['j']} (相似度: {pair['sim']:.6f})")
            print(f"      [Query {pair['i']}]: {pair['query_i'][:200]}...")
            print(f"      [Query {pair['j']}]: {pair['query_j'][:200]}...")

    return {
        "counter": counter,
        "num_queries": len(queries),
        "similarities": similarities,
        "queries": queries,
        "high_sim_pairs": high_sim_pairs,
        "min": float(np.min(similarities)),
        "max": float(np.max(similarities)),
        "mean": float(np.mean(similarities)),
        "std": float(np.std(similarities)),
    }

async def analyze_dataset(lamp_type: int, strip_prefix: bool = False) -> Dict:
    """分析单个数据集

    Args:
        lamp_type: 数据集类型
        strip_prefix: 是否去除固定前缀
    """
    config = DATASET_CONFIG[lamp_type]
    input_address = config["input_address"]
    dataset_name = config["name"]
    format_type = config.get("format", "jsonl")

    suffix = " (无前缀)" if strip_prefix else " (带前缀)"

    print(f"\n{'='*80}")
    print(f"📊 分析 {dataset_name}{suffix}")
    print(f"{'='*80}")
    print(f"   数据文件: {input_address}")
    print(f"   格式: {format_type.upper()}")
    print(f"   模式: {'去除固定前缀' if strip_prefix else '保留固定前缀（原始格式）'}")

    # 获取数据集大小
    dataset_size = get_dataset_size(input_address, format_type)
    print(f"   数据集大小: {dataset_size}")

    # 随机选取 counter
    random.seed(RANDOM_SEED)
    counters = random.sample(range(dataset_size), min(NUM_COUNTERS, dataset_size))
    print(f"   选取的 Counter: {counters}")

    # 分析每个 counter
    all_results = []
    all_similarities = []

    for counter in counters:
        result = await analyze_single_counter(lamp_type, counter, config, strip_prefix=strip_prefix)
        if "error" in result:
            print(f"   ⚠️ Counter {counter}: {result['error']}")
        else:
            all_results.append(result)
            all_similarities.extend(result["similarities"])
            print(f"   ✅ Counter {counter}: {result['num_queries']} queries, "
                  f"sim=[{result['min']:.4f}, {result['max']:.4f}], mean={result['mean']:.4f}")

    # 汇总统计
    if all_similarities:
        all_similarities = np.array(all_similarities)
        summary = {
            "dataset": dataset_name,
            "num_counters": len(all_results),
            "total_pairs": len(all_similarities),
            "min": float(np.min(all_similarities)),
            "max": float(np.max(all_similarities)),
            "mean": float(np.mean(all_similarities)),
            "std": float(np.std(all_similarities)),
            "percentiles": {
                "25%": float(np.percentile(all_similarities, 25)),
                "50%": float(np.percentile(all_similarities, 50)),
                "75%": float(np.percentile(all_similarities, 75)),
                "90%": float(np.percentile(all_similarities, 90)),
                "95%": float(np.percentile(all_similarities, 95)),
            }
        }

        print(f"\n   📈 汇总统计 ({summary['num_counters']} counters, {summary['total_pairs']} pairs):")
        print(f"      Min:   {summary['min']:.4f}")
        print(f"      Max:   {summary['max']:.4f}")
        print(f"      Mean:  {summary['mean']:.4f}")
        print(f"      Std:   {summary['std']:.4f}")
        print(f"      P25:   {summary['percentiles']['25%']:.4f}")
        print(f"      P50:   {summary['percentiles']['50%']:.4f}")
        print(f"      P75:   {summary['percentiles']['75%']:.4f}")
        print(f"      P90:   {summary['percentiles']['90%']:.4f}")
        print(f"      P95:   {summary['percentiles']['95%']:.4f}")

        # 高相似度分析
        high_sim_threshold = 0.85
        high_sim_count = np.sum(all_similarities >= high_sim_threshold)
        high_sim_ratio = high_sim_count / len(all_similarities) * 100
        summary['high_sim_ratio'] = high_sim_ratio  # 添加到 summary 中
        print(f"\n   ⚠️ 高相似度 (≥{high_sim_threshold}) 的 Query 对: {high_sim_count} ({high_sim_ratio:.1f}%)")

        return summary, all_results

    return None, all_results

async def main():
    """主函数"""
    print("="*80)
    print("🔍 LongLaMP Query 文本相似度分析")
    print("="*80)
    print(f"配置: 随机选取 {NUM_COUNTERS} 个 counter, 随机种子 = {RANDOM_SEED}")

    with_prefix = {}
    without_prefix = {}

    # ========== 实验1：带前缀（原始格式）==========
    print("\n" + "="*80)
    print("🔬 实验1：带前缀（原始格式）")
    print("="*80)
    for lamp_type in [8, 9, 10]:
        summary, _ = await analyze_dataset(lamp_type, strip_prefix=False)
        if summary:
            with_prefix[lamp_type] = summary

    # ========== 实验2：去掉前缀 ==========
    print("\n" + "="*80)
    print("🔬 实验2：去掉前缀（只用内容）")
    print("="*80)
    for lamp_type in [8, 9, 10]:
        summary, _ = await analyze_dataset(lamp_type, strip_prefix=True)
        if summary:
            without_prefix[lamp_type] = summary

    # ========== 最终对比 ==========
    print("\n" + "="*100)
    print("📊 最终对比：带前缀 vs 去掉前缀")
    print("="*100)
    print(f"{'数据集':<35} {'模式':<12} {'Mean':<10} {'Max':<10} {'P90':<10} {'P95':<10} {'高相似度%':<10}")
    print("-"*100)

    for lamp_type in [8, 9, 10]:
        if lamp_type in with_prefix:
            s = with_prefix[lamp_type]
            name = s['dataset']
            print(f"{name:<35} {'带前缀':<12} {s['mean']:<10.4f} {s['max']:<10.4f} "
                  f"{s['percentiles']['90%']:<10.4f} {s['percentiles']['95%']:<10.4f} "
                  f"{s.get('high_sim_ratio', 0):<10.1f}")
        if lamp_type in without_prefix:
            s = without_prefix[lamp_type]
            print(f"{'':<35} {'去掉前缀':<12} {s['mean']:<10.4f} {s['max']:<10.4f} "
                  f"{s['percentiles']['90%']:<10.4f} {s['percentiles']['95%']:<10.4f} "
                  f"{s.get('high_sim_ratio', 0):<10.1f}")
        print("-"*100)

    # ========== 计算差值 ==========
    print("\n📈 相似度降低幅度（带前缀 - 去掉前缀）:")
    print("-"*60)
    for lamp_type in [8, 9, 10]:
        if lamp_type in with_prefix and lamp_type in without_prefix:
            w = with_prefix[lamp_type]
            wo = without_prefix[lamp_type]
            mean_diff = w['mean'] - wo['mean']
            p90_diff = w['percentiles']['90%'] - wo['percentiles']['90%']
            p95_diff = w['percentiles']['95%'] - wo['percentiles']['95%']
            print(f"   LongLaMP-{lamp_type}: Mean降低 {mean_diff:.4f}, P90降低 {p90_diff:.4f}, P95降低 {p95_diff:.4f}")

    print("\n✅ 对比实验完成")

if __name__ == "__main__":
    asyncio.run(main())
