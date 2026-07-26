"""
PersonaAgent Data Loading Script

This script loads and processes multiple datasets for PersonaAgent, generating:
- EpisodicMemory: List of [{query, ground_truth, metadata}] dictionaries
- SemanticMemory: LLM-extracted high-level user preferences and behavioral patterns
- input: Dataset-specific input format
- output: Dataset-specific output format

Supported datasets:
- LaMP 4: Headline generation
- LaMP 5: Title generation for abstracts
- LaMP 8: Abstract generation (LongLaMP)
- LaMP 9: Product review generation (LongLaMP)
- LaMP 10: Reddit post generation (LongLaMP)
- UltraChat: Multi-turn dialogue
- WildChat: Multi-turn dialogue
- PrefEval: Preference evaluation with persona
"""

import os
import json
import time
import random
from typing import List, Dict, Tuple, Any, Optional, Union
from openai import OpenAI
import httpx as _httpx
import ijson  # For streaming JSON parsing of large files

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Hyperparameters
# ============================================================================

# Number of recent EpisodicMemory entries to use for SemanticMemory generation
# If EpisodicMemory has fewer entries, all available entries will be used
SEMANTIC_MEMORY_HISTORY_COUNT = 10

# LaMP datasets specific hyperparameters
# Threshold for determining EpisodicMemory count based on profile size
LAMP_PROFILE_THRESHOLD_SMALL = 10   # If profile < 10, use small context
LAMP_PROFILE_THRESHOLD_LARGE = 20   # If profile >= 20, use large context

# EpisodicMemory count for different profile sizes
LAMP_EPISODIC_MEMORY_COUNT_SMALL = 5   # For profile < 10 entries
LAMP_EPISODIC_MEMORY_COUNT_LARGE = 10  # For profile >= 10 entries

# Number of entries from EpisodicMemory to use for SemanticMemory generation (last M entries)
LAMP_SEMANTIC_MEMORY_COUNT = 5
# Total number of test entries target (current input/output + additional from profile)
LAMP_TOTAL_TEST_COUNT = 10

# Dataset type mapping
DATASET_TYPES = {
    "lamp4": 4,
    "lamp_4": 4,
    "LaMP_4": 4,
    "lamp5": 5,
    "lamp_5": 5,
    "LaMP_5": 5,
    "lamp8": 8,
    "lamp_8": 8,
    "LaMP_8": 8,
    "abstract_generation": 8,
    "lamp9": 9,
    "lamp_9": 9,
    "LaMP_9": 9,
    "product_review": 9,
    "lamp10": 10,
    "lamp_10": 10,
    "LaMP_10": 10,
    "topic_writing": 10,
    "ultrachat": 0,
    "UltraChat": 0,
    "wildchat": -1,
    "WildChat": -1,
    "prefeval": -2,
    "PrefEval": -2,
}

# Default file paths
DEFAULT_PATHS = {
    4: {
        "input": os.path.join(PROJECT_ROOT, "APOHF-main", "time", "LaMP_4", "train", "train_questions.json"),
        "output": os.path.join(PROJECT_ROOT, "APOHF-main", "time", "LaMP_4", "train", "train_outputs.json"),
    },
    5: {
        "input": os.path.join(PROJECT_ROOT, "APOHF-main", "time", "LaMP_5", "train", "train_questions.json"),
        "output": os.path.join(PROJECT_ROOT, "APOHF-main", "time", "LaMP_5", "train", "train_outputs.json"),
    },
    8: {
        "input": os.path.join(PROJECT_ROOT, "APOHF-main", "longLaMP", "abstract_generation", "temporal_train.json"),
        "output": None,
    },
    9: {
        "input": os.path.join(PROJECT_ROOT, "APOHF-main", "longLaMP", "product_review", "temporal_train.json"),
        "output": None,
    },
    10: {
        "input": os.path.join(PROJECT_ROOT, "APOHF-main", "longLaMP", "topic_writing", "temporal_train.json"),
        "output": None,
    },
    0: {
        "input": os.path.join(PROJECT_ROOT, "ultrachat_multiturn", "ultrachat_long_dialogues_with_response.json"),
        "output": None,
    },
    -1: {
        "input": os.path.join(PROJECT_ROOT, "wildchat", "wildchat_long_dialogues_with_response.json"),
        "output": None,
    },
    -2: {
        "input": os.path.join(PROJECT_ROOT, "PrefEval_dataset", "PrefEval_persona.json"),
        "output": None,
    },
}

# OpenAI client for semantic memory extraction
_proxy_url = os.environ.get("https_proxy") or os.environ.get("http_proxy")
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY", "API_KEY_NOT_SET"),
    base_url="https://openrouter.ai/api/v1",
    http_client=_httpx.Client(proxy=_proxy_url) if _proxy_url else None,
)


def get_dataset_type(dataset_identifier: Union[str, int]) -> int:
    """Convert dataset identifier (name or id) to dataset type integer."""
    if isinstance(dataset_identifier, int):
        return dataset_identifier
    return DATASET_TYPES.get(dataset_identifier, dataset_identifier)


def load_json_file(file_path: str) -> Any:
    """Load a JSON file (standard JSON format)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_json_item_streaming(file_path: str, index: int) -> Optional[Dict]:
    """
    Load a single item from a large JSON array file using streaming.

    This function uses ijson to parse the file incrementally, only keeping
    the item at the specified index in memory. Much more efficient for
    large files (e.g., 3GB+ JSON files).

    Supports files with multiple concatenated JSON arrays (e.g., "][" patterns).

    Args:
        file_path: Path to the JSON file containing an array (or multiple arrays)
        index: The index of the item to retrieve (0-based, counting across all arrays)

    Returns:
        The item at the specified index, or None if not found
    """
    print(f"   📖 Streaming JSON file: {file_path}")
    print(f"   🎯 Looking for item at index: {index}")

    # 首先尝试使用备用方法（更可靠地处理多数组拼接的文件）
    # 因为 ijson 的 multiple_values 在某些情况下仍然会失败
    try:
        result = _load_json_item_streaming_fallback(file_path, index)
        if result is not None:
            return result
    except Exception as e:
        print(f"   ⚠️ 备用方法失败: {e}，尝试标准ijson解析...")

    # 如果备用方法失败，尝试标准 ijson 解析
    current_index = 0
    try:
        with open(file_path, 'rb') as f:
            parser = ijson.items(f, 'item', multiple_values=True)

            for item in parser:
                if current_index == index:
                    print(f"   ✅ Found item at index {index}")
                    return item
                current_index += 1

                if current_index % 100 == 0:
                    print(f"   ... scanned {current_index} items", end='\r')

    except ijson.common.IncompleteJSONError as e:
        print(f"   ⚠️ ijson 解析失败: {e}")
        return None

    print(f"   ⚠️ Index {index} not found (file has {current_index} items)")
    return None


def _load_json_item_streaming_fallback(file_path: str, index: int) -> Optional[Dict]:
    """
    备用方法：处理多个JSON数组拼接的文件。

    适用于文件格式为多个JSON数组拼接（如 }]}][{ 模式）的情况。
    这种格式常见于 LongLaMP 数据集，其中每个数组包含约1000个用户。
    """
    import json
    import re

    print(f"   📖 [Fallback] 处理多数组拼接格式...")

    # 读取整个文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 将多个拼接的数组转换为单个有效的JSON数组
    # 模式: }]}] [{ -> }]}, [{
    # 首先找到分割点
    split_pattern = r'\}\]\}\]\s*\[\{'

    # 检查是否存在这种分割模式
    if re.search(split_pattern, content):
        print(f"   🔧 检测到多数组拼接格式，正在合并...")
        # 替换分割模式：}]}] [{ -> }]}, [{
        merged_content = re.sub(split_pattern, '}]}, [{', content)
        # 包装成单个数组
        merged_content = '[' + merged_content + ']'
    else:
        # 可能是简单的 ][ 分割
        simple_split = r'\]\s*\['
        if re.search(simple_split, content):
            print(f"   🔧 检测到简单数组拼接格式，正在合并...")
            merged_content = re.sub(simple_split, ', ', content)
        else:
            merged_content = content

    # 解析合并后的JSON
    try:
        data = json.loads(merged_content)
        print(f"   ✅ 成功解析，共 {len(data)} 个顶层数组/用户组")

        # 展平所有用户组中的项目
        current_index = 0
        for user_group in data:
            # user_group 可能是一个用户对象列表
            if isinstance(user_group, list):
                for item in user_group:
                    if current_index == index:
                        print(f"   ✅ Found item at index {index}")
                        return item
                    current_index += 1
            elif isinstance(user_group, dict):
                # 直接是一个用户对象
                if current_index == index:
                    print(f"   ✅ Found item at index {index}")
                    return user_group
                current_index += 1

            if current_index % 1000 == 0:
                print(f"   ... scanned {current_index} items")

        print(f"   ⚠️ Index {index} not found (file has {current_index} items)")
        return None

    except json.JSONDecodeError as e:
        print(f"   ❌ JSON解析失败: {e}")
        # 如果合并后仍然无法解析，尝试逐块解析
        return _load_json_item_by_chunks(file_path, index)


def _load_json_item_by_chunks(file_path: str, index: int) -> Optional[Dict]:
    """
    逐块解析多数组拼接的JSON文件。
    每个块是一个独立的JSON数组。
    """
    import json
    import re

    print(f"   📖 [Chunks] 逐块解析多数组文件...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则表达式分割多个数组
    # 模式: }]}] 后面可能跟着 \n 然后 [{
    split_pattern = r'\}\]\}\]\s*(?=\[\{)'
    chunks = re.split(split_pattern, content)

    print(f"   🔧 分割成 {len(chunks)} 个块")

    current_index = 0
    for chunk_idx, chunk in enumerate(chunks):
        # 修复块的格式
        chunk = chunk.strip()
        if not chunk.startswith('['):
            chunk = '[' + chunk
        if not chunk.endswith(']'):
            chunk = chunk + '}]}]'  # 补全结尾

        try:
            chunk_data = json.loads(chunk)

            # 遍历块中的项目
            for item in chunk_data:
                if current_index == index:
                    print(f"   ✅ Found item at index {index} (in chunk {chunk_idx})")
                    return item
                current_index += 1

            if (chunk_idx + 1) % 5 == 0:
                print(f"   ... processed {chunk_idx + 1} chunks, {current_index} items")

        except json.JSONDecodeError as e:
            print(f"   ⚠️ Chunk {chunk_idx} 解析失败: {e}")
            continue

    print(f"   ⚠️ Index {index} not found (file has {current_index} items)")
    return None


def load_jsonl_file(file_path: str) -> List[Dict]:
    """Load a JSONL file (one JSON object per line)."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, list):
                        data.extend(obj)
                    else:
                        data.append(obj)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON parse error in {file_path}: {e}")
                    continue
    return data


def extract_semantic_memory(episodic_memory: List[List]) -> str:
    """
    Extract high-level user preferences and behavioral patterns from EpisodicMemory
    using LLM.

    Uses the last SEMANTIC_MEMORY_HISTORY_COUNT entries from EpisodicMemory
    (or all available if fewer than SEMANTIC_MEMORY_HISTORY_COUNT).

    NOTE: This function uses the SAME prompt template as POHF's Summarization_lamp
    to ensure fair comparison between PersonaAgent and POHF algorithms.

    Args:
        episodic_memory: List of [query, ground_truth, metadata] representing user history

    Returns:
        String containing LLM-extracted user preferences and behavioral patterns
    """
    # Use the last N entries from EpisodicMemory (or all if fewer than N)
    history_count = SEMANTIC_MEMORY_HISTORY_COUNT
    if len(episodic_memory) <= history_count:
        selected_history = episodic_memory
    else:
        selected_history = episodic_memory[-history_count:]

    # Format EpisodicMemory data for LLM
    profile_parts = []
    for i, entry in enumerate(selected_history):
        query = entry[0] if len(entry) > 0 else ""
        ground_truth = entry[1] if len(entry) > 1 else ""
        metadata = entry[2] if len(entry) > 2 else {}

        # Format each entry with query, ground_truth, and metadata
        entry_text = f"Entry {i+1}:\n"
        entry_text += f"  Query: {query}\n"
        entry_text += f"  Response: {ground_truth}\n"
        if metadata:
            metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items() if v])
            if metadata_str:
                entry_text += f"  Metadata: {metadata_str}"
        profile_parts.append(entry_text)

    profile_text = "\n---\n".join(profile_parts)

    # 🔧 [一致性修改] 使用与 POHF 的 Summarization_lamp 相同的 prompt 模板
    # 确保 PersonaAgent 和 POHF 使用相同的 persona summary 生成方式
    prompt = f"""You are an advanced profile analysis and summarization model. Based on the user's historical data provided below, generate an **informative and coherent user persona summary** that reflects the user's characteristics, style, and behavior patterns.

### User History Data:
{profile_text}

### Instructions:
1. **CORE ANCHORING** (Most Important):
   - Identify 2-3 SPECIFIC, CONCRETE details from the history (e.g., specific topics, specific writing styles, specific interests revealed)
   - These concrete details MUST appear in your summary - they are the "anchors" that define this user
   - Example anchors: specific hobbies, specific professional domains, specific communication quirks

2. **BEHAVIORAL PATTERNS**:
   - Focus on the user's **personality traits, interests, tone, communication style, preferences, goals**
   - Describe what they care about
   - Describe how they express themselves

3. **SPECIFICITY OVER ABSTRACTION**:
   - Use CONCRETE language, avoid vague terms
   - Prefer specific examples over general categories

Please provide the final user persona summary below (one paragraph, 150-300 words):"""

    system_prompt = """You are an advanced profile analysis and summarization model.
Your goal is to generate a coherent, SPECIFIC, and psychologically insightful user persona summary.

Critical Guidelines:
- **ANCHOR ON SPECIFICS**: Always include 2-3 concrete, verifiable details from the history as "anchors"
- **NO ROLE DRIFT**: Describe who the user IS based on evidence, not who they COULD become
- **CONCRETE > ABSTRACT**: Use specific examples rather than general categories
- Focus on: personality traits, interests, tone, communication style, preferences, goals, behavior patterns
- Be **concise but information-rich** — aim for a natural paragraph (150-300 words)
- Avoid generic statements; describe **HOW** and **WHY** they behave or communicate that way
- Never include labels like "Summary:" or formatting headers
- Maintain a neutral, descriptive tone — avoid praise or flattery
- Use third-person perspective (e.g., "They prefer...", "This user...")"""

    max_retries = 5
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-v3.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                timeout=30.0
            )

            if not response or not response.choices or len(response.choices) == 0:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️ SemanticMemory API invalid response, retrying in {delay:.1f}s ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("SemanticMemory API returned invalid response")

            content = response.choices[0].message.content
            if content is None or not content.strip():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️ SemanticMemory API empty content, retrying in {delay:.1f}s ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("SemanticMemory API returned empty content")
            print("original summary = ", content.strip())
            return content.strip()

        except Exception as e:
            if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):
                # Fallback: return truncated profile text
                return f"User profile summary (auto-generated): {profile_text[:500]}..."

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"⚠️ SemanticMemory API error ({e}), retrying in {delay:.1f}s ({attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"⚠️ SemanticMemory extraction failed after {max_retries} retries")
                return f"User profile summary (fallback): {profile_text[:500]}..."

    return f"User profile summary (fallback): {profile_text[:500]}..."


def _split_profile_for_lamp(profile: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    根据profile数量动态决定EpisodicMemory和test_profile的分割

    规则：
    - 如果profile < 10条：前3条作为EpisodicMemory，剩余作为test entries
    - 如果profile >= 10条但 < 20条：前10条作为EpisodicMemory，剩余作为test entries
    - 如果profile >= 20条：前10条作为EpisodicMemory，第11-19条作为test entries

    Args:
        profile: 全部profile列表

    Returns:
        tuple: (episodic_profile, test_profile)
    """
    profile_count = len(profile)

    if profile_count < LAMP_PROFILE_THRESHOLD_SMALL:
        # Profile不足10条：前3条作为EpisodicMemory，剩余作为test entries
        episodic_count = LAMP_EPISODIC_MEMORY_COUNT_SMALL
        episodic_profile = profile[:episodic_count]
        test_profile = profile[episodic_count:]
    elif profile_count < LAMP_PROFILE_THRESHOLD_LARGE:
        # Profile在10-19条之间：前10条作为EpisodicMemory，剩余作为test entries
        episodic_count = LAMP_EPISODIC_MEMORY_COUNT_LARGE
        episodic_profile = profile[:episodic_count]
        test_profile = profile[episodic_count:]

    else:
        # Profile >= 20条：前10条作为EpisodicMemory，第11-19条作为test entries
        episodic_count = LAMP_EPISODIC_MEMORY_COUNT_LARGE
        max_test = LAMP_TOTAL_TEST_COUNT - 1  # 最多9条额外的test entries
        episodic_profile = profile[:episodic_count]
        test_profile = profile[episodic_count:episodic_count + max_test]


    return episodic_profile, test_profile


# ============================================================================
# LaMP 4: Headline Generation
# ============================================================================

def build_episodic_memory_lamp4(profile: List[Dict]) -> List[List]:
    """
    Build EpisodicMemory for LaMP 4 (Headline Generation).
    Profile: {"text": "article_content", "title": "title", "id": "...", "date": "..."}
    Query: "Generate a headline for the following article: [article_content]"
    Ground truth: title
    Metadata: date

    Returns: [[query1, ground_truth1, metadata1], [query2, ground_truth2, metadata2], ...]
    """
    episodic_memory = []
    for item in profile:
        text = item.get("text", "")
        title = item.get("title", "")
        date = item.get("date", "")

        episodic_memory.append([
            f"Generate a headline for the following article: {text}",
            title,
            {"date": date, "id": item.get("id", "")}
        ])
    return episodic_memory


def load_lamp4(input_path: str, output_path: str, counter: int) -> Tuple[List[List], str, List[str], List[str]]:
    """
    Load LaMP 4 data and return (EpisodicMemory, SemanticMemory, inputs_list, outputs_list).

    For LaMP datasets:
    - Profile分割根据数量动态决定：
      - < 10条：前3条作为EpisodicMemory，剩余作为test entries
      - >= 10但 < 20条：前10条作为EpisodicMemory，剩余作为test entries
      - >= 20条：前10条作为EpisodicMemory，第11-19条作为test entries
    - Last LAMP_SEMANTIC_MEMORY_COUNT (5) entries of EpisodicMemory -> SemanticMemory generation
    - Current input/output + additional profile entries -> inputs_list, outputs_list

    优化：使用流式加载单个用户数据，减少内存占用
    """
    # 使用流式加载单个用户数据
    data_item = load_json_item_streaming(input_path, counter)
    if data_item is None:
        raise ValueError(f"Could not find item at index {counter} in {input_path}")

    item_id = data_item["id"]
    profile = data_item.get("profile", [])
    current_input = data_item.get("input", "")

    # 加载输出数据并查找对应的 output
    # 注意：output 文件通常较小，可以直接加载
    current_output = ""
    if output_path:
        output_data = load_json_file(output_path)
        if "golds" in output_data:
            for item in output_data["golds"]:
                if item["id"] == item_id:
                    current_output = item["output"]
                    break

    # 根据profile数量动态分割
    episodic_profile, test_profile = _split_profile_for_lamp(profile)

    # Build EpisodicMemory from episodic_profile
    episodic_memory = build_episodic_memory_lamp4(episodic_profile)

    # Extract SemanticMemory from last M entries of EpisodicMemory
    semantic_count = min(LAMP_SEMANTIC_MEMORY_COUNT, len(episodic_memory))
    semantic_entries = episodic_memory[-semantic_count:] if semantic_count > 0 else episodic_memory
    semantic_memory = extract_semantic_memory(semantic_entries)

    # Build inputs_list and outputs_list
    inputs_list = [current_input]
    outputs_list = [current_output]

    # Add entries from test_profile
    for item in test_profile:
        text = item.get("text", "")
        title = item.get("title", "")
        query = f"Generate a headline for the following article: {text}"
        inputs_list.append(query)
        outputs_list.append(title)

    return episodic_memory, semantic_memory, inputs_list, outputs_list


# ============================================================================
# LaMP 5: Title Generation for Abstracts
# ============================================================================

def build_episodic_memory_lamp5(profile: List[Dict]) -> List[List]:
    """
    Build EpisodicMemory for LaMP 5 (Title Generation).
    Profile: {"abstract": "abstract_content", "title": "title", "date": "...", "id": "..."}
    Query: "Generate a title for the following abstract: [abstract_content]"
    Ground truth: title
    Metadata: date

    Returns: [[query1, ground_truth1, metadata1], [query2, ground_truth2, metadata2], ...]
    """
    episodic_memory = []
    for item in profile:
        abstract = item.get("abstract", "")
        title = item.get("title", "")
        date = item.get("date", "")

        episodic_memory.append([
            f"Generate a title for the following abstract: {abstract}",
            title,
            {"date": date, "id": item.get("id", "")}
        ])
    return episodic_memory


def load_lamp5(input_path: str, output_path: str, counter: int) -> Tuple[List[List], str, List[str], List[str]]:
    """
    Load LaMP 5 data and return (EpisodicMemory, SemanticMemory, inputs_list, outputs_list).

    For LaMP datasets:
    - Profile分割根据数量动态决定
    - Last LAMP_SEMANTIC_MEMORY_COUNT (5) entries of EpisodicMemory -> SemanticMemory generation
    - Current input/output + additional profile entries -> inputs_list, outputs_list

    优化：使用流式加载单个用户数据，减少内存占用
    """
    # 使用流式加载单个用户数据
    data_item = load_json_item_streaming(input_path, counter)
    if data_item is None:
        raise ValueError(f"Could not find item at index {counter} in {input_path}")

    item_id = data_item["id"]
    profile = data_item.get("profile", [])
    current_input = data_item.get("input", "")

    # 加载输出数据并查找对应的 output
    current_output = ""
    if output_path:
        output_data = load_json_file(output_path)
        if "golds" in output_data:
            for item in output_data["golds"]:
                if item["id"] == item_id:
                    current_output = item["output"]
                    break

    # 根据profile数量动态分割
    episodic_profile, test_profile = _split_profile_for_lamp(profile)

    # Build EpisodicMemory from episodic_profile
    episodic_memory = build_episodic_memory_lamp5(episodic_profile)

    # Extract SemanticMemory from last M entries of EpisodicMemory
    semantic_count = min(LAMP_SEMANTIC_MEMORY_COUNT, len(episodic_memory))
    semantic_entries = episodic_memory[-semantic_count:] if semantic_count > 0 else episodic_memory
    semantic_memory = extract_semantic_memory(semantic_entries)

    # Build inputs_list and outputs_list
    inputs_list = [current_input]
    outputs_list = [current_output]

    for item in test_profile:
        abstract = item.get("abstract", "")
        title = item.get("title", "")
        query = f"Generate a title for the following abstract: {abstract}"
        inputs_list.append(query)
        outputs_list.append(title)

    return episodic_memory, semantic_memory, inputs_list, outputs_list


# ============================================================================
# LaMP 8: Abstract Generation (LongLaMP)
# ============================================================================

def build_episodic_memory_lamp8(profile: List[Dict]) -> List[List]:
    """
    Build EpisodicMemory for LaMP 8 (Abstract Generation).
    Profile: {"abstract": "abstract", "title": "title", "id": "...", "year": ...}
    Query: "Generate an abstract for the title [title]"
    Ground truth: abstract
    Metadata: year

    Returns: [[query1, ground_truth1, metadata1], [query2, ground_truth2, metadata2], ...]
    """
    episodic_memory = []
    for item in profile:
        abstract = item.get("abstract", "")
        title = item.get("title", "")
        year = item.get("year", "")

        episodic_memory.append([
            f"Generate an abstract for the title: {title}",
            abstract,
            {"year": year, "id": item.get("id", "")}
        ])
    return episodic_memory


def load_lamp8(input_path: str, counter: int) -> Tuple[List[List], str, List[str], List[str]]:
    """
    Load LaMP 8 data and return (EpisodicMemory, SemanticMemory, inputs_list, outputs_list).

    Uses streaming JSON parsing for large files (3GB+).

    For LaMP datasets:
    - Profile分割根据数量动态决定
    - Last LAMP_SEMANTIC_MEMORY_COUNT (5) entries of EpisodicMemory -> SemanticMemory generation
    - Current input/output + additional profile entries -> inputs_list, outputs_list
    """
    data_item = load_json_item_streaming(input_path, counter)

    if data_item is None:
        raise ValueError(f"Could not find item at index {counter} in {input_path}")

    profile = data_item.get("profile", [])
    current_input = data_item.get("input", "")
    current_output = data_item.get("output", "")

    # 根据profile数量动态分割
    episodic_profile, test_profile = _split_profile_for_lamp(profile)

    # Build EpisodicMemory from episodic_profile
    episodic_memory = build_episodic_memory_lamp8(episodic_profile)

    # Extract SemanticMemory from last M entries of EpisodicMemory
    semantic_count = min(LAMP_SEMANTIC_MEMORY_COUNT, len(episodic_memory))
    semantic_entries = episodic_memory[-semantic_count:] if semantic_count > 0 else episodic_memory
    semantic_memory = extract_semantic_memory(semantic_entries)

    # Build inputs_list and outputs_list
    inputs_list = [current_input]
    outputs_list = [current_output]

    for item in test_profile:
        title = item.get("title", "")
        abstract = item.get("abstract", "")
        query = f"Generate an abstract for the title: {title}"
        inputs_list.append(query)
        outputs_list.append(abstract)

    return episodic_memory, semantic_memory, inputs_list, outputs_list


# ============================================================================
# LaMP 9: Product Review Generation (LongLaMP)
# ============================================================================

def build_episodic_memory_lamp9(profile: List[Dict]) -> List[List]:
    """
    Build EpisodicMemory for LaMP 9 (Product Review Generation).
    Profile: {"description": "product_desc", "overall": rating, "reviewText": "review_content", "summary": "review_summary"}
    Query: "Generate the review text for a product with description [product_desc] and rating [rating]"
    Ground truth: review_content
    Metadata: review_summary

    Returns: [[query1, ground_truth1, metadata1], [query2, ground_truth2, metadata2], ...]
    """
    episodic_memory = []
    for item in profile:
        description = item.get("description", "")
        rating = item.get("overall", "")
        review_text = item.get("reviewText", "")
        summary = item.get("summary", "")

        episodic_memory.append([
            f"Generate the review text for a product with description: {description} and rating {rating}",
            review_text,
            {"summary": summary}
        ])
    return episodic_memory


def load_lamp9(input_path: str, counter: int) -> Tuple[List[List], str, List[str], List[str]]:
    """
    Load LaMP 9 data and return (EpisodicMemory, SemanticMemory, inputs_list, outputs_list).

    Uses streaming JSON parsing for large files.

    For LaMP datasets:
    - Profile分割根据数量动态决定
    - Last LAMP_SEMANTIC_MEMORY_COUNT (5) entries of EpisodicMemory -> SemanticMemory generation
    - Current input/output + additional profile entries -> inputs_list, outputs_list
    """
    data_item = load_json_item_streaming(input_path, counter)

    if data_item is None:
        raise ValueError(f"Could not find item at index {counter} in {input_path}")

    profile = data_item.get("profile", [])
    current_input = data_item.get("input", "")
    current_output = data_item.get("output", "")

    # 根据profile数量动态分割
    episodic_profile, test_profile = _split_profile_for_lamp(profile)

    # Build EpisodicMemory from episodic_profile
    episodic_memory = build_episodic_memory_lamp9(episodic_profile)

    # Extract SemanticMemory from last M entries of EpisodicMemory
    semantic_count = min(LAMP_SEMANTIC_MEMORY_COUNT, len(episodic_memory))
    semantic_entries = episodic_memory[-semantic_count:] if semantic_count > 0 else episodic_memory
    semantic_memory = extract_semantic_memory(semantic_entries)

    # Build inputs_list and outputs_list
    inputs_list = [current_input]
    outputs_list = [current_output]

    for item in test_profile:
        description = item.get("description", "")
        rating = item.get("overall", "")
        review_text = item.get("reviewText", "")
        query = f"Generate the review text for a product with description: {description} and rating {rating}"
        inputs_list.append(query)
        outputs_list.append(review_text)

    return episodic_memory, semantic_memory, inputs_list, outputs_list


# ============================================================================
# LaMP 10: Reddit Post Generation (LongLaMP)
# ============================================================================

def build_episodic_memory_lamp10(profile: List[Dict]) -> List[List]:
    """
    Build EpisodicMemory for LaMP 10 (Reddit Post Generation).
    Profile: {"author": "author", "content": "content", "id": "...", "summary": "summary"}
    Query: "Generate the content for a reddit post [summary]"
    Ground truth: content
    Metadata: empty

    Returns: [[query1, ground_truth1, metadata1], [query2, ground_truth2, metadata2], ...]
    """
    episodic_memory = []
    for item in profile:
        content = item.get("content", "")
        summary = item.get("summary", "")

        episodic_memory.append([
            f"Generate the content for a reddit post: {summary}",
            content,
            {}
        ])
    return episodic_memory


def load_lamp10(input_path: str, counter: int) -> Tuple[List[List], str, List[str], List[str]]:
    """
    Load LaMP 10 data and return (EpisodicMemory, SemanticMemory, inputs_list, outputs_list).

    Uses streaming JSON parsing for large files.

    For LaMP datasets:
    - Profile分割根据数量动态决定
    - Last LAMP_SEMANTIC_MEMORY_COUNT (5) entries of EpisodicMemory -> SemanticMemory generation
    - Current input/output + additional profile entries -> inputs_list, outputs_list
    """
    data_item = load_json_item_streaming(input_path, counter)

    if data_item is None:
        raise ValueError(f"Could not find item at index {counter} in {input_path}")

    profile = data_item.get("profile", [])
    current_input = data_item.get("input", "")
    current_output = data_item.get("output", "")

    # 根据profile数量动态分割
    episodic_profile, test_profile = _split_profile_for_lamp(profile)

    # Build EpisodicMemory from episodic_profile
    episodic_memory = build_episodic_memory_lamp10(episodic_profile)

    # Extract SemanticMemory from last M entries of EpisodicMemory
    semantic_count = min(LAMP_SEMANTIC_MEMORY_COUNT, len(episodic_memory))
    semantic_entries = episodic_memory[-semantic_count:] if semantic_count > 0 else episodic_memory
    semantic_memory = extract_semantic_memory(semantic_entries)

    # Build inputs_list and outputs_list
    inputs_list = [current_input]
    outputs_list = [current_output]

    for item in test_profile:
        summary = item.get("summary", "")
        content = item.get("content", "")
        query = f"Generate the content for a reddit post: {summary}"
        inputs_list.append(query)
        outputs_list.append(content)

    return episodic_memory, semantic_memory, inputs_list, outputs_list


# ============================================================================
# UltraChat / WildChat: Multi-turn Dialogue
# ============================================================================

def build_episodic_memory_multiturn(profile: List[Dict], max_past_turns: int = 5) -> List[List]:
    """
    Build EpisodicMemory for UltraChat/WildChat (Multi-turn Dialogue).
    Task: Predict next user query
    Query: "Predict the user's next query based on the past conversation: [past_conversation]"
    Ground truth: User's question in the current turn
    Past conversation: Maximum of 5 most recent conversation turns
    Metadata: Current turn number

    Note: Starting from turn 2 since turn 1 has no past conversation

    Returns: [[query1, ground_truth1, metadata1], [query2, ground_truth2, metadata2], ...]
    """
    episodic_memory = []

    for turn_idx in range(1, len(profile)):  # Start from turn 2 (index 1)
        # Get past conversation (up to max_past_turns)
        start_idx = max(0, turn_idx - max_past_turns)
        past_turns = profile[start_idx:turn_idx]

        # Format past conversation
        past_conversation_parts = []
        for turn in past_turns:
            if isinstance(turn, dict):
                user_msg = turn.get("user", "")
                response_msg = turn.get("response", "")
                past_conversation_parts.append(f"User: {user_msg}\nAssistant: {response_msg}")
            else:
                past_conversation_parts.append(f"User: {turn}")

        past_conversation = "\n".join(past_conversation_parts)

        # Get current turn's user query as ground truth
        current_turn = profile[turn_idx]
        if isinstance(current_turn, dict):
            ground_truth = current_turn.get("user", "")
        else:
            ground_truth = str(current_turn)

        episodic_memory.append([
            f"Predict the user's next query based on the past conversation: {past_conversation}",
            ground_truth,
            {"turn_number": turn_idx + 1}
        ])

    return episodic_memory


def load_multiturn_dialogue(input_path: str, counter: int, _dataset_type: int) -> Tuple[List[Dict], str, str, str]:
    """
    Load UltraChat/WildChat data and return (EpisodicMemory, SemanticMemory, input, output).

    优化：使用流式加载单个用户数据，减少内存占用
    """
    # 使用流式加载单个用户数据
    data_item = load_json_item_streaming(input_path, counter)
    if data_item is None:
        raise ValueError(f"Could not find item at index {counter} in {input_path}")

    profile = data_item.get("profile", [])
    output_text = data_item.get("output", "")

    # Fixed input for multi-turn dialogue
    input_text = "predict user's next query"

    # Build episodic memory from profile
    episodic_memory = build_episodic_memory_multiturn(profile)

    # Extract semantic memory from episodic memory
    semantic_memory = extract_semantic_memory(episodic_memory)

    return episodic_memory, semantic_memory, input_text, output_text


# ============================================================================
# PrefEval: Preference Evaluation with Persona
# ============================================================================

def build_episodic_memory_prefeval(conversation: Dict) -> List[List]:
    """
    Build EpisodicMemory for PrefEval.
    Query: User's question in each conversation turn
    Ground truth: AI's response in each conversation turn
    Metadata: Current turn number

    Returns: [[query1, ground_truth1, metadata1], [query2, ground_truth2, metadata2], ...]
    """
    episodic_memory = []

    # Sort conversation keys to ensure correct order
    sorted_keys = sorted(conversation.keys(), key=lambda x: int(x))

    for i, key in enumerate(sorted_keys):
        turn = conversation[key]
        user_query = turn.get("user", "")
        assistant_response = turn.get("assistant", "")

        # Query is the user's question, ground_truth is the AI's response
        episodic_memory.append([
            user_query,
            assistant_response,
            {"turn_number": i + 1}
        ])

    return episodic_memory


# PrefEval response 缓存目录 (与 FTPERSLLM 共享)
PREFEVAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "prefeval_output")


def get_prefeval_response_with_cache(
    input_id: int,
    input_text: str,
    episodic_memory: List[List],
    preference: str,
    explanation: str,
    persona: str
) -> str:
    """
    获取 PrefEval response，优先从缓存加载，不存在则生成并保存

    Args:
        input_id: 数据的唯一标识符
        input_text: The question to answer
        episodic_memory: List of [query, ground_truth, metadata] representing conversation history
        preference: User's preference
        explanation: Explanation of the preference
        persona: User's persona description

    Returns:
        str: response 文本
    """
    import os

    # 确保缓存目录存在
    os.makedirs(PREFEVAL_OUTPUT_DIR, exist_ok=True)

    # 缓存文件路径
    cache_file = os.path.join(PREFEVAL_OUTPUT_DIR, f"response_{input_id}.json")

    # 尝试从缓存加载
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                response = cached_data.get("response", "")
                if response:
                    print(f"   📂 [PrefEval] 从缓存加载 response (id={input_id})")
                    return response
        except Exception as e:
            print(f"   ⚠️ [PrefEval] 缓存加载失败 (id={input_id}): {e}")

    # 缓存不存在或加载失败，生成新的 response
    print(f"   🔄 [PrefEval] 生成新 response (id={input_id})")
    response = generate_prefeval_response(input_text, episodic_memory, preference, explanation, persona)

    # 保存到缓存
    try:
        cache_data = {
            "input_id": input_id,
            "question": input_text,
            "preference": preference,
            "explanation": explanation,
            "persona": persona,
            "response": response
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"   💾 [PrefEval] response 已保存到缓存 (id={input_id})")
    except Exception as e:
        print(f"   ⚠️ [PrefEval] 缓存保存失败 (id={input_id}): {e}")

    return response


def generate_prefeval_response(
    input_text: str,
    episodic_memory: List[List],
    preference: str,
    explanation: str,
    persona: str
) -> str:
    """
    Generate a response for PrefEval using LLM based on input, EpisodicMemory, and output info.

    Args:
        input_text: The question to answer
        episodic_memory: List of [query, ground_truth, metadata] representing conversation history
        preference: User's preference
        explanation: Explanation of the preference
        persona: User's persona description

    Returns:
        Generated response string
    """
    # Format episodic memory as conversation history
    conversation_history = ""
    for item in episodic_memory:
        query, response, metadata = item
        turn_num = metadata.get("turn_number", "")
        conversation_history += f"Turn {turn_num}:\nUser: {query}\nAssistant: {response}\n\n"

    prompt = f"""Based on the following information, generate an appropriate response to the user's question.

    ### User Persona:
    {persona}

    ### User Preference:
    {preference}

    ### Explanation of Preference:
    {explanation}

    ### Conversation History:
    {conversation_history}

    ### Current Question:
    {input_text}

    ### Instructions:
    Generate a response that:
    1. Addresses the user's question
    2. Aligns with the user's stated preferences
    3. Is consistent with the persona and conversation history
    4. Is natural and helpful

    Response:"""

    system_prompt = """You are a helpful assistant that generates personalized responses based on user preferences and conversation history. Your responses should be natural, helpful, and aligned with the user's stated preferences."""

    max_retries = 5
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-v3.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                timeout=30.0
            )

            if not response or not response.choices or len(response.choices) == 0:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️ PrefEval response API invalid response, retrying in {delay:.1f}s ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("PrefEval response API returned invalid response")

            content = response.choices[0].message.content
            if content is None or not content.strip():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️ PrefEval response API empty content, retrying in {delay:.1f}s ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("PrefEval response API returned empty content")

            return content.strip()

        except Exception as e:
            if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):
                # Fallback: return a basic response
                return f"Based on your preferences, here is my response to your question: {input_text}"

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"⚠️ PrefEval response API error ({e}), retrying in {delay:.1f}s ({attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"⚠️ PrefEval response generation failed after {max_retries} retries")
                return f"Based on your preferences, here is my response to your question: {input_text}"

    return f"Based on your preferences, here is my response to your question: {input_text}"


def load_prefeval(input_path: str, counter: int) -> Tuple[List[List], str, str, str]:
    """
    Load PrefEval data and return (EpisodicMemory, SemanticMemory, input, output).

    For PrefEval:
    - input = "question" field
    - output = LLM-generated response (with cache support)

    优化：使用流式加载单个用户数据，减少内存占用
    """
    # 使用流式加载单个用户数据
    data_item = load_json_item_streaming(input_path, counter)
    if data_item is None:
        raise ValueError(f"Could not find item at index {counter} in {input_path}")

    input_id = data_item.get("id", counter)
    conversation = data_item.get("conversation", {})
    input_text = data_item.get("question", "")

    # Get preference info for response generation
    preference = data_item.get("preference", "")
    explanation = data_item.get("explanation", "")
    persona = data_item.get("persona", "")

    # Build episodic memory first, then extract semantic memory from it
    episodic_memory = build_episodic_memory_prefeval(conversation)
    semantic_memory = extract_semantic_memory(episodic_memory)

    # Generate response using LLM with cache support
    output = get_prefeval_response_with_cache(
        input_id=input_id,
        input_text=input_text,
        episodic_memory=episodic_memory,
        preference=preference,
        explanation=explanation,
        persona=persona
    )

    return episodic_memory, semantic_memory, input_text, output


# ============================================================================
# Main Function: load_template_data
# ============================================================================

def load_template_data(
    dataset_identifier: Union[str, int],
    counter: int,
    input_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> Tuple[List[List], str, Any, Any]:
    """
    Main function to load and process data for PersonaAgent.

    Args:
        dataset_identifier: Dataset name or type ID
            - Names: "lamp4", "lamp5", "lamp8", "lamp9", "lamp10",
                     "ultrachat", "wildchat", "prefeval"
            - IDs: 4, 5, 8, 9, 10, 0, -1, -2
        counter: Index of the data item to load
        input_path: Optional custom input file path (uses default if None)
        output_path: Optional custom output file path (uses default if None)

    Returns:
        Tuple of (EpisodicMemory, SemanticMemory, inputs, outputs)
        - EpisodicMemory: List of [[query, ground_truth, metadata], ...]
        - SemanticMemory: String containing LLM-extracted user preferences
        - inputs: For LaMP 4/5/8/9/10: List[str] of input queries
                  For other datasets: str (single input)
        - outputs: For LaMP 4/5/8/9/10: List[str] of expected outputs
                   For other datasets: str (single output)
    """
    dataset_type = get_dataset_type(dataset_identifier)

    # Get default paths if not provided
    if input_path is None:
        paths = DEFAULT_PATHS.get(dataset_type, {})
        input_path = paths.get("input")
        if input_path is None:
            raise ValueError(f"No default input path for dataset type {dataset_type}")

    if output_path is None:
        paths = DEFAULT_PATHS.get(dataset_type, {})
        output_path = paths.get("output")

    print(f"📂 Loading dataset type {dataset_type} (counter={counter})")
    print(f"   Input: {input_path}")
    if output_path:
        print(f"   Output: {output_path}")

    # Route to appropriate loader
    if dataset_type == 4:  # LaMP 4
        return load_lamp4(input_path, output_path, counter)

    elif dataset_type == 5:  # LaMP 5
        return load_lamp5(input_path, output_path, counter)

    elif dataset_type == 8:  # LaMP 8 (LongLaMP Abstract Generation)
        return load_lamp8(input_path, counter)

    elif dataset_type == 9:  # LaMP 9 (LongLaMP Product Review)
        return load_lamp9(input_path, counter)

    elif dataset_type == 10:  # LaMP 10 (LongLaMP Topic Writing)
        return load_lamp10(input_path, counter)

    elif dataset_type in [0, -1]:  # UltraChat (0) or WildChat (-1)
        return load_multiturn_dialogue(input_path, counter, dataset_type)

    elif dataset_type == -2:  # PrefEval
        return load_prefeval(input_path, counter)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_dataset_size(dataset_identifier: Union[str, int], input_path: Optional[str] = None) -> int:
    """
    Get the total number of items in a dataset.

    Args:
        dataset_identifier: Dataset name or type ID
        input_path: Optional custom input file path

    Returns:
        Number of items in the dataset
    """
    dataset_type = get_dataset_type(dataset_identifier)

    if input_path is None:
        paths = DEFAULT_PATHS.get(dataset_type, {})
        input_path = paths.get("input")
        if input_path is None:
            raise ValueError(f"No default input path for dataset type {dataset_type}")

    # LongLaMP datasets are JSONL format
    if dataset_type in [8, 9, 10]:
        data = load_jsonl_file(input_path)
    else:
        data = load_json_file(input_path)

    return len(data)


# ============================================================================
# Convenience Functions for Batch Processing
# ============================================================================

def load_all_data(
    dataset_identifier: Union[str, int],
    max_items: Optional[int] = None,
    input_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> List[Tuple[List[Dict], str, Any, Any]]:
    """
    Load all items from a dataset.

    Args:
        dataset_identifier: Dataset name or type ID
        max_items: Maximum number of items to load (None for all)
        input_path: Optional custom input file path
        output_path: Optional custom output file path

    Returns:
        List of (EpisodicMemory, SemanticMemory, input, output) tuples
    """
    dataset_size = get_dataset_size(dataset_identifier, input_path)

    if max_items is not None:
        dataset_size = min(dataset_size, max_items)

    results = []
    for i in range(dataset_size):
        try:
            result = load_template_data(dataset_identifier, i, input_path, output_path)
            results.append(result)

            if (i + 1) % 100 == 0:
                print(f"   Loaded {i + 1}/{dataset_size} items...")
        except Exception as e:
            print(f"⚠️ Error loading item {i}: {e}")
            continue

    print(f"✅ Loaded {len(results)} items from dataset")
    return results


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    # Test loading a single item from each dataset
    print("=" * 60)
    print("PersonaAgent Data Loader Test")
    print("=" * 60)

    # Test LaMP 4
    try:
        print("\n--- Testing LaMP 4 ---")
        em, sm, inp, out = load_template_data("lamp4", 0)
        print(f"EpisodicMemory items: {len(em)}")
        print(f"SemanticMemory length: {len(sm)}")
        print(f"Input: {inp[:100]}...")
        print(f"Output: {out[:100] if out else 'N/A'}...")
    except Exception as e:
        print(f"Error: {e}")

    # Test PrefEval
    try:
        print("\n--- Testing PrefEval ---")
        em, sm, inp, out = load_template_data("prefeval", 0)
        print(f"EpisodicMemory items: {len(em)}")
        print(f"SemanticMemory length: {len(sm)}")
        print(f"Input: {inp[:100]}...")
        print(f"Output (preference, explanation, persona): {out[0][:50] if out[0] else 'N/A'}...")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
