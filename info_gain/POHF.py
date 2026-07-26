import os, sys
import math
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# 抑制常见的警告信息
warnings.filterwarnings("ignore", message="Extension saving to grad_batch")
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")

from rouge_score import rouge_scorer
import hashlib
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# 设置matplotlib后端为Agg（非GUI模式，适合服务器/并行环境）
import matplotlib
matplotlib.use('Agg')

#临时把当前目录加入到工作目录中
cwd = os.getcwd()
sys.path.append(cwd)

# 导入information_second_term模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from information_second_term import PairwiseInformationManager, ContextualPairwiseInformationManager


def is_lamp_dataset(lamp_type: int = None) -> bool:
    """
    判断当前数据集是否为LaMP数据集。

    🔧 [统一] 现在所有数据集都使用contextual模式(2048维输入: query concat persona)。
    此函数仅用于识别LaMP数据集类型，不再影响embedding维度。

    Args:
        lamp_type: LaMP类型，如果为None则从环境变量获取

    Returns:
        bool: True如果是LaMP数据集(4,5,8,9,10)
    """
    if lamp_type is None:
        lamp_type = int(os.environ.get('POHF_LAMP_TYPE', 0))

    # LaMP数据集类型: 4, 5, 8, 9, 10
    return lamp_type in [4, 5, 8, 9, 10]


def get_input_dim_for_dataset(lamp_type: int = None, config: dict = None) -> int:
    """
    根据数据集类型获取正确的输入维度。

    🔧 [统一] 现在所有数据集都使用2048维输入（query concat persona）。

    Args:
        lamp_type: LaMP类型（保留兼容性，不再影响维度）
        config: 配置字典

    Returns:
        int: 输入维度（统一为2048）
    """
    try:
        from POHF_parameters import CONTEXTUAL_BANDIT_CONFIG
        contextual_dim = CONTEXTUAL_BANDIT_CONFIG.get("contextual_input_dim", 2048)
    except ImportError:
        contextual_dim = 2048

    # 🔧 [统一] 所有数据集都使用2048维
    return contextual_dim


def should_use_contextual_mode(lamp_type: int = None) -> bool:
    """
    判断是否应该使用contextual dueling bandit模式。

    🔧 [统一] 现在所有数据集都使用contextual模式（query concat persona）。

    Args:
        lamp_type: LaMP类型（保留兼容性，不再影响模式选择）

    Returns:
        bool: 始终返回True（所有数据集都使用contextual模式）
    """
    # 🔧 [统一] 所有数据集都使用contextual模式
    return True


# 全局响应缓存
_response_cache = {}
import os
import torch

# 设置 multiprocessing 启动方法为 spawn（避免 CUDA 多进程问题）
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已经设置过了

def detect_and_setup_gpu():
    """
    检测并设置 GPU 设备

    注意：此函数应该在每个进程中调用，以确保正确读取 CUDA_VISIBLE_DEVICES
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    # 统一使用cuda:0（通过CUDA_VISIBLE_DEVICES控制实际物理GPU）
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    return torch.device("cuda:0")

def get_device():
    """获取当前设备（延迟初始化）"""
    if not hasattr(get_device, '_device'):
        get_device._device = detect_and_setup_gpu()
    return get_device._device

def reset_device():
    """重置设备缓存（用于子进程）"""
    if hasattr(get_device, '_device'):
        delattr(get_device, '_device')

# 延迟初始化：不在模块加载时设置 GPU
# 这样子进程可以在启动后根据自己的 CUDA_VISIBLE_DEVICES 正确初始化
detected_device = None  # 将在需要时通过 get_device() 获取

tkwargs = {
    "device": torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
    "dtype": torch.float32,
}
import torch.nn.functional as F

from load_data import load_templated_data
import nltk

from openai import OpenAI, AsyncOpenAI
import asyncio

###整理时删除
# 同步客户端（保留用于向后兼容）
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY", "API_KEY_NOT_SET"),
    base_url="https://openrouter.ai/api/v1"
)

# 异步客户端（用于并行 API 调用）
async_client = AsyncOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY", "API_KEY_NOT_SET"),
    base_url="https://openrouter.ai/api/v1"
)

def get_dataset_name(lamp_type):
    """
    根据LaMP_type获取数据集名称

    Args:
        lamp_type: 数据集类型
            - 0: UltraChat
            - -1: WildChat
            - 其他: LaMP数据集

    Returns:
        str: 数据集名称
    """
    if lamp_type == 0:
        return "ultrachat"
    elif lamp_type == -1:
        return "wildchat"
    else:
        return f"lamp{lamp_type}"


def get_filename_prefix(lamp_type):

    # 如果已经是字符串形式的数据集名称
    if isinstance(lamp_type, str):
        if lamp_type in ['ultrachat', 'wildchat']:
            return lamp_type
        elif lamp_type == 'unknown':
            return 'unknown'
        else:
            # 可能是 "lamp4" 这样的格式，直接返回
            return lamp_type

    # 如果是整数形式的 LaMP_type
    if lamp_type == 0:
        return "ultrachat"
    elif lamp_type == -1:
        return "wildchat"
    else:
        return f"lamp{lamp_type}"

###
def prompt_reformer(input_data, instruction_index, summary_index, lamp_type=None, query_index=0):
    """
    生成prompt

    Args:
        input_data: 模板化的输入数据
        instruction_index: 指令索引（已弃用，保留兼容性）
        summary_index: summary变体索引
        lamp_type: 数据集类型
        query_index: query索引（对于LaMP数据集，用于选择特定的query）

    Returns:
        格式化的prompt字符串
    """
    # ========== 处理UltraChat和WildChat数据集 ==========
    if lamp_type in [0, -1]:

        dataset_name = "ultrachat" if lamp_type == 0 else "wildchat"

        prompt = "\n".join([
            "### Task Instruction:",
            input_data[2],
            "",
            "### Conversation History:",
            "\n".join(str(item) for item in input_data[5]) if isinstance(input_data[5], list) else str(input_data[5]),
            "",
             "### Important Words:",
            input_data[4],
            "",
            "### Personality and Style Description:",
            input_data[3][summary_index],  # summary
            "",
            "### Output Requirement:",
            "IMPORTANT: Your response must strongly reflect the personality traits described in 'Personality and Style Description'. "
            "Based on the conversation history and user profile, predict what the user will ask next. "
            "Your prediction should be consistent with the user's communication style, personality and interests shown in the conversation history. "
            "Do not contain things like 'user:'"
            "Please provide only the predicted user query below:"
        ])

        return prompt

    # ========== 处理原始LaMP和LongLaMP数据集 ==========
    # 处理input_data[1] - Current Task Details (query)
    # 对于LaMP数据集，query是一个列表，使用query_index选择特定的query
    query_data = input_data[1]
    if isinstance(query_data, list):
        # 确保query_index在有效范围内
        if query_index < len(query_data):
            current_task = str(query_data[query_index])
        else:
            # 如果索引超出范围，使用第一个query
            current_task = str(query_data[0]) if len(query_data) > 0 else ""
    else:
        current_task = str(query_data) if query_data else ""

    # 处理input_data[4] - Important Words (synthesis)，可能是列表或字符串
    important_words = input_data[4] if isinstance(input_data[4], str) else ", ".join(str(w) for w in input_data[4]) if isinstance(input_data[4], list) else str(input_data[4])

    # ========== 处理 input_data[5] - ranked_entries ==========
    # input_data[5] 现在是 ranked_entries_list（列表的列表）
    # 每个元素对应一个 query 的 ranked_entries
    # 根据 query_index 选择对应的 ranked_entries
    ranked_entries_data = input_data[5]

    if isinstance(ranked_entries_data, list) and len(ranked_entries_data) > 0:
        # 检查是否是列表的列表（新格式：每个query对应一组ranked_entries）
        if isinstance(ranked_entries_data[0], list):
            # 新格式：ranked_entries_list[query_index]
            if query_index < len(ranked_entries_data):
                current_ranked_entries = ranked_entries_data[query_index]
            else:
                # 索引超出范围，使用第一个
                current_ranked_entries = ranked_entries_data[0]
        else:
            # 旧格式：单一的 ranked_entries 列表（兼容性）
            current_ranked_entries = ranked_entries_data
    else:
        current_ranked_entries = []

    # 格式化 history context，每个条目截断到前 256 个字符
    def truncate_entry(entry, max_chars=256):
        """将条目截断到指定字符数"""
        if isinstance(entry, dict):
            text = entry.get('text', str(entry))
        else:
            text = str(entry)
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    history_context_str = "\n".join(truncate_entry(item) for item in current_ranked_entries) if current_ranked_entries else ""

    prompt = "\n".join([
        "### Task Instruction:",
        input_data[2],
        "",
        "### Current Task Details:",
        current_task,
        "",
        "### Personality and Style Description:",
        input_data[3][summary_index],
        "",
        "### Important Words:",
        important_words,
        "",
        "### History context:",
        history_context_str,
        "",
        "### Output Requirement:",
        "IMPORTANT: Your response must strongly reflect the personality traits described in 'Personality and Style Description'. "
        "Important Words are keywords extracted from the user's historical information and may be used as supplementary references."
        "Your output must follow the Task Instruction and Current Task Details and should be written in the style described in the Personality and Style Description section. "
        "Please provide only the response required to complete the task, without including any additional information or descriptions."
        "please generate content below:"
    ])

    return prompt


async def response_generator_async(prompt, config=None, personality_description=None):
    """
    异步生成响应，支持缓存和确定性输出

    Args:
        prompt: 输入提示
        config: LLM配置参数
        personality_description: 用户人格描述（可选），用于增强system role
    """
    global _response_cache

    if config is None:
        from POHF_parameters import LLM_CONFIG
        config = LLM_CONFIG

    # 构建缓存key时包含personality_description
    cache_key_content = prompt + (personality_description or "")
    if config.get("use_cache", True):
        prompt_hash = hashlib.md5(cache_key_content.encode()).hexdigest()
        if prompt_hash in _response_cache:
            return _response_cache[prompt_hash]
    else:
        prompt_hash = None

    # 构建system role，如果提供了personality_description则强化
    if personality_description:
        system_content = (
            "You are an assistant specialized in faithfully mimicking what human would do. "
            "Your top priority is to generate text that **perfectly reflects the personality, tone, and reasoning style** "
            "of the user described below.\n\n"
            f"### User Personality and Style (MUST FOLLOW):\n{personality_description}\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. Your response must strongly reflect the personality traits described above.\n"
            "2. Adapt your tone, vocabulary, sentence structure, and reasoning style to match the user's personality.\n"
            "3. Your response must strictly follow the requirements stated in the Task Instruction.\n"
            "4. Ensure the final output feels human-written and consistent with the persona."
        )
    else:
        system_content = (
            "You are an assistant specialized in faithfully mimicking what human would do. "
            "Your top priority is to generate text based on Current Task Details that **perfectly reflects the personality, tone, and reasoning style** "
            "described in the 'User Personality and Style Description'. "
            "your response must **strictly follow the requirements and objectives stated in the Task Instruction**. "
            "Ensure the final output feels human-written and consistent with the persona."
        )

    # 从API_CONFIG获取模型名称
    from POHF_parameters import API_CONFIG
    model_name = API_CONFIG.get("openai_model", "deepseek/deepseek-v3.2")

    request_params = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": config.get("temperature", 0.4),
        "top_p": config.get("top_p", 1.0),
        "frequency_penalty": config.get("frequency_penalty", 0.0),
        "presence_penalty": config.get("presence_penalty", 0.0),
    }

    if config.get("use_seed", True):
        request_params["seed"] = config.get("seed", 0)

    # 🔧 从配置获取重试参数
    max_retries = config.get("max_retries", 5)
    retry_delay = config.get("retry_delay", 2.0)
    retry_backoff = config.get("retry_backoff", 2.0)

    last_error = None
    for attempt in range(max_retries):
        try:
            response = await async_client.chat.completions.create(**request_params)
            result = response.choices[0].message.content

            max_output_length = config.get("max_output_length", None)
            if max_output_length and len(result) > max_output_length:
                result = result[:max_output_length] + "..."

            if config.get("use_cache", True) and prompt_hash:
                cache_size = config.get("cache_size", 1000)
                if len(_response_cache) >= cache_size:
                    oldest_key = next(iter(_response_cache))
                    del _response_cache[oldest_key]
                _response_cache[prompt_hash] = result
            return result

        except Exception as e:
            last_error = e
            error_message = str(e).lower()
            error_type_name = type(e).__name__

            # 🔧 检测是否是内容安全错误（不重试，直接使用安全提示）
            if 'content' in error_message and ('filter' in error_message or 'policy' in error_message or 'safety' in error_message):
                safer_prompt = (
                    "Please generate a safe, appropriate, and professional response that complies with content policies. "
                    "Avoid any potentially sensitive topics.\n\n" + prompt
                )

                try:
                    response = await async_client.chat.completions.create(**{
                        **request_params,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful, safe, and respectful assistant. Always prioritize content safety."
                            },
                            {
                                "role": "user",
                                "content": safer_prompt
                            }
                        ]
                    })
                    result = response.choices[0].message.content

                    if config.get("use_cache", True) and prompt_hash:
                        cache_size = config.get("cache_size", 1000)
                        if len(_response_cache) >= cache_size:
                            oldest_key = next(iter(_response_cache))
                            del _response_cache[oldest_key]
                        _response_cache[prompt_hash] = result

                    return result
                except Exception as retry_e:
                    print(f"⚠️ [response_generator_async] 安全提示重试失败: {retry_e}")
                    return f"Error generating response: {str(e)}"

            # 🔧 检查是否是可重试的错误类型
            is_retryable = any(keyword in error_message for keyword in [
                'timeout', 'connection', 'rate', 'limit', '429', '500', '502', '503', '504',
                'expecting value', 'json', 'decode', 'reset', 'closed'
            ])

            if is_retryable and attempt < max_retries - 1:
                wait_time = retry_delay * (retry_backoff ** attempt)
                print(f"⚠️ [response_generator_async] 调用失败 (尝试 {attempt + 1}/{max_retries}) [{error_type_name}]: {e}")
                print(f"   等待 {wait_time:.1f}s 后重试...", flush=True)
                await asyncio.sleep(wait_time)
            elif attempt < max_retries - 1:
                # 非可重试错误，但仍尝试一次
                wait_time = retry_delay
                print(f"⚠️ [response_generator_async] 非典型错误 (尝试 {attempt + 1}/{max_retries}) [{error_type_name}]: {e}")
                print(f"   等待 {wait_time:.1f}s 后重试...", flush=True)
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ [response_generator_async] 已达最大重试次数 ({max_retries}) [{error_type_name}]: {e}")

    return f"Error generating response: {str(last_error)}"


def response_generator(prompt, config=None, personality_description=None):
    """
    同步版本的 response_generator（向后兼容）
    内部调用异步版本

    Args:
        prompt: 输入提示
        config: LLM配置参数
        personality_description: 用户人格描述（可选）
    """
    try:
        loop = asyncio.get_running_loop()
        # 如果在异步上下文中，创建一个任务
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, response_generator_async(prompt, config, personality_description))
            return future.result()
    except RuntimeError:
        # 没有运行中的事件循环，直接运行
        return asyncio.run(response_generator_async(prompt, config, personality_description))



import numpy as np


async def rouge_score_comparison_async(ground_truth, response1, response2, rouge_type, LLM_as_judge=False, arm1_idx=None, arm2_idx=None, ground_truth_index=0):
    """
    异步版本的 rouge_score_comparison

    Args:
        ground_truth: 真实值（可以是字符串或列表）
        response1, response2: 两个响应
        rouge_type: ROUGE类型
        LLM_as_judge: 是否使用LLM评判
        arm1_idx, arm2_idx: 臂索引
        ground_truth_index: 当ground_truth是列表时，使用的索引（默认0）

    Returns:
        (score1, score2, score, preference, p_)
    """
    # 设置随机种子以保持一致性
    from POHF_parameters import EXPERIMENT_CONFIG
    random_seed = EXPERIMENT_CONFIG.get("random_seed", 0)
    np.random.seed(random_seed)

    # 处理ground_truth：如果是列表，使用指定索引的元素
    if isinstance(ground_truth, list):
        if ground_truth_index < len(ground_truth):
            ground_truth = str(ground_truth[ground_truth_index])
        elif len(ground_truth) > 0:
            # 如果索引超出范围，使用第一个元素
            ground_truth = str(ground_truth[0])
        else:
            ground_truth = ""
    elif not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)

    # 确保response1和response2是字符串
    if isinstance(response1, list):
        response1 = " ".join(str(item) for item in response1)
    elif not isinstance(response1, str):
        response1 = str(response1)

    if isinstance(response2, list):
        response2 = " ".join(str(item) for item in response2)
    elif not isinstance(response2, str):
        response2 = str(response2)

    if LLM_as_judge == False:
        scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        scores1 = scorer.score(ground_truth, response1)
        scores2 = scorer.score(ground_truth, response2)
        scores1 = scores1['rougeL'].fmeasure
        scores2 = scores2['rougeL'].fmeasure

        from POHF_parameters import ROUGE_CONFIG
        a = ROUGE_CONFIG.get("a", 2.0)
        b = ROUGE_CONFIG.get("b", 0.01)
        beta_min = ROUGE_CONFIG.get("beta_min", 1.0)
        beta_max = ROUGE_CONFIG.get("beta_max", 50.0)

        # 计算自适应beta值
        beta = a / (b + (scores1 + scores2) / 2)
        beta = np.clip(beta, beta_min, beta_max)

        # 计算概率性preference
        p_ = 1 / (1 + np.exp(-beta * (scores1 - scores2)))
        if scores1 == scores2:
            preference = 1
        else:
            preference = np.random.binomial(1, p_)

        score = scores1 if preference == 1 else scores2
        return scores1, scores2, score, preference, p_
    else:
        from POHF_parameters import LLM_CONFIG
        import re

        system_role = (
            "You are an expert evaluator. Your task is to compare two responses and determine which one is better "
            "based on the given ground truth. The primary consideration is whether the sentence was likely written by the same person as the ground truth, supplemented by assessments of accuracy, relevance, completeness, and clarity. "
            "You must respond with ONLY a single digit: 1 if Response 1 is better, or 0 if Response 2 is better. "
            "Do not include any other text, explanation, or formatting."
        )

        prompt = f"""Ground Truth:
                {ground_truth}

                Response 1:
                {response1}

                Response 2:
                {response2}

                Which response is closer to Ground Truth? Reply with only 1 (Response 1 is better) or 0 (Response 2 is better).
                If they are the same, return 1 by default.
                Please provide the final result (1 or 0 only) below:"""

        # 从API_CONFIG获取模型名称
        from POHF_parameters import API_CONFIG
        model_name = API_CONFIG.get("openai_model", "deepseek/deepseek-v3.2")

        request_params = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_role
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": LLM_CONFIG.get("temperature", 0.0),
            "top_p": LLM_CONFIG.get("top_p", 1.0),
        }

        # 重试配置
        max_retries = LLM_CONFIG.get("max_retries", 3)
        retry_delay = LLM_CONFIG.get("retry_delay", 2.0)
        retry_backoff = LLM_CONFIG.get("retry_backoff", 2.0)

        preference = None

        for attempt in range(max_retries):
            try:
                response = await async_client.chat.completions.create(**request_params)
                result = response.choices[0].message.content
                if result is None:
                    result = ""
                result = result.strip()

                # 解析结果，只接受1或0
                if result == "1":
                    preference = 1
                elif result == "0":
                    preference = 0
                else:
                    # 如果返回的不是1或0，尝试提取数字
                    match = re.search(r'[01]', result)
                    if match:
                        preference = int(match.group())
                    else:
                        preference = 1

                break

            except Exception as e:
                error_type_name = type(e).__name__

                # 检查是否是可重试的错误类型
                is_retryable = any(keyword in str(e).lower() for keyword in [
                    'timeout', 'connection', 'rate', 'limit', '429', '500', '502', '503', '504',
                    'expecting value', 'json', 'decode'
                ])

                if attempt < max_retries - 1:
                    wait_time = retry_delay * (retry_backoff ** attempt)
                    print(f"⚠️ [rouge_score_comparison_async] LLM调用失败 (尝试 {attempt + 1}/{max_retries}) [{error_type_name}]: {e}")
                    print(f"   等待 {wait_time:.1f}s 后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"❌ [rouge_score_comparison_async] LLM判断失败，已达最大重试次数 ({max_retries}) [{error_type_name}]: {e}")
                    preference = 1

        # 如果所有重试都失败，使用默认值
        if preference is None:
            print(f"❌ [rouge_score_comparison_async] 所有重试均失败，使用默认 preference=1")
            preference = 1

        # 当使用LLM作为评判时，score_1和score_2根据preference设为1或0
        score_1 = 1 if preference == 1 else 0
        score_2 = 0 if preference == 1 else 1
        return score_1, score_2, None, preference, None


def rouge_score_comparison(ground_truth, response1, response2, rouge_type, LLM_as_judge=False, arm1_idx=None, arm2_idx=None):
    """
    同步版本的 rouge_score_comparison（向后兼容）
    """
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                rouge_score_comparison_async(ground_truth, response1, response2, rouge_type, LLM_as_judge, arm1_idx, arm2_idx)
            )
            return future.result()
    except RuntimeError:
        return asyncio.run(rouge_score_comparison_async(ground_truth, response1, response2, rouge_type, LLM_as_judge, arm1_idx, arm2_idx))


async def batch_generate_responses_async(prompts, config=None, personality_descriptions=None):
    """
    批量并行生成多个响应

    Args:
        prompts: 提示列表
        config: LLM配置
        personality_descriptions: 人格描述列表（可选），与prompts一一对应

    Returns:
        响应列表（与输入顺序对应）
    """
    if personality_descriptions is None:
        tasks = [response_generator_async(prompt, config) for prompt in prompts]
    else:
        tasks = [response_generator_async(prompt, config, pd) for prompt, pd in zip(prompts, personality_descriptions)]
    return await asyncio.gather(*tasks)


async def generate_and_compare_async(prompt1, prompt2, ground_truth, config, LLM_as_judge=False, arm1_idx=None, arm2_idx=None, personality1=None, personality2=None, ground_truth_index=0):
    """
    并行生成两个响应并比较

    Args:
        prompt1, prompt2: 两个提示
        ground_truth: 真实值（对于LaMP数据集是列表）
        config: LLM配置
        LLM_as_judge: 是否使用LLM评判
        arm1_idx, arm2_idx: 臂索引
        personality1, personality2: 两个arm对应的人格描述（可选）
        ground_truth_index: 当ground_truth是列表时，使用的索引（默认0，即第一个真实output）

    Returns:
        (response1, response2, score_1, score_2, score, preference, p_)
    """
    # 并行生成两个响应
    response1, response2 = await asyncio.gather(
        response_generator_async(prompt1, config, personality1),
        response_generator_async(prompt2, config, personality2)
    )

    # 比较响应（必须在两个响应都生成后）
    score_1, score_2, score, preference, p_ = await rouge_score_comparison_async(
        ground_truth, response1, response2, 'rougeL',
        LLM_as_judge=LLM_as_judge, arm1_idx=arm1_idx, arm2_idx=arm2_idx,
        ground_truth_index=ground_truth_index
    )

    return response1, response2, score_1, score_2, score, preference, p_


import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import copy   # 用于 deepcopy

# BackPACK: 用于 per-sample gradient (grad_batch)
from backpack import extend,backpack
from backpack.extensions import BatchGrad


class Network(nn.Module):
    def __init__(self, input_dim, config=None, init_params=None):
        super(Network, self).__init__()

        # 从配置文件获取参数，如果没有配置则使用默认值
        if config is None:
            from POHF_parameters import NETWORK_CONFIG
            config = NETWORK_CONFIG

        hidden_size = config.get("hidden_size", 128)
        depth = config.get("depth", 2)
        dropout_rate = config.get("dropout_rate", 0.2)
        activation = config.get("activation", "GELU")

        self.dropout_rate = dropout_rate

        # 恢复原始POHF网络结构，确保BackPACK兼容性
        layers = []

        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.GELU())  # 使用GELU激活函数

        # 添加额外的隐藏层（depth-1层）
        current_dim = hidden_size
        for i in range(depth - 1):
            next_dim = current_dim  # 保持相同维度，这是原始POHF的做法
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.GELU())  # 使用GELU激活函数
            current_dim = next_dim

        # 最后的输出层
        layers.append(nn.Linear(current_dim, 1))

        self.model = nn.Sequential(*layers)
        self._initialize(init_params)

    def _initialize(self, init_params):
        linear_layers = [layer for layer in self.model if isinstance(layer, nn.Linear)]
        if init_params is None:
            for layer in linear_layers:
                init.kaiming_normal_(layer.weight, nonlinearity='relu')
                init.zeros_(layer.bias)
        else:
            for i, layer in enumerate(linear_layers):
                layer.weight.data = init_params[i*2]
                layer.bias.data = init_params[i*2+1]

    def forward(self, x):
        return self.model(x).squeeze(-1)
    
class NeuralDB:
    """
    升级版NeuralDB - 使用配置文件参数

    """

    def __init__(self, input_dim, config=None):
        """
        初始化NeuralDB

        Args:
            input_dim: 输入维度（如果为None，将根据数据集类型自动确定）
            config: 配置参数
        """
        # 从配置文件加载参数
        if config is None:
            from POHF_parameters import NETWORK_CONFIG, TRAINING_CONFIG, POHF_CONFIG, DEVICE_CONFIG
            network_config = NETWORK_CONFIG.copy()
            training_config = TRAINING_CONFIG
            pohf_config = POHF_CONFIG
            device_config = DEVICE_CONFIG
        else:
            network_config = config.get("network", {})
            training_config = config.get("training", {})
            pohf_config = config.get("pohf", {})
            device_config = config.get("device", {})

        # 设备配置
        device_type = device_config.get("device", "cuda:1")
        self.device = torch.device(device_type if torch.cuda.is_available() else "cpu")

        # GPU内存管理
        if self.device.type == 'cuda' and device_config.get("clear_cache", True):
            torch.cuda.empty_cache()

        # 确定输入维度：优先使用传入的input_dim
        # 否则根据数据集类型自动确定（LaMP使用2048维，其他使用1024维）
        if input_dim is not None:
            self.input_dim = input_dim
        else:
            self.input_dim = get_input_dim_for_dataset(config=config)

        network_config["input_dim"] = self.input_dim  # 设置实际的输入维度

        # POHF算法参数
        self.version = pohf_config.get("version", "matrix")
        self.lamb = pohf_config.get("lambda", 1.0)
        self.nu = pohf_config.get("nu", 0.2)

        # 创建网络
        self.func = extend(Network(self.input_dim, config=network_config)).to(self.device)

        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.init_model_weight = copy.deepcopy(self.func.state_dict())

        # 训练配置
        self.lr = training_config.get("learning_rate", 1e-3)
        self.epoch = training_config.get("epochs", 100)
        self.weight_decay = training_config.get("weight_decay", 1.0)

        # 优化器配置
        optimizer_type = training_config.get("optimizer", "AdamW")
        if optimizer_type == "AdamW":
            self.optimizer_fn = optim.AdamW
        elif optimizer_type == "Adam":
            self.optimizer_fn = optim.Adam
        else:
            self.optimizer_fn = optim.AdamW  # 默认使用AdamW

        # 不extend optimizer，只extend model即可
        self.optimizer = self.optimizer_fn(self.func.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
        ## 协方差矩阵初始化，使用配置参数

        # 从配置获取内存优化参数
        max_params_for_matrix = pohf_config.get("max_params_for_matrix", 10000)

        if self.total_param > max_params_for_matrix:  # 如果参数太多，强制使用diag版本
            self.version = "diag"

        # 获取协方差矩阵初始化参数
        cov_init_enabled = pohf_config.get("cov_init_enabled", True)
        cov_init_value = pohf_config.get("cov_init_value", 0.01)

        if self.version == "diag":
            if cov_init_enabled:
                self.S = torch.ones(self.total_param, dtype=torch.float32, device=self.device) * cov_init_value
                self.Sinv = 1.0 / self.S.clamp(min=1e-16)
            else:
                self.S = self.lamb * torch.ones(self.total_param, dtype=torch.float32, device=self.device)
                self.Sinv = 1.0 / self.S
        elif self.version == "matrix":
            # 只有在参数数量合理时才使用matrix版本
            if self.total_param <= max_params_for_matrix:
                if cov_init_enabled:
                    self.S = torch.eye(self.total_param, dtype=torch.float32, device=self.device) * cov_init_value
                    self.Sinv = torch.eye(self.total_param, dtype=torch.float32, device=self.device) / cov_init_value
                else:
                    self.S = self.lamb * torch.eye(self.total_param, dtype=torch.float32, device=self.device)
                    self.Sinv = torch.inverse(self.S)
            else:
                self.version = "diag"
                if cov_init_enabled:
                    self.S = torch.ones(self.total_param, dtype=torch.float32, device=self.device) * cov_init_value
                    self.Sinv = 1.0 / self.S.clamp(min=1e-16)
                else:
                    self.S = self.lamb * torch.ones(self.total_param, dtype=torch.float32, device=self.device)
                    self.Sinv = 1.0 / self.S


    def restart_model(self, N):
        self.func.load_state_dict(copy.deepcopy(self.init_model_weight))

        # 从配置获取训练参数
        from POHF_parameters import TRAINING_CONFIG

        fixed_weight_decay = self.weight_decay  # 新版本：固定不变

        self.optimizer = self.optimizer_fn(
            self.func.parameters(),
            lr=self.lr,
            weight_decay=fixed_weight_decay  # 使用固定的weight_decay
        )

        # 使用配置的学习率调度器
        scheduler_type = TRAINING_CONFIG.get("scheduler_type", "CosineAnnealingLR")
        min_lr_ratio = TRAINING_CONFIG.get("min_lr_ratio", 0.01)

        if scheduler_type == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epoch,  # 完整的训练周期
                eta_min=self.lr * min_lr_ratio,  # 最小学习率
                last_epoch=-1
            )
        else:
            # 默认使用余弦退火
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epoch,
                eta_min=self.lr * min_lr_ratio,
                last_epoch=-1
            )

        # 标记需要跳过第一次 scheduler.step()
        self._first_epoch = True

    def _reset_optimizer_only(self):
        """
        只重置优化器和学习率调度器，保留网络权重（用于增量训练）

        与 restart_model 的区别：
        - restart_model: 重置网络权重 + 优化器（每次从头学习）
        - _reset_optimizer_only: 只重置优化器（保留已学习的知识，继续学习）
        """
        from POHF_parameters import TRAINING_CONFIG

        fixed_weight_decay = self.weight_decay

        self.optimizer = self.optimizer_fn(
            self.func.parameters(),
            lr=self.lr,
            weight_decay=fixed_weight_decay
        )

        scheduler_type = TRAINING_CONFIG.get("scheduler_type", "CosineAnnealingLR")
        min_lr_ratio = TRAINING_CONFIG.get("min_lr_ratio", 0.01)

        if scheduler_type == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epoch,
                eta_min=self.lr * min_lr_ratio,
                last_epoch=-1
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epoch,
                eta_min=self.lr * min_lr_ratio,
                last_epoch=-1
            )

        self._first_epoch = True

    def save_query_start_weights(self):
        """
        保存当前网络权重作为该 query 的起点
        在每个 query 开始时调用
        """
        self._query_start_weights = copy.deepcopy(self.func.state_dict())

    def restore_to_query_start(self):
        """
        恢复到该 query 开始时的网络权重
        在 query 内部每次迭代前调用（配合累积数据重训练）
        """
        if hasattr(self, '_query_start_weights') and self._query_start_weights is not None:
            self.func.load_state_dict(copy.deepcopy(self._query_start_weights))
        # 重置优化器
        self._reset_optimizer_only()

    def train_model(self, X1, X2, Y, incremental=False, reset_to_query_start=False):
        """
        训练模型

        Args:
            X1, X2, Y: 训练数据
            incremental: 是否使用增量训练模式（已废弃，用 reset_to_query_start 替代）
                - False (默认): 重置网络权重到初始权重，从头训练
                - True: 保留网络权重，只重置优化器，继续训练
            reset_to_query_start: 是否重置到 query 开始时的权重
                - False (默认): 不使用此模式
                - True: 恢复到该 query 开始时保存的权重，然后用累积数据训练

        训练模式优先级：reset_to_query_start > incremental > 默认

        新增训练策略（query内重训练 + query间增量）：
            - Query 开始时调用 save_query_start_weights() 保存权重
            - Query 内部每次迭代使用 reset_to_query_start=True，用累积数据重训练
            - Query 之间自然保留网络权重（增量学习）
        """
        if reset_to_query_start:
            # Query内重训练模式：恢复到该query开始时的权重
            self.restore_to_query_start()
        elif incremental:
            # 增量训练：保留网络权重，只重置优化器
            self._reset_optimizer_only()
        else:
            # 原有行为：重置网络权重到初始权重
            self.restart_model(Y.shape[0])

        self.func.train()
        self.func.to(self.device)

        # 从配置获取训练参数
        from POHF_parameters import TRAINING_CONFIG
        batch_size = TRAINING_CONFIG.get("batch_size", 32)
        gradient_clip_norm = TRAINING_CONFIG.get("gradient_clip_norm", 1.0)

        # 早停配置
        early_stopping = TRAINING_CONFIG.get("early_stopping", False)
        patience = TRAINING_CONFIG.get("early_stopping_patience", 5)
        min_delta = TRAINING_CONFIG.get("early_stopping_min_delta", 1e-4)

        X1_tensor = torch.tensor(X1, dtype=torch.float32)
        X2_tensor = torch.tensor(X2, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        dataset = TensorDataset(X1_tensor, X2_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 早停状态
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, self.epoch + 1):
            epoch_loss = 0.0
            batch_count = 0

            for batch_X1, batch_X2, batch_Y in dataloader:
                batch_X1, batch_X2, batch_Y = batch_X1.to(self.device), batch_X2.to(self.device), batch_Y.to(self.device)
                self.func.zero_grad()
                self.optimizer.zero_grad()

                score_1 = self.func(batch_X1)
                score_2 = self.func(batch_X2)
                loss = F.binary_cross_entropy_with_logits(score_1 - score_2, batch_Y)

                loss.backward()

                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.func.parameters(), max_norm=gradient_clip_norm)

                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            self.scheduler.step()
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0

            # 早停检查
            if early_stopping:
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

    def calculate_greedy_score(self, items):
        """计算greedy分数和基于初始参数的梯度"""
        import copy

        # 1. 保存当前模型参数
        current_state = copy.deepcopy(self.func.state_dict())

        # 2. 加载初始参数计算梯度
        self.func.load_state_dict(self.init_model_weight)
        self.func.eval()

        # 转换输入数据
        if isinstance(items, np.ndarray):
            items = torch.from_numpy(items).to(dtype=torch.float32, device=self.device)
        else:
            items = torch.from_numpy(np.array(items)).to(dtype=torch.float32, device=self.device)

        # 基于初始参数计算输出并求梯度
        outputs_init = self.func(items)
        outputs_init = torch.sigmoid(outputs_init)

        # 计算基于初始参数的梯度
        self.func.zero_grad()
        with backpack(BatchGrad()):
            outputs_init.sum().backward()

        # 提取基于初始参数的梯度
        init_grads_batch = torch.cat(
            [p.grad_batch.flatten(1) for p in self.func.parameters() if hasattr(p, 'grad_batch') and p.grad_batch is not None],
            dim=1
        )

        # 3. 恢复当前参数计算最终分数
        self.func.load_state_dict(current_state)
        self.func.eval()

        # 基于当前参数计算最终输出分数
        with torch.no_grad():  # 不需要梯度，只要分数
            outputs_current = self.func(items)
            outputs_current = torch.sigmoid(outputs_current)

        # print(f"🔄 梯度基于初始参数计算，分数基于当前参数计算")

        # 清理 grad_batch 属性，释放 BackPACK 存储的每样本梯度
        for p in self.func.parameters():
            if hasattr(p, 'grad_batch'):
                del p.grad_batch

        return outputs_current, init_grads_batch

    def calculate_scores_only(self, items):
        """只计算 greedy 分数，不计算梯度（内存优化版本）

        适用于 POHF-InfoGain 等不需要完整梯度的算法
        显存占用：~10 MB（vs calculate_greedy_score 的 ~3 GB）
        """
        # 转换输入数据
        if isinstance(items, np.ndarray):
            items = torch.from_numpy(items).to(dtype=torch.float32, device=self.device)
        else:
            items = torch.from_numpy(np.array(items)).to(dtype=torch.float32, device=self.device)

        # 基于当前参数计算分数（不需要梯度）
        self.func.eval()
        with torch.no_grad():
            outputs = self.func(items)
            outputs = torch.sigmoid(outputs)

        return outputs

    def calculate_gradients_for_arms(self, items, arm_indices):
        """只计算指定 arm 的梯度（内存优化版本）

        Args:
            items: 所有 arm 的 embeddings
            arm_indices: 需要计算梯度的 arm 索引列表

        Returns:
            dict: {arm_index: gradient_tensor}

        显存占用：~10 MB（vs calculate_greedy_score 的 ~3 GB）
        """
        import copy

        # 转换输入数据
        if isinstance(items, np.ndarray):
            items_tensor = torch.from_numpy(items).to(dtype=torch.float32, device=self.device)
        else:
            items_tensor = torch.from_numpy(np.array(items)).to(dtype=torch.float32, device=self.device)

        # 只选择需要的 arm
        selected_items = items_tensor[arm_indices]

        # 保存当前模型参数
        current_state = copy.deepcopy(self.func.state_dict())

        # 加载初始参数计算梯度
        self.func.load_state_dict(self.init_model_weight)
        self.func.eval()

        # 计算梯度
        outputs = self.func(selected_items)
        outputs = torch.sigmoid(outputs)

        self.func.zero_grad()
        with backpack(BatchGrad()):
            outputs.sum().backward()

        # 提取梯度
        grads = torch.cat(
            [p.grad_batch.flatten(1) for p in self.func.parameters()
             if hasattr(p, 'grad_batch') and p.grad_batch is not None],
            dim=1
        )

        # 清理 grad_batch
        for p in self.func.parameters():
            if hasattr(p, 'grad_batch'):
                del p.grad_batch

        # 恢复当前参数
        self.func.load_state_dict(current_state)

        # 构建返回字典
        result = {}
        for i, arm_idx in enumerate(arm_indices):
            result[arm_idx] = grads[i].clone()

        del grads
        torch.cuda.empty_cache()

        return result

    def calculate_ucb_scores_memory_efficient(self, items, greedy_arm_index, current_iteration=0, total_iterations=100, batch_size=50):

        import copy

        # 转换输入数据
        if isinstance(items, np.ndarray):
            items_tensor = torch.from_numpy(items).to(dtype=torch.float32, device=self.device)
        else:
            items_tensor = torch.from_numpy(np.array(items)).to(dtype=torch.float32, device=self.device)

        num_arms = items_tensor.shape[0]

        # 1. 先计算所有 arm 的 greedy 分数（不需要梯度）
        self.func.eval()
        with torch.no_grad():
            greedy_scores = self.func(items_tensor)
            greedy_scores = torch.sigmoid(greedy_scores)

        # 2. 计算 greedy arm 的梯度（基于初始参数）
        current_state = copy.deepcopy(self.func.state_dict())
        self.func.load_state_dict(self.init_model_weight)
        self.func.eval()

        greedy_item = items_tensor[greedy_arm_index:greedy_arm_index+1]
        outputs_greedy = self.func(greedy_item)
        outputs_greedy = torch.sigmoid(outputs_greedy)

        self.func.zero_grad()
        with backpack(BatchGrad()):
            outputs_greedy.sum().backward()

        greedy_grad = torch.cat(
            [p.grad_batch.flatten(1) for p in self.func.parameters()
             if hasattr(p, 'grad_batch') and p.grad_batch is not None],
            dim=1
        ).squeeze(0)  # [num_params]

        # 清理 grad_batch
        for p in self.func.parameters():
            if hasattr(p, 'grad_batch'):
                del p.grad_batch

        # 3. 计算动态 nu 值
        dynamic_nu = self.get_dynamic_nu(current_iteration, total_iterations)

        # 4. 分批计算 UCB 分数
        sigma_squared_list = []

        for i in range(0, num_arms, batch_size):
            end_idx = min(i + batch_size, num_arms)
            batch_items = items_tensor[i:end_idx]

            # 计算当前批次的梯度
            outputs_batch = self.func(batch_items)
            outputs_batch = torch.sigmoid(outputs_batch)

            self.func.zero_grad()
            with backpack(BatchGrad()):
                outputs_batch.sum().backward()

            batch_grads = torch.cat(
                [p.grad_batch.flatten(1) for p in self.func.parameters()
                 if hasattr(p, 'grad_batch') and p.grad_batch is not None],
                dim=1
            )

            # 清理 grad_batch
            for p in self.func.parameters():
                if hasattr(p, 'grad_batch'):
                    del p.grad_batch

            # 计算差异和不确定性
            diff = batch_grads - greedy_grad.unsqueeze(0)

            if self.version == "matrix":
                batch_sigma_squared = torch.sum((diff @ self.Sinv) * diff, dim=1)
            elif self.version == "diag":
                batch_sigma_squared = torch.sum(diff**2 * self.Sinv, dim=1)

            sigma_squared_list.append(batch_sigma_squared)

            # 释放批次梯度
            del batch_grads, diff
            torch.cuda.empty_cache()

        # 恢复当前参数
        self.func.load_state_dict(current_state)

        # 5. 合并结果计算 UCB 分数
        sigma_squared = torch.cat(sigma_squared_list, dim=0)
        sigma = torch.sqrt(sigma_squared + 1e-12)

        import math
        time_factor = math.log(0.1 * current_iteration + 1) + 1
        ucb_scores = greedy_scores + dynamic_nu * sigma * time_factor

        return greedy_scores, ucb_scores

    def get_dynamic_nu(self, current_iteration, total_iterations):
        """计算动态nu值"""
        # 从配置获取动态衰减参数
        from POHF_parameters import POHF_CONFIG

        nu_decay_enabled = POHF_CONFIG.get("nu_decay_enabled", True)
        if not nu_decay_enabled:
            return self.nu

        nu_decay_factor = POHF_CONFIG.get("nu_decay_factor", 0.95)
        nu_min = POHF_CONFIG.get("nu_min", 0.05)
        nu_decay_start = POHF_CONFIG.get("nu_decay_start", 0)
        nu_decay_type = POHF_CONFIG.get("nu_decay_type", "exponential")

        # 如果还没到衰减开始时间
        if current_iteration < nu_decay_start:
            return self.nu

        # 计算衰减后的nu值
        decay_iterations = current_iteration - nu_decay_start

        if nu_decay_type == "exponential":
            # 指数衰减: nu = nu_initial * (decay_factor ^ iterations)
            dynamic_nu = self.nu * (nu_decay_factor ** decay_iterations)
        elif nu_decay_type == "linear":
            # 线性衰减: nu = nu_initial * (1 - progress)
            progress = min(1.0, decay_iterations / (total_iterations - nu_decay_start))
            dynamic_nu = self.nu * (1.0 - progress * (1.0 - nu_min / self.nu))
        elif nu_decay_type == "step":
            # 阶梯衰减: 每10次迭代衰减一次
            step_size = 10
            steps = decay_iterations // step_size
            dynamic_nu = self.nu * (nu_decay_factor ** steps)
        else:
            dynamic_nu = self.nu

        # 确保不低于最小值
        dynamic_nu = max(nu_min, dynamic_nu)

        return dynamic_nu

    def get_covariance_matrix_stats(self, current_iteration=0):
        """获取协方差矩阵的统计信息（返回字典格式）"""
        stats_data = {
            "iteration": current_iteration,
            "version": self.version,
            "total_param": self.total_param,
            "lambda": self.lamb,
            "nu": self.nu
        }

        if self.version == "diag":
            # 对角版本的统计信息
            S_stats = {
                "mean": self.S.mean().item(),
                "std": self.S.std().item(),
                "min": self.S.min().item(),
                "max": self.S.max().item(),
                "median": self.S.median().item(),
            }

            Sinv_stats = {
                "mean": self.Sinv.mean().item(),
                "std": self.Sinv.std().item(),
                "min": self.Sinv.min().item(),
                "max": self.Sinv.max().item(),
                "median": self.Sinv.median().item(),
            }

            stats_data.update({
                "S_matrix_stats": S_stats,
                "Sinv_matrix_stats": Sinv_stats,
                "uncertainty_scale_estimate": Sinv_stats['mean']
            })

        elif self.version == "matrix":
            # 矩阵版本的统计信息
            S_eigenvals = torch.linalg.eigvals(self.S).real
            S_eigenvals = S_eigenvals[S_eigenvals > 1e-10]  # 过滤接近零的特征值

            if len(S_eigenvals) > 0:
                condition_number = S_eigenvals.max() / S_eigenvals.min()
                matrix_stats = {
                    "matrix_shape": list(self.S.shape),
                    "frobenius_norm": torch.norm(self.S, 'fro').item(),
                    "condition_number": condition_number.item(),
                    "max_eigenvalue": S_eigenvals.max().item(),
                    "min_eigenvalue": S_eigenvals.min().item(),
                    "num_eigenvalues": len(S_eigenvals)
                }
            else:
                matrix_stats = {
                    "matrix_shape": list(self.S.shape),
                    "frobenius_norm": torch.norm(self.S, 'fro').item(),
                    "condition_number": float('inf'),
                    "max_eigenvalue": 0.0,
                    "min_eigenvalue": 0.0,
                    "num_eigenvalues": 0,
                    "warning": "Matrix is nearly singular"
                }

            stats_data.update({
                "S_matrix_stats": matrix_stats
            })

        return stats_data

    def print_covariance_matrix_stats(self, current_iteration=0):
        """输出协方差矩阵的统计信息（静默模式）"""
        pass

    def calculate_UCB_score(self, greedy_scores, grads_batch, greedy_arm_index, current_iteration=0, total_iterations=100):
        """计算UCB分数 (内存优化版本，支持动态nu衰减)"""


        # 计算动态nu值
        dynamic_nu = self.get_dynamic_nu(current_iteration, total_iterations)

        greedy_grad = grads_batch[greedy_arm_index]
        greedy_grad = greedy_grad.unsqueeze(0)

        num_arms = grads_batch.shape[0]
        batch_size = 50  # 分批处理，避免内存不足

        sigma_squared_list = []

        # 分批计算以节省内存
        for i in range(0, num_arms, batch_size):
            end_idx = min(i + batch_size, num_arms)

            # 计算当前批次的差异
            batch_grads = grads_batch[i:end_idx]
            diff = batch_grads - greedy_grad

            # 计算不确定性 sigma^2
            if self.version == "matrix":
                batch_sigma_squared = torch.sum((diff @ self.Sinv) * diff, dim=1)
            elif self.version == "diag":
                batch_sigma_squared = torch.sum(diff**2 * self.Sinv, dim=1)

            sigma_squared_list.append(batch_sigma_squared)


        # 合并结果
        sigma_squared = torch.cat(sigma_squared_list, dim=0)
        sigma = torch.sqrt(sigma_squared + 1e-12)


        import math
        time_factor = math.log(0.1 * current_iteration + 1) + 1
        ucb_scores = greedy_scores + dynamic_nu * sigma * time_factor

        return ucb_scores
    def update_matrix(self, gradient_vector_1, gradient_vector_2):
        if self.version == "matrix":
            diff_vector = gradient_vector_2 - gradient_vector_1
            outer_product = torch.outer(diff_vector, diff_vector)
            self.S += outer_product
            self.Sinv = torch.inverse(self.S)


        elif self.version == "diag":
            diff_vector = gradient_vector_2 - gradient_vector_1
            self.S += diff_vector**2
            self.Sinv = 1.0 / self.S.clamp(min=1e-16)



from typing import List
import numpy as np
import pickle
import os
import json

class EmbeddingManager:
    """管理embedding的文件存储和加载，支持domain-prompt映射"""

    def __init__(self, filename="embedding_using.pkl", mapping_filename="domain_prompt_mapping.pkl"):
        self.filename = filename
        self.mapping_filename = mapping_filename
        self.embeddings_dict = {}
        self.domain_to_prompt_mapping = None

    def save_embeddings(self, embeddings_tensor, domain_to_prompt_mapping=None, clear_memory=True):
        """保存embeddings和映射关系到文件并清理内存"""
        # 转换为numpy并保存
        embeddings_np = embeddings_tensor.cpu().numpy()

        # 保存embeddings到文件
        with open(self.filename, 'wb') as f:
            pickle.dump(embeddings_np, f)

        # 保存映射关系到文件
        if domain_to_prompt_mapping is not None:
            with open(self.mapping_filename, 'wb') as f:
                pickle.dump(domain_to_prompt_mapping, f)

        if clear_memory:
            # 清理GPU内存
            del embeddings_tensor
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        return embeddings_np.shape

    def load_mapping(self):
        """加载domain-prompt映射关系"""
        if self.domain_to_prompt_mapping is None:
            try:
                with open(self.mapping_filename, 'rb') as f:
                    self.domain_to_prompt_mapping = pickle.load(f)
            except FileNotFoundError as e:
                print(f"⚠️ [EmbeddingManager.load_mapping] 映射文件不存在: {self.mapping_filename}")
                self.domain_to_prompt_mapping = []
        return self.domain_to_prompt_mapping

    def get_prompt_indices(self, instruction_idx, summary_idx):
        """根据instruction和summary索引获取对应的prompt索引"""
        mapping = self.load_mapping()
        indices = []
        for i, (inst_idx, summ_idx) in enumerate(mapping):
            if inst_idx == instruction_idx and summ_idx == summary_idx:
                indices.append(i)
        return indices

    def load_embedding(self, index):
        """加载指定索引的embedding"""
        if not hasattr(self, '_cached_embeddings'):
            with open(self.filename, 'rb') as f:
                self._cached_embeddings = pickle.load(f)

        return torch.from_numpy(self._cached_embeddings[index]).cuda()

    def load_embeddings_batch(self, indices):
        """批量加载embeddings"""
        if not hasattr(self, '_cached_embeddings'):
            with open(self.filename, 'rb') as f:
                self._cached_embeddings = pickle.load(f)

        batch_embeddings = []
        for idx in indices:
            batch_embeddings.append(self._cached_embeddings[idx])

        return torch.from_numpy(np.array(batch_embeddings)).cuda()

    def get_all_embeddings(self):
        """获取所有embeddings (仅在必要时使用)"""
        with open(self.filename, 'rb') as f:
            embeddings_np = pickle.load(f)
        return torch.from_numpy(embeddings_np).cuda()

    def clear_cache(self):
        """清理内存中的缓存"""
        if hasattr(self, '_cached_embeddings'):
            del self._cached_embeddings
        import gc
        gc.collect()
import asyncio
import aiohttp

##该代码也许需要修改

class EmbeddingClient:

    def __init__(self, api_url: str = "http://127.0.0.1:7777/v1/embeddings"):
        self.api_url = api_url

    def normalize_l2(self, x):
        """L2标准化"""
        x = np.array(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x
            return x / norm
        else:
            norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
            return np.where(norm == 0, x, x / norm)

    async def get_embedding(self, text, max_retries=5, retry_delay=2.0, retry_backoff=2.0):
        """获取单个文本的embedding（异步版本，带重试机制）"""
        payload = {
            "model": EMBEDDING_MODEL,
            "input": [text]
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.api_url, json=payload, timeout=30) as response:
                        if response.status == 200:
                            result = await response.json()
                            # 适配标准OpenAI embedding API格式：data[0]['embedding']
                            return result["data"][0]["embedding"]
                        else:
                            error_msg = f"HTTP {response.status}"
                            if attempt < max_retries - 1:
                                wait_time = retry_delay * (retry_backoff ** attempt)
                                print(f"⚠️ [EmbeddingClient.get_embedding] 请求失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
                                print(f"   等待 {wait_time:.1f}s 后重试...", flush=True)
                                await asyncio.sleep(wait_time)
                            else:
                                print(f"❌ [EmbeddingClient.get_embedding] 已达最大重试次数: {error_msg}")
                                return None
            except Exception as e:
                last_error = e
                error_type_name = type(e).__name__
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (retry_backoff ** attempt)
                    print(f"⚠️ [EmbeddingClient.get_embedding] 编码失败 (尝试 {attempt + 1}/{max_retries}) [{error_type_name}]: {e}")
                    print(f"   等待 {wait_time:.1f}s 后重试...", flush=True)
                    await asyncio.sleep(wait_time)
                else:
                    print(f"❌ [EmbeddingClient.get_embedding] 已达最大重试次数 [{error_type_name}]: {e}")
                    return None

        return None

    async def encode_texts(self, texts: List[str], normalize: bool = True) -> List[np.ndarray]:
        """
        批量编码文本为嵌入向量（异步版本）

        Args:
            texts: 要编码的文本列表
            normalize: 是否进行L2标准化

        Returns:
            嵌入向量列表
        """
        embeddings = []

        for i, text in enumerate(texts):
            # 使用异步API获取embedding
            embedding = await self.get_embedding(text)

            if embedding is not None:
                # API成功，处理embedding
                emb_array = np.array(embedding, dtype=np.float32)
                # 截取前1024维
                emb_array = emb_array[:1024] if len(emb_array) > 1024 else emb_array
                if normalize:
                    emb_array = self.normalize_l2(emb_array)
                embeddings.append(emb_array)
            else:
                raise RuntimeError(f"Failed to get embedding for text {i}: {text[:50]}...")

        return embeddings

    async def encode_single_text(self, text: str, normalize: bool = True) -> np.ndarray:
        result = await self.encode_texts([text], normalize=normalize)
        return result[0]






import random
import torch
import numpy as np

def monitor_gpu_memory(stage_name=""):
    """监控GPU内存使用情况"""
    if torch.cuda.is_available():
        current_gpu = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(current_gpu) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(current_gpu) / 1024**3
        memory_total = torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3
        memory_free = memory_total - memory_allocated

        # 内存警告
        if memory_free < 2.0:
            torch.cuda.empty_cache()

def check_current_gpu_status():
    """检查当前GPU使用状态（静默版本）"""
    pass

def verify_embedding_prompt_correspondence(domain_texts, init_instructions, domain_to_prompt_mapping, input_data):
    # 检查数量一致性
    if len(domain_texts) == len(init_instructions) == len(domain_to_prompt_mapping):
        pass
    else:
        return False

    # 随机检查几个对应关系
    import random
    check_indices = random.sample(range(len(domain_texts)), min(3, len(domain_texts)))

    correct_count = 0
    for idx in check_indices:
        inst_idx, summ_idx = domain_to_prompt_mapping[idx]
        domain_text = domain_texts[idx]
        full_prompt = init_instructions[idx]

        # 🔧 适配新逻辑的验证：
        expected_instruction = input_data[2]  # 现在是单个instruction字符串
        expected_summary = input_data[3][summ_idx]      # 对应的summary

        # 检查full prompt是否包含正确的instruction和summary
        instruction_match = expected_instruction in full_prompt
        summary_match = expected_summary in full_prompt

        # 🔧 新逻辑：domain text只应该包含summary，不包含instruction
        domain_summary_match = expected_summary in domain_text
        domain_instruction_absent = expected_instruction not in domain_text

        # 🔧 新的验证条件
        if instruction_match and summary_match and domain_summary_match and domain_instruction_absent:
            correct_count += 1

    return correct_count == len(check_indices)


# ========== 简化日志输出工具 ==========
class ProgressLogger:
    """
    进度日志记录器，用于并行和串行模式
    """
    def __init__(self, counter, run_index, total_counters, algorithm_name="POHF-InfoGain", verbose=True):
        self.counter = counter
        self.run_index = run_index
        self.total_counters = total_counters
        self.algorithm_name = algorithm_name
        self.verbose = verbose
        self._last_progress = -1
        self._progress_interval = 10  # 每10%输出一次进度

    def _get_prefix(self):
        """获取日志前缀"""
        return f"[Counter {self.run_index + 1}/{self.total_counters}][{self.algorithm_name}]"

    def log_progress(self, current_iter, total_iter, extra_info=""):
        """输出进度信息"""
        if total_iter == 0:
            return
        progress = int(current_iter / total_iter * 100)
        # 每 10% 输出一次，或者在最后一次迭代时输出
        if progress >= self._last_progress + self._progress_interval or current_iter == total_iter:
            self._last_progress = progress
            extra = f" | {extra_info}" if extra_info else ""
            print(f"{self._get_prefix()} 迭代 {current_iter}/{total_iter} ({progress}%){extra}", flush=True)

    def log_start(self):
        """输出开始信息"""
        print(f"{self._get_prefix()} 🚀 开始运行 counter={self.counter}", flush=True)

    def log_complete(self, best_score=None, greedy_arm=None):
        """输出完成信息"""
        info = f"best_score={best_score:.4f}, greedy_arm={greedy_arm}" if best_score is not None else ""
        print(f"{self._get_prefix()} ✅ 完成! {info}", flush=True)

    def log_error(self, error_msg):
        """输出错误信息"""
        print(f"{self._get_prefix()} ❌ 错误: {error_msg}", flush=True)

    def log_verbose(self, msg):
        """仅在 verbose 模式下输出详细信息"""
        if self.verbose:
            print(f"{self._get_prefix()} {msg}", flush=True)


def run_single_counter_process(args):
    """
    在独立进程中运行单个 counter 的包装函数

    这个函数在子进程中执行，每个进程独立加载模块和 GPU 资源。

    Args:
        args: 元组 (counter, run_index, total_counters, config_dict)

    Returns:
        dict: 包含运行状态和结果
    """
    counter, run_index, total_counters, config_dict = args

    import os
    import sys
    import gc
    import warnings

    # 在子进程中抑制警告
    warnings.filterwarnings("ignore", message="Extension saving to grad_batch")
    warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")

    # 🔧 简化日志：使用简洁的进程标识
    log_prefix = f"[Counter {run_index + 1}/{total_counters}]"

    try:
        import torch
        import asyncio

        # 🔧 重置设备缓存，确保子进程使用正确的 CUDA_VISIBLE_DEVICES
        reset_device()

        # 设置 GPU（静默模式）- 根据 CUDA_VISIBLE_DEVICES 使用 cuda:0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()

        # 运行单个 counter
        result = asyncio.run(run_single_counter_async(
            counter=counter,
            run_index=run_index,
            total_counters=total_counters,
            config=config_dict
        ))

        # 清理 GPU 显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            'status': 'success',
            'counter': counter,
            'all_results': result.get('all_results', []),
            'greedy_arm_results': result.get('greedy_arm_results', [])
        }

    except Exception as e:
        import traceback
        print(f"❌ [run_single_counter_process] Counter {counter} 运行失败: {e}")
        print(f"   Traceback: {traceback.format_exc()}")

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as cleanup_e:
            print(f"⚠️ [run_single_counter_process] GPU清理失败: {cleanup_e}")
        gc.collect()

        return {
            'status': 'failed',
            'counter': counter,
            'error': str(e),
            'all_results': [],
            'greedy_arm_results': []
        }


async def run_single_counter_async(counter, run_index, total_counters, config):
    """
    异步运行单个 counter - 调用主 run() 函数但只处理一个 counter
    """
    # 创建只包含单个 counter 的配置副本
    import copy
    single_config = copy.deepcopy(config)

    # 设置单 counter 模式
    single_config['_single_counter_mode'] = True
    single_config['_target_counter'] = counter
    single_config['_run_index'] = run_index
    single_config['_total_counters'] = total_counters

    # 调用主 run 函数
    results = await run(config=single_config)

    return {
        'all_results': results if isinstance(results, list) else [results],
        'greedy_arm_results': []  # 将在 run() 中处理
    }


async def run(config=None):
    """
    运行POHF算法，使用配置文件参数
    """
    # 🔧 在运行开始时重置并初始化 GPU 设备
    # 确保使用当前进程的 CUDA_VISIBLE_DEVICES 设置
    reset_device()
    if torch.cuda.is_available():
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
        torch.cuda.set_device(0)  # 使用 CUDA_VISIBLE_DEVICES 映射后的 cuda:0
        torch.cuda.empty_cache()

    if config is None:
        from POHF_parameters import get_all_configs, generate_counter_array
        config = get_all_configs()
    else:
        from POHF_parameters import generate_counter_array

    experiment_config = config.get("experiment", {})
    data_config = config.get("data", {})
    path_config = config.get("path", {})
    api_config = config.get("api", {})
    device_config = config.get("device", {})
    rouge_config = config.get("rouge", {})

    LLM_as_judge = rouge_config.get("LLM_as_judge", False)


    # 🔧 [清理] 移除未使用的 n_init, total_iter, max_iter（现在统一使用 unified_training_rounds）
    random_seed = experiment_config.get("random_seed", 0)


    input_address = path_config.get("input_address")
    output_address = path_config.get("output_address")
    LaMP_type = path_config.get("LaMP_type", 4)

    if 'POHF_LAMP_TYPE' in os.environ:
        lamp_type_override = int(os.environ['POHF_LAMP_TYPE'])
        input_address_override = os.environ.get('POHF_INPUT_ADDRESS')
        output_address_override = os.environ.get('POHF_OUTPUT_ADDRESS')

        # 🔧 支持从环境变量覆盖 counter_array_length
        counter_array_length_override = os.environ.get('POHF_COUNTER_ARRAY_LENGTH')

        LaMP_type = lamp_type_override
        input_address = input_address_override
        if output_address_override:
            output_address = output_address_override

        # 更新 data_config 中的 counter_array_length
        if counter_array_length_override:
            data_config['counter_array_length'] = int(counter_array_length_override)

        # 更新 path_config 中的 LaMP_type（用于 generate_counter_array）
        path_config['LaMP_type'] = LaMP_type

        # 更新 config 字典以确保 generate_counter_array 使用新配置
        config['data'] = data_config
        config['path'] = path_config

        device_config['device'] = 'cuda:0'


    max_len = data_config.get("max_len", 100)
    times = data_config.get("times", 50)


    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    counter_array = generate_counter_array(config, input_address)

    # 检查是否为单 counter 模式
    single_counter_mode = config.get('_single_counter_mode', False)
    if single_counter_mode:
        target_counter = config.get('_target_counter')
        counter_array = [target_counter]

    # 获取并行配置
    parallel_config = config.get('parallel', {})
    parallel_enabled = parallel_config.get('parallel_enabled', True)
    parallel_counters = parallel_config.get('parallel_counters', 3)
    timeout_per_counter = parallel_config.get('timeout_per_counter', 36000)

    # 🔧 支持从环境变量覆盖 parallel_counters
    parallel_counters_override = os.environ.get('POHF_PARALLEL_COUNTERS')
    if parallel_counters_override:
        parallel_counters = int(parallel_counters_override)

    if single_counter_mode:
        parallel_enabled = False

    # ========== 并行运行模式 ==========
    if parallel_enabled and len(counter_array) > 1 and parallel_counters > 1:
        from tqdm import tqdm

        task_args = [
            (counter, idx, len(counter_array), config)
            for idx, counter in enumerate(counter_array)
        ]

        all_results = []
        all_greedy_arm_results = []
        failed_counters = []
        max_workers = min(parallel_counters, len(counter_array))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_counter = {
                executor.submit(run_single_counter_process, args): args[0]
                for args in task_args
            }

            # 使用 tqdm 进度条
            with tqdm(total=len(counter_array), desc="🚀 并行运行", unit="counter",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                for future in as_completed(future_to_counter):
                    counter = future_to_counter[future]
                    try:
                        result = future.result(timeout=timeout_per_counter)

                        if result['status'] == 'success':
                            all_results.extend(result.get('all_results', []))
                            all_greedy_arm_results.extend(result.get('greedy_arm_results', []))
                            pbar.set_postfix_str(f"✅ Counter {counter}")
                        else:
                            failed_counters.append({
                                'counter': counter,
                                'error': result.get('error', 'Unknown error')
                            })
                            pbar.set_postfix_str(f"❌ Counter {counter}")

                    except Exception as e:
                        print(f"❌ [并行运行] Counter {counter} 异常: {e}")
                        failed_counters.append({
                            'counter': counter,
                            'error': str(e)
                        })
                        pbar.set_postfix_str(f"❌ Counter {counter}")

                    pbar.update(1)

        # 输出最终统计
        print(f"\n✅ 完成: {len(all_results)}/{len(counter_array)} counters 成功, {len(failed_counters)} 失败")
        if failed_counters:
            print(f"❌ 失败的 counters: {[fc['counter'] for fc in failed_counters]}")
            for fc in failed_counters:
                print(f"   Counter {fc['counter']}: {fc['error']}")

        # 按 counter 顺序排序结果
        all_results.sort(key=lambda x: x.get('counter', 0))
        all_greedy_arm_results.sort(key=lambda x: x.get('counter', 0))

        # 生成平均图表
        if not LLM_as_judge and all_results and len(all_results) > 1:
            # 🔧 [简化] 从配置获取统一训练轮次
            try:
                from POHF_parameters import CONTEXTUAL_BANDIT_CONFIG
                unified_training_rounds = CONTEXTUAL_BANDIT_CONFIG.get("unified_training_rounds", 10)
            except ImportError:
                unified_training_rounds = 10

            # 🔧 [修复] 收集所有query的完整数据用于JSON导出
            full_results_for_export = []
            for result in all_results:
                full_data = result.get('best_instruction_over_iter', [])
                baseline_data = result.get('baseline_results', {})

                # 🔧 [修复] 传递完整数据，包含所有query的信息
                full_result = {
                    'algorithm': result.get('algorithm', 'POHF-InfoGain'),
                    'counter': result.get('counter', 0),
                    'best_instruction_over_iter': full_data,  # 所有query的数据
                    'baseline_results': baseline_data,  # 所有query的baseline数据
                    'total_arms': result.get('total_arms', 0),
                    'contextual_mode': result.get('contextual_mode', False),
                    'num_queries': result.get('num_queries', 1),
                    'total_iterations': len(full_data)
                }
                full_results_for_export.append(full_result)

            if full_results_for_export:
                plot_counter_average_results(full_results_for_export, get_dataset_name(LaMP_type))
                print(f"   📊 [图1 Counter Average] 已生成Counter平均图（包含所有Query数据）")

            # 🔧 [图2] 生成跨Query进步的Counter Average
            all_query_progress_data = []
            for result in all_results:
                full_data = result.get('best_instruction_over_iter', [])
                baseline_data = result.get('baseline_results', {})
                counter = result.get('counter', 0)

                # 🔧 [简化] 统一训练轮次
                q_iters = unified_training_rounds
                num_queries = len(full_data) // q_iters if q_iters > 0 else 1

                # 🔧 [修改] 包含第一个query（query_0）的数据，因为现在所有query使用统一训练配置
                if num_queries >= 1:
                    query_final_values = {}
                    for q_idx in range(0, num_queries):  # 从0开始，包含第一个query
                        start = q_idx * q_iters
                        end = start + q_iters
                        if start >= len(full_data):
                            continue

                        main_vals = [item[2] if isinstance(item, tuple) else item for item in full_data[start:end]]
                        all_vals = list(main_vals)
                        bl_vals_map = {}
                        for bl, bd in baseline_data.items():
                            if bd.get('values'):
                                bv = bd['values'][start:end]
                                bl_vals_map[bl] = bv
                                all_vals.extend(bv)

                        if all_vals:
                            vmin, vmax = min(all_vals), max(all_vals)
                            vrange = vmax - vmin if vmax > vmin else 1.0
                            norm = lambda v, vmin=vmin, vrange=vrange: (v - vmin) / vrange

                            if main_vals:
                                query_final_values.setdefault(result.get('algorithm', 'POHF-InfoGain'), {})[q_idx] = {'final_value_normalized': norm(main_vals[-1]), 'final_value_raw': main_vals[-1]}

                            for bl, bv in bl_vals_map.items():
                                if bv:
                                    entry = {'final_value_normalized': norm(bv[-1]), 'final_value_raw': bv[-1]}
                                    if bl == 'Random':
                                        entry['avg_value_normalized'] = np.mean([norm(v) for v in bv])
                                    query_final_values.setdefault(bl, {})[q_idx] = entry

                    if query_final_values:
                        q_indices = sorted(set(q for alg in query_final_values.values() for q in alg.keys()))
                        if q_indices:
                            rand_avg = {q: query_final_values.get('Random', {}).get(q, {}).get('avg_value_normalized', 0.0) for q in q_indices}
                            alg_progress = {alg: [data[q]['final_value_normalized'] - rand_avg.get(q, 0.0) for q in q_indices if q in data] for alg, data in query_final_values.items()}
                            alg_progress = {k: v for k, v in alg_progress.items() if v}

                            q_prog_data = {'counter': counter, 'query_indices': q_indices, 'algorithms': alg_progress}
                            all_query_progress_data.append(q_prog_data)

            if all_query_progress_data:
                plot_query_progress_counter_average(all_query_progress_data, get_dataset_name(LaMP_type))
                print(f"   📊 [图2 Counter Average] 已生成跨Query进步的Counter平均图")

        return all_results

    # ========== 串行运行模式 ==========

    api_url = api_config.get("embedding_api_url", "http://127.0.0.1:7777/v1/embeddings")
    embedding_client = EmbeddingClient(api_url=api_url)

    # ========== Contextual Embedding 计算辅助函数 ==========
    async def compute_contextual_embeddings(query_text, summary_texts, embedding_client, is_lamp, tkwargs):
        """
        计算contextual embedding: [query_emb | summary_emb]

        Args:
            query_text: 当前query文本
            summary_texts: summary文本列表
            embedding_client: embedding客户端
            is_lamp: 是否是LaMP数据集
            tkwargs: tensor参数

        Returns:
            embeddings: [num_summaries, dim] 的tensor
                - LaMP数据集: dim=2048 (query_emb 1024 + summary_emb 1024)
                - 其他数据集: dim=1024 (仅summary_emb)
        """
        if is_lamp:
            # LaMP数据集: [query_emb | summary_emb] = 2048维
            # 1. 计算query的embedding
            query_emb_list = await embedding_client.encode_texts([query_text], normalize=True)
            query_emb = torch.from_numpy(query_emb_list[0])  # [1024]

            # 2. 计算每个summary的embedding
            summary_emb_list = await embedding_client.encode_texts(summary_texts, normalize=True)
            summary_embs = torch.stack([torch.from_numpy(emb) for emb in summary_emb_list])  # [num_summaries, 1024]

            # 3. Concat: [query_emb | summary_emb]
            query_emb_expanded = query_emb.unsqueeze(0).expand(summary_embs.shape[0], -1)  # [num_summaries, 1024]
            embeddings = torch.cat([query_emb_expanded, summary_embs], dim=1)  # [num_summaries, 2048]
            embeddings = embeddings.to(**tkwargs)

            return embeddings, query_emb
        else:
            # 非LaMP数据集: 仅summary embedding = 1024维
            summary_emb_list = await embedding_client.encode_texts(summary_texts, normalize=True)
            embeddings = torch.stack([torch.from_numpy(emb) for emb in summary_emb_list])
            embeddings = embeddings.to(**tkwargs)

            return embeddings, None

    max_history_items = config.get('data', {}).get('max_history_items', 20)

    # ========== Contextual Dueling Bandit 模式 ==========
    # 🔧 [统一] 所有数据集都使用contextual模式（query concat persona = 2048维）
    contextual_mode_enabled = True  # 统一使用contextual模式

    # 从配置获取contextual bandit参数
    try:
        from POHF_parameters import CONTEXTUAL_BANDIT_CONFIG
        # 🔧 [简化] 统一配置：无random sample阶段，所有query统一训练轮次
        unified_training_rounds = CONTEXTUAL_BANDIT_CONFIG.get("unified_training_rounds", 10)
        contextual_input_dim = CONTEXTUAL_BANDIT_CONFIG.get("contextual_input_dim", 2048)
    except ImportError:
        unified_training_rounds = 10
        contextual_input_dim = 2048

    dataset_name = get_dataset_name(LaMP_type)
    print(f"🔄 [Contextual Mode] {dataset_name}数据集启用Contextual Dueling Bandit模式")
    print(f"   输入维度: {contextual_input_dim} (query + persona concat)")
    print(f"   每个query: {unified_training_rounds}轮偏好反馈（无random sample阶段）")

    all_results = []

    all_greedy_arm_results = []

    # 🔧 新增：用于收集所有counter的第一个query结果（用于图1的counter_average）
    all_first_query_results = []

    # 🔧 新增：用于收集所有counter的跨Query进步数据（用于图2的counter_average）
    all_query_progress_data = []

    # Persona保存目录
    persona_output_dir = os.path.join(PROJECT_ROOT, "persona_results")
    os.makedirs(persona_output_dir, exist_ok=True)

    algorithms_to_run = [
        {"name": "POHF-InfoGain", "info_gain_enabled": True}
    ]

    # 循环运行每个counter值
    for run_index, counter in enumerate(counter_array):

        counter_seed = random_seed
        random.seed(counter_seed)
        np.random.seed(counter_seed)
        torch.manual_seed(counter_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(counter_seed)
            torch.cuda.manual_seed_all(counter_seed)

        input_data, ground_truth = load_templated_data(input_address, output_address, LaMP_type, max_len, None, times, counter, max_history_items)
        original_summary_for_counter = input_data[3][0] if len(input_data) > 3 and len(input_data[3]) > 0 else ""

        # ========== 确定query数量和训练轮次 ==========
        query_data = input_data[1]
        ground_truth_data = ground_truth[1]

        # 🔧 [统一] 判断num_queries的逻辑
        # LaMP数据集(4,5,8,9,10): 支持多个query
        # 非LaMP数据集(ultrachat=0, wildchat=-1, prefevel=-2): 只使用1个query
        if is_lamp_dataset(LaMP_type) and isinstance(query_data, list):
            num_queries = len(query_data)
            print(f"  🔄 [Contextual Mode] LaMP数据集检测到 {num_queries} 组query/output对")
        else:
            num_queries = 1  # 非LaMP数据集只使用一个query
            print(f"  🔄 [Contextual Mode] 非LaMP数据集使用单query模式")

        # 存储当前counter的所有算法结果
        counter_results = []

        # 存储每个query的最终值（用于图2 - 跨Query进步图）
        query_final_values = {}

        # 🔧 [问题1修复] 初始化用于收集所有算法所有query的persona结果
        counter_persona_data = {
            "counter": counter,
            "lamp_type": LaMP_type,
            "contextual_mode": contextual_mode_enabled,
            "num_queries": 0,  # 将在确定num_queries后更新
            "queries": {}  # 每个query_idx对应的所有算法结果
        }

        # 预先计算所有summary的embedding（summary不变，只需计算一次）
        domain_texts_base = []
        for j in range(times):
            domain_text = f"### Summary:\n{input_data[3][j]}"
            domain_texts_base.append(domain_text)

        # 打乱顺序的索引（在所有query迭代中保持一致）
        indices = list(range(times))
        random.shuffle(indices)
        domain_texts = [domain_texts_base[i] for i in indices]

        # 预先计算summary embeddings（只需计算一次，因为summary不变）
        summary_emb_list = await embedding_client.encode_texts(domain_texts, normalize=True)
        summary_embs_base = torch.stack([torch.from_numpy(emb) for emb in summary_emb_list])  # [times, 1024]

        embedding_dir = os.path.join(PROJECT_ROOT, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)

        # 🔧 [问题1修复] 更新num_queries到counter_persona_data
        counter_persona_data["num_queries"] = num_queries

        # 循环运行每种算法
        for algorithm_config in algorithms_to_run:
            algorithm_name = algorithm_config["name"]
            info_gain_enabled = algorithm_config["info_gain_enabled"]

            # 🔧 创建进度记录器
            progress_logger = ProgressLogger(counter, run_index, len(counter_array), algorithm_name)

            # 为当前算法重置变量（跨query迭代累积）
            num_group = []
            x_train = []
            y_train = []
            select_idx_history = []
            instruction_select_history = []
            second_arm_selections = []
            llm_config = config.get("llm", {})

            # 用于存储跨query迭代的模型和info_manager
            l = None  # NeuralDB模型

            # ========== Query迭代循环 (Contextual Dueling Bandit核心) ==========
            for query_idx in range(num_queries):
                # 🔧 [统一] 所有数据集：无random sample阶段，统一训练轮次
                current_n_init = 0  # 无random sample阶段
                current_max_iter = unified_training_rounds  # 统一使用配置的训练轮次

                # 获取当前query和ground_truth
                if isinstance(query_data, list) and query_idx < len(query_data):
                    current_query = str(query_data[query_idx])
                else:
                    current_query = str(query_data) if not isinstance(query_data, list) else str(query_data[0])

                # 当前query对应的ground_truth索引
                current_gt_index = query_idx if isinstance(ground_truth_data, list) and query_idx < len(ground_truth_data) else 0

                print(f"\n  📍 [Query {query_idx+1}/{num_queries}] 训练配置: {current_max_iter}轮偏好反馈")

                # 生成当前query的init_instructions
                init_instructions = []
                domain_to_prompt_mapping = []
                for j in range(times):
                    original_j = indices[j]  # 使用打乱后的索引映射回原始summary索引
                    full_prompt = prompt_reformer(input_data, 0, original_j, lamp_type=LaMP_type, query_index=query_idx)
                    init_instructions.append(full_prompt)
                    domain_to_prompt_mapping.append((query_idx, original_j))

                # ========== 计算当前query的Contextual Embedding ==========
                # 🔧 [统一] 所有数据集都使用: [query_emb | persona_emb] = 2048维
                query_emb_list = await embedding_client.encode_texts([current_query], normalize=True)
                query_emb = torch.from_numpy(query_emb_list[0])  # [1024]

                # Concat: [query_emb | persona_emb]
                query_emb_expanded = query_emb.unsqueeze(0).expand(summary_embs_base.shape[0], -1)  # [times, 1024]
                sen_embeddings = torch.cat([query_emb_expanded, summary_embs_base], dim=1)  # [times, 2048]
                sen_embeddings = sen_embeddings.to(**tkwargs)

                if query_idx == 0:
                    print(f"  📐 Contextual embedding: query({query_emb.shape}) + persona({summary_embs_base.shape[1]}) = {sen_embeddings.shape}")

                # 保存当前query的embedding
                embedding_filename = os.path.join(embedding_dir, f"embedding_using_{get_dataset_name(LaMP_type)}_counter{counter}_query{query_idx}.pkl")
                mapping_filename = os.path.join(embedding_dir, f"domain_prompt_mapping_{get_dataset_name(LaMP_type)}_counter{counter}_query{query_idx}.pkl")

                embedding_manager = EmbeddingManager(embedding_filename, mapping_filename)
                embedding_manager.save_embeddings(
                    sen_embeddings,
                    domain_to_prompt_mapping=domain_to_prompt_mapping,
                    clear_memory=False  # 不清理，后续还需要使用
                )

                # 🔧 Query开始时保存网络权重（用于Query内重训练 + Query间增量）
                from POHF_parameters import TRAINING_CONFIG
                use_query_level_incremental = TRAINING_CONFIG.get("query_level_incremental", True)
                if use_query_level_incremental and hasattr(l, 'save_query_start_weights'):
                    l.save_query_start_weights()
                    if query_idx == 0:
                        print(f"  🔄 启用Query级增量训练：Query内累积重训练，Query间增量")

                # ========== 随机采样阶段（每个query都有） ==========
                # 记录当前query random_sample开始前的数据位置，用于后续只更新当前query的数据
                query_random_sample_start_idx = len(select_idx_history)

                if current_n_init > 0:
                    init_pairs = []
                    for i in range(current_n_init):
                        upper_bound = times
                        num1, num2 = random.sample(range(upper_bound), 2)
                        while (num1, num2) in num_group:
                            num1, num2 = random.sample(range(upper_bound), 2)
                        num_group.append((num1, num2))
                        init_pairs.append((num1, num2))

                    # 并行生成所有初始响应
                    async def process_init_pair(pair_idx, num1, num2, gt_idx):
                        """并行处理单个初始化对"""
                        personality1 = domain_texts[num1] if num1 < len(domain_texts) else None
                        personality2 = domain_texts[num2] if num2 < len(domain_texts) else None
                        response1, response2, score_1, score_2, _, new_y, p_ = await generate_and_compare_async(
                            init_instructions[num1], init_instructions[num2],
                            ground_truth[1], llm_config,
                            LLM_as_judge=LLM_as_judge, arm1_idx=num1, arm2_idx=num2,
                            personality1=personality1, personality2=personality2,
                            ground_truth_index=gt_idx
                        )
                        return {
                            'pair_idx': pair_idx,
                            'num1': num1, 'num2': num2,
                            'response1': response1, 'response2': response2,
                            'score_1': score_1, 'score_2': score_2,
                            'new_y': new_y
                        }

                    print(f"  📋 随机采样阶段: 并行生成 {len(init_pairs)} 对响应...", flush=True)
                    init_tasks = [process_init_pair(i, num1, num2, current_gt_index) for i, (num1, num2) in enumerate(init_pairs)]
                    init_results = await asyncio.gather(*init_tasks)

                    for result in sorted(init_results, key=lambda x: x['pair_idx']):
                        num1, num2 = result['num1'], result['num2']
                        score_1, score_2, new_y = result['score_1'], result['score_2'], result['new_y']

                        winner = "arm1" if new_y == 1 else "arm2"
                        print(f"    [Init {result['pair_idx']+1}/{len(init_pairs)}] arm1={num1}(score={score_1:.4f}) vs arm2={num2}(score={score_2:.4f}) → {winner}", flush=True)

                        emb1 = embedding_manager.load_embedding(num1)
                        emb2 = embedding_manager.load_embedding(num2)
                        x_train += [torch.cat([emb1.reshape(1,1,-1), emb2.reshape(1,1,-1)])]
                        y_train += [new_y]
                        select_idx_history += [[num1, num2]]
                        instruction_select_history += [(init_instructions[num1], score_1, init_instructions[num2], score_2)]

                # ========== 训练/更新模型 ==========
                x_train_new = x_train
                y_train_new = y_train

                if len(x_train) > 0:
                    x_train_tensor = torch.cat(x_train, dim=1)
                else:
                    # 后续query没有随机采样，使用之前的数据
                    x_train_tensor = torch.zeros(1, 0, sen_embeddings.shape[-1])

                random.seed(random_seed)
                np.random.seed(random_seed)
                torch.manual_seed(random_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(random_seed)
                    torch.cuda.manual_seed_all(random_seed)

                # 使用POHF代码中定义的NeuralDB类，传入配置参数
                if l is None or query_idx == 0:
                    # 第一个query或者模型未初始化：创建新模型
                    l = NeuralDB(input_dim=sen_embeddings.shape[-1], config=config)
                # 注意：后续query复用之前的模型，继续训练

                # 准备训练数据 - NeuralDB需要X1, X2, Y格式
                X1_train = []
                X2_train = []
                Y_train = []

                for i, (x_pair, y) in enumerate(zip(x_train_new, y_train)):
                    # x_pair shape: [2, 1, embedding_dim] -> [2, embedding_dim]
                    x_pair_reshaped = x_pair.squeeze(1)
                    X1_train.append(x_pair_reshaped[0].cpu().numpy())
                    X2_train.append(x_pair_reshaped[1].cpu().numpy())
                    Y_train.append(y)

                X1_train = np.array(X1_train) if X1_train else np.array([]).reshape(0, sen_embeddings.shape[-1])
                X2_train = np.array(X2_train) if X2_train else np.array([]).reshape(0, sen_embeddings.shape[-1])
                Y_train = np.array(Y_train) if Y_train else np.array([])

                if len(X1_train) > 0:
                    # 🔧 训练模式选择
                    from POHF_parameters import TRAINING_CONFIG
                    use_incremental = TRAINING_CONFIG.get("incremental_training", False)
                    use_query_level = TRAINING_CONFIG.get("query_level_incremental", True)

                    if use_query_level and use_incremental:
                        # 🔧 Query级增量训练：使用累积数据 + reset_to_query_start
                        # Query内部：用该Query内所有累积数据重训练（从Query开始时的权重）
                        # Query之间：保留网络权重（自然增量）
                        l.train_model(X1_train, X2_train, Y_train, reset_to_query_start=True)
                        l._has_trained = True
                    else:
                        # 旧模式：逐个pair增量训练
                        for i in range(len(X1_train)):
                            X1_single = X1_train[i:i+1]
                            X2_single = X2_train[i:i+1]
                            Y_single = Y_train[i:i+1]
                            is_first_training = not hasattr(l, '_has_trained') or not l._has_trained
                            l.train_model(X1_single, X2_single, Y_single, incremental=use_incremental and not is_first_training)
                            l._has_trained = True

                # 从文件加载所有embeddings用于计算梯度
                with open(embedding_filename, 'rb') as f:
                    embeddings_np = pickle.load(f)

                # 🔧 内存优化：只计算需要的 arm 的梯度
                unique_arms = set()
                for arm_pair in select_idx_history:
                    unique_arms.add(arm_pair[0])
                    unique_arms.add(arm_pair[1])
                unique_arms_list = list(unique_arms)

                # 计算 greedy_scores 用于 info_manager 初始化
                greedy_scores = l.calculate_scores_only(embeddings_np)

                # 只计算需要的 arm 的梯度（如果有随机采样阶段的数据）
                if unique_arms_list:
                    arm_grads_init = l.calculate_gradients_for_arms(embeddings_np, unique_arms_list)

                    # 使用random selection阶段收集的arm pairs更新矩阵
                    matrix_updates = 0
                    for i, (arm_pair, y_value) in enumerate(zip(select_idx_history, y_train)):
                        arm1_idx, arm2_idx = arm_pair
                        if arm1_idx in arm_grads_init and arm2_idx in arm_grads_init:
                            grad1 = arm_grads_init[arm1_idx]
                            grad2 = arm_grads_init[arm2_idx]
                            l.update_matrix(grad1, grad2)
                            matrix_updates += 1

                    del arm_grads_init
                    torch.cuda.empty_cache()

                # 清理 embeddings_np
                del embeddings_np
                torch.cuda.empty_cache()

                # 初始化 Information Manager（仅在第一个query或需要时）
                if algorithm_name == "POHF-InfoGain":
                    from information_second_term import ContextualPairwiseInformationManager, PairwiseInformationManager
                    bayesian_alpha = config.get("pohf", {}).get("bayesian_alpha", 1.0)

                    # 🔧 [统一] 所有数据集都使用ContextualPairwiseInformationManager
                    if not hasattr(l, 'contextual_info_manager') or l.contextual_info_manager is None:
                        l.contextual_info_manager = ContextualPairwiseInformationManager(
                            len(greedy_scores), use_optimized=True, bayesian_alpha=bayesian_alpha
                        )
                    # 为当前query初始化或更新概率矩阵
                    # 使用当前query的embedding作为input特征
                    current_input_embedding = query_emb.cpu().numpy()
                    l.info_manager = l.contextual_info_manager.initialize_for_new_input(
                        input_index=query_idx, input_embedding=current_input_embedding
                    )
                    print(f"   🔄 [Contextual] 使用ContextualPairwiseInformationManager (query_index={query_idx})")

                    # 使用当前query的随机采样阶段的feedback更新info_manager
                    # 只使用当前query新增的数据（从query_random_sample_start_idx开始）
                    current_query_pairs = select_idx_history[query_random_sample_start_idx:]
                    current_query_labels = y_train[query_random_sample_start_idx:]

                    if len(current_query_pairs) > 0:
                        print(f"   📊 更新info_manager: 使用当前query的 {len(current_query_pairs)} 对random_sample数据")
                        from information_second_term import update_information_with_feedback
                        for arm_pair, y_value in zip(current_query_pairs, current_query_labels):
                            arm1_idx, arm2_idx = arm_pair
                            arm1_wins = bool(y_value)
                            info_gain = update_information_with_feedback(l.info_manager, arm1_idx, arm2_idx, arm1_wins)
                            # 🔧 记录增量比较结果（用于历史迁移）
                            if hasattr(l, 'contextual_info_manager') and l.contextual_info_manager is not None:
                                l.contextual_info_manager.record_comparison(arm1_idx, arm2_idx, arm1_wins)

                # 🔧 输出当前query的迭代进度
                progress_logger.log_progress(0, current_max_iter, f"(Query {query_idx+1}/{num_queries} 开始迭代)")

                # 🔧 [修复] 每个query独立的best_score：在每个query开始时重置best_r
                # best_score不能跨query累积，因为不同query有不同的ground truth
                best_r = -np.inf
                best_index = -1

                # 初始化跟踪变量（在query循环内）
                if query_idx == 0:
                    best_values = []
                    now_values = []
                    best_instruction_over_iter = []
                    # 🔧 [修复] 用于记录每个query每个算法的best_score
                    per_query_best_scores = {}  # {query_idx: {"POHF-InfoGain": score, "POHF": score, ...}}

                    # 初始化baseline算法（仅第一个query）
                    from LLM_regression import create_algorithm

                    baseline_algorithms = {}
                    baseline_results = {}

                    try:
                        from POHF_parameters import BASELINE_CONFIG
                        enabled_baselines = BASELINE_CONFIG.get("enabled_baselines", [])

                        input_dim = sen_embeddings.shape[-1]
                        algorithm_creators = {
                            'POHF': lambda: NeuralDB(input_dim=input_dim, config=config),
                            'Random': lambda: create_algorithm('Random', input_dim=input_dim, config=config),
                            'POHF-Random': lambda: create_algorithm('POHFRandom', input_dim=input_dim, config=config),
                            'POHF-RandomPair': lambda: create_algorithm('POHFRandomPair', input_dim=input_dim, config=config),
                            'DoubleTS': lambda: create_algorithm('DoubleTS', input_dim=input_dim, config=config),
                            # POHF-InfoGain-NoHistory: 消融实验，不使用历史数据
                            # 注意：第一个query时此算法复用主算法结果，此处的实例仅用于后续query
                            'POHF-InfoGain-NoHistory': lambda: NeuralDB(input_dim=input_dim, config=config),
                            # Linear-InfoGain: 线性模型 + Information Gain（神经网络的线性对照组）
                            'Linear-InfoGain': lambda: create_algorithm('LinearInfoGain', input_dim=input_dim, config=config),
                        }

                        for alg_name in enabled_baselines:
                            if alg_name in algorithm_creators:
                                # POHF-InfoGain-NoHistory 在第一个query时不需要初始化，复用主算法结果
                                if alg_name == 'POHF-InfoGain-NoHistory':
                                    # 仅初始化结果存储，算法实例在后续query时创建
                                    baseline_results[alg_name] = {'values': [], 'best_values': [], 'greedy_arm_index': 0, 'best_greedy_arm_index': 0}
                                    continue

                                random.seed(random_seed)
                                np.random.seed(random_seed)
                                torch.manual_seed(random_seed)
                                if torch.cuda.is_available():
                                    torch.cuda.manual_seed(random_seed)
                                    torch.cuda.manual_seed_all(random_seed)
                                baseline_algorithms[alg_name] = algorithm_creators[alg_name]()
                                baseline_results[alg_name] = {'values': [], 'best_values': [], 'greedy_arm_index': 0, 'best_greedy_arm_index': 0}

                    except Exception as e:
                        progress_logger.log_error(f"Baseline算法初始化失败: {e}")
                        baseline_algorithms = {}
                        baseline_results = {}

                    # 🔧 为baseline算法训练
                    if baseline_algorithms and len(X1_train) > 0:
                        from POHF_parameters import TRAINING_CONFIG
                        use_incremental = TRAINING_CONFIG.get("incremental_training", False)
                        use_query_level = TRAINING_CONFIG.get("query_level_incremental", True)

                        for alg_name, alg in baseline_algorithms.items():
                            try:
                                if hasattr(alg, 'train_model'):
                                    import inspect
                                    sig = inspect.signature(alg.train_model)
                                    supports_query_level = 'reset_to_query_start' in sig.parameters

                                    if use_query_level and use_incremental and supports_query_level:
                                        # Query级增量：使用累积数据 + reset_to_query_start
                                        alg.train_model(X1_train, X2_train, Y_train, reset_to_query_start=True)
                                        alg._has_trained = True
                                    else:
                                        # 旧模式：逐个pair增量训练
                                        supports_incremental = 'incremental' in sig.parameters
                                        for i in range(len(X1_train)):
                                            X1_single = X1_train[i:i+1]
                                            X2_single = X2_train[i:i+1]
                                            Y_single = Y_train[i:i+1]
                                            is_first = not hasattr(alg, '_has_trained') or not alg._has_trained
                                            if supports_incremental:
                                                alg.train_model(X1_single, X2_single, Y_single,
                                                              incremental=use_incremental and not is_first)
                                            else:
                                                alg.train_model(X1_single, X2_single, Y_single)
                                            alg._has_trained = True
                            except Exception as e:
                                progress_logger.log_error(f"{alg_name}训练失败: {e}")

                    baseline_training_data = {}
                    input_dim = sen_embeddings.shape[-1]
                    for alg_name in baseline_algorithms.keys():
                        if alg_name != 'Random':
                            # 🔧 [修复] 即使没有随机采样数据也要初始化 baseline_training_data
                            # 这样后续迭代时可以正确累积训练数据
                            if len(X1_train) > 0:
                                baseline_training_data[alg_name] = {
                                    'X1': X1_train.copy(),
                                    'X2': X2_train.copy(),
                                    'Y': Y_train.copy()
                                }
                            else:
                                baseline_training_data[alg_name] = {
                                    'X1': np.array([]).reshape(0, input_dim),
                                    'X2': np.array([]).reshape(0, input_dim),
                                    'Y': np.array([])
                                }

                # 🔧 [修复] 每个query开始时重置baseline_best_values
                # best_score不能跨query累积，因为不同query有不同的ground truth
                if baseline_algorithms:
                    baseline_best_values = {name: -np.inf for name in baseline_algorithms.keys()}

                    # 🔧 Query级增量训练：为每个baseline保存query开始时的权重
                    from POHF_parameters import TRAINING_CONFIG
                    use_query_level = TRAINING_CONFIG.get("query_level_incremental", True)
                    use_incremental = TRAINING_CONFIG.get("incremental_training", False)
                    if use_query_level and use_incremental:
                        for alg_name, alg in baseline_algorithms.items():
                            if hasattr(alg, 'save_query_start_weights'):
                                alg.save_query_start_weights()
                # 确保 POHF-InfoGain-NoHistory 也有 best_values 记录
                if 'POHF-InfoGain-NoHistory' in baseline_results:
                    if 'baseline_best_values' not in locals():
                        baseline_best_values = {}
                    baseline_best_values['POHF-InfoGain-NoHistory'] = -np.inf

                # ========== POHF-InfoGain-NoHistory: 每个query重置网络和概率矩阵 ==========
                # 消融实验：不使用历史数据，每个query都使用全新的未训练网络和均匀概率矩阵
                if 'POHF-InfoGain-NoHistory' in baseline_results and query_idx > 0:
                    try:
                        from information_second_term import ContextualPairwiseInformationManager, PairwiseInformationManager
                        input_dim = sen_embeddings.shape[-1]

                        # 重置随机种子（确保可复现性）
                        random.seed(random_seed + query_idx)  # 使用不同的种子，但保持可复现
                        np.random.seed(random_seed + query_idx)
                        torch.manual_seed(random_seed + query_idx)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(random_seed + query_idx)
                            torch.cuda.manual_seed_all(random_seed + query_idx)

                        # 创建全新的NeuralDB实例（未训练）
                        alg_nih_new = NeuralDB(input_dim=input_dim, config=config)
                        baseline_algorithms['POHF-InfoGain-NoHistory'] = alg_nih_new

                        # 初始化 ContextualPairwiseInformationManager
                        num_arms_nih = len(init_instructions)
                        bayesian_alpha_nih = config.get("pohf", {}).get("bayesian_alpha", 1.0)
                        alg_nih_new.contextual_info_manager = ContextualPairwiseInformationManager(
                            num_arms_nih, use_optimized=True, bayesian_alpha=bayesian_alpha_nih
                        )

                        # 使用均匀矩阵初始化（不使用历史数据）
                        # 🔧 [统一] query_emb 现在对所有数据集都有定义
                        current_input_embedding_nih = query_emb.cpu().numpy()
                        alg_nih_new.info_manager = alg_nih_new.contextual_info_manager.initialize_without_history(
                            input_index=query_idx, input_embedding=current_input_embedding_nih
                        )

                        print(f"  🔄 [POHF-InfoGain-NoHistory] Query {query_idx}: 重置网络和概率矩阵 (均匀初始化)")

                        # 🔧 [消融实验核心] 每个query都重置训练数据
                        # 与主算法不同，POHF-InfoGain-NoHistory 不跨query累积训练数据
                        if 'baseline_training_data' not in locals():
                            baseline_training_data = {}
                        # 强制重置（不是检查是否存在）
                        baseline_training_data['POHF-InfoGain-NoHistory'] = {
                            'X1': np.array([]).reshape(0, input_dim),
                            'X2': np.array([]).reshape(0, input_dim),
                            'Y': np.array([])
                        }
                        print(f"  🔄 [POHF-InfoGain-NoHistory] Query {query_idx}: 重置训练数据")
                    except Exception as e:
                        print(f"⚠️ [POHF-InfoGain-NoHistory] 重置失败: {e}")
                        import traceback
                        traceback.print_exc()

                # 🔧 [说明] baseline_training_data 的行为与主算法 POHF-InfoGain 保持一致：
                # - POHF-InfoGain 的 x_train_new 跨 query 累积（第 2155 行初始化，后续累积）
                # - baseline_training_data 也应该跨 query 累积（保持公平比较）
                # - 注意：这意味着两者都混合了不同 query 的 embedding，这是当前设计的特性
                # 🔧 [修复] 确保 baseline_training_data 和所有算法的键都存在
                try:
                    _ = baseline_training_data
                except NameError:
                    baseline_training_data = {}

                input_dim = sen_embeddings.shape[-1]
                for alg_name in baseline_algorithms.keys():
                    if alg_name != 'Random' and alg_name not in baseline_training_data:
                        baseline_training_data[alg_name] = {
                            'X1': np.array([]).reshape(0, input_dim),
                            'X2': np.array([]).reshape(0, input_dim),
                            'Y': np.array([])
                        }

                # ========== 当前query的偏好反馈循环 ==========
                for t in range(current_max_iter):
                    # 🔧 简化日志：使用进度记录器
                    progress_logger.log_progress(t, current_max_iter, f"[Query {query_idx+1}/{num_queries}]")

                    # 从文件加载embeddings
                    with open(embedding_filename, 'rb') as f:
                        embeddings_np = pickle.load(f)

                    # 内存优化：根据算法类型选择不同的计算方式
                    if algorithm_name == "POHF-InfoGain":
                        greedy_scores = l.calculate_scores_only(embeddings_np)
                        ucb_scores_precomputed = None
                    else:
                        # 原始 POHF 使用分批计算的 UCB
                        greedy_scores_temp = l.calculate_scores_only(embeddings_np)
                        greedy_arm_temp = torch.argmax(greedy_scores_temp).item()
                        greedy_scores, ucb_scores_precomputed = l.calculate_ucb_scores_memory_efficient(
                            embeddings_np,
                            greedy_arm_index=greedy_arm_temp,
                            current_iteration=t,
                            total_iterations=current_max_iter
                        )

                    # 运行baseline算法（静默模式）
                    baseline_iteration_results = {}
                    if baseline_algorithms or 'POHF-InfoGain-NoHistory' in baseline_results:
                        embeddings_list = []
                        with open(embedding_filename, 'rb') as f:
                            embeddings_np_baseline = pickle.load(f)
                            for i in range(len(init_instructions)):
                                embeddings_list.append(embeddings_np_baseline[i])

                        # ========== POHF-InfoGain-NoHistory 特殊处理 ==========
                        # 第一个query时复用主算法结果（在主算法迭代后处理），后续query独立运行
                        if 'POHF-InfoGain-NoHistory' in baseline_results:
                            alg_name_nih = 'POHF-InfoGain-NoHistory'
                            try:
                                if query_idx == 0:
                                    # 第一个query: 跳过独立运行，在主算法迭代完成后复用结果
                                    # 见下方 "POHF-InfoGain-NoHistory 第一个query复用结果" 部分
                                    pass
                                else:
                                    # 后续query: 独立运行（使用全新网络和均匀概率矩阵）
                                    # 算法实例在每个query开始时已经重新创建（见下方query循环开始处）
                                    if alg_name_nih in baseline_algorithms:
                                        alg_nih = baseline_algorithms[alg_name_nih]
                                        # 使用与主算法POHF-InfoGain相同的选择逻辑
                                        greedy_scores_nih = alg_nih.calculate_scores_only(embeddings_np_baseline)
                                        arm1_nih = torch.argmax(greedy_scores_nih).item()

                                        # 使用info_manager的information gain选择arm2
                                        if hasattr(alg_nih, 'info_manager') and alg_nih.info_manager is not None:
                                            info_gains_nih = alg_nih.info_manager.get_all_information_gains(arm1_nih)
                                            if info_gains_nih:
                                                arm2_nih = max(info_gains_nih.keys(), key=lambda a: info_gains_nih[a])
                                            else:
                                                # 如果没有信息增益，随机选择
                                                available_arms = [i for i in range(len(init_instructions)) if i != arm1_nih]
                                                arm2_nih = random.choice(available_arms) if available_arms else arm1_nih
                                        else:
                                            # 没有info_manager时随机选择
                                            available_arms = [i for i in range(len(init_instructions)) if i != arm1_nih]
                                            arm2_nih = random.choice(available_arms) if available_arms else arm1_nih

                                        # 生成响应并比较
                                        personality1_nih = domain_texts[arm1_nih] if arm1_nih < len(domain_texts) else None
                                        personality2_nih = domain_texts[arm2_nih] if arm2_nih < len(domain_texts) else None
                                        response1_nih, response2_nih, score_1_nih, score_2_nih, score_nih, preference_nih, p_nih = await generate_and_compare_async(
                                            init_instructions[arm1_nih], init_instructions[arm2_nih],
                                            ground_truth[1], llm_config,
                                            LLM_as_judge=LLM_as_judge, arm1_idx=arm1_nih, arm2_idx=arm2_nih,
                                            personality1=personality1_nih, personality2=personality2_nih,
                                            ground_truth_index=query_idx
                                        )

                                        # 记录结果（使用greedy arm的分数）
                                        greedy_score_nih = score_1_nih
                                        baseline_results[alg_name_nih]['values'].append(greedy_score_nih)
                                        baseline_results[alg_name_nih]['greedy_arm_index'] = arm1_nih
                                        if greedy_score_nih > baseline_best_values.get(alg_name_nih, -np.inf):
                                            baseline_best_values[alg_name_nih] = greedy_score_nih
                                            baseline_results[alg_name_nih]['best_greedy_arm_index'] = arm1_nih

                                        # 更新算法（Fisher矩阵和info_manager）
                                        arm_grads_nih = alg_nih.calculate_gradients_for_arms(embeddings_np_baseline, [arm1_nih, arm2_nih])
                                        grad1_nih = arm_grads_nih[arm1_nih]
                                        grad2_nih = arm_grads_nih[arm2_nih]
                                        alg_nih.update_matrix(grad1_nih, grad2_nih)

                                        # 更新info_manager
                                        if hasattr(alg_nih, 'info_manager') and alg_nih.info_manager is not None:
                                            arm1_wins_nih = bool(preference_nih == 1)
                                            alg_nih.info_manager.update_pairwise_probability_with_transitive(arm1_nih, arm2_nih, arm1_wins_nih)

                                        # 🔧 方案B：NoHistory只用新增的1个pair进行增量训练
                                        # - 每个query开始时创建新的算法实例（网络/概率矩阵自动重置）
                                        # - query内的迭代只用当前pair增量训练
                                        if hasattr(alg_nih, 'train_model'):
                                            try:
                                                emb1_nih_loaded = embedding_manager.load_embedding(arm1_nih)
                                                emb2_nih_loaded = embedding_manager.load_embedding(arm2_nih)
                                                emb1_np_nih = emb1_nih_loaded.cpu().numpy().reshape(1, -1) if hasattr(emb1_nih_loaded, 'cpu') else emb1_nih_loaded.reshape(1, -1)
                                                emb2_np_nih = emb2_nih_loaded.cpu().numpy().reshape(1, -1) if hasattr(emb2_nih_loaded, 'cpu') else emb2_nih_loaded.reshape(1, -1)

                                                # 仍然累积数据（用于记录）
                                                baseline_training_data[alg_name_nih]['X1'] = np.vstack([baseline_training_data[alg_name_nih]['X1'], emb1_np_nih])
                                                baseline_training_data[alg_name_nih]['X2'] = np.vstack([baseline_training_data[alg_name_nih]['X2'], emb2_np_nih])
                                                baseline_training_data[alg_name_nih]['Y'] = np.append(baseline_training_data[alg_name_nih]['Y'], preference_nih)

                                                # NoHistory: 每个query第一次迭代重置，后续增量
                                                is_first_iteration_in_query = len(baseline_training_data[alg_name_nih]['Y']) == 1
                                                alg_nih.train_model(
                                                    emb1_np_nih, emb2_np_nih, np.array([preference_nih]),
                                                    incremental=not is_first_iteration_in_query
                                                )
                                            except Exception as e:
                                                print(f"⚠️ [{alg_name_nih}] 训练失败: {e}")
                                                import traceback
                                                traceback.print_exc()

                                        winner_nih = "arm1" if preference_nih == 1 else "arm2"
                                        print(f"    [{alg_name_nih}] arm1={arm1_nih}(score={score_1_nih:.4f}) vs arm2={arm2_nih}(score={score_2_nih:.4f}) → {winner_nih}", flush=True)

                                        # 存储本轮结果
                                        baseline_iteration_results[alg_name_nih] = {
                                            'arms': (arm1_nih, arm2_nih),
                                            'scores': (score_1_nih, score_2_nih),
                                            'current_value': greedy_score_nih,
                                            'best_value': baseline_best_values.get(alg_name_nih, greedy_score_nih)
                                        }

                                        del arm_grads_nih, grad1_nih, grad2_nih
                                        torch.cuda.empty_cache()
                            except Exception as e:
                                print(f"⚠️ [Baseline迭代] {alg_name_nih} 失败: {e}")
                                import traceback
                                traceback.print_exc()
                                baseline_results[alg_name_nih]['values'].append(0.0)

                        # ========== 其他baseline算法 ==========
                        for alg_name, alg in baseline_algorithms.items():
                            # 跳过POHF-InfoGain-NoHistory，已在上面单独处理
                            if alg_name == 'POHF-InfoGain-NoHistory':
                                continue

                            try:
                                if alg_name == 'POHF':
                                    greedy_scores_baseline = alg.calculate_scores_only(embeddings_np_baseline)
                                    arm1 = torch.argmax(greedy_scores_baseline).item()
                                    greedy_scores_baseline, ucb_scores_baseline = alg.calculate_ucb_scores_memory_efficient(
                                        embeddings_np_baseline,
                                        greedy_arm_index=arm1,
                                        current_iteration=t,
                                        total_iterations=current_max_iter
                                    )
                                    ucb_scores_copy = ucb_scores_baseline.clone()
                                    ucb_scores_copy[arm1] = float('-inf')
                                    arm2 = torch.argmax(ucb_scores_copy).item()
                                elif alg_name == 'POHF-Random':
                                    arm1, arm2 = alg.select_arm(embeddings_list, history=select_idx_history)
                                elif alg_name == 'POHF-RandomPair':
                                    arm1, arm2 = alg.select_arm(embeddings_list, history=select_idx_history)
                                elif alg_name == 'Linear-InfoGain':
                                    # Linear-InfoGain: 使用线性模型计算 greedy scores + Information Gain 选择 arm2
                                    greedy_scores_linear = alg.calculate_scores_only(embeddings_np_baseline)
                                    arm1 = torch.argmax(greedy_scores_linear).item()

                                    # 初始化 Information Manager（如果需要）
                                    if not hasattr(alg, 'info_manager') or alg.info_manager is None:
                                        from information_second_term import ContextualPairwiseInformationManager, PairwiseInformationManager
                                        bayesian_alpha_linear = config.get("pohf", {}).get("bayesian_alpha", 1.0)

                                        # 使用 ContextualPairwiseInformationManager
                                        if not hasattr(alg, 'contextual_info_manager') or alg.contextual_info_manager is None:
                                            alg.contextual_info_manager = ContextualPairwiseInformationManager(
                                                len(greedy_scores_linear), use_optimized=True, bayesian_alpha=bayesian_alpha_linear
                                            )
                                        # 为当前 query 初始化
                                        current_input_embedding_linear = query_emb.cpu().numpy()
                                        alg.info_manager = alg.contextual_info_manager.initialize_for_new_input(
                                            input_index=query_idx, input_embedding=current_input_embedding_linear
                                        )

                                    # 使用 Information Gain 选择第二个 arm
                                    from POHF_parameters import POHF_CONFIG
                                    info_gain_normalize_linear = POHF_CONFIG.get("info_gain_normalize", True)
                                    info_gain_scale_linear = POHF_CONFIG.get("info_gain_scale", 1.0)

                                    available_arms_linear = list(range(len(greedy_scores_linear)))
                                    if info_gain_normalize_linear:
                                        info_gains_linear = alg.info_manager.get_normalized_information_gains(arm1, available_arms_linear)
                                    else:
                                        info_gains_linear = alg.info_manager.get_all_information_gains(arm1, available_arms_linear)

                                    combined_scores_linear = greedy_scores_linear.clone()
                                    for arm_idx, info_gain in info_gains_linear.items():
                                        combined_scores_linear[arm_idx] = greedy_scores_linear[arm_idx] + info_gain_scale_linear * info_gain

                                    combined_scores_linear[arm1] = float('-inf')
                                    arm2 = torch.argmax(combined_scores_linear).item()
                                else:
                                    arm1, arm2 = alg.select_arm(embeddings_list, history=select_idx_history)

                                # 🚀 异步并行生成响应并比较
                                # 获取对应arm的personality description
                                personality1_baseline = domain_texts[arm1] if arm1 < len(domain_texts) else None
                                personality2_baseline = domain_texts[arm2] if arm2 < len(domain_texts) else None
                                # 🔧 [修复] 添加 ground_truth_index=query_idx，确保使用当前 query 的 ground truth
                                response1_baseline, response2_baseline, score_1_baseline, score_2_baseline, score_baseline, preference_baseline, p_baseline = await generate_and_compare_async(
                                    init_instructions[arm1], init_instructions[arm2],
                                    ground_truth[1], llm_config,
                                    LLM_as_judge=LLM_as_judge, arm1_idx=arm1, arm2_idx=arm2,
                                    personality1=personality1_baseline, personality2=personality2_baseline,
                                    ground_truth_index=query_idx  # 使用当前query的ground truth
                                )

                                # 📊 输出 baseline 算法迭代详情
                                winner_baseline = "arm1" if preference_baseline == 1 else "arm2"
                                print(f"    [{alg_name}] arm1={arm1}(score={score_1_baseline:.4f}) vs arm2={arm2}(score={score_2_baseline:.4f}) → {winner_baseline}", flush=True)

                                # 记录结果
                                if alg_name == 'POHF':
                                    greedy_score = score_1_baseline
                                    baseline_results[alg_name]['values'].append(greedy_score)
                                    baseline_results[alg_name]['greedy_arm_index'] = arm1
                                    if greedy_score > baseline_best_values[alg_name]:
                                        baseline_best_values[alg_name] = greedy_score
                                        baseline_results[alg_name]['best_greedy_arm_index'] = arm1
                                elif alg_name == 'Random':
                                    # 使用两个随机 arm 得分的平均值
                                    current_avg_score = (score_1_baseline + score_2_baseline) / 2
                                    baseline_results[alg_name]['values'].append(current_avg_score)
                                    best_arm_random = arm1 if score_1_baseline >= score_2_baseline else arm2
                                    baseline_results[alg_name]['greedy_arm_index'] = best_arm_random
                                    if current_avg_score > baseline_best_values[alg_name]:
                                        baseline_best_values[alg_name] = current_avg_score
                                        baseline_results[alg_name]['best_greedy_arm_index'] = best_arm_random
                                elif alg_name == 'POHF-Random':
                                    greedy_score = score_1_baseline
                                    baseline_results[alg_name]['values'].append(greedy_score)
                                    baseline_results[alg_name]['greedy_arm_index'] = arm1
                                    if greedy_score > baseline_best_values[alg_name]:
                                        baseline_best_values[alg_name] = greedy_score
                                        baseline_results[alg_name]['best_greedy_arm_index'] = arm1
                                elif alg_name == 'POHF-RandomPair':
                                    greedy_scores_randompair = alg.calculate_scores_only(embeddings_np_baseline)
                                    greedy_arm_randompair = torch.argmax(greedy_scores_randompair).item()
                                    personality_randompair = domain_texts[greedy_arm_randompair] if greedy_arm_randompair < len(domain_texts) else None
                                    response_greedy_randompair = await response_generator_async(init_instructions[greedy_arm_randompair], llm_config, personality_randompair)
                                    # 🔧 [修复] 使用当前 query_idx 获取正确的 ground truth
                                    gt_for_scoring = ground_truth[1]
                                    if isinstance(gt_for_scoring, list):
                                        # 使用 query_idx 获取当前 query 对应的 ground truth
                                        if query_idx < len(gt_for_scoring):
                                            gt_for_scoring = str(gt_for_scoring[query_idx])
                                        elif len(gt_for_scoring) > 0:
                                            gt_for_scoring = str(gt_for_scoring[0])
                                        else:
                                            gt_for_scoring = ""
                                    elif not isinstance(gt_for_scoring, str):
                                        gt_for_scoring = str(gt_for_scoring)
                                    if LLM_as_judge:
                                        greedy_score_1, _, _, _, _ = await rouge_score_comparison_async(
                                            ground_truth[1], response_greedy_randompair, response_greedy_randompair, 'rougeL', LLM_as_judge=False, ground_truth_index=query_idx
                                        )
                                        greedy_score_randompair = greedy_score_1
                                    else:
                                        from rouge_score import rouge_scorer
                                        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                                        greedy_score_randompair = scorer.score(gt_for_scoring, response_greedy_randompair)['rougeL'].fmeasure
                                    baseline_results[alg_name]['values'].append(greedy_score_randompair)
                                    baseline_results[alg_name]['greedy_arm_index'] = greedy_arm_randompair
                                    if greedy_score_randompair > baseline_best_values[alg_name]:
                                        baseline_best_values[alg_name] = greedy_score_randompair
                                        baseline_results[alg_name]['best_greedy_arm_index'] = greedy_arm_randompair
                                elif alg_name == 'DoubleTS':
                                    current_max_score = max(score_1_baseline, score_2_baseline)
                                    baseline_results[alg_name]['values'].append(current_max_score)
                                    best_arm_doublts = arm1 if score_1_baseline >= score_2_baseline else arm2
                                    baseline_results[alg_name]['greedy_arm_index'] = best_arm_doublts
                                    if current_max_score > baseline_best_values[alg_name]:
                                        baseline_best_values[alg_name] = current_max_score
                                        baseline_results[alg_name]['best_greedy_arm_index'] = best_arm_doublts
                                elif alg_name == 'Linear-InfoGain':
                                    # Linear-InfoGain: 记录 greedy arm 的分数（与 POHF-InfoGain 一致）
                                    greedy_score_linear = score_1_baseline
                                    baseline_results[alg_name]['values'].append(greedy_score_linear)
                                    baseline_results[alg_name]['greedy_arm_index'] = arm1
                                    if greedy_score_linear > baseline_best_values[alg_name]:
                                        baseline_best_values[alg_name] = greedy_score_linear
                                        baseline_results[alg_name]['best_greedy_arm_index'] = arm1

                                # 更新算法
                                if alg_name == 'POHF':
                                    arm_grads_baseline = alg.calculate_gradients_for_arms(embeddings_np_baseline, [arm1, arm2])
                                    grad1_baseline = arm_grads_baseline[arm1]
                                    grad2_baseline = arm_grads_baseline[arm2]
                                    alg.update_matrix(grad1_baseline, grad2_baseline)
                                    del arm_grads_baseline, grad1_baseline, grad2_baseline
                                    torch.cuda.empty_cache()
                                elif alg_name == 'POHF-Random':
                                    pass
                                elif alg_name == 'Linear-InfoGain':
                                    # 更新 Information Manager（与 POHF-InfoGain 相同的逻辑）
                                    if hasattr(alg, 'info_manager') and alg.info_manager is not None:
                                        arm1_wins_linear = (score_1_baseline >= score_2_baseline)
                                        from information_second_term import update_information_with_feedback
                                        update_information_with_feedback(alg.info_manager, arm1, arm2, arm1_wins_linear)
                                        # 记录增量比较结果（用于历史迁移）
                                        if hasattr(alg, 'contextual_info_manager') and alg.contextual_info_manager is not None:
                                            alg.contextual_info_manager.record_comparison(arm1, arm2, arm1_wins_linear)
                                else:
                                    alg.update(arm1, arm2, preference_baseline)

                                # 存储本轮结果
                                baseline_iteration_results[alg_name] = {
                                    'arms': (arm1, arm2),
                                    'scores': (score_1_baseline, score_2_baseline),
                                    'current_value': baseline_results[alg_name]['values'][-1],
                                    'best_value': baseline_best_values[alg_name]
                                }

                                # 为baseline算法准备新的训练数据
                                emb1_baseline = embedding_manager.load_embedding(arm1)
                                emb2_baseline = embedding_manager.load_embedding(arm2)

                                # 训练baseline算法
                                if hasattr(alg, 'train_model') and alg_name != 'Random':
                                    try:
                                        emb1_np = emb1_baseline.cpu().numpy().reshape(1, -1) if hasattr(emb1_baseline, 'cpu') else emb1_baseline.reshape(1, -1)
                                        emb2_np = emb2_baseline.cpu().numpy().reshape(1, -1) if hasattr(emb2_baseline, 'cpu') else emb2_baseline.reshape(1, -1)

                                        # 累积数据
                                        baseline_training_data[alg_name]['X1'] = np.vstack([baseline_training_data[alg_name]['X1'], emb1_np])
                                        baseline_training_data[alg_name]['X2'] = np.vstack([baseline_training_data[alg_name]['X2'], emb2_np])
                                        baseline_training_data[alg_name]['Y'] = np.append(baseline_training_data[alg_name]['Y'], preference_baseline)

                                        from POHF_parameters import TRAINING_CONFIG
                                        use_incremental = TRAINING_CONFIG.get("incremental_training", False)
                                        use_query_level = TRAINING_CONFIG.get("query_level_incremental", True)

                                        import inspect
                                        sig = inspect.signature(alg.train_model)
                                        supports_query_level = 'reset_to_query_start' in sig.parameters

                                        if use_query_level and use_incremental and supports_query_level:
                                            # 🔧 Query级增量：使用该Query内累积的所有数据训练
                                            alg.train_model(
                                                baseline_training_data[alg_name]['X1'],
                                                baseline_training_data[alg_name]['X2'],
                                                baseline_training_data[alg_name]['Y'],
                                                reset_to_query_start=True
                                            )
                                            alg._has_trained = True
                                        else:
                                            # 旧模式：只用新增的1个pair训练
                                            is_first = not hasattr(alg, '_has_trained') or not alg._has_trained
                                            if 'incremental' in sig.parameters:
                                                alg.train_model(emb1_np, emb2_np, np.array([preference_baseline]),
                                                              incremental=use_incremental and not is_first)
                                            else:
                                                alg.train_model(emb1_np, emb2_np, np.array([preference_baseline]))
                                            alg._has_trained = True
                                    except Exception as e:
                                        progress_logger.log_error(f"{alg_name} 训练失败: {e}")

                            except Exception as e:
                                print(f"⚠️ [Baseline迭代] {alg_name} 失败: {e}")
                                baseline_results[alg_name]['values'].append(0.0)
                                baseline_results[alg_name]['best_values'].append(baseline_best_values[alg_name])

                        # 清理baseline embeddings
                        del embeddings_np_baseline
                        del embeddings_list
                        torch.cuda.empty_cache()

                    # 找到贪心最优arm
                    greedy_arm_index = torch.argmax(greedy_scores).item()

                    # 🔧 内存优化：根据算法类型处理 UCB 分数
                    if algorithm_name == "POHF-InfoGain":
                        ucb_scores = greedy_scores.clone()
                    else:
                        ucb_scores = ucb_scores_precomputed

                    # POHF策略：选择一个greedy arm和一个UCB arm
                    arm_select1 = greedy_arm_index

                    # 使用当前算法循环的配置参数
                    from POHF_parameters import POHF_CONFIG
                    base_info_gain_scale = POHF_CONFIG.get("info_gain_scale", 1.0)
                    info_gain_normalize = POHF_CONFIG.get("info_gain_normalize", True)

                    # Information Gain衰减参数
                    info_gain_decay_enabled = POHF_CONFIG.get("info_gain_decay_enabled", True)
                    info_gain_decay_factor = POHF_CONFIG.get("info_gain_decay_factor", 0.995)
                    info_gain_min_scale = POHF_CONFIG.get("info_gain_min_scale", 0.1)
                    info_gain_decay_start = POHF_CONFIG.get("info_gain_decay_start", 10)
                    info_gain_decay_type = POHF_CONFIG.get("info_gain_decay_type", "exponential")

                    # 计算当前轮次的Information Gain缩放值
                    if info_gain_decay_enabled and t >= info_gain_decay_start:
                        decay_iterations = t - info_gain_decay_start
                        if info_gain_decay_type == "exponential":
                            info_gain_scale = base_info_gain_scale * (info_gain_decay_factor ** decay_iterations)
                        elif info_gain_decay_type == "linear":
                            decay_rate = (base_info_gain_scale - info_gain_min_scale) / (current_max_iter - info_gain_decay_start)
                            info_gain_scale = max(info_gain_min_scale, base_info_gain_scale - decay_rate * decay_iterations)
                        elif info_gain_decay_type == "step":
                            step_size = 10
                            steps = decay_iterations // step_size
                            info_gain_scale = base_info_gain_scale * (info_gain_decay_factor ** steps)
                        else:
                            info_gain_scale = base_info_gain_scale
                        info_gain_scale = max(info_gain_min_scale, info_gain_scale)
                    else:
                        info_gain_scale = base_info_gain_scale

                    # 选择第二个arm的策略
                    if algorithm_name == "POHF-InfoGain":
                        if not hasattr(l, 'info_manager'):
                            bayesian_alpha = config.get("pohf", {}).get("bayesian_alpha", 1.0)
                            l.info_manager = PairwiseInformationManager(len(greedy_scores), use_optimized=True, bayesian_alpha=bayesian_alpha)

                        available_arms = list(range(len(greedy_scores)))
                        if info_gain_normalize:
                            info_gains = l.info_manager.get_normalized_information_gains(arm_select1, available_arms)
                        else:
                            info_gains = l.info_manager.get_all_information_gains(arm_select1, available_arms)

                        combined_scores = greedy_scores.clone()
                        for arm_idx, info_gain in info_gains.items():
                            combined_scores[arm_idx] = greedy_scores[arm_idx] + info_gain_scale * info_gain

                        combined_scores[arm_select1] = float('-inf')
                        arm_select2 = torch.argmax(combined_scores).item()
                    else:
                        ucb_scores_copy = ucb_scores.clone()
                        ucb_scores_copy[arm_select1] = float('-inf')
                        arm_select2 = torch.argmax(ucb_scores_copy).item()

                    select_idx_history += [[arm_select1, arm_select2]]

                    if algorithm_name == "POHF-InfoGain":
                        second_arm_selections.append(arm_select2)

                    # 🚀 使用异步并行生成响应和比较
                    llm_config = config.get("llm", {})
                    personality1 = domain_texts[arm_select1] if arm_select1 < len(domain_texts) else None
                    personality2 = domain_texts[arm_select2] if arm_select2 < len(domain_texts) else None
                    response1, response2, score_1, score_2, score, new_y, p_ = await generate_and_compare_async(
                        init_instructions[arm_select1], init_instructions[arm_select2],
                        ground_truth[1], llm_config,
                        LLM_as_judge=LLM_as_judge, arm1_idx=arm_select1, arm2_idx=arm_select2,
                        personality1=personality1, personality2=personality2,
                        ground_truth_index=query_idx  # 使用当前query的ground truth
                    )

                    # 📊 输出迭代详情
                    winner = "arm1" if new_y == 1 else "arm2"
                    if LLM_as_judge:
                        print(f"  [Query {query_idx+1}][Iter {t+1}/{current_max_iter}] arm1={arm_select1} vs arm2={arm_select2} → {winner} wins (LLM judge)", flush=True)
                    else:
                        print(f"  [Query {query_idx+1}][Iter {t+1}/{current_max_iter}] arm1={arm_select1}(score={score_1:.4f}) vs arm2={arm_select2}(score={score_2:.4f}) → {winner} wins", flush=True)

                    instruction_select_history += [(init_instructions[arm_select1], score_1, init_instructions[arm_select2], score_2)]

                    # 找到当前最佳arm
                    if LLM_as_judge:
                        pass
                    else:
                        if score_1 >= score_2:
                            best_arm = arm_select1
                            best_score = score_1
                        else:
                            best_arm = arm_select2
                            best_score = score_2
                        now_values += [score]
                        # 🔧 [修复] 在元组中添加query_index，便于后续按query分组
                        best_instruction_over_iter += [(t, init_instructions[arm_select1], score_1, query_idx)]
                        if best_score > best_r:
                            best_index = best_arm
                            best_r = best_score
                            print(f"  🏆 新的最佳分数: {best_r:.4f} (arm={best_index})", flush=True)

                    # 训练模型
                    emb1 = embedding_manager.load_embedding(arm_select1)
                    emb2 = embedding_manager.load_embedding(arm_select2)

                    # 累积数据
                    x_train_new += [torch.cat([emb1.reshape(1,1,-1), emb2.reshape(1,1,-1)])]
                    y_train_new += [new_y]

                    # 🔧 训练模式选择
                    from POHF_parameters import TRAINING_CONFIG
                    use_incremental = TRAINING_CONFIG.get("incremental_training", False)
                    use_query_level = TRAINING_CONFIG.get("query_level_incremental", True)

                    if use_query_level and use_incremental:
                        # 🔧 Query级增量训练：用该Query内所有累积数据重训练
                        # 构建该Query内的累积训练数据
                        X1_accum = []
                        X2_accum = []
                        Y_accum = []
                        for x_pair, y_val in zip(x_train_new, y_train_new):
                            x_pair_reshaped = x_pair.squeeze(1)
                            X1_accum.append(x_pair_reshaped[0].cpu().numpy())
                            X2_accum.append(x_pair_reshaped[1].cpu().numpy())
                            Y_accum.append(y_val)
                        X1_accum = np.array(X1_accum)
                        X2_accum = np.array(X2_accum)
                        Y_accum = np.array(Y_accum)

                        # 使用累积数据 + reset_to_query_start
                        l.train_model(X1_accum, X2_accum, Y_accum, reset_to_query_start=True)
                        l._has_trained = True
                    else:
                        # 旧模式：只用新增的1个pair进行增量训练
                        X1_single = emb1.cpu().numpy().reshape(1, -1)
                        X2_single = emb2.cpu().numpy().reshape(1, -1)
                        Y_single = np.array([new_y])
                        is_first_training = not hasattr(l, '_has_trained') or not l._has_trained
                        l.train_model(X1_single, X2_single, Y_single, incremental=use_incremental and not is_first_training)
                        l._has_trained = True

                    arm_grads = l.calculate_gradients_for_arms(embeddings_np, [arm_select1, arm_select2])
                    grad1 = arm_grads[arm_select1]
                    grad2 = arm_grads[arm_select2]
                    l.update_matrix(grad1, grad2)

                    del arm_grads, grad1, grad2
                    del embeddings_np
                    torch.cuda.empty_cache()

                    # 更新Information Manager
                    if algorithm_name == "POHF-InfoGain" and hasattr(l, 'info_manager'):
                        arm1_wins = (score_1 >= score_2)
                        from information_second_term import update_information_with_feedback
                        info_gain = update_information_with_feedback(l.info_manager, arm_select1, arm_select2, arm1_wins)
                        # 🔧 记录增量比较结果（用于历史迁移）
                        if hasattr(l, 'contextual_info_manager') and l.contextual_info_manager is not None:
                            l.contextual_info_manager.record_comparison(arm_select1, arm_select2, arm1_wins)

                    # 计算得分分解
                    if algorithm_name == "POHF-InfoGain":
                        ucb_arm_greedy_score = greedy_scores[arm_select2].item()
                        ucb_arm_uncertainty = 0.0
                        ucb_arm_total_score = combined_scores[arm_select2].item()
                    else:
                        ucb_arm_greedy_score = greedy_scores[arm_select2].item()
                        ucb_arm_uncertainty = (ucb_scores[arm_select2] - greedy_scores[arm_select2]).item()
                        ucb_arm_total_score = ucb_scores[arm_select2].item()

                    if not LLM_as_judge:
                        best_values.append(best_r)

                    # 收集当前轮次的所有输出数据
                    iteration_output = {
                        "iteration": t,
                        "query_index": query_idx,
                        "selected_arms": {
                            "arm1": arm_select1,
                            "arm2": arm_select2
                        },
                        "scores": {
                            "reward": score if not LLM_as_judge else None,
                            "arm1_score": score_1,
                            "arm2_score": score_2
                        },
                        "rouge_comparison": {
                            "probability_p": p_ if not LLM_as_judge else None,
                            "preference": new_y
                        },
                        "second_arm_analysis": {
                            "algorithm": "InfoGain" if info_gain_enabled else "UCB",
                            "selected_arm": arm_select2,
                            "greedy_score": ucb_arm_greedy_score,
                            "uncertainty_score": ucb_arm_uncertainty,
                            "total_score": ucb_arm_total_score
                        },
                        "greedy_arm_index": greedy_arm_index,
                        "instructions": {
                            "arm1_instruction": init_instructions[arm_select1],
                            "arm2_instruction": init_instructions[arm_select2]
                        },
                        "responses": {
                            "response1": response1,
                            "response2": response2
                        },
                        "baseline_results": baseline_iteration_results if baseline_algorithms else {}
                    }

                    if not LLM_as_judge:
                        iteration_output["best_performance"] = {
                            "best_value": best_r,
                            "best_arm": best_index
                        }

                    # ========== POHF-InfoGain-NoHistory 第一个query复用结果 ==========
                    # 在主算法迭代完成后，将结果复制到 POHF-InfoGain-NoHistory
                    if query_idx == 0 and 'POHF-InfoGain-NoHistory' in baseline_results:
                        alg_name_nih = 'POHF-InfoGain-NoHistory'
                        # 复用主算法的 greedy arm 分数
                        baseline_results[alg_name_nih]['values'].append(score_1)
                        baseline_results[alg_name_nih]['greedy_arm_index'] = greedy_arm_index
                        if score_1 > baseline_best_values.get(alg_name_nih, -np.inf):
                            baseline_best_values[alg_name_nih] = score_1
                            baseline_results[alg_name_nih]['best_greedy_arm_index'] = greedy_arm_index
                        print(f"    [POHF-InfoGain-NoHistory] Query 0: 复用主算法结果 (greedy_arm={greedy_arm_index}, score={score_1:.4f})", flush=True)

                        # 存储本轮结果到 baseline_iteration_results
                        baseline_iteration_results[alg_name_nih] = {
                            'arms': (arm_select1, arm_select2),
                            'scores': (score_1, score_2),
                            'current_value': score_1,
                            'best_value': baseline_best_values.get(alg_name_nih, score_1)
                        }

                # ========== Query循环结束后保存结果 ==========
                # 🔧 保存当前 query 的增量比较结果到历史中
                if algorithm_name == "POHF-InfoGain" and hasattr(l, 'contextual_info_manager') and l.contextual_info_manager is not None:
                    l.contextual_info_manager.finalize_query()

                # 🔧 为 Linear-InfoGain 也保存 query 的增量比较结果
                if 'Linear-InfoGain' in baseline_algorithms:
                    alg_linear = baseline_algorithms['Linear-InfoGain']
                    if hasattr(alg_linear, 'contextual_info_manager') and alg_linear.contextual_info_manager is not None:
                        alg_linear.contextual_info_manager.finalize_query()

                # 保存当前算法的结果
                final_greedy_arm_index = greedy_arm_index if 'greedy_arm_index' in locals() else 0
                algorithm_result = {
                    "algorithm": algorithm_name,
                    "counter": counter,
                    "run_index": run_index,
                    "final_greedy_arm_index": final_greedy_arm_index,
                    # 🔧 修复：baseline_results 可能包含 POHF-InfoGain-NoHistory，即使它不在 baseline_algorithms 中
                    "baseline_results": baseline_results if (baseline_algorithms or 'baseline_results' in locals()) else {},
                    "second_arm_selections": second_arm_selections if algorithm_name == "POHF-InfoGain" else [],
                    "total_arms": len(greedy_scores) if 'greedy_scores' in locals() else times,
                    "contextual_mode": contextual_mode_enabled,
                    "num_queries": num_queries  # 🔧 [统一] 所有数据集都使用实际的num_queries
                }

                if not LLM_as_judge:
                    algorithm_result["best_instruction_over_iter"] = best_instruction_over_iter
                    algorithm_result["best_score"] = best_r
                    algorithm_result["best_index"] = best_index
                    algorithm_result["total_iterations"] = len(best_instruction_over_iter)

                counter_results.append(algorithm_result)

                # ========== [问题1修复] 收集每个query的所有算法Persona数据 ==========
                query_key = f"query_{query_idx}"
                if query_key not in counter_persona_data["queries"]:
                    counter_persona_data["queries"][query_key] = {
                        "query_index": query_idx,
                        "query_text": current_query if 'current_query' in locals() else "",
                        "ground_truth": ground_truth_data[query_idx] if isinstance(ground_truth_data, list) and query_idx < len(ground_truth_data) else str(ground_truth_data),
                        "algorithms": {}
                    }

                # 收集POHF-InfoGain的结果
                if algorithm_name == "POHF-InfoGain" and final_greedy_arm_index < len(domain_texts):
                    greedy_persona = domain_texts[final_greedy_arm_index].replace("### Summary:\n", "")
                    # 生成该query对应的greedy arm response
                    greedy_prompt = init_instructions[final_greedy_arm_index] if final_greedy_arm_index < len(init_instructions) else ""
                    greedy_personality = domain_texts[final_greedy_arm_index] if final_greedy_arm_index < len(domain_texts) else None
                    greedy_response = await response_generator_async(greedy_prompt, llm_config, greedy_personality)

                    pohf_infogain_data = {
                        "greedy_arm_index": final_greedy_arm_index,
                        "persona_summary": greedy_persona,
                        "response": greedy_response,
                    }
                    # Only add greedy_score if not in LLM_as_judge mode
                    if not LLM_as_judge:
                        pohf_infogain_data["greedy_score"] = best_r
                    counter_persona_data["queries"][query_key]["algorithms"]["POHF-InfoGain"] = pohf_infogain_data

                # 收集所有baseline算法的结果
                if baseline_algorithms and baseline_results:
                    for baseline_name, baseline_data in baseline_results.items():
                        baseline_greedy_idx = baseline_data.get('greedy_arm_index', 0)
                        if baseline_greedy_idx < len(domain_texts):
                            baseline_persona = domain_texts[baseline_greedy_idx].replace("### Summary:\n", "")
                            # 生成该baseline算法的greedy arm response
                            baseline_prompt = init_instructions[baseline_greedy_idx] if baseline_greedy_idx < len(init_instructions) else ""
                            baseline_personality = domain_texts[baseline_greedy_idx] if baseline_greedy_idx < len(domain_texts) else None
                            baseline_response = await response_generator_async(baseline_prompt, llm_config, baseline_personality)

                            baseline_algo_data = {
                                "greedy_arm_index": baseline_greedy_idx,
                                "persona_summary": baseline_persona,
                                "response": baseline_response,
                            }
                            # Only add greedy_score if not in LLM_as_judge mode
                            if not LLM_as_judge:
                                # 🔧 根据算法类型选择正确的分数
                                # Random 和 DoubleTS: 使用最后一轮选择的较大分数
                                # 其他算法 (POHF, POHF-Random, POHF-RandomPair): 使用 greedy arm 的分数
                                if baseline_name in ['Random', 'DoubleTS']:
                                    # 最后一轮选择的较大分数
                                    baseline_score = baseline_data['values'][-1] if baseline_data.get('values') else None
                                else:
                                    # greedy arm 的分数（已在迭代中记录）
                                    baseline_score = baseline_best_values.get(baseline_name, None) if 'baseline_best_values' in locals() else None
                                baseline_algo_data["greedy_score"] = baseline_score

                            counter_persona_data["queries"][query_key]["algorithms"][baseline_name] = baseline_algo_data

            # 🔧 简化日志：输出 counter 完成信息
            if not LLM_as_judge:
                progress_logger.log_complete(best_score=best_r, greedy_arm=final_greedy_arm_index)
            else:
                progress_logger.log_complete()

        # 算法循环结束，保存当前counter的所有算法结果
        for result in counter_results:
            all_results.append(result)

        # ========== 作图逻辑 ==========
        if not LLM_as_judge and counter_results:
            # 获取主算法结果
            main_result = next((res for res in counter_results if res.get('algorithm') == algorithm_name), None)

            if main_result and main_result.get('best_instruction_over_iter'):
                full_data = main_result['best_instruction_over_iter']
                baseline_data = main_result.get('baseline_results', {})

                # 🔧 [简化] 统一训练轮次
                q_iters = unified_training_rounds

                # ========== 图1：query 0 的学习曲线 ==========
                q0_data = full_data[:q_iters]
                q0_baselines = {bl: {'values': bd['values'][:q_iters]} for bl, bd in baseline_data.items() if bd.get('values')}

                q0_result = {
                    'algorithm': algorithm_name,
                    'counter': counter,
                    'best_instruction_over_iter': q0_data,
                    'baseline_results': q0_baselines,
                    'total_arms': main_result.get('total_arms', times),
                    'second_arm_selections': main_result.get('second_arm_selections', [])[:q_iters]
                }

                plot_dual_algorithm_comparison([q0_result], counter, get_dataset_name(LaMP_type))
                if q0_result.get('second_arm_selections'):
                    plot_second_arm_selection_stats(q0_result['second_arm_selections'], counter, q0_result['total_arms'], get_dataset_name(LaMP_type))
                print(f"   📊 [图1] Query 0 学习曲线 (counter={counter}, 轮次={len(q0_data)})")

                # 🔧 [修复] 保存完整结果用于 counter_average（包含所有query数据）
                # 使用 main_result 而不是 q0_result，以便导出所有query的数据
                full_result_for_avg = {
                    'algorithm': algorithm_name,
                    'counter': counter,
                    'best_instruction_over_iter': full_data,  # 所有query的数据
                    'baseline_results': baseline_data,  # 所有query的baseline数据
                    'total_arms': main_result.get('total_arms', times),
                    'second_arm_selections': main_result.get('second_arm_selections', []),
                    'contextual_mode': contextual_mode_enabled,
                    'num_queries': num_queries,
                    'total_iterations': len(full_data)
                }
                all_first_query_results.append({'counter': counter, 'results': [full_result_for_avg]})

                # ========== 图2：query 0-N 的进步（包含第一个query） ==========
                # 🔧 [修改] 包含第一个query（query_0）的数据，因为现在所有query使用统一训练配置
                if num_queries >= 1 and contextual_mode_enabled:
                    for q_idx in range(0, num_queries):  # 从0开始，包含第一个query
                        start = q_idx * q_iters
                        end = start + q_iters
                        if start >= len(full_data):
                            continue

                        # 收集所有值用于归一化
                        main_vals = [item[2] if isinstance(item, tuple) else item for item in full_data[start:end]]
                        all_vals = list(main_vals)
                        bl_vals_map = {}
                        for bl, bd in baseline_data.items():
                            if bd.get('values'):
                                bv = bd['values'][start:end]
                                bl_vals_map[bl] = bv
                                all_vals.extend(bv)

                        if all_vals:
                            vmin, vmax = min(all_vals), max(all_vals)
                            vrange = vmax - vmin if vmax > vmin else 1.0
                            norm = lambda v: (v - vmin) / vrange

                            if main_vals:
                                query_final_values.setdefault(algorithm_name, {})[q_idx] = {'final_value_normalized': norm(main_vals[-1]), 'final_value_raw': main_vals[-1]}

                            for bl, bv in bl_vals_map.items():
                                if bv:
                                    entry = {'final_value_normalized': norm(bv[-1]), 'final_value_raw': bv[-1]}
                                    if bl == 'Random':
                                        entry['avg_value_normalized'] = np.mean([norm(v) for v in bv])
                                    query_final_values.setdefault(bl, {})[q_idx] = entry

                    if query_final_values:
                        q_indices = sorted(set(q for alg in query_final_values.values() for q in alg.keys()))
                        if q_indices:
                            rand_avg = {q: query_final_values.get('Random', {}).get(q, {}).get('avg_value_normalized', 0.0) for q in q_indices}
                            alg_progress = {alg: [data[q]['final_value_normalized'] - rand_avg.get(q, 0.0) for q in q_indices if q in data] for alg, data in query_final_values.items()}
                            alg_progress = {k: v for k, v in alg_progress.items() if v}

                            # 🔍 诊断输出：每个query的原始分数和归一化分数
                            print(f"\n   📊 [诊断] Query进度详情 (counter={counter}):")
                            for q_idx in q_indices:
                                print(f"      Query {q_idx}:")
                                for alg_name, alg_data in query_final_values.items():
                                    if q_idx in alg_data:
                                        raw = alg_data[q_idx].get('final_value_raw', 'N/A')
                                        norm = alg_data[q_idx].get('final_value_normalized', 'N/A')
                                        rand_norm = rand_avg.get(q_idx, 0.0)
                                        improvement = norm - rand_norm if isinstance(norm, (int, float)) else 'N/A'
                                        print(f"         {alg_name}: raw={raw:.4f}, norm={norm:.4f}, rand_avg_norm={rand_norm:.4f}, improvement={improvement:.4f}" if isinstance(raw, (int, float)) else f"         {alg_name}: {raw}")

                            q_prog_data = {'counter': counter, 'query_indices': q_indices, 'algorithms': alg_progress}
                            plot_query_progress(q_prog_data, counter, get_dataset_name(LaMP_type))
                            print(f"   📊 [图2] 跨Query进步图 (counter={counter})")
                            all_query_progress_data.append(q_prog_data)

        else:
            # LLM_as_judge模式：收集greedy arm信息（静默模式）
            # 适配contextual模式：每个query都有对应的persona和response
            counter_greedy_data = {
                "counter": counter,
                "query": input_data[1],
                "instruction": input_data[2],
                "ground_truth": ground_truth[1],
                "history_context": input_data[5] if len(input_data) > 5 else "",
                "original_summary": original_summary_for_counter,
                "contextual_mode": contextual_mode_enabled if 'contextual_mode_enabled' in locals() else False,
                "num_queries": num_queries if 'num_queries' in locals() else 1,
                "algorithms": {}
            }

            llm_config = config.get("llm", {})

            for result in counter_results:
                algorithm_name = result.get('algorithm', 'Unknown')
                final_greedy_arm_idx = result.get('final_greedy_arm_index', 0)

                if final_greedy_arm_idx < len(init_instructions):
                    greedy_prompt = init_instructions[final_greedy_arm_idx]
                    # 获取对应arm的personality description (persona原文)
                    greedy_personality = domain_texts[final_greedy_arm_idx] if final_greedy_arm_idx < len(domain_texts) else None
                    # 从domain_text提取纯summary内容（去除前缀）
                    greedy_persona_text = greedy_personality.replace("### Summary:\n", "") if greedy_personality else None
                    # 构建缓存key时包含personality
                    cache_key = greedy_prompt + (greedy_personality or "")
                    prompt_hash = hashlib.md5(cache_key.encode()).hexdigest()
                    if prompt_hash in _response_cache:
                        greedy_response = _response_cache[prompt_hash]
                    else:
                        # 🚀 使用异步生成响应
                        greedy_response = await response_generator_async(greedy_prompt, llm_config, greedy_personality)
                else:
                    greedy_prompt = "Index out of range"
                    greedy_response = "N/A"
                    greedy_persona_text = None

                # 保存结果，包含persona原文
                counter_greedy_data["algorithms"][algorithm_name] = {
                    "greedy_arm_index": final_greedy_arm_idx,
                    "persona": greedy_persona_text,  # 添加persona原文
                    "response": greedy_response
                }

                baseline_results_data = result.get('baseline_results', {})
                for baseline_name, baseline_data in baseline_results_data.items():
                    if baseline_name not in counter_greedy_data["algorithms"]:
                        final_baseline_greedy_idx = baseline_data.get('greedy_arm_index', 0)

                        if final_baseline_greedy_idx < len(init_instructions):
                            baseline_prompt = init_instructions[final_baseline_greedy_idx]
                            # 获取对应arm的personality description (persona原文)
                            baseline_personality = domain_texts[final_baseline_greedy_idx] if final_baseline_greedy_idx < len(domain_texts) else None
                            # 从domain_text提取纯summary内容（去除前缀）
                            baseline_persona_text = baseline_personality.replace("### Summary:\n", "") if baseline_personality else None
                            # 构建缓存key时包含personality
                            baseline_cache_key = baseline_prompt + (baseline_personality or "")
                            baseline_prompt_hash = hashlib.md5(baseline_cache_key.encode()).hexdigest()
                            if baseline_prompt_hash in _response_cache:
                                baseline_response = _response_cache[baseline_prompt_hash]
                            else:
                                # 🚀 使用异步生成响应
                                baseline_response = await response_generator_async(baseline_prompt, llm_config, baseline_personality)
                        else:
                            baseline_prompt = "Index out of range"
                            baseline_response = "N/A"
                            baseline_persona_text = None

                        # 保存结果，包含persona原文
                        counter_greedy_data["algorithms"][baseline_name] = {
                            "greedy_arm_index": final_baseline_greedy_idx,
                            "persona": baseline_persona_text,  # 添加persona原文
                            "response": baseline_response
                        }

            all_greedy_arm_results.append(counter_greedy_data)

            # 增量保存结果（静默模式）
            try:
                import json
                save_dir = os.path.join(PROJECT_ROOT, "final_su")
                os.makedirs(save_dir, exist_ok=True)

                json_filename = f"{get_dataset_name(LaMP_type)}_greedy_prompts.json"
                json_filepath = os.path.join(save_dir, json_filename)

                existing_results = None
                if os.path.exists(json_filepath):
                    try:
                        with open(json_filepath, 'r', encoding='utf-8') as f:
                            existing_results = json.load(f)
                    except (json.JSONDecodeError, IOError) as json_err:
                        print(f"⚠️ [增量保存] 读取现有JSON失败: {json_err}")
                        existing_results = None

                if existing_results and isinstance(existing_results.get('counters'), list):
                    existing_results['counters'].append(counter_greedy_data)
                    existing_results['total_counters'] = len(existing_results['counters'])
                    final_results_incremental = existing_results
                else:
                    final_results_incremental = {
                        "dataset": get_dataset_name(LaMP_type),
                        "total_counters": 1,
                        "counters": [counter_greedy_data]
                    }

                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(final_results_incremental, f, ensure_ascii=False, indent=2)

            except Exception as save_err:
                progress_logger.log_error(f"增量保存失败: {save_err}")

        # ========== [问题1修复] 保存该counter的完整persona数据到单个文件 ==========
        import json
        from datetime import datetime

        # 更新保存时间
        counter_persona_data["saved_at"] = datetime.now().isoformat()

        persona_save_dir = os.path.join(PROJECT_ROOT, "persona_results")
        os.makedirs(persona_save_dir, exist_ok=True)

        # Add LLM suffix when LLM_as_judge is True
        if LLM_as_judge:
            persona_filename = os.path.join(
                persona_save_dir,
                f"persona_{get_dataset_name(LaMP_type)}_counter{counter}_all_algorithms_LLM.json"
            )
        else:
            persona_filename = os.path.join(
                persona_save_dir,
                f"persona_{get_dataset_name(LaMP_type)}_counter{counter}_all_algorithms.json"
            )

        try:
            with open(persona_filename, 'w', encoding='utf-8') as f:
                json.dump(counter_persona_data, f, ensure_ascii=False, indent=2)
            print(f"   💾 [Persona] 保存所有算法persona数据: {persona_filename}")
        except Exception as e:
            print(f"   ⚠️ [Persona] 保存失败: {e}")

        # 清理资源（静默模式）
        embedding_manager.clear_cache()
        if os.path.exists(embedding_filename):
            pass

        if LLM_as_judge:
            _response_cache.clear()

        # 清理显存（静默模式）
        if 'baseline_algorithms' in locals():
            for alg_name in list(baseline_algorithms.keys()):
                if alg_name in baseline_algorithms:
                    del baseline_algorithms[alg_name]
            baseline_algorithms.clear()
            del baseline_algorithms

        if 'baseline_training_data' in locals():
            for alg_name in list(baseline_training_data.keys()):
                if alg_name in baseline_training_data:
                    del baseline_training_data[alg_name]
            baseline_training_data.clear()
            del baseline_training_data

        if 'l' in locals():
            del l

        if 'x_train_tensor' in locals():
            del x_train_tensor
        if 'X1_train' in locals():
            del X1_train
        if 'X2_train' in locals():
            del X2_train
        if 'Y_train' in locals():
            del Y_train

        if 'sen_embeddings' in locals():
            del sen_embeddings
        if 'sen_embeddings_list' in locals():
            del sen_embeddings_list

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()




    # ========== 最终结果汇总（静默模式）==========

    # 生成counter平均图表（静默模式）
    # 只在非单counter模式下生成图表，避免并行模式下子进程重复生成
    if not LLM_as_judge and len(all_results) > 1 and not single_counter_mode:
        # 🔧 图1 Counter Average：使用第一个query的结果生成
        if all_first_query_results:
            # 将所有counter的第一个query结果合并成一个列表
            first_query_combined_results = []
            for fq_data in all_first_query_results:
                first_query_combined_results.extend(fq_data['results'])

            if first_query_combined_results:
                plot_counter_average_results(first_query_combined_results, get_dataset_name(LaMP_type))
                print(f"   📊 [图1 Counter Average] 已生成第一个Query的Counter平均图")
        else:
            # 如果没有收集到first_query_results，使用原有逻辑
            plot_counter_average_results(all_results, get_dataset_name(LaMP_type))

        # 🔧 图2 Counter Average：使用跨Query进步数据生成
        if all_query_progress_data:
            plot_query_progress_counter_average(all_query_progress_data, get_dataset_name(LaMP_type))
            print(f"   📊 [图2 Counter Average] 已生成跨Query进步的Counter平均图")

    return all_results


def get_unique_filename(directory, base_filename):
    """
    生成唯一的文件名，避免覆盖已存在的文件

    Args:
        directory: 目标目录
        base_filename: 基础文件名

    Returns:
        str: 唯一的完整文件路径
    """
    full_path = os.path.join(directory, base_filename)

    # 如果文件不存在，直接返回
    if not os.path.exists(full_path):
        return full_path

    # 如果文件存在，添加序号
    name, ext = os.path.splitext(base_filename)
    counter = 1

    while True:
        new_filename = f"{name}_{counter:03d}{ext}"
        new_full_path = os.path.join(directory, new_filename)

        if not os.path.exists(new_full_path):
            return new_full_path

        counter += 1

        # 防止无限循环
        if counter > 999:
            import time
            timestamp_suffix = str(int(time.time()))
            new_filename = f"{name}_{timestamp_suffix}{ext}"
            new_full_path = os.path.join(directory, new_filename)
            return new_full_path


def get_plot_config_from_baseline():
    """
    从BASELINE_CONFIG获取绘图配置

    Returns:
        tuple: (colors, markers, algorithms_with_range, exclude_from_minmax)
    """
    from POHF_parameters import BASELINE_CONFIG

    display_config = BASELINE_CONFIG.get("algorithm_display_config", {})
    plot_config = BASELINE_CONFIG.get("plot_config", {})

    # 提取颜色和标记
    colors = {}
    markers = {}

    for alg_name, config in display_config.items():
        if config.get('show_in_plots', True):
            colors[alg_name] = config.get('color', '#333333')
            markers[alg_name] = config.get('marker', 'o')

    # 获取显示范围的算法和排除min-max的算法
    algorithms_with_range = set(plot_config.get("show_range_for_algorithms", ['Random']))
    exclude_from_minmax = plot_config.get("exclude_from_minmax", ['Random'])

    return colors, markers, algorithms_with_range, exclude_from_minmax


def generate_minmax_counter_average_plot(algorithm_average_curves_minmax, algorithm_max_curves_minmax, algorithm_min_curves_minmax, timestamp, counter_count, lamp_type=None):
    """
    生成Min-Max归一化的Counter Average折线图

    Args:
        algorithm_average_curves_minmax: Min-Max归一化的算法平均曲线数据
        algorithm_max_curves_minmax: Min-Max归一化的算法上界曲线数据
        algorithm_min_curves_minmax: Min-Max归一化的算法下界曲线数据
        timestamp: 时间戳
        counter_count: counter数量
        lamp_type: LaMP数据集类型（用于文件命名）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    if not algorithm_average_curves_minmax:
        return

    # 生成折线图
    plt.figure(figsize=(14, 10))

    # 🔧 从配置中获取绘图参数
    colors, markers, algorithms_with_range, _ = get_plot_config_from_baseline()

    # 绘制每个算法的Min-Max归一化平均曲线和SE范围（所有算法都显示）
    for alg_name, average_curve in algorithm_average_curves_minmax.items():
        if not average_curve:
            continue

        x_values = list(range(len(average_curve)))
        upper_curve = algorithm_max_curves_minmax[alg_name]  # 平均值+标准误
        lower_curve = algorithm_min_curves_minmax[alg_name]  # 平均值-标准误
        color = colors.get(alg_name, '#333333')

        # 为所有算法绘制SE范围填充
        plt.fill_between(x_values, lower_curve, upper_curve,
                        color=color, alpha=0.2)

        # 绘制平均曲线（所有算法都有）
        final_score = average_curve[-1] if average_curve else 0
        label = f"{alg_name} (Final: {final_score:.3f})"

        plt.plot(x_values, average_curve,
                marker=markers.get(alg_name, 'o'),
                linestyle='-',
                linewidth=3,
                markersize=8,
                color=color,
                label=label,
                alpha=0.9,
                markerfacecolor='white',
                markeredgewidth=2)

    # 图表美化
    # 使用传入的counter_count参数，而不是重新从配置读取
    plt.title(f'Counter Average Performance Curves (Min-Max Normalized)\n(Min-Max normalized scores averaged across {counter_count} counters, all algorithms show mean±1SE range)',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Min-Max Normalized Score', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')

    # 添加图例
    plt.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.02, 0.98), ncol=1)

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 设置坐标轴 - 动态调整Y轴范围
    plt.xlim(0, max([len(curve) for curve in algorithm_average_curves_minmax.values()]) - 1)

    # 计算所有曲线的最小值和最大值，动态设置Y轴范围
    all_values = []
    for curve in algorithm_average_curves_minmax.values():
        all_values.extend(curve)

    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min

        # 添加5%的边距
        margin = max(0.05, y_range * 0.05)
        plt.ylim(y_min - margin, y_max + margin)
    else:
        plt.ylim(0, 1.05)

    # 美化坐标轴
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()

    # 创建保存目录并保存图片
    save_dir = os.path.join(PROJECT_ROOT, "past_average")
    os.makedirs(save_dir, exist_ok=True)

    # 使用传入的lamp_type参数，如果没有则尝试从配置读取
    if lamp_type is None:
        try:
            from POHF_parameters import PATH_CONFIG
            lamp_type = PATH_CONFIG.get('LaMP_type', 'unknown')
        except:
            lamp_type = 'unknown'

    base_filename = f"{get_filename_prefix(lamp_type)}_counter_average_curves_minmax_{timestamp}.pdf"

    # 确保文件名唯一，避免覆盖
    filename = get_unique_filename(save_dir, base_filename)

    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    except Exception as e:
        print(f"❌ [generate_minmax_counter_average_plot] 保存图片失败: {e}")

    plt.close()


def plot_counter_average_results(all_results, lamp_type=None):
    """
    生成所有counter的平均折线图
    将每个counter的结果合成一个平均的折线图，y轴是分数，x轴是轮次

    归一化方式说明：
    - 使用下界为0的归一化：normalized_score = score / max_score
    - 每个counter内所有算法使用相同的max_score进行归一化
    - 归一化后的分数表示相对于该counter最佳性能的百分比
    - 这种方式保持了绝对性能关系，分数为0的算法归一化后仍为0

    Args:
        all_results: 所有counter的结果列表
                    [{'counter': int, 'best_instruction_over_iter': [...], 'baseline_results': {...}}, ...]
        lamp_type: LaMP数据集类型（用于文件命名）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    if not all_results:
        return

    # 收集所有算法在所有counter中的轮次数据
    algorithm_iteration_data = {}
    max_iterations = 0

    for result in all_results:
        counter = result['counter']
        best_instruction_over_iter = result.get('best_instruction_over_iter', [])
        baseline_results = result.get('baseline_results', {})

        # 处理主算法数据（现在是POHF-InfoGain）
        algorithm_name = result.get('algorithm', 'POHF-InfoGain')
        if best_instruction_over_iter:
            if algorithm_name not in algorithm_iteration_data:
                algorithm_iteration_data[algorithm_name] = []

            # 🔧 [修复] 提取主算法每轮的得分，兼容3元组和4元组格式
            # 3元组: (t, instruction, score)
            # 4元组: (t, instruction, score, query_idx)
            main_scores = []
            for item in best_instruction_over_iter:
                if len(item) >= 3:
                    main_scores.append(item[2])  # score 始终在第3个位置
                elif isinstance(item, (int, float)):
                    main_scores.append(item)
            algorithm_iteration_data[algorithm_name].append(main_scores)
            max_iterations = max(max_iterations, len(main_scores))

        # 处理baseline算法数据
        for alg_name, alg_results in baseline_results.items():
            if 'values' in alg_results and alg_results['values']:
                if alg_name not in algorithm_iteration_data:
                    algorithm_iteration_data[alg_name] = []

                # 提取baseline算法每轮的得分
                baseline_scores = alg_results['values']
                algorithm_iteration_data[alg_name].append(baseline_scores)
                max_iterations = max(max_iterations, len(baseline_scores))

    # 首先确定最大轮次数
    max_iterations = 0
    for counter_data_list in algorithm_iteration_data.values():
        for scores in counter_data_list:
            max_iterations = max(max_iterations, len(scores))

    # 按counter-轮次分组收集分数（每个counter的每个轮次收集该counter所有算法的分数）
    counter_iteration_grouped_scores = {}

    for counter_idx, result in enumerate(all_results):
        counter = result['counter']
        counter_iteration_grouped_scores[counter] = {}

        for iteration_idx in range(max_iterations):
            counter_iteration_grouped_scores[counter][iteration_idx] = []

            # 收集该counter该轮次的所有算法分数（排除Random）
            for alg_name, counter_data_list in algorithm_iteration_data.items():
                if alg_name != 'Random' and counter_idx < len(counter_data_list):
                    scores = counter_data_list[counter_idx]
                    if iteration_idx < len(scores):
                        counter_iteration_grouped_scores[counter][iteration_idx].append(scores[iteration_idx])




    # ========== 按counter级别归一化 ==========
    # 🔧 获取配置中需要排除的算法
    _, _, _, exclude_from_minmax = get_plot_config_from_baseline()

    # 步骤1: 按counter计算归一化所需的max值
    # 数据结构: counter_normalization[counter_idx] = {'max': x, 'min': y}
    counter_normalization = {}

    for counter_idx in range(len(all_results)):
        # 收集该counter所有算法（排除Random）的所有分数
        all_scores_for_counter = []

        for alg_name, counter_data_list in algorithm_iteration_data.items():
            if alg_name not in exclude_from_minmax and counter_idx < len(counter_data_list):
                scores = counter_data_list[counter_idx]
                all_scores_for_counter.extend(scores)

        # 计算该counter的max/min
        if all_scores_for_counter:
            counter_normalization[counter_idx] = {
                'max': max(all_scores_for_counter),
                'min': min(all_scores_for_counter)
            }
        else:
            counter_normalization[counter_idx] = {'max': 1.0, 'min': 0.0}

    # 步骤2: 对每个算法按counter级别归一化并计算平均值
    algorithm_average_curves = {}
    algorithm_max_curves = {}
    algorithm_min_curves = {}

    for alg_name, counter_data_list in algorithm_iteration_data.items():

        # 存储所有counter归一化后的曲线
        normalized_curves = []

        for counter_idx, scores in enumerate(counter_data_list):
            # 按counter级别归一化：使用该counter的max值
            norm_info = counter_normalization.get(counter_idx)
            normalized_scores = []
            for score in scores:
                if norm_info and norm_info['max'] > 0:
                    normalized_scores.append(score / norm_info['max'])
                else:
                    normalized_scores.append(0.0)

            normalized_curves.append(normalized_scores)

        # 🔧 [问题2修复] 计算平均曲线时只使用有数据的counter
        if normalized_curves:
            # 找到最长的曲线长度
            max_len = max(len(curve) for curve in normalized_curves)

            # 对每个位置计算平均值、标准误上下界（只使用有数据的counter）
            average_curve = []
            upper_curve = []  # 平均值 + 1倍标准误
            lower_curve = []  # 平均值 - 1倍标准误

            for i in range(max_len):
                # 🔧 [问题2修复] 只收集该位置有真实数据的值（不填充缺失值）
                values_at_position = []
                for curve in normalized_curves:
                    if i < len(curve):  # 只有当该curve在该位置有数据时才添加
                        values_at_position.append(curve[i])
                    # 注意：不再使用curve[-1]填充缺失数据

                if values_at_position:  # 确保有数据参与计算
                    n_valid = len(values_at_position)  # 🔧 使用实际参与计算的counter数量
                    mean_val = np.mean(values_at_position)
                    if n_valid > 1:
                        std_val = np.std(values_at_position, ddof=1)  # 计算样本标准差
                        se_val = std_val / np.sqrt(n_valid)  # 🔧 使用有效样本数计算标准误
                    else:
                        se_val = 0.0

                    average_curve.append(mean_val)
                    upper_curve.append(mean_val + se_val)  # 平均值 + 1倍标准误
                    lower_curve.append(mean_val - se_val)  # 平均值 - 1倍标准误
                # 注意：如果该位置没有数据，不添加任何值到曲线

            algorithm_average_curves[alg_name] = average_curve
            algorithm_max_curves[alg_name] = upper_curve  # 重用变量名，实际存储上界
            algorithm_min_curves[alg_name] = lower_curve  # 重用变量名，实际存储下界

    # ========== Min-Max归一化按counter级别 ==========
    algorithm_average_curves_minmax = {}
    algorithm_max_curves_minmax = {}
    algorithm_min_curves_minmax = {}

    for alg_name, counter_data_list in algorithm_iteration_data.items():

        # 存储所有counter的min-max归一化后的曲线
        minmax_normalized_curves = []

        for counter_idx, scores in enumerate(counter_data_list):
            # 按counter级别归一化：使用该counter的max/min值
            norm_info = counter_normalization.get(counter_idx)
            minmax_normalized_scores = []
            for score in scores:
                if norm_info and norm_info['max'] > norm_info['min']:
                    minmax_normalized_scores.append(
                        (score - norm_info['min']) / (norm_info['max'] - norm_info['min'])
                    )
                elif norm_info and norm_info['max'] == norm_info['min']:
                    minmax_normalized_scores.append(1.0)  # 所有值相同
                else:
                    minmax_normalized_scores.append(0.0)

            minmax_normalized_curves.append(minmax_normalized_scores)

        # 🔧 [问题2修复] 计算min-max归一化的平均曲线时只使用有数据的counter
        if minmax_normalized_curves:
            # 找到最长的曲线长度
            max_len = max(len(curve) for curve in minmax_normalized_curves)

            # 对每个位置计算平均值、标准误上下界（只使用有数据的counter）
            minmax_average_curve = []
            minmax_upper_curve = []  # 平均值 + 1倍标准误
            minmax_lower_curve = []  # 平均值 - 1倍标准误

            for i in range(max_len):
                # 🔧 [问题2修复] 只收集该位置有真实数据的值（不填充缺失值）
                values_at_position = []
                for curve in minmax_normalized_curves:
                    if i < len(curve):  # 只有当该curve在该位置有数据时才添加
                        values_at_position.append(curve[i])
                    # 注意：不再使用curve[-1]填充缺失数据

                if values_at_position:  # 确保有数据参与计算
                    n_valid = len(values_at_position)  # 🔧 使用实际参与计算的counter数量
                    mean_val = np.mean(values_at_position)
                    if n_valid > 1:
                        std_val = np.std(values_at_position, ddof=1)  # 计算样本标准差
                        se_val = std_val / np.sqrt(n_valid)  # 🔧 使用有效样本数计算标准误
                    else:
                        se_val = 0.0

                    minmax_average_curve.append(mean_val)
                    minmax_upper_curve.append(mean_val + se_val)  # 平均值 + 1倍标准误
                    minmax_lower_curve.append(mean_val - se_val)  # 平均值 - 1倍标准误
                # 如果该位置没有数据，不添加任何值到曲线

            algorithm_average_curves_minmax[alg_name] = minmax_average_curve
            algorithm_max_curves_minmax[alg_name] = minmax_upper_curve
            algorithm_min_curves_minmax[alg_name] = minmax_lower_curve
        else:
            algorithm_average_curves_minmax[alg_name] = []
            algorithm_max_curves_minmax[alg_name] = []
            algorithm_min_curves_minmax[alg_name] = []

    # 生成折线图
    plt.figure(figsize=(14, 10))

    # 🔧 从配置中获取绘图参数
    colors, markers, algorithms_with_range, _ = get_plot_config_from_baseline()

    # 绘制每个算法的平均曲线和SE范围（所有算法都显示）
    for alg_name, average_curve in algorithm_average_curves.items():
        x_values = list(range(len(average_curve)))
        upper_curve = algorithm_max_curves[alg_name]  # 实际存储的是平均值+标准误
        lower_curve = algorithm_min_curves[alg_name]  # 实际存储的是平均值-标准误

        color = colors.get(alg_name, '#999999')

        # 为所有算法绘制SE范围填充
        plt.fill_between(x_values, lower_curve, upper_curve,
                        color=color, alpha=0.2)

        # 绘制平均曲线（所有算法都有）
        label = alg_name

        plt.plot(x_values, average_curve,
                marker=markers.get(alg_name, 'o'),
                linestyle='-',
                linewidth=3,
                markersize=8,
                color=color,
                label=label,
                alpha=0.9,
                markerfacecolor='white',
                markeredgewidth=2)

    # 图表美化
    from POHF_parameters import DATA_CONFIG
    counter_count = DATA_CONFIG.get("counter_array_length", 40)
    plt.title(f'Counter Average Performance Curves\n(Normalized scores averaged across {counter_count} counters, all algorithms show mean±1SE range)',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Normalized Score', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')

    # 添加图例（调整位置以适应更多信息）
    plt.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.02, 0.98), ncol=1)

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 设置坐标轴 - 动态调整Y轴范围
    plt.xlim(0, max([len(curve) for curve in algorithm_average_curves.values()]) - 1)

    # 计算所有曲线的最小值和最大值，动态设置Y轴范围
    all_values = []
    for curve in algorithm_average_curves.values():
        all_values.extend(curve)

    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min

        # 添加5%的边距
        margin = max(0.05, y_range * 0.05)
        plt.ylim(y_min - margin, y_max + margin)
    else:
        plt.ylim(0, 1.05)

    plt.tight_layout()

    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建保存目录
    save_dir = os.path.join(PROJECT_ROOT, "past_average")
    os.makedirs(save_dir, exist_ok=True)

    # 使用传入的lamp_type参数，如果没有则尝试从配置读取
    if lamp_type is None:
        try:
            from POHF_parameters import PATH_CONFIG
            lamp_type = PATH_CONFIG.get('LaMP_type', 'unknown')
        except:
            lamp_type = 'unknown'

    base_filename = f"{get_filename_prefix(lamp_type)}_counter_average_curves_with_range_{timestamp}.pdf"

    # 确保文件名唯一，避免覆盖
    filename = get_unique_filename(save_dir, base_filename)

    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    except Exception as e:
        print(f"❌ [plot_counter_average_results] 保存图片失败: {e}")

    plt.close()

    generate_minmax_counter_average_plot(algorithm_average_curves_minmax, algorithm_max_curves_minmax, algorithm_min_curves_minmax, timestamp, counter_count, lamp_type)

    # 保存数据到文件以便复现图像
    # 使用传入的lamp_type参数（已在前面确定）
    base_data_filename = f"{get_filename_prefix(lamp_type)}_counter_average_data_{timestamp}.json"

    # 确保JSON文件名唯一，避免覆盖
    data_filename = get_unique_filename(save_dir, base_data_filename)

    # 准备要保存的数据
    # 🔧 [修复] 保存所有query的所有iteration数据，按query分组
    export_data = {}

    # 为每个counter收集所有算法的原始数据（包括所有query）
    for counter_idx, result in enumerate(all_results):
        counter = result['counter']
        counter_key = f"counter_{counter}"

        if counter_key not in export_data:
            export_data[counter_key] = {
                "metadata": {
                    "contextual_mode": result.get('contextual_mode', False),
                    "num_queries": result.get('num_queries', 1),
                    "total_iterations": result.get('total_iterations', 0)
                },
                "all_iterations": {},  # 所有iteration的数据（兼容旧格式）
                "by_query": {}  # 按query分组的数据（新格式）
            }

        # 收集该counter的所有算法原始数据（所有iteration）
        for alg_name, counter_data_list in algorithm_iteration_data.items():
            if counter_idx < len(counter_data_list):
                export_data[counter_key]["all_iterations"][alg_name] = counter_data_list[counter_idx]

        # 🔧 [新增] 按query分组保存数据
        best_instruction_over_iter = result.get('best_instruction_over_iter', [])
        baseline_results_data = result.get('baseline_results', {})

        if best_instruction_over_iter:
            # 按query_idx分组主算法数据
            query_grouped_main = {}
            for item in best_instruction_over_iter:
                if len(item) >= 4:
                    # 4元组格式: (t, instruction, score, query_idx)
                    t, instruction, score, query_idx = item
                elif len(item) >= 3:
                    # 3元组格式: (t, instruction, score) - 假设query_idx=0
                    t, instruction, score = item
                    query_idx = 0
                else:
                    continue

                query_key = f"query_{query_idx}"
                if query_key not in query_grouped_main:
                    query_grouped_main[query_key] = []
                query_grouped_main[query_key].append(score)

            # 保存主算法的按query分组数据
            algorithm_name = result.get('algorithm', 'POHF-InfoGain')
            for query_key, scores in query_grouped_main.items():
                if query_key not in export_data[counter_key]["by_query"]:
                    export_data[counter_key]["by_query"][query_key] = {}
                export_data[counter_key]["by_query"][query_key][algorithm_name] = scores

            # 🔧 [新增] 按query分组保存baseline算法数据
            # 需要根据主算法的query分组信息来分割baseline数据
            if baseline_results_data and query_grouped_main:
                # 计算每个query的iteration数量
                query_iter_counts = {k: len(v) for k, v in query_grouped_main.items()}

                for bl_name, bl_data in baseline_results_data.items():
                    if 'values' in bl_data and bl_data['values']:
                        bl_values = bl_data['values']
                        offset = 0
                        for query_key in sorted(query_grouped_main.keys(), key=lambda x: int(x.split('_')[1])):
                            n_iters = query_iter_counts.get(query_key, 0)
                            if offset + n_iters <= len(bl_values):
                                query_bl_values = bl_values[offset:offset + n_iters]
                            else:
                                query_bl_values = bl_values[offset:]

                            if query_bl_values:
                                if query_key not in export_data[counter_key]["by_query"]:
                                    export_data[counter_key]["by_query"][query_key] = {}
                                export_data[counter_key]["by_query"][query_key][bl_name] = query_bl_values

                            offset += n_iters

    # 写入JSON文件
    import json
    try:
        with open(data_filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"   📄 [JSON] 保存完整数据到: {data_filename}")
    except Exception as e:
        print(f"❌ [plot_counter_average_results] 保存JSON数据失败: {e}")




def plot_pohf_results_with_baselines(pohf_results, baseline_results=None, save_to_file=True, show_plot=False, filename_prefix=None):
    """
    绘制POHF结果图并与baseline算法进行比较

    Args:
        pohf_results: POHF运行结果 [(iteration, instruction, score), ...]
        baseline_results: baseline算法结果字典 {'Random': {'values': [...], 'best_values': [...]}, ...}
        save_to_file: 是否保存到文件
        show_plot: 是否显示图片（需要GUI环境）
        filename_prefix: 文件名前缀，用于区分不同的数据组
    """
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    if not pohf_results:
        return

    # 提取POHF数据
    pohf_x = [p[0] for p in pohf_results]  # iterations
    pohf_y = [p[2] for p in pohf_results]  # scores

    # 创建图形
    plt.figure(figsize=(14, 10))

    # 绘制POHF结果 (根据是否启用Information Gain来确定标签和颜色)
    from POHF_parameters import POHF_CONFIG
    info_gain_enabled = POHF_CONFIG.get("info_gain_enabled", False)

    if info_gain_enabled:
        pohf_label = 'POHF-InfoGain'
        pohf_color = '#8B2E86'
        pohf_marker = '*'
    else:
        pohf_label = 'POHF'
        pohf_color = '#2E86AB'
        pohf_marker = 'o'

    plt.plot(pohf_x, pohf_y, marker=pohf_marker, linestyle='-', linewidth=3, markersize=8,
             color=pohf_color, markerfacecolor='#A23B72', markeredgecolor='white',
             markeredgewidth=2, label=pohf_label, alpha=0.9)

    # 绘制baseline算法结果
    if baseline_results:
        # 🔧 从配置中获取绘图参数
        colors, markers, _, _ = get_plot_config_from_baseline()

        for alg_name, results in baseline_results.items():
            if 'values' in results and results['values']:
                x_baseline = list(range(len(results['values'])))
                y_baseline = results['values']  # 直接使用values数组进行绘图

                # 为不同算法添加描述性标签
                if alg_name == 'Random':
                    label = f'{alg_name} (Current Max)'
                elif alg_name == 'LinearDuelingBandits':
                    label = f'{alg_name} (Greedy Score)'
                elif alg_name == 'DoubleTS':
                    label = f'{alg_name} (Current Max)'
                else:
                    label = alg_name

                plt.plot(x_baseline, y_baseline,
                        marker=markers.get(alg_name, 'o'),
                        linestyle='--',
                        linewidth=2,
                        markersize=6,
                        color=colors.get(alg_name, '#999999'),
                        label=label,
                        alpha=0.8)

    # 设置图形样式
    plt.title('POHF vs Baseline Algorithms Performance Comparison',
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Best Score', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9)

    # 美化坐标轴
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()

    if save_to_file:
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = os.path.join(PROJECT_ROOT, "plots")
        os.makedirs(save_dir, exist_ok=True)

        if filename_prefix:
            # 使用自定义前缀
            filename = os.path.join(save_dir, f"{filename_prefix}_comparison_{timestamp}.pdf")
        else:
            # 使用默认前缀
            filename = os.path.join(save_dir, f"pohf_baseline_comparison_{timestamp}.pdf")

        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

        # 如果没有自定义前缀，也保存一个固定名称的版本
        if not filename_prefix:
            latest_filename = os.path.join(save_dir, "pohf_baseline_comparison_latest.pdf")
            plt.savefig(latest_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

    if show_plot:
        try:
            plt.show()
        except:
            pass
    else:
        plt.close()  # 关闭图形以释放内存


def plot_pohf_results(outcome, save_to_file=True, show_plot=False, filename_prefix=None):
    """
    绘制POHF结果图并保存到文件

    Args:
        outcome: POHF运行结果 [(iteration, instruction, score), ...]
        save_to_file: 是否保存到文件
        show_plot: 是否显示图片（需要GUI环境）
        filename_prefix: 文件名前缀，用于区分不同的数据组
    """
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    if not outcome:
        return

    # 提取数据
    x = [p[0] for p in outcome]  # iterations
    y = [p[2] for p in outcome]  # scores

    # 创建图形
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=8,
             color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=2)

    # 设置标签和标题
    plt.xlabel("Iterations", fontsize=14, fontweight='bold')
    plt.ylabel("Reward Score", fontsize=14, fontweight='bold')
    plt.title("POHF Algorithm: Reward Evolution Over Iterations", fontsize=16, fontweight='bold', pad=20)

    # 设置纵坐标范围，最低值始终为0.0
    max_score = max(y)
    plt.ylim(0.0, max_score * 1.1)  # 最低值0.0，最高值留10%余量

    # 美化图形
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)

    # 添加数值标注
    for xi, yi in zip(x, y):
        plt.annotate(f'{yi:.4f}', (xi, yi), textcoords="offset points",
                    xytext=(0,15), ha='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

    # 添加统计信息
    min_score = min(y)
    avg_score = sum(y) / len(y)

    stats_text = f"Max: {max_score:.4f}\\nMin: {min_score:.4f}\\nAvg: {avg_score:.4f}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
             fontsize=10)

    plt.tight_layout()

    if save_to_file:
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = os.path.join(PROJECT_ROOT, "plots")
        os.makedirs(save_dir, exist_ok=True)

        if filename_prefix:
            # 使用自定义前缀
            filename = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.pdf")
        else:
            # 使用默认前缀
            filename = os.path.join(save_dir, f"pohf_results_{timestamp}.pdf")

        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

        # 如果没有自定义前缀，也保存一个固定名称的版本
        if not filename_prefix:
            latest_filename = os.path.join(save_dir, "pohf_latest_results.pdf")
            plt.savefig(latest_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

    if show_plot:
        try:
            plt.show()
        except:
            pass
    else:
        plt.close()  # 关闭图形以释放内存


def plot_dual_algorithm_comparison(algorithm_results, counter, lamp_type=None):
    """
    生成完整算法对比图，显示POHF、POHF-InfoGain、Random和LinearBandit的性能对比

    Args:
        algorithm_results: 当前counter的所有算法结果列表
        counter: 当前counter值
        lamp_type: LaMP数据集类型（用于文件命名）
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 10))

    # 🔧 从配置中获取绘图参数
    colors, markers, _, _ = get_plot_config_from_baseline()

    # 用于跟踪已绘制的算法，避免重复
    plotted_algorithms = set()
    plotted_baselines = set()

    # 绘制主算法的学习曲线
    for result in algorithm_results:
        algorithm_name = result['algorithm']

        # 防止重复绘制同一个算法
        if algorithm_name in plotted_algorithms:
            continue

        best_instruction_over_iter = result['best_instruction_over_iter']

        # 🔧 [修复] 提取分数数据，兼容3元组和4元组格式
        # 3元组: (t, instruction, score)
        # 4元组: (t, instruction, score, query_idx)
        if best_instruction_over_iter and isinstance(best_instruction_over_iter[0], tuple):
            # 提取分数（第三个元素）
            best_values = [item[2] for item in best_instruction_over_iter if len(item) >= 3]
        else:
            # 如果数据格式不同，直接使用
            best_values = best_instruction_over_iter

        x_values = list(range(len(best_values)))
        color = colors.get(algorithm_name, '#999999')
        marker = markers.get(algorithm_name, 'o')

        plt.plot(x_values, best_values,
                marker=marker, linestyle='-', linewidth=3, markersize=8,
                color=color, label=algorithm_name, alpha=0.9)

        plotted_algorithms.add(algorithm_name)

        # 绘制baseline算法结果（只绘制一次）
        baseline_results = result.get('baseline_results', {})
        if baseline_results:
            for baseline_name, baseline_data in baseline_results.items():
                if baseline_name not in plotted_baselines and 'values' in baseline_data and baseline_data['values']:
                    baseline_x = list(range(len(baseline_data['values'])))
                    baseline_y = baseline_data['values']

                    baseline_color = colors.get(baseline_name, '#999999')
                    baseline_marker = markers.get(baseline_name, 'o')

                    plt.plot(baseline_x, baseline_y,
                            marker=baseline_marker, linestyle='--', linewidth=2, markersize=6,
                            color=baseline_color, label=baseline_name, alpha=0.8)

                    plotted_baselines.add(baseline_name)

    # 设置图形样式
    plt.title(f'Algorithm Performance Comparison (Counter {counter})',
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Best Score', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)

    # 设置坐标轴样式
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)

    # 保存对比图
    # 使用传入的lamp_type参数，如果没有则尝试从配置读取
    if lamp_type is None:
        try:
            from POHF_parameters import PATH_CONFIG
            lamp_type = PATH_CONFIG.get('LaMP_type', 'unknown')
        except:
            lamp_type = 'unknown'

    save_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{get_filename_prefix(lamp_type)}_algorithm_comparison_counter_{counter}.pdf")

    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()



def plot_second_arm_selection_stats(second_arm_selections, counter, total_arms, lamp_type=None):
    """
    生成第二个arm选择统计图，显示每个arm被选择的频次和轮次分布

    Args:
        second_arm_selections: 每轮选择的第二个arm列表
        counter: 当前counter值
        total_arms: 总arm数量
        lamp_type: LaMP数据集类型（用于文件命名）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    if not second_arm_selections:
        return

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图像，包含两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1: 每个arm被选择的频次统计
    arm_counts = Counter(second_arm_selections)
    arms = list(range(total_arms))
    frequencies = [arm_counts.get(arm, 0) for arm in arms]

    bars = ax1.bar(arms, frequencies, alpha=0.7, color='steelblue', edgecolor='navy', linewidth=0.8)
    ax1.set_xlabel('Arm Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Selection Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Second Arm Selection Frequency (Counter {counter})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 在柱状图上添加数值标签
    for bar, freq in zip(bars, frequencies):
        if freq > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(freq), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 设置x轴刻度
    ax1.set_xticks(arms)
    ax1.set_xlim(-0.5, total_arms - 0.5)

    # 子图2: 轮次选择时间线
    iterations = list(range(len(second_arm_selections)))
    colors = plt.cm.tab20(np.linspace(0, 1, total_arms))

    # 为每个arm分配颜色
    arm_colors = [colors[arm] for arm in second_arm_selections]

    scatter = ax2.scatter(iterations, second_arm_selections, c=arm_colors,
                         alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Selected Arm Index', fontsize=12, fontweight='bold')
    ax2.set_title(f'Second Arm Selection Timeline (Counter {counter})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 设置y轴刻度
    ax2.set_yticks(arms)
    ax2.set_ylim(-0.5, total_arms - 0.5)

    # 添加统计信息文本
    total_selections = len(second_arm_selections)
    unique_arms = len(arm_counts)
    most_selected_arm = max(arm_counts, key=arm_counts.get) if arm_counts else 0
    most_selected_count = arm_counts[most_selected_arm] if arm_counts else 0

    stats_text = f"""Statistics:
    Total Iterations: {total_selections}
    Unique Arms Selected: {unique_arms}/{total_arms}
    Most Selected Arm: {most_selected_arm} ({most_selected_count} times)
    Selection Rate: {most_selected_count/total_selections*100:.1f}%"""

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 调整布局
    plt.tight_layout()

    # 使用传入的lamp_type参数，如果没有则尝试从配置读取
    if lamp_type is None:
        try:
            from POHF_parameters import PATH_CONFIG
            lamp_type = PATH_CONFIG.get('LaMP_type', 'unknown')
        except:
            lamp_type = 'unknown'

    save_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{get_filename_prefix(lamp_type)}_second_arm_selection_counter_{counter}.pdf")

    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def plot_query_progress(query_progress_data, counter, lamp_type=None):
    """
    生成跨Query进步图：显示各算法相对于Random baseline的提升

    Args:
        query_progress_data: 每个query的进步数据
            {
                'query_indices': [0, 1, 2, 3, ...],  # query索引（从0开始，包含所有query）
                'algorithms': {
                    'POHF-InfoGain': [val1, val2, ...],  # 每个query最后一轮值 - Random平均值
                    'DoubleTS': [...],
                    ...
                }
            }
        counter: 当前counter值
        lamp_type: LaMP数据集类型（用于文件命名）
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not query_progress_data or not query_progress_data.get('query_indices'):
        return

    query_indices = query_progress_data['query_indices']
    algorithms_data = query_progress_data['algorithms']

    if not algorithms_data:
        return

    plt.figure(figsize=(12, 8))

    # 🔧 从配置中获取绘图参数
    colors, markers, _, _ = get_plot_config_from_baseline()

    # 绘制每个算法的进步曲线
    for alg_name, values in algorithms_data.items():
        if not values or alg_name == 'Random':  # 不绘制Random（因为它是基准线）
            continue

        color = colors.get(alg_name, '#999999')
        marker = markers.get(alg_name, 'o')

        plt.plot(query_indices, values,
                marker=marker, linestyle='-', linewidth=2.5, markersize=8,
                color=color, label=alg_name, alpha=0.9,
                markerfacecolor='white', markeredgewidth=2)

    # 绘制y=0的参考线（Random baseline）
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random Baseline')

    # 设置图形样式
    plt.title(f'Algorithm Progress Across Queries (Counter {counter})\n(Value = Final Score - Random Average)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Query Index', fontsize=14, fontweight='bold')
    plt.ylabel('Improvement over Random', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)

    # 设置x轴刻度为整数
    plt.xticks(query_indices)

    # 设置坐标轴样式
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()

    # 保存图片
    if lamp_type is None:
        try:
            from POHF_parameters import PATH_CONFIG
            lamp_type = PATH_CONFIG.get('LaMP_type', 'unknown')
        except:
            lamp_type = 'unknown'

    save_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{get_filename_prefix(lamp_type)}_query_progress_counter_{counter}.pdf")

    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   📊 [Query Progress] 保存跨Query进步图: {filename}")


def plot_query_progress_counter_average(all_query_progress_data, lamp_type=None):
    """
    生成跨Query进步图的Counter Average版本

    Args:
        all_query_progress_data: 所有counter的query进步数据列表
            [
                {'counter': 0, 'query_indices': [...], 'algorithms': {...}},
                {'counter': 1, 'query_indices': [...], 'algorithms': {...}},
                ...
            ]
        lamp_type: LaMP数据集类型
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    if not all_query_progress_data:
        return

    # 收集所有算法名称和query索引范围
    all_algorithms = set()
    min_query_idx = float('inf')
    max_query_idx = -1
    for data in all_query_progress_data:
        if data and 'algorithms' in data:
            all_algorithms.update(data['algorithms'].keys())
            if data.get('query_indices'):
                min_query_idx = min(min_query_idx, min(data['query_indices']))
                max_query_idx = max(max_query_idx, max(data['query_indices']))

    # 🔧 [修复] 支持从query_0开始的情况
    if not all_algorithms or max_query_idx < 0:
        return

    # 如果没有找到有效的min，设置为0
    if min_query_idx == float('inf'):
        min_query_idx = 0

    # 收集每个算法在每个query位置的所有值（从min_query_idx到max_query_idx）
    algorithm_query_values = {alg: {q: [] for q in range(min_query_idx, max_query_idx + 1)} for alg in all_algorithms}

    for data in all_query_progress_data:
        if not data or 'algorithms' not in data:
            continue
        query_indices = data.get('query_indices', [])
        for alg_name, values in data['algorithms'].items():
            for i, q_idx in enumerate(query_indices):
                if i < len(values) and q_idx in algorithm_query_values.get(alg_name, {}):
                    algorithm_query_values[alg_name][q_idx].append(values[i])

    # 计算平均值和标准误
    algorithm_average = {}
    algorithm_se_upper = {}
    algorithm_se_lower = {}

    for alg_name in all_algorithms:
        if alg_name == 'Random':
            continue
        avg_values = []
        upper_values = []
        lower_values = []
        valid_queries = []

        for q_idx in range(min_query_idx, max_query_idx + 1):  # 🔧 从min_query_idx开始
            values = algorithm_query_values[alg_name].get(q_idx, [])
            if values:
                mean_val = np.mean(values)
                if len(values) > 1:
                    std_val = np.std(values, ddof=1)
                    se_val = std_val / np.sqrt(len(values))
                else:
                    se_val = 0.0
                avg_values.append(mean_val)
                upper_values.append(mean_val + se_val)
                lower_values.append(mean_val - se_val)
                valid_queries.append(q_idx)

        if avg_values:
            algorithm_average[alg_name] = (valid_queries, avg_values)
            algorithm_se_upper[alg_name] = upper_values
            algorithm_se_lower[alg_name] = lower_values

    if not algorithm_average:
        return

    # 绘图
    plt.figure(figsize=(14, 10))

    colors, markers, _, _ = get_plot_config_from_baseline()

    for alg_name, (query_indices, avg_values) in algorithm_average.items():
        color = colors.get(alg_name, '#999999')
        marker = markers.get(alg_name, 'o')
        upper = algorithm_se_upper[alg_name]
        lower = algorithm_se_lower[alg_name]

        # 绘制SE范围
        plt.fill_between(query_indices, lower, upper, color=color, alpha=0.2)

        # 绘制平均曲线
        plt.plot(query_indices, avg_values,
                marker=marker, linestyle='-', linewidth=3, markersize=8,
                color=color, label=alg_name, alpha=0.9,
                markerfacecolor='white', markeredgewidth=2)

    # 绘制y=0的参考线
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random Baseline')

    # 设置图形样式
    counter_count = len(all_query_progress_data)
    plt.title(f'Query Progress Counter Average\n(Averaged across {counter_count} counters, showing mean±1SE)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Query Index', fontsize=14, fontweight='bold')
    plt.ylabel('Improvement over Random', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存图片
    if lamp_type is None:
        try:
            from POHF_parameters import PATH_CONFIG
            lamp_type = PATH_CONFIG.get('LaMP_type', 'unknown')
        except:
            lamp_type = 'unknown'

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PROJECT_ROOT, "past_average")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{get_filename_prefix(lamp_type)}_query_progress_counter_average_{timestamp}.pdf")

    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   📊 [Query Progress Average] 保存跨Query进步Counter Average图: {filename}")


if __name__ == '__main__':
    # 使用配置文件运行POHF算法
    from POHF_parameters import get_all_configs, print_config_summary

    # 下载必要的NLTK数据
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    # 打印配置摘要
    print_config_summary()

    # 获取配置并运行
    config = get_all_configs()

    # 使用异步运行
    import asyncio
    all_results = asyncio.run(run(config=config))
