import os, sys
import math
import warnings

warnings.filterwarnings("ignore", message="Extension saving to grad_batch")
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")

from rouge_score import rouge_scorer
import hashlib
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_Completed
import traceback

COMPARISON_CACHE = {}

def _swap_comparison_result(result):
    response1, response2, score_1, score_2, score, preference, p_ = result
    new_response1, new_response2 = response2, response1
    new_score_1, new_score_2 = score_2, score_1
    new_preference = 1 - preference if preference is not None else None
    new_p_ = 1 - p_ if p_ is not None else None
    new_score = new_score_1 if new_preference == 1 else new_score_2
    return (new_response1, new_response2, new_score_1, new_score_2, new_score, new_preference, new_p_)

def clear_comparison_cache():
    global COMPARISON_CACHE
    COMPARISON_CACHE = {}

def get_comparison_cache_stats():
    return len(COMPARISON_CACHE)

import matplotlib
matplotlib.use('Agg')

cwd = os.getcwd()
sys.path.append(cwd)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from information_second_term import PairwiseInformationManager, ContextualPairwiseInformationManager

def is_lamp_dataset(lamp_type: int = None) -> bool:
    if lamp_type is None:
        lamp_type = int(os.environ.get('POHF_LAMP_TYPE', 0))

    return lamp_type in [4, 5, 8, 9, 10]

_query_similarity_cache = {}

def clear_query_similarity_cache():
    global _query_similarity_cache
    _query_similarity_cache.clear()

def compute_query_weight(data_query_idx: int, current_query_idx: int,
                         query_embeddings: dict, training_config: dict) -> float:
    query_decay_enabled = training_config.get("query_decay_enabled", False)
    query_similarity_enabled = training_config.get("query_similarity_enabled", False)
    if not query_decay_enabled and not query_similarity_enabled:
        return 1.0

    global _query_similarity_cache

    query_decay_gamma = training_config.get("query_decay_gamma", 0.8)

    if query_decay_enabled:
        decay_weight = query_decay_gamma ** (current_query_idx - data_query_idx)
    else:
        decay_weight = 1.0

    query_similarity_enabled = training_config.get("query_similarity_enabled", False)

    if query_similarity_enabled and data_query_idx != current_query_idx:
        cache_key = (data_query_idx, current_query_idx)
        if cache_key in _query_similarity_cache:
            similarity_weight = _query_similarity_cache[cache_key]
            return decay_weight * similarity_weight

        steepness = training_config.get("query_similarity_steepness", 11.0)
        midpoint = training_config.get("query_similarity_midpoint", 0.5)

        current_emb = query_embeddings.get(current_query_idx)
        data_emb = query_embeddings.get(data_query_idx)

        if current_emb is not None and data_emb is not None:
            if hasattr(current_emb, 'cpu'):
                current_emb = current_emb.cpu().numpy()
            if hasattr(data_emb, 'cpu'):
                data_emb = data_emb.cpu().numpy()

            current_emb = current_emb.flatten()
            data_emb = data_emb.flatten()

            norm1 = np.linalg.norm(current_emb)
            norm2 = np.linalg.norm(data_emb)

            if norm1 > 1e-8 and norm2 > 1e-8:
                cosine_sim = np.dot(current_emb, data_emb) / (norm1 * norm2)
                cosine_sim = max(0.0, float(cosine_sim))
            else:
                cosine_sim = 0.0

            similarity_weight = 1.0 / (1.0 + np.exp(-(cosine_sim - midpoint) * steepness))
        else:
            similarity_weight = 1.0

        _query_similarity_cache[cache_key] = similarity_weight
    else:
        similarity_weight = 1.0

    return decay_weight * similarity_weight

def get_input_dim_for_dataset(lamp_type: int = None, config: dict = None) -> int:
    try:
        from IDS_TAP_parameters.py import CONTEXTUAL_BANDIT_CONFIG
        contextual_dim = CONTEXTUAL_BANDIT_CONFIG.get("contextual_input_dim", 2048)
    except ImportError:
        contextual_dim = 2048

    return contextual_dim

def should_use_contextual_mode(lamp_type: int = None) -> bool:
    return True

_response_cache = {}

import os
import torch

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

def detect_and_setup_gpu():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    return torch.device("cuda:0")

def get_device():
    if not hasattr(get_device, '_device'):
        get_device._device = detect_and_setup_gpu()
    return get_device._device

def reset_device():
    if hasattr(get_device, '_device'):
        delattr(get_device, '_device')

detected_device = None

tkwargs = {
    "device": torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
    "dtype": torch.float32,
}
import torch.nn.functional as F

from load_data import load_templated_data
import nltk

from openai import OpenAI, AsyncOpenAI
import asyncio

client = OpenAI(
    api_key="YOUR_API_KEY_HERE",
    base_url="YOUR_OPENAI_BASE_URL"
)

async_client = AsyncOpenAI(
    api_key="YOUR_API_KEY_HERE",
    base_url="YOUR_OPENAI_BASE_URL"
)

def get_dataset_name(lamp_type):
    if lamp_type == 0:
        return "ultrachat"
    elif lamp_type == -1:
        return "wildchat"
    elif lamp_type == -2:
        return "prefeval"
    else:
        return f"lamp{lamp_type}"

def get_filename_prefix(lamp_type):

    if isinstance(lamp_type, str):
        if lamp_type in ['ultrachat', 'wildchat', 'prefeval']:
            return lamp_type
        elif lamp_type == 'unknown':
            return 'unknown'
        else:
            return lamp_type

    if lamp_type == 0:
        return "ultrachat"
    elif lamp_type == -1:
        return "wildchat"
    elif lamp_type == -2:
        return "prefeval"
    else:
        return f"lamp{lamp_type}"

def prompt_reformer(input_data, instruction_index, summary_index, lamp_type=None, query_index=0):
    if lamp_type == -2:

        conversation_history = ""
        if isinstance(input_data[5], list):
            for i, turn in enumerate(input_data[5]):
                conversation_history += f"{turn}\n"
        else:
            conversation_history = str(input_data[5])

        prompt = "\n".join([
            "### Task Instruction:",
            input_data[2],
            "",
            "### Current Question:",
            str(input_data[1]) if input_data[1] else "",
            "",
            "### Conversation History:",
            conversation_history.strip(),
            "",
            "### User Personality Profile:",
            input_data[3][summary_index],  
            "",
            "### Key Characteristics:",
            str(input_data[4]) if input_data[4] else "",
            "",
            "### Output Requirement:",
            "Based on the conversation history and user profile, generate a response that: "
            "1. Directly addresses the current question "
            "2. Is consistent with the personality traits shown in the conversation history "
            "3. Supplement by Reflecting the user's communication style and preferences "
            "Please provide only the response below:"
        ])

        return prompt

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
            input_data[3][summary_index],
            "",
            "### Output Requirement:",
            "Based on the conversation history and user profile, predict what the user will ask next. "
            "You should prioritize task accuracy and relevance, while refer to the Personality and Style Description for writing tone. "
            "Please provide only the predicted user query below:"
        ])

        return prompt
    query_data = input_data[1]
    if isinstance(query_data, list):
        if query_index < len(query_data):
            current_task = str(query_data[query_index])
        else:
            current_task = str(query_data[0]) if len(query_data) > 0 else ""
    else:
        current_task = str(query_data) if query_data else ""

    important_words = input_data[4] if isinstance(input_data[4], str) else ", ".join(str(w) for w in input_data[4]) if isinstance(input_data[4], list) else str(input_data[4])

    ranked_entries_data = input_data[5]

    if isinstance(ranked_entries_data, list) and len(ranked_entries_data) > 0:
        if isinstance(ranked_entries_data[0], list):
            if query_index < len(ranked_entries_data):
                current_ranked_entries = ranked_entries_data[query_index]
            else:
                current_ranked_entries = ranked_entries_data[0]
        else:
            current_ranked_entries = ranked_entries_data
    else:
        current_ranked_entries = []

    def truncate_entry(entry, max_chars=1000000):
        if isinstance(entry, dict):
            text = entry.get('text', str(entry))
        else:
            text = str(entry)
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    history_context_str = "\n".join([str(entry) for entry in current_ranked_entries]) if current_ranked_entries else ""

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
        "### Important Words from history:",
        important_words,
        "",
        "### History context:",
        history_context_str,
        "",
        "### Output Requirement:",
        "Complete the task according to the Task Instruction and Current Task Details. "
        "You should prioritize task accuracy and relevance, while refer to the Personality and Style Description for writing tone. "
        "Please provide only the required response below:"
    ])

    return prompt

async def response_generator_async(prompt, config=None, personality_description=None):
    global _response_cache

    if config is None:
        from IDS_TAP_parameters.py import LLM_CONFIG
        config = LLM_CONFIG

    cache_key_content = prompt + (personality_description or "")
    if config.get("use_cache", True):
        prompt_hash = hashlib.md5(cache_key_content.encode()).hexdigest()
        if prompt_hash in _response_cache:
            return _response_cache[prompt_hash]
    else:
        prompt_hash = None

    system_content = (
        "You are a helpful assistant. Your task is to provide a helpful and accurate response "
        "to the user's request based on the given instruction and context."
    )

    from IDS_TAP_parameters.py import API_CONFIG
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
                    print(f"⚠️ [response_generator_async] Safety prompt retry failed: {retry_e}")
                    return f"Error generating response: {str(e)}"

            is_retryable = any(keyword in error_message for keyword in [
                'timeout', 'connection', 'rate', 'limit', '429', '500', '502', '503', '504',
                'expecting value', 'json', 'decode', 'reset', 'closed'
            ])

            if is_retryable and attempt < max_retries - 1:
                wait_time = retry_delay * (retry_backoff ** attempt)
                print(f"⚠️ [response_generator_async] Call failed (Attempt {attempt + 1}/{max_retries}) [{error_type_name}]: {e}")
                print(f"   Waiting {wait_time:.1f}s  before retry...", flush=True)
                await asyncio.sleep(wait_time)
            elif attempt < max_retries - 1:
                wait_time = retry_delay
                print(f"⚠️ [response_generator_async] Atypical error (Attempt {attempt + 1}/{max_retries}) [{error_type_name}]: {e}")
                print(f"   Waiting {wait_time:.1f}s  before retry...", flush=True)
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ [response_generator_async] Max retries reached ({max_retries}) [{error_type_name}]: {e}")

    return f"Error generating response: {str(last_error)}"

def response_generator(prompt, config=None, personality_description=None):
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, response_generator_async(prompt, config, personality_description))
            return future.result()
    except RuntimeError:
        return asyncio.run(response_generator_async(prompt, config, personality_description))

import numpy as np

async def rouge_score_comparison_async(ground_truth, response1, response2, rouge_type, LLM_as_judge=False, arm1_idx=None, arm2_idx=None, ground_truth_index=0, counter=None, query_idx=None):
    from IDS_TAP_parameters.py import EXPERIMENT_CONFIG
    random_seed = EXPERIMENT_CONFIG.get("random_seed", 0)
    np.random.seed(random_seed)

    if isinstance(ground_truth, list):
        if ground_truth_index < len(ground_truth):
            ground_truth = str(ground_truth[ground_truth_index])
        elif len(ground_truth) > 0:
            ground_truth = str(ground_truth[0])
        else:
            ground_truth = ""
    elif not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)

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

        from IDS_TAP_parameters.py import ROUGE_CONFIG
        a = ROUGE_CONFIG.get("a", 2.0)
        b = ROUGE_CONFIG.get("b", 0.01)
        beta_min = ROUGE_CONFIG.get("beta_min", 1.0)
        beta_max = ROUGE_CONFIG.get("beta_max", 50.0)

        beta = a / (b + (scores1 + scores2) / 2)
        beta = np.clip(beta, beta_min, beta_max)

        p_ = 1 / (1 + np.exp(-beta * (scores1 - scores2)))
        if scores1 == scores2:
            preference = 1
        else:
            preference = np.random.binomial(1, p_)

        score = scores1 if preference == 1 else scores2
        return scores1, scores2, score, preference, p_
    else:
        from IDS_TAP_parameters.py import LLM_CONFIG
        import re

        system_role = (
            "You are an expert evaluator. Your task is to compare two responses and determine which one is better "
            "based on the given ground truth. The PRIMARY consideration is: text similarity, content accuracy, structure alignment, and semantic coherence. "
            "The SECONDARY consideration is: language style and personality characteristics. "
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

        from IDS_TAP_parameters.py import API_CONFIG
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
            "temperature": 0.0,
            "top_p": 1.0,
        }

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

                if result == "1":
                    preference = 1
                elif result == "0":
                    preference = 0
                else:
                    match = re.search(r'[01]', result)
                    if match:
                        preference = int(match.group())
                    else:
                        preference = 1

                break

            except Exception as e:
                error_type_name = type(e).__name__

                is_retryable = any(keyword in str(e).lower() for keyword in [
                    'timeout', 'connection', 'rate', 'limit', '429', '500', '502', '503', '504',
                    'expecting value', 'json', 'decode'
                ])

                if attempt < max_retries - 1:
                    wait_time = retry_delay * (retry_backoff ** attempt)
                    print(f"⚠️ [rouge_score_comparison_async] LLM call failed (Attempt {attempt + 1}/{max_retries}) [{error_type_name}]: {e}")
                    print(f"   Waiting {wait_time:.1f}s  before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"❌ [rouge_score_comparison_async] LLM judgment failed，Max retries reached ({max_retries}) [{error_type_name}]: {e}")
                    preference = 1

        if preference is None:
            print(f"❌ [rouge_score_comparison_async] All retries failed，Using default preference=1")
            preference = 1

        score_1 = 1 if preference == 1 else 0
        score_2 = 0 if preference == 1 else 1
        return score_1, score_2, None, preference, None

def rouge_score_comparison(ground_truth, response1, response2, rouge_type, LLM_as_judge=False, arm1_idx=None, arm2_idx=None):
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
    if personality_descriptions is None:
        tasks = [response_generator_async(prompt, config) for prompt in prompts]
    else:
        tasks = [response_generator_async(prompt, config, pd) for prompt, pd in zip(prompts, personality_descriptions)]
    return await asyncio.gather(*tasks)

async def generate_and_compare_async(prompt1, prompt2, ground_truth, config, LLM_as_judge=False, arm1_idx=None, arm2_idx=None, personality1=None, personality2=None, ground_truth_index=0, counter=None, query_idx=None):
    global COMPARISON_CACHE

    use_cache = (counter is not None and query_idx is not None and
                 arm1_idx is not None and arm2_idx is not None)

    if use_cache:
        if arm1_idx <= arm2_idx:
            cache_key = (counter, query_idx, arm1_idx, arm2_idx)
            swapped = False
        else:
            cache_key = (counter, query_idx, arm2_idx, arm1_idx)
            swapped = True

        if cache_key in COMPARISON_CACHE:
            cached_result = COMPARISON_CACHE[cache_key]
            if swapped:
                result = _swap_comparison_result(cached_result)
                print(f"    💾 [Cache hit-swap] counter={counter}, query={query_idx}, arm1={arm1_idx}, arm2={arm2_idx}")
            else:
                result = cached_result
                print(f"    💾 [Cache hit] counter={counter}, query={query_idx}, arm1={arm1_idx}, arm2={arm2_idx}")
            return result

    response1, response2 = await asyncio.gather(
        response_generator_async(prompt1, config, personality1),
        response_generator_async(prompt2, config, personality2)
    )

    score_1, score_2, score, preference, p_ = await rouge_score_comparison_async(
        ground_truth, response1, response2, 'rougeL',
        LLM_as_judge=LLM_as_judge, arm1_idx=arm1_idx, arm2_idx=arm2_idx,
        ground_truth_index=ground_truth_index,
        counter=counter, query_idx=query_idx
    )

    result = (response1, response2, score_1, score_2, score, preference, p_)

    if use_cache:
        if arm1_idx <= arm2_idx:
            cache_key = (counter, query_idx, arm1_idx, arm2_idx)
            COMPARISON_CACHE[cache_key] = result
        else:
            cache_key = (counter, query_idx, arm2_idx, arm1_idx)
            COMPARISON_CACHE[cache_key] = _swap_comparison_result(result)
        print(f"    💾 [Cache store] counter={counter}, query={query_idx}, arm1={arm1_idx}, arm2={arm2_idx}, cache size={len(COMPARISON_CACHE)}")

    return result

import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import copy

from backpack import extend,backpack
from backpack.extensions import BatchGrad

class Network(nn.Module):
    def __init__(self, input_dim, config=None, init_params=None):
        super(Network, self).__init__()

        if config is None:
            from IDS_TAP_parameters.py import NETWORK_CONFIG
            config = NETWORK_CONFIG

        hidden_size = config.get("hidden_size", 128)
        depth = config.get("depth", 2)
        dropout_rate = config.get("dropout_rate", 0.2)
        activation = config.get("activation", "GELU")

        self.dropout_rate = dropout_rate

        if activation.upper() == "GELU":
            self.activation_fn = nn.GELU()
        else:
            self.activation_fn = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout_rate)

        layers = []

        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.GELU())

        current_dim = hidden_size
        for i in range(depth - 1):
            next_dim = current_dim
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.GELU())
            current_dim = next_dim

        layers.append(nn.Linear(current_dim, 1))

        self.model = nn.Sequential(*layers)
        self._initialize(init_params)

        print(f"🧠 Network init: init=kaiming_normal_, dropout={dropout_rate}, "
              f"hidden={hidden_size}, depth={depth}")

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
        y = x
        layers = list(self.model)
        for i, layer in enumerate(layers[:-1]):
            y = layer(y)
            if isinstance(layer, (nn.GELU, nn.ReLU)):
                y = self.dropout(y)
        y = layers[-1](y)
        return y.squeeze(-1)
    
class NeuralDB:

    def __init__(self, input_dim, config=None):
        if config is None:
            from IDS_TAP_parameters.py import NETWORK_CONFIG, TRAINING_CONFIG, POHF_CONFIG, DEVICE_CONFIG
            network_config = NETWORK_CONFIG.copy()
            training_config = TRAINING_CONFIG
            pohf_config = POHF_CONFIG
            device_config = DEVICE_CONFIG
        else:
            network_config = config.get("network", {})
            training_config = config.get("training", {})
            pohf_config = config.get("pohf", {})
            device_config = config.get("device", {})

        device_type = device_config.get("device", "cuda:1")
        self.device = torch.device(device_type if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cuda' and device_config.get("clear_cache", True):
            torch.cuda.empty_cache()

        if input_dim is not None:
            self.input_dim = input_dim
        else:
            self.input_dim = get_input_dim_for_dataset(config=config)

        network_config["input_dim"] = self.input_dim

        self.version = pohf_config.get("version", "matrix")
        self.lamb = pohf_config.get("lambda", 1.0)
        self.nu = pohf_config.get("nu", 0.2)

        self.func = extend(Network(self.input_dim, config=network_config)).to(self.device)

        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.init_model_weight = copy.deepcopy(self.func.state_dict())

        self.lr = training_config.get("learning_rate", 1e-3)
        self.epoch = training_config.get("epochs", 100)
        self.weight_decay = training_config.get("weight_decay", 1.0)

        optimizer_type = training_config.get("optimizer", "AdamW")
        if optimizer_type == "AdamW":
            self.optimizer_fn = optim.AdamW
        elif optimizer_type == "Adam":
            self.optimizer_fn = optim.Adam
        else:
            self.optimizer_fn = optim.AdamW

        self.optimizer = self.optimizer_fn(self.func.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
        max_params_for_matrix = pohf_config.get("max_params_for_matrix", 10000)

        if self.total_param > max_params_for_matrix:
            self.version = "diag"

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

        from IDS_TAP_parameters.py import TRAINING_CONFIG

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

    def _reset_optimizer_only(self):
        from IDS_TAP_parameters.py import TRAINING_CONFIG

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
        self._query_start_weights = copy.deepcopy(self.func.state_dict())

    def restore_to_query_start(self):
        if hasattr(self, '_query_start_weights') and self._query_start_weights is not None:
            self.func.load_state_dict(copy.deepcopy(self._query_start_weights))
        self._reset_optimizer_only()

    def train_model(self, X1, X2, Y, incremental=False, reset_to_query_start=False, weights=None):
        if reset_to_query_start:
            self.restore_to_query_start()
        elif incremental:
            self._reset_optimizer_only()
        else:
            self.restart_model(Y.shape[0] if hasattr(Y, 'shape') else len(Y))

        self.func.train()
        self.func.to(self.device)

        from IDS_TAP_parameters.py import TRAINING_CONFIG
        batch_size = TRAINING_CONFIG.get("batch_size", 32)
        gradient_clip_norm = TRAINING_CONFIG.get("gradient_clip_norm", 1.0)
        early_stopping = TRAINING_CONFIG.get("early_stopping", False)
        patience = TRAINING_CONFIG.get("early_stopping_patience", 5)
        min_delta = TRAINING_CONFIG.get("early_stopping_min_delta", 1e-4)
        debug_training = TRAINING_CONFIG.get("debug_training", False)

        if isinstance(X1, torch.Tensor):
            X1_np = X1.cpu().numpy()
            X2_np = X2.cpu().numpy()
            Y_np = Y.cpu().numpy()
        else:
            X1_np, X2_np, Y_np = X1, X2, Y

        if weights is not None:
            if isinstance(weights, torch.Tensor):
                W_np = weights.cpu().numpy()
            else:
                W_np = np.array(weights)
        else:
            W_np = np.ones(len(Y_np))

        X1_tensor = torch.tensor(X1_np, dtype=torch.float32)
        X2_tensor = torch.tensor(X2_np, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_np, dtype=torch.float32)
        W_tensor = torch.tensor(W_np, dtype=torch.float32)

        dataset = TensorDataset(X1_tensor, X2_tensor, Y_tensor, W_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0
        loss_history = []
        stopped_early = False
        final_epoch = self.epoch

        for epoch in range(1, self.epoch + 1):
            epoch_loss = 0.0
            batch_count = 0
            total_weight = 0.0

            for batch_X1, batch_X2, batch_Y, batch_W in dataloader:
                batch_X1 = batch_X1.to(self.device)
                batch_X2 = batch_X2.to(self.device)
                batch_Y = batch_Y.to(self.device)
                batch_W = batch_W.to(self.device)

                self.func.zero_grad()
                self.optimizer.zero_grad()

                score_1 = self.func(batch_X1)
                score_2 = self.func(batch_X2)

                per_sample_loss = F.binary_cross_entropy_with_logits(
                    score_1 - score_2, batch_Y, reduction='none'
                )
                loss = (per_sample_loss * batch_W).sum() / batch_W.sum()

                loss.backward()

                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.func.parameters(), max_norm=gradient_clip_norm)

                self.optimizer.step()

                epoch_loss += loss.item() * batch_W.sum().item()
                total_weight += batch_W.sum().item()
                batch_count += 1

            self.scheduler.step()
            avg_loss = epoch_loss / total_weight if total_weight > 0 else 0
            loss_history.append(avg_loss)

            if early_stopping:
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        stopped_early = True
                        final_epoch = epoch
                        break

        if debug_training:
            n_samples = len(Y_np)
            first_loss = loss_history[0] if loss_history else 0
            last_loss = loss_history[-1] if loss_history else 0
            loss_reduction = first_loss - last_loss

            early_stop_status = ""
            if early_stopping:
                if stopped_early:
                    early_stop_status = f"⚡early stop@epoch{final_epoch}"
                else:
                    early_stop_status = f"✓full{self.epoch}epochs"

            print(f"      🎓 [Training] samples={n_samples}, loss: {first_loss:.4f}→{last_loss:.4f} "
                  f"(Δ={loss_reduction:.4f}), {early_stop_status}")

    def calculate_greedy_score(self, items):
        import copy

        current_state = copy.deepcopy(self.func.state_dict())

        self.func.load_state_dict(self.init_model_weight)
        self.func.eval()

        if isinstance(items, np.ndarray):
            items = torch.from_numpy(items).to(dtype=torch.float32, device=self.device)
        else:
            items = torch.from_numpy(np.array(items)).to(dtype=torch.float32, device=self.device)

        outputs_init = self.func(items)
        outputs_init = torch.sigmoid(outputs_init)

        self.func.zero_grad()
        with backpack(BatchGrad()):
            outputs_init.sum().backward()

        init_grads_batch = torch.cat(
            [p.grad_batch.flatten(1) for p in self.func.parameters() if hasattr(p, 'grad_batch') and p.grad_batch is not None],
            dim=1
        )

        self.func.load_state_dict(current_state)
        self.func.eval()

        with torch.no_grad():
            outputs_current = self.func(items)
            outputs_current = torch.sigmoid(outputs_current)

        for p in self.func.parameters():
            if hasattr(p, 'grad_batch'):
                del p.grad_batch

        return outputs_current, init_grads_batch

    def calculate_scores_only(self, items):
        try:
            from IDS_TAP_parameters.py import POHF_CONFIG
            temperature = POHF_CONFIG.get("softmax_temperature", 1.0)
        except ImportError:
            temperature = 1.0

        if isinstance(items, np.ndarray):
            items = torch.from_numpy(items).to(dtype=torch.float32, device=self.device)
        else:
            items = torch.from_numpy(np.array(items)).to(dtype=torch.float32, device=self.device)

        self.func.eval()
        with torch.no_grad():
            outputs = self.func(items)
            outputs = outputs.squeeze()
            outputs = torch.softmax(outputs / temperature, dim=0)
            outputs = outputs.unsqueeze(-1)

        return outputs

    def calculate_gradients_for_arms(self, items, arm_indices):
        import copy

        if isinstance(items, np.ndarray):
            items_tensor = torch.from_numpy(items).to(dtype=torch.float32, device=self.device)
        else:
            items_tensor = torch.from_numpy(np.array(items)).to(dtype=torch.float32, device=self.device)

        selected_items = items_tensor[arm_indices]

        current_state = copy.deepcopy(self.func.state_dict())

        self.func.load_state_dict(self.init_model_weight)
        self.func.eval()

        outputs = self.func(selected_items)
        outputs = torch.sigmoid(outputs)

        self.func.zero_grad()
        with backpack(BatchGrad()):
            outputs.sum().backward()

        grads = torch.cat(
            [p.grad_batch.flatten(1) for p in self.func.parameters()
             if hasattr(p, 'grad_batch') and p.grad_batch is not None],
            dim=1
        )

        for p in self.func.parameters():
            if hasattr(p, 'grad_batch'):
                del p.grad_batch

        self.func.load_state_dict(current_state)

        result = {}
        for i, arm_idx in enumerate(arm_indices):
            result[arm_idx] = grads[i].clone()

        del grads
        torch.cuda.empty_cache()

        return result

    def calculate_ucb_scores_memory_efficient(self, items, greedy_arm_index, current_iteration=0, total_iterations=100, batch_size=50):

        import copy

        if isinstance(items, np.ndarray):
            items_tensor = torch.from_numpy(items).to(dtype=torch.float32, device=self.device)
        else:
            items_tensor = torch.from_numpy(np.array(items)).to(dtype=torch.float32, device=self.device)

        num_arms = items_tensor.shape[0]

        self.func.eval()
        with torch.no_grad():
            greedy_scores = self.func(items_tensor)
            greedy_scores = torch.sigmoid(greedy_scores)

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
        ).squeeze(0)

        for p in self.func.parameters():
            if hasattr(p, 'grad_batch'):
                del p.grad_batch

        dynamic_nu = self.get_dynamic_nu(current_iteration, total_iterations)

        sigma_squared_list = []

        for i in range(0, num_arms, batch_size):
            end_idx = min(i + batch_size, num_arms)
            batch_items = items_tensor[i:end_idx]

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

            for p in self.func.parameters():
                if hasattr(p, 'grad_batch'):
                    del p.grad_batch

            diff = batch_grads - greedy_grad.unsqueeze(0)

            if self.version == "matrix":
                batch_sigma_squared = torch.sum((diff @ self.Sinv) * diff, dim=1)
            elif self.version == "diag":
                batch_sigma_squared = torch.sum(diff**2 * self.Sinv, dim=1)

            sigma_squared_list.append(batch_sigma_squared)

            del batch_grads, diff
            torch.cuda.empty_cache()

        self.func.load_state_dict(current_state)

        sigma_squared = torch.cat(sigma_squared_list, dim=0)
        sigma = torch.sqrt(sigma_squared + 1e-12)

        import math
        time_factor = math.log(0.1 * current_iteration + 1) + 1
        ucb_scores = greedy_scores + dynamic_nu * sigma * time_factor

        return greedy_scores, ucb_scores

    def get_dynamic_nu(self, current_iteration, total_iterations):
        from IDS_TAP_parameters.py import POHF_CONFIG

        nu_decay_enabled = POHF_CONFIG.get("nu_decay_enabled", True)
        if not nu_decay_enabled:
            return self.nu

        nu_decay_factor = POHF_CONFIG.get("nu_decay_factor", 0.95)
        nu_min = POHF_CONFIG.get("nu_min", 0.05)
        nu_decay_start = POHF_CONFIG.get("nu_decay_start", 0)
        nu_decay_type = POHF_CONFIG.get("nu_decay_type", "exponential")

        if current_iteration < nu_decay_start:
            return self.nu

        decay_iterations = current_iteration - nu_decay_start

        if nu_decay_type == "exponential":
            dynamic_nu = self.nu * (nu_decay_factor ** decay_iterations)
        elif nu_decay_type == "linear":
            progress = min(1.0, decay_iterations / (total_iterations - nu_decay_start))
            dynamic_nu = self.nu * (1.0 - progress * (1.0 - nu_min / self.nu))
        elif nu_decay_type == "step":
            step_size = 10
            steps = decay_iterations // step_size
            dynamic_nu = self.nu * (nu_decay_factor ** steps)
        else:
            dynamic_nu = self.nu

        dynamic_nu = max(nu_min, dynamic_nu)

        return dynamic_nu

    def get_covariance_matrix_stats(self, current_iteration=0):
        stats_data = {
            "iteration": current_iteration,
            "version": self.version,
            "total_param": self.total_param,
            "lambda": self.lamb,
            "nu": self.nu
        }

        if self.version == "diag":
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
            S_eigenvals = torch.linalg.eigvals(self.S).real
            S_eigenvals = S_eigenvals[S_eigenvals > 1e-10]

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
        pass

    def calculate_UCB_score(self, greedy_scores, grads_batch, greedy_arm_index, current_iteration=0, total_iterations=100):

        dynamic_nu = self.get_dynamic_nu(current_iteration, total_iterations)

        greedy_grad = grads_batch[greedy_arm_index]
        greedy_grad = greedy_grad.unsqueeze(0)

        num_arms = grads_batch.shape[0]
        batch_size = 50

        sigma_squared_list = []

        for i in range(0, num_arms, batch_size):
            end_idx = min(i + batch_size, num_arms)

            batch_grads = grads_batch[i:end_idx]
            diff = batch_grads - greedy_grad

            if self.version == "matrix":
                batch_sigma_squared = torch.sum((diff @ self.Sinv) * diff, dim=1)
            elif self.version == "diag":
                batch_sigma_squared = torch.sum(diff**2 * self.Sinv, dim=1)

            sigma_squared_list.append(batch_sigma_squared)

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

    def __init__(self, filename="embedding_using.pkl", mapping_filename="domain_prompt_mapping.pkl"):
        self.filename = filename
        self.mapping_filename = mapping_filename
        self.embeddings_dict = {}
        self.domain_to_prompt_mapping = None

    def save_embeddings(self, embeddings_tensor, domain_to_prompt_mapping=None, clear_memory=True):
        embeddings_np = embeddings_tensor.cpu().numpy()

        with open(self.filename, 'wb') as f:
            pickle.dump(embeddings_np, f)

        if domain_to_prompt_mapping is not None:
            with open(self.mapping_filename, 'wb') as f:
                pickle.dump(domain_to_prompt_mapping, f)

        if clear_memory:
            del embeddings_tensor
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        return embeddings_np.shape

    def load_mapping(self):
        if self.domain_to_prompt_mapping is None:
            try:
                with open(self.mapping_filename, 'rb') as f:
                    self.domain_to_prompt_mapping = pickle.load(f)
            except FileNotFoundError as e:
                print(f"⚠️ [EmbeddingManager.load_mapping] Mapping file not found: {self.mapping_filename}")
                self.domain_to_prompt_mapping = []
        return self.domain_to_prompt_mapping

    def get_prompt_indices(self, instruction_idx, summary_idx):
        mapping = self.load_mapping()
        indices = []
        for i, (inst_idx, summ_idx) in enumerate(mapping):
            if inst_idx == instruction_idx and summ_idx == summary_idx:
                indices.append(i)
        return indices

    def load_embedding(self, index):
        if not hasattr(self, '_cached_embeddings'):
            with open(self.filename, 'rb') as f:
                self._cached_embeddings = pickle.load(f)

        return torch.from_numpy(self._cached_embeddings[index]).cuda()

    def load_embeddings_batch(self, indices):
        if not hasattr(self, '_cached_embeddings'):
            with open(self.filename, 'rb') as f:
                self._cached_embeddings = pickle.load(f)

        batch_embeddings = []
        for idx in indices:
            batch_embeddings.append(self._cached_embeddings[idx])

        return torch.from_numpy(np.array(batch_embeddings)).cuda()

    def get_all_embeddings(self):
        with open(self.filename, 'rb') as f:
            embeddings_np = pickle.load(f)
        return torch.from_numpy(embeddings_np).cuda()

    def clear_cache(self):
        if hasattr(self, '_cached_embeddings'):
            del self._cached_embeddings
        import gc
        gc.collect()
import asyncio
import aiohttp

class EmbeddingClient:

    def __init__(self, api_url: str = "YOUR_EMBEDDING_API_URL"):
        self.api_url = api_url

    def normalize_l2(self, x):
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
        payload = {
            "model": "/workspace/users/zhiwei/qwen3",
            "input": [text]
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.api_url, json=payload, timeout=30) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result["data"][0]["embedding"]
                        else:
                            error_msg = f"HTTP {response.status}"
                            if attempt < max_retries - 1:
                                wait_time = retry_delay * (retry_backoff ** attempt)
                                print(f"⚠️ [EmbeddingClient.get_embedding] Request failed (Attempt {attempt + 1}/{max_retries}): {error_msg}")
                                print(f"   Waiting {wait_time:.1f}s  before retry...", flush=True)
                                await asyncio.sleep(wait_time)
                            else:
                                print(f"❌ [EmbeddingClient.get_embedding] Max retries reached: {error_msg}")
                                return None
            except Exception as e:
                last_error = e
                error_type_name = type(e).__name__
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (retry_backoff ** attempt)
                    print(f"⚠️ [EmbeddingClient.get_embedding] Encoding failed (Attempt {attempt + 1}/{max_retries}) [{error_type_name}]: {e}")
                    print(f"   Waiting {wait_time:.1f}s  before retry...", flush=True)
                    await asyncio.sleep(wait_time)
                else:
                    print(f"❌ [EmbeddingClient.get_embedding] Max retries reached [{error_type_name}]: {e}")
                    return None

        return None

    async def encode_texts(self, texts: List[str], normalize: bool = True) -> List[np.ndarray]:
        embeddings = []

        for i, text in enumerate(texts):
            embedding = await self.get_embedding(text)

            if embedding is not None:
                emb_array = np.array(embedding, dtype=np.float32)
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
    if torch.cuda.is_available():
        current_gpu = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(current_gpu) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(current_gpu) / 1024**3
        memory_total = torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3
        memory_free = memory_total - memory_allocated

        if memory_free < 2.0:
            torch.cuda.empty_cache()

def check_current_gpu_status():
    pass

def verify_embedding_prompt_correspondence(domain_texts, init_instructions, domain_to_prompt_mapping, input_data):
    if len(domain_texts) == len(init_instructions) == len(domain_to_prompt_mapping):
        pass
    else:
        return False

    import random
    check_indices = random.sample(range(len(domain_texts)), min(3, len(domain_texts)))

    correct_count = 0
    for idx in check_indices:
        inst_idx, summ_idx = domain_to_prompt_mapping[idx]
        domain_text = domain_texts[idx]
        full_prompt = init_instructions[idx]

        expected_instruction = input_data[2]
        expected_summary = input_data[3][summ_idx]

        instruction_match = expected_instruction in full_prompt
        summary_match = expected_summary in full_prompt

        domain_summary_match = expected_summary in domain_text
        domain_instruction_absent = expected_instruction not in domain_text

        if instruction_match and summary_match and domain_summary_match and domain_instruction_absent:
            correct_count += 1

    return correct_count == len(check_indices)

class ProgressLogger:
    def __init__(self, counter, run_index, total_counters, algorithm_name="POHF-InfoGain", verbose=True):
        self.counter = counter
        self.run_index = run_index
        self.total_counters = total_counters
        self.algorithm_name = algorithm_name
        self.verbose = verbose
        self._last_progress = -1
        self._progress_interval = 10

    def _get_prefix(self):
        return f"[Counter {self.run_index + 1}/{self.total_counters}][{self.algorithm_name}]"

    def log_progress(self, current_iter, total_iter, extra_info=""):
        if total_iter == 0:
            return
        progress = int(current_iter / total_iter * 100)
        if progress >= self._last_progress + self._progress_interval or current_iter == total_iter:
            self._last_progress = progress
            extra = f" | {extra_info}" if extra_info else ""
            print(f"{self._get_prefix()} Iteration {current_iter}/{total_iter} ({progress}%){extra}", flush=True)

    def log_start(self):
        print(f"{self._get_prefix()} 🚀 Starting counter={self.counter}", flush=True)

    def log_Complete(self, best_score=None, greedy_arm=None):
        info = f"best_score={best_score:.4f}, greedy_arm={greedy_arm}" if best_score is not None else ""
        print(f"{self._get_prefix()} ✅ Complete! {info}", flush=True)

    def log_error(self, error_msg):
        print(f"{self._get_prefix()} ❌ Error: {error_msg}", flush=True)

    def log_verbose(self, msg):
        if self.verbose:
            print(f"{self._get_prefix()} {msg}", flush=True)

def run_single_counter_process(args):
    counter, run_index, total_counters, config_dict = args

    import os
    import sys
    import gc
    import warnings

    warnings.filterwarnings("ignore", message="Extension saving to grad_batch")
    warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")

    log_prefix = f"[Counter {run_index + 1}/{total_counters}]"

    try:
        import torch
        import asyncio

        reset_device()

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()

        result = asyncio.run(run_single_counter_async(
            counter=counter,
            run_index=run_index,
            total_counters=total_counters,
            config=config_dict
        ))

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
        print(f"❌ [run_single_counter_process] Counter {counter} failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as cleanup_e:
            print(f"⚠️ [run_single_counter_process] GPU cleanup failed: {cleanup_e}")
        gc.collect()

        return {
            'status': 'failed',
            'counter': counter,
            'error': str(e),
            'all_results': [],
            'greedy_arm_results': []
        }

async def run_single_counter_async(counter, run_index, total_counters, config):
    import copy
    single_config = copy.deepcopy(config)

    single_config['_single_counter_mode'] = True
    single_config['_target_counter'] = counter
    single_config['_run_index'] = run_index
    single_config['_total_counters'] = total_counters

    results = await run(config=single_config)

    return {
        'all_results': results if isinstance(results, list) else [results],
        'greedy_arm_results': []
    }

async def run(config=None):
    reset_device()
    if torch.cuda.is_available():
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()

    if config is None:
        from IDS_TAP_parameters.py import get_all_configs, generate_counter_array
        config = get_all_configs()
    else:
        from IDS_TAP_parameters.py import generate_counter_array

    experiment_config = config.get("experiment", {})
    data_config = config.get("data", {})
    path_config = config.get("path", {})
    api_config = config.get("api", {})
    device_config = config.get("device", {})
    rouge_config = config.get("rouge", {})

    LLM_as_judge = rouge_config.get("LLM_as_judge", False)

    random_seed = experiment_config.get("random_seed", 0)

    input_address = path_config.get("input_address")
    output_address = path_config.get("output_address")
    LaMP_type = path_config.get("LaMP_type", 4)

    if 'POHF_LAMP_TYPE' in os.environ:
        lamp_type_override = int(os.environ['POHF_LAMP_TYPE'])
        input_address_override = os.environ.get('POHF_INPUT_ADDRESS')
        output_address_override = os.environ.get('POHF_OUTPUT_ADDRESS')

        counter_array_length_override = os.environ.get('POHF_COUNTER_ARRAY_LENGTH')

        LaMP_type = lamp_type_override
        input_address = input_address_override
        if output_address_override:
            output_address = output_address_override

        if counter_array_length_override:
            data_config['counter_array_length'] = int(counter_array_length_override)

        path_config['LaMP_type'] = LaMP_type

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

    specific_counters_str = os.environ.get('POHF_SPECIFIC_COUNTERS')
    if specific_counters_str:
        specific_counters = [int(c.strip()) for c in specific_counters_str.split(',') if c.strip()]
        counter_array = specific_counters
        print(f"📋 Using specified counter list: {counter_array}")

    single_counter_mode = config.get('_single_counter_mode', False)
    if single_counter_mode:
        target_counter = config.get('_target_counter')
        counter_array = [target_counter]

    parallel_config = config.get('parallel', {})
    parallel_enabled = parallel_config.get('parallel_enabled', True)
    parallel_counters = parallel_config.get('parallel_counters', 3)
    timeout_per_counter = parallel_config.get('timeout_per_counter', 36000)

    parallel_counters_override = os.environ.get('POHF_PARALLEL_COUNTERS')
    if parallel_counters_override:
        parallel_counters = int(parallel_counters_override)

    if single_counter_mode:
        parallel_enabled = False

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

            with tqdm(total=len(counter_array), desc="🚀 Parallel run", unit="counter",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                for future in as_Completed(future_to_counter):
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
                        print(f"❌ [parallel execution] Counter {counter} exception: {e}")
                        failed_counters.append({
                            'counter': counter,
                            'error': str(e)
                        })
                        pbar.set_postfix_str(f"❌ Counter {counter}")

                    pbar.update(1)

        print(f"\n✅ Complete: {len(all_results)}/{len(counter_array)} counters success, {len(failed_counters)} failed")
        if failed_counters:
            print(f"❌ Failed counters: {[fc['counter'] for fc in failed_counters]}")
            for fc in failed_counters:
                print(f"   Counter {fc['counter']}: {fc['error']}")

        all_results.sort(key=lambda x: x.get('counter', 0))
        all_greedy_arm_results.sort(key=lambda x: x.get('counter', 0))

        if not LLM_as_judge and all_results and len(all_results) > 1:
            try:
                from IDS_TAP_parameters.py import CONTEXTUAL_BANDIT_CONFIG
                unified_training_rounds = CONTEXTUAL_BANDIT_CONFIG.get("unified_training_rounds", 10)
            except ImportError:
                unified_training_rounds = 10

            full_results_for_export = []
            for result in all_results:
                full_data = result.get('best_instruction_over_iter', [])
                baseline_data = result.get('baseline_results', {})

                full_result = {
                    'algorithm': result.get('algorithm', 'POHF-InfoGain'),
                    'counter': result.get('counter', 0),
                    'best_instruction_over_iter': full_data,
                    'baseline_results': baseline_data,
                    'total_arms': result.get('total_arms', 0),
                    'contextual_mode': result.get('contextual_mode', False),
                    'num_queries': result.get('num_queries', 1),
                    'total_iterations': len(full_data)
                }
                full_results_for_export.append(full_result)

            if full_results_for_export:
                plot_counter_average_results(full_results_for_export, get_dataset_name(LaMP_type))
                print(f"   📊 [Figure 1 Counter Average] Generated Counter average plot（Including all Query data）")

            all_query_progress_data = []
            for result in all_results:
                full_data = result.get('best_instruction_over_iter', [])
                baseline_data = result.get('baseline_results', {})
                counter = result.get('counter', 0)

                q_iters = unified_training_rounds
                num_queries = len(full_data) // q_iters if q_iters > 0 else 1

                if num_queries >= 1:
                    query_final_values = {}
                    for q_idx in range(0, num_queries):
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
                            vmax = max(all_vals)
                            norm = lambda v, vmax=vmax: v / vmax if vmax > 0 else 0.0

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
                            alg_progress = {alg: [data[q]['final_value_normalized'] for q in q_indices if q in data] for alg, data in query_final_values.items()}
                            alg_progress = {k: v for k, v in alg_progress.items() if v}

                            rand_avg_line = None
                            if 'Random' in query_final_values:
                                rand_avgs = [query_final_values['Random'].get(q, {}).get('avg_value_normalized', None) for q in q_indices]
                                rand_avgs = [v for v in rand_avgs if v is not None]
                                if rand_avgs:
                                    rand_avg_line = np.mean(rand_avgs)

                            q_prog_data = {'counter': counter, 'query_indices': q_indices, 'algorithms': alg_progress, 'random_avg_line': rand_avg_line}
                            all_query_progress_data.append(q_prog_data)

            if all_query_progress_data:
                plot_query_progress_counter_average(all_query_progress_data, get_dataset_name(LaMP_type))
                print(f"   📊 [Figure 2 Counter Average] Generated cross-Query progress Counter average plot")

        return all_results

    api_url = api_config.get("embedding_api_url", "YOUR_EMBEDDING_API_URL")
    embedding_client = EmbeddingClient(api_url=api_url)

    async def compute_contextual_embeddings(query_text, summary_texts, embedding_client, is_lamp, tkwargs):
        if is_lamp:
            query_emb_list = await embedding_client.encode_texts([query_text], normalize=True)
            query_emb = torch.from_numpy(query_emb_list[0])

            summary_emb_list = await embedding_client.encode_texts(summary_texts, normalize=True)
            summary_embs = torch.stack([torch.from_numpy(emb) for emb in summary_emb_list])

            query_emb_expanded = query_emb.unsqueeze(0).expand(summary_embs.shape[0], -1)
            embeddings = torch.cat([query_emb_expanded, summary_embs], dim=1)
            embeddings = embeddings.to(**tkwargs)

            return embeddings, query_emb
        else:
            summary_emb_list = await embedding_client.encode_texts(summary_texts, normalize=True)
            embeddings = torch.stack([torch.from_numpy(emb) for emb in summary_emb_list])
            embeddings = embeddings.to(**tkwargs)

            return embeddings, None

    max_history_items = config.get('data', {}).get('max_history_items', 20)

    contextual_mode_enabled = True

    try:
        from IDS_TAP_parameters.py import CONTEXTUAL_BANDIT_CONFIG
        unified_training_rounds = CONTEXTUAL_BANDIT_CONFIG.get("unified_training_rounds", 10)
        contextual_input_dim = CONTEXTUAL_BANDIT_CONFIG.get("contextual_input_dim", 2048)
    except ImportError:
        unified_training_rounds = 10
        contextual_input_dim = 2048

    dataset_name = get_dataset_name(LaMP_type)
    print(f"🔄 [Contextual Mode] {dataset_name}Dataset enabled Contextual Dueling Bandit mode")
    print(f"   input dimension: {contextual_input_dim} (query + persona concat)")
    print(f"   Each query: {unified_training_rounds} rounds preference feedback（No random sample phase）")

    all_results = []

    all_greedy_arm_results = []

    all_first_query_results = []

    all_query_progress_data = []

    persona_output_dir = "./persona_results"
    os.makedirs(persona_output_dir, exist_ok=True)

    algorithms_to_run = [
        {"name": "POHF-InfoGain", "info_gain_enabled": True}
    ]

    for run_index, counter in enumerate(counter_array):

        clear_query_similarity_cache()

        clear_comparison_cache()

        counter_seed = random_seed
        random.seed(counter_seed)
        np.random.seed(counter_seed)
        torch.manual_seed(counter_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(counter_seed)
            torch.cuda.manual_seed_all(counter_seed)

        input_data, ground_truth = load_templated_data(input_address, output_address, LaMP_type, max_len, None, times, counter, max_history_items)

        print("\n" + "="*80)
        print(f"📋 [DEBUG] load_templated_data return value details (counter={counter}, LaMP_type={LaMP_type})")
        print("="*80)

        print(f"\n🔹 input_data length: {len(input_data)}")
        print(f"   [0] input_id: {input_data[0]}")
        print(f"   [1] query (type={type(input_data[1]).__name__}):")
        if isinstance(input_data[1], list):
            print(f"       List length: {len(input_data[1])}")
            for i, q in enumerate(input_data[1][:3]):
                q_preview = str(q)[:200] + "..." if len(str(q)) > 200 else str(q)
                print(f"       [{i}]: {q_preview}")
            if len(input_data[1]) > 3:
                print(f"       ... plus {len(input_data[1]) - 3} items")
        else:
            q_preview = str(input_data[1])[:300] + "..." if len(str(input_data[1])) > 300 else str(input_data[1])
            print(f"       {q_preview}")

        print(f"   [2] single_instruction: {str(input_data[2])[:200]}...")

        print(f"   [3] allVersions_summary (type={type(input_data[3]).__name__}):")
        if isinstance(input_data[3], list):
            print(f"       List length: {len(input_data[3])}")
            for i, s in enumerate(input_data[3][:2]):
                s_preview = str(s)[:150] + "..." if len(str(s)) > 150 else str(s)
                print(f"       [{i}]: {s_preview}")

        print(f"   [4] synthesis: {str(input_data[4])[:200]}...")

        print(f"   [5] ranked_entries (type={type(input_data[5]).__name__}):")
        if isinstance(input_data[5], list):
            if len(input_data[5]) > 0 and isinstance(input_data[5][0], list):
                print(f"       Nested list: {len(input_data[5])} groups，Each group {len(input_data[5][0]) if input_data[5] else 0} items")
            else:
                print(f"       List length: {len(input_data[5])}")
                for i, entry in enumerate(input_data[5][:2]):
                    entry_preview = str(entry)[:100] + "..." if len(str(entry)) > 100 else str(entry)
                    print(f"       [{i}]: {entry_preview}")

        print(f"\n🔹 ground_truth length: {len(ground_truth)}")
        print(f"   [0] output_id: {ground_truth[0]}")
        print(f"   [1] output (type={type(ground_truth[1]).__name__}):")
        if isinstance(ground_truth[1], list):
            print(f"       List length: {len(ground_truth[1])}")
            for i, o in enumerate(ground_truth[1][:3]):
                o_preview = str(o)[:150] + "..." if len(str(o)) > 150 else str(o)
                print(f"       [{i}]: {o_preview}")
            if len(ground_truth[1]) > 3:
                print(f"       ... plus {len(ground_truth[1]) - 3} items")
        else:
            o_preview = str(ground_truth[1])[:300] + "..." if len(str(ground_truth[1])) > 300 else str(ground_truth[1])
            print(f"       {o_preview}")

        print("="*80 + "\n")

        original_summary_for_counter = input_data[3][0] if len(input_data) > 3 and len(input_data[3]) > 0 else ""

        query_data = input_data[1]
        ground_truth_data = ground_truth[1]

        if is_lamp_dataset(LaMP_type) and isinstance(query_data, list):
            num_queries = len(query_data)
            print(f"  🔄 [Contextual Mode] LaMP dataset detected {num_queries}  query/output pairs")
        else:
            num_queries = 1
            print(f"  🔄 [Contextual Mode] Non-LaMP dataset using single query mode")

        counter_results = []

        query_final_values = {}

        counter_persona_data = {
            "counter": counter,
            "lamp_type": LaMP_type,
            "contextual_mode": contextual_mode_enabled,
            "num_queries": 0,
            "queries": {}
        }

        domain_texts_base = []
        for j in range(times):
            domain_text = f"### Summary:\n{input_data[3][j]}"
            domain_texts_base.append(domain_text)

        indices = list(range(times))
        random.shuffle(indices)
        domain_texts = [domain_texts_base[i] for i in indices]

        original_persona_index = indices.index(0)
        print(f"  🎯 [Original Persona] Original persona (index 0) Position after shuffling: {original_persona_index}")

        print(f"\n  📋 [Embedding Input] Summary text overview (Total {len(domain_texts)} items):")
        for i, text in enumerate(domain_texts[:3]):
            preview = text.replace('\n', ' ')[:80]
            print(f"     [{i}]: {preview}...")
        if len(domain_texts) > 3:
            print(f"     ... plus {len(domain_texts) - 3}  summaries")

        summary_emb_list = await embedding_client.encode_texts(domain_texts, normalize=True)
        summary_embs_base = torch.stack([torch.from_numpy(emb) for emb in summary_emb_list])

        embedding_dir = "./embeddings"
        os.makedirs(embedding_dir, exist_ok=True)

        counter_persona_data["num_queries"] = num_queries

        for algorithm_config in algorithms_to_run:
            algorithm_name = algorithm_config["name"]
            info_gain_enabled = algorithm_config["info_gain_enabled"]

            progress_logger = ProgressLogger(counter, run_index, len(counter_array), algorithm_name)

            num_group = []
            x_train = []
            y_train = []
            query_indices = []
            query_embeddings = {}
            select_idx_history = []
            instruction_select_history = []
            second_arm_selections = []
            llm_config = config.get("llm", {})

            l = None

            for query_idx in range(num_queries):
                current_n_init = 0
                current_max_iter = unified_training_rounds

                if isinstance(query_data, list) and query_idx < len(query_data):
                    current_query = str(query_data[query_idx])
                else:
                    current_query = str(query_data) if not isinstance(query_data, list) else str(query_data[0])

                current_gt_index = query_idx if isinstance(ground_truth_data, list) and query_idx < len(ground_truth_data) else 0

                print(f"\n  📍 [Query {query_idx+1}/{num_queries}] Training configuration: {current_max_iter} rounds preference feedback")

                print(f"  📝 [Embedding Input] Query {query_idx} Full text:")
                print(f"     ───────────────────────────────────────────────────────────")
                query_lines = [current_query[i:i+100] for i in range(0, len(current_query), 100)]
                for line in query_lines[:10]:
                    print(f"     {line}")
                if len(query_lines) > 10:
                    print(f"     ... (Total {len(current_query)} characters，Truncated for display)")
                print(f"     ───────────────────────────────────────────────────────────")

                init_instructions = []
                domain_to_prompt_mapping = []
                for j in range(times):
                    original_j = indices[j]
                    full_prompt = prompt_reformer(input_data, 0, original_j, lamp_type=LaMP_type, query_index=query_idx)
                    init_instructions.append(full_prompt)
                    domain_to_prompt_mapping.append((query_idx, original_j))

                query_emb_list = await embedding_client.encode_texts([current_query], normalize=True)
                query_emb = torch.from_numpy(query_emb_list[0])

                query_emb_expanded = query_emb.unsqueeze(0).expand(summary_embs_base.shape[0], -1)
                sen_embeddings = torch.cat([query_emb_expanded, summary_embs_base], dim=1)
                sen_embeddings = sen_embeddings.to(**tkwargs)

                from IDS_TAP_parameters.py import TRAINING_CONFIG
                query_similarity_enabled = TRAINING_CONFIG.get("query_similarity_enabled", False)
                query_decay_enabled = TRAINING_CONFIG.get("query_decay_enabled", False)

                if query_similarity_enabled or query_decay_enabled:
                    query_embeddings[query_idx] = query_emb.cpu().numpy() if hasattr(query_emb, 'cpu') else query_emb

                    if query_idx > 0 and query_similarity_enabled:
                        print(f"  📊 [Query {query_idx}] Similarity with historical queries:")
                        current_emb_flat = query_embeddings[query_idx].flatten()
                        norm_current = np.linalg.norm(current_emb_flat)

                        steepness = TRAINING_CONFIG.get("query_similarity_steepness", 11.0)
                        midpoint = TRAINING_CONFIG.get("query_similarity_midpoint", 0.5)

                        for hist_idx in range(query_idx):
                            hist_emb_flat = query_embeddings[hist_idx].flatten()
                            norm_hist = np.linalg.norm(hist_emb_flat)
                            if norm_current > 1e-8 and norm_hist > 1e-8:
                                cosine_sim = np.dot(current_emb_flat, hist_emb_flat) / (norm_current * norm_hist)
                                cosine_sim = max(0.0, float(cosine_sim))
                            else:
                                cosine_sim = 0.0
                            sim_weight = 1.0 / (1.0 + np.exp(-(cosine_sim - midpoint) * steepness))
                            print(f"      Query {hist_idx}: cosine_sim={cosine_sim:.4f} → weight={sim_weight:.4f}")

                            _query_similarity_cache[(hist_idx, query_idx)] = sim_weight

                if query_idx == 0:
                    print(f"  📐 Contextual embedding: query({query_emb.shape}) + persona({summary_embs_base.shape[1]}) = {sen_embeddings.shape}")

                embedding_filename = os.path.join(embedding_dir, f"embedding_using_{get_dataset_name(LaMP_type)}_counter{counter}_query{query_idx}.pkl")
                mapping_filename = os.path.join(embedding_dir, f"domain_prompt_mapping_{get_dataset_name(LaMP_type)}_counter{counter}_query{query_idx}.pkl")

                embedding_manager = EmbeddingManager(embedding_filename, mapping_filename)
                embedding_manager.save_embeddings(
                    sen_embeddings,
                    domain_to_prompt_mapping=domain_to_prompt_mapping,
                    clear_memory=False
                )

                query_random_sample_start_idx = len(select_idx_history)

                from IDS_TAP_parameters.py import TRAINING_CONFIG
                cross_query_incremental = TRAINING_CONFIG.get("cross_query_incremental", False)
                if cross_query_incremental and query_idx > 0:
                    print(f"  🔄 [Cross-Query incremental mode] Clearing training data，Keeping network weights")
                    x_train = []
                    y_train = []
                    query_indices = []

                if cross_query_incremental and l is not None:
                    l.save_query_start_weights()
                    print(f"  💾 [Cross-Query incremental mode] Saving Query {query_idx} start network weights")

                if current_n_init > 0:
                    init_pairs = []
                    for i in range(current_n_init):
                        upper_bound = times
                        num1, num2 = random.sample(range(upper_bound), 2)
                        while (num1, num2) in num_group:
                            num1, num2 = random.sample(range(upper_bound), 2)
                        num_group.append((num1, num2))
                        init_pairs.append((num1, num2))

                    async def process_init_pair(pair_idx, num1, num2, gt_idx):
                        personality1 = domain_texts[num1] if num1 < len(domain_texts) else None
                        personality2 = domain_texts[num2] if num2 < len(domain_texts) else None
                        response1, response2, score_1, score_2, _, new_y, p_ = await generate_and_compare_async(
                            init_instructions[num1], init_instructions[num2],
                            ground_truth[1], llm_config,
                            LLM_as_judge=LLM_as_judge, arm1_idx=num1, arm2_idx=num2,
                            personality1=personality1, personality2=personality2,
                            ground_truth_index=gt_idx,
                            counter=counter, query_idx=query_idx
                        )
                        return {
                            'pair_idx': pair_idx,
                            'num1': num1, 'num2': num2,
                            'response1': response1, 'response2': response2,
                            'score_1': score_1, 'score_2': score_2,
                            'new_y': new_y
                        }

                    print(f"  📋 Random sampling phase: Parallel generating {len(init_pairs)}  responses...", flush=True)
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
                        query_indices += [query_idx]
                        select_idx_history += [[num1, num2]]
                        instruction_select_history += [(init_instructions[num1], score_1, init_instructions[num2], score_2)]

                if len(x_train) > 0:
                    x_train_tensor = torch.cat(x_train, dim=1)
                else:
                    x_train_tensor = torch.zeros(1, 0, sen_embeddings.shape[-1])

                random.seed(random_seed)
                np.random.seed(random_seed)
                torch.manual_seed(random_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(random_seed)
                    torch.cuda.manual_seed_all(random_seed)

                if l is None or query_idx == 0:
                    l = NeuralDB(input_dim=sen_embeddings.shape[-1], config=config)

                X1_train = []
                X2_train = []
                Y_train = []
                W_train = []

                from IDS_TAP_parameters.py import TRAINING_CONFIG
                cross_query_incremental = TRAINING_CONFIG.get("cross_query_incremental", False)
                _query_decay_enabled = TRAINING_CONFIG.get("query_decay_enabled", False)
                _query_similarity_enabled = TRAINING_CONFIG.get("query_similarity_enabled", False)
                _use_simple_weight = cross_query_incremental or (not _query_decay_enabled and not _query_similarity_enabled)

                for i, (x_pair, y, q_idx) in enumerate(zip(x_train, y_train, query_indices)):
                    x_pair_reshaped = x_pair.squeeze(1)
                    X1_train.append(x_pair_reshaped[0].cpu().numpy())
                    X2_train.append(x_pair_reshaped[1].cpu().numpy())
                    Y_train.append(y)
                    if _use_simple_weight:
                        weight = 1.0
                    else:
                        weight = compute_query_weight(q_idx, query_idx, query_embeddings, TRAINING_CONFIG)
                    W_train.append(weight)

                X1_train = np.array(X1_train) if X1_train else np.array([]).reshape(0, sen_embeddings.shape[-1])
                X2_train = np.array(X2_train) if X2_train else np.array([]).reshape(0, sen_embeddings.shape[-1])
                Y_train = np.array(Y_train) if Y_train else np.array([])
                W_train = np.array(W_train) if W_train else np.array([])

                if len(X1_train) > 0:
                    if cross_query_incremental:
                        l.train_model(X1_train, X2_train, Y_train, reset_to_query_start=True, weights=W_train)
                    else:
                        l.train_model(X1_train, X2_train, Y_train, incremental=False, weights=W_train)
                    l._has_trained = True

                with open(embedding_filename, 'rb') as f:
                    embeddings_np = pickle.load(f)

                unique_arms = set()
                for arm_pair in select_idx_history:
                    unique_arms.add(arm_pair[0])
                    unique_arms.add(arm_pair[1])
                unique_arms_list = list(unique_arms)

                greedy_scores = l.calculate_scores_only(embeddings_np)

                if unique_arms_list:
                    arm_grads_init = l.calculate_gradients_for_arms(embeddings_np, unique_arms_list)

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

                del embeddings_np
                torch.cuda.empty_cache()

                if algorithm_name == "POHF-InfoGain":
                    from information_second_term import ContextualPairwiseInformationManager, PairwiseInformationManager
                    bayesian_alpha = config.get("pohf", {}).get("bayesian_alpha", 1.0)
                    bt_isolated_arm_mode = config.get("pohf", {}).get("bt_isolated_arm_mode", "unknown_isolated")
                    reset_info_matrix_per_query = config.get("pohf", {}).get("reset_info_matrix_per_query", False)

                    if not hasattr(l, 'contextual_info_manager') or l.contextual_info_manager is None:
                        l.contextual_info_manager = ContextualPairwiseInformationManager(
                            len(greedy_scores), use_optimized=True, bayesian_alpha=bayesian_alpha,
                            bt_isolated_arm_mode=bt_isolated_arm_mode
                        )
                    current_input_embedding = query_emb.cpu().numpy()

                    if reset_info_matrix_per_query:
                        l.info_manager = l.contextual_info_manager.initialize_without_history(
                            input_index=query_idx, input_embedding=current_input_embedding
                        )
                        print(f"   🔄 [Contextual] Info matrix reset to uniform distribution (reset_info_matrix_per_query=True, query_index={query_idx})")
                    else:
                        l.info_manager = l.contextual_info_manager.initialize_for_new_input(
                            input_index=query_idx, input_embedding=current_input_embedding
                        )
                        print(f"   🔄 [Contextual] UsingContextualPairwiseInformationManager (query_index={query_idx})")

                    current_query_pairs = select_idx_history[query_random_sample_start_idx:]
                    current_query_labels = y_train[query_random_sample_start_idx:]

                    if len(current_query_pairs) > 0:
                        print(f"   📊 Updating info_manager: Using current query's {len(current_query_pairs)}  random_sample pairs")
                        from information_second_term import update_information_with_feedback
                        for arm_pair, y_value in zip(current_query_pairs, current_query_labels):
                            arm1_idx, arm2_idx = arm_pair
                            arm1_wins = bool(y_value)
                            info_gain = update_information_with_feedback(l.info_manager, arm1_idx, arm2_idx, arm1_wins)
                            if hasattr(l, 'contextual_info_manager') and l.contextual_info_manager is not None:
                                l.contextual_info_manager.record_comparison(arm1_idx, arm2_idx, arm1_wins)

                progress_logger.log_progress(0, current_max_iter, f"(Query {query_idx+1}/{num_queries} Starting iteration)")

                best_r = -np.inf
                best_index = -1

                if query_idx == 0:
                    best_values = []
                    now_values = []
                    best_instruction_over_iter = []
                    per_query_best_scores = {}

                    from LLM_regression import create_algorithm

                    baseline_algorithms = {}
                    baseline_results = {}

                    try:
                        from IDS_TAP_parameters.py import BASELINE_CONFIG
                        enabled_baselines = BASELINE_CONFIG.get("enabled_baselines", [])

                        input_dim = sen_embeddings.shape[-1]
                        algorithm_creators = {
                            'POHF': lambda: NeuralDB(input_dim=input_dim, config=config),
                            'Random': lambda: create_algorithm('Random', input_dim=input_dim, config=config),
                            'POHF-Random': lambda: create_algorithm('POHFRandom', input_dim=input_dim, config=config),
                            'POHF-RandomPair': lambda: create_algorithm('POHFRandomPair', input_dim=input_dim, config=config),
                            'DoubleTS': lambda: create_algorithm('DoubleTS', input_dim=input_dim, config=config),
                            'POHF-InfoGain-NoHistory': lambda: NeuralDB(input_dim=input_dim, config=config),
                            'Linear-InfoGain': lambda: create_algorithm('LinearInfoGain', input_dim=input_dim, config=config),
                        }

                        for alg_name in enabled_baselines:
                            if alg_name in algorithm_creators:
                                if alg_name == 'POHF-InfoGain-NoHistory':
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
                        progress_logger.log_error(f"Baseline algorithm init failed: {e}")
                        baseline_algorithms = {}
                        baseline_results = {}

                    if baseline_algorithms and len(X1_train) > 0:
                        for alg_name, alg in baseline_algorithms.items():
                            try:
                                if hasattr(alg, 'train_model'):
                                    alg.train_model(X1_train, X2_train, Y_train, incremental=False, weights=W_train)
                                    alg._has_trained = True
                            except Exception as e:
                                progress_logger.log_error(f"{alg_name}Training failed: {e}")

                    baseline_training_data = {}
                    input_dim = sen_embeddings.shape[-1]
                    for alg_name in baseline_algorithms.keys():
                        if alg_name != 'Random':
                            if len(X1_train) > 0:
                                baseline_training_data[alg_name] = {
                                    'X1': X1_train.copy(),
                                    'X2': X2_train.copy(),
                                    'Y': Y_train.copy(),
                                    'query_indices': list(query_indices)
                                }
                            else:
                                baseline_training_data[alg_name] = {
                                    'X1': np.array([]).reshape(0, input_dim),
                                    'X2': np.array([]).reshape(0, input_dim),
                                    'Y': np.array([]),
                                    'query_indices': []
                                }

                    if 'POHF-InfoGain-NoHistory' in baseline_results:
                        baseline_training_data['POHF-InfoGain-NoHistory'] = {
                            'X1': np.array([]).reshape(0, input_dim),
                            'X2': np.array([]).reshape(0, input_dim),
                            'Y': np.array([]),
                            'query_indices': []
                        }

                if baseline_algorithms:
                    baseline_best_values = {name: -np.inf for name in baseline_algorithms.keys()}
                if 'POHF-InfoGain-NoHistory' in baseline_results:
                    if 'baseline_best_values' not in locals():
                        baseline_best_values = {}
                    baseline_best_values['POHF-InfoGain-NoHistory'] = -np.inf

                if 'POHF-InfoGain-NoHistory' in baseline_results and query_idx > 0:
                    try:
                        from information_second_term import ContextualPairwiseInformationManager, PairwiseInformationManager
                        input_dim = sen_embeddings.shape[-1]

                        random.seed(random_seed + query_idx)
                        np.random.seed(random_seed + query_idx)
                        torch.manual_seed(random_seed + query_idx)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(random_seed + query_idx)
                            torch.cuda.manual_seed_all(random_seed + query_idx)

                        alg_nih_new = NeuralDB(input_dim=input_dim, config=config)
                        baseline_algorithms['POHF-InfoGain-NoHistory'] = alg_nih_new

                        num_arms_nih = len(init_instructions)
                        bayesian_alpha_nih = config.get("pohf", {}).get("bayesian_alpha", 1.0)
                        bt_isolated_arm_mode_nih = config.get("pohf", {}).get("bt_isolated_arm_mode", "unknown_isolated")
                        alg_nih_new.contextual_info_manager = ContextualPairwiseInformationManager(
                            num_arms_nih, use_optimized=True, bayesian_alpha=bayesian_alpha_nih,
                            bt_isolated_arm_mode=bt_isolated_arm_mode_nih
                        )

                        current_input_embedding_nih = query_emb.cpu().numpy()
                        alg_nih_new.info_manager = alg_nih_new.contextual_info_manager.initialize_without_history(
                            input_index=query_idx, input_embedding=current_input_embedding_nih
                        )

                        print(f"  🔄 [POHF-InfoGain-NoHistory] Query {query_idx}: Resetting network and probability matrix (Uniform initialization)")

                        if 'baseline_training_data' not in locals():
                            baseline_training_data = {}
                        baseline_training_data['POHF-InfoGain-NoHistory'] = {
                            'X1': np.array([]).reshape(0, input_dim),
                            'X2': np.array([]).reshape(0, input_dim),
                            'Y': np.array([]),
                            'query_indices': []
                        }
                        print(f"  🔄 [POHF-InfoGain-NoHistory] Query {query_idx}: Resetting training data")
                    except Exception as e:
                        print(f"⚠️ [POHF-InfoGain-NoHistory] Reset failed: {e}")
                        import traceback
                        traceback.print_exc()

                try:
                    _ = baseline_training_data
                except NameError:
                    baseline_training_data = {}

                input_dim = sen_embeddings.shape[-1]

                from IDS_TAP_parameters.py import TRAINING_CONFIG
                cross_query_incremental = TRAINING_CONFIG.get("cross_query_incremental", False)
                if cross_query_incremental and baseline_algorithms:
                    for alg_name, alg in baseline_algorithms.items():
                        if alg_name != 'Random' and alg_name != 'POHF-InfoGain-NoHistory':
                            if hasattr(alg, 'save_query_start_weights'):
                                alg.save_query_start_weights()
                            if query_idx > 0:
                                baseline_training_data[alg_name] = {
                                    'X1': np.array([]).reshape(0, input_dim),
                                    'X2': np.array([]).reshape(0, input_dim),
                                    'Y': np.array([]),
                                    'query_indices': []
                                }
                    if query_idx > 0:
                        print(f"  🔄 [Cross-Query incremental mode] Clearing baseline training data，Saving network weights")

                for alg_name in baseline_algorithms.keys():
                    if alg_name != 'Random' and alg_name not in baseline_training_data:
                        baseline_training_data[alg_name] = {
                            'X1': np.array([]).reshape(0, input_dim),
                            'X2': np.array([]).reshape(0, input_dim),
                            'Y': np.array([]),
                            'query_indices': []
                        }

                for t in range(current_max_iter):
                    progress_logger.log_progress(t, current_max_iter, f"[Query {query_idx+1}/{num_queries}]")

                    with open(embedding_filename, 'rb') as f:
                        embeddings_np = pickle.load(f)

                    if algorithm_name == "POHF-InfoGain":
                        greedy_scores = l.calculate_scores_only(embeddings_np)
                        ucb_scores_precomputed = None
                    else:
                        greedy_scores_temp = l.calculate_scores_only(embeddings_np)
                        greedy_arm_temp = torch.argmax(greedy_scores_temp).item()
                        greedy_scores, ucb_scores_precomputed = l.calculate_ucb_scores_memory_efficient(
                            embeddings_np,
                            greedy_arm_index=greedy_arm_temp,
                            current_iteration=t,
                            total_iterations=current_max_iter
                        )

                    baseline_iteration_results = {}
                    if baseline_algorithms or 'POHF-InfoGain-NoHistory' in baseline_results:
                        embeddings_list = []
                        with open(embedding_filename, 'rb') as f:
                            embeddings_np_baseline = pickle.load(f)
                            for i in range(len(init_instructions)):
                                embeddings_list.append(embeddings_np_baseline[i])

                        if 'POHF-InfoGain-NoHistory' in baseline_results:
                            alg_name_nih = 'POHF-InfoGain-NoHistory'
                            try:
                                if query_idx == 0:
                                    pass
                                else:
                                    if alg_name_nih in baseline_algorithms:
                                        alg_nih = baseline_algorithms[alg_name_nih]
                                        greedy_scores_nih = alg_nih.calculate_scores_only(embeddings_np_baseline)
                                        arm1_nih = torch.argmax(greedy_scores_nih).item()

                                        if hasattr(alg_nih, 'info_manager') and alg_nih.info_manager is not None:
                                            info_gains_nih = alg_nih.info_manager.get_all_information_gains(arm1_nih)
                                            if info_gains_nih:
                                                arm2_nih = max(info_gains_nih.keys(), key=lambda a: info_gains_nih[a])
                                            else:
                                                available_arms = [i for i in range(len(init_instructions)) if i != arm1_nih]
                                                arm2_nih = random.choice(available_arms) if available_arms else arm1_nih
                                        else:
                                            available_arms = [i for i in range(len(init_instructions)) if i != arm1_nih]
                                            arm2_nih = random.choice(available_arms) if available_arms else arm1_nih

                                        personality1_nih = domain_texts[arm1_nih] if arm1_nih < len(domain_texts) else None
                                        personality2_nih = domain_texts[arm2_nih] if arm2_nih < len(domain_texts) else None
                                        response1_nih, response2_nih, score_1_nih, score_2_nih, score_nih, preference_nih, p_nih = await generate_and_compare_async(
                                            init_instructions[arm1_nih], init_instructions[arm2_nih],
                                            ground_truth[1], llm_config,
                                            LLM_as_judge=LLM_as_judge, arm1_idx=arm1_nih, arm2_idx=arm2_nih,
                                            personality1=personality1_nih, personality2=personality2_nih,
                                            ground_truth_index=query_idx,
                                            counter=counter, query_idx=query_idx
                                        )

                                        greedy_score_nih = score_1_nih
                                        baseline_results[alg_name_nih]['values'].append(greedy_score_nih)
                                        baseline_results[alg_name_nih]['greedy_arm_index'] = arm1_nih
                                        if greedy_score_nih > baseline_best_values.get(alg_name_nih, -np.inf):
                                            baseline_best_values[alg_name_nih] = greedy_score_nih
                                            baseline_results[alg_name_nih]['best_greedy_arm_index'] = arm1_nih

                                        arm_grads_nih = alg_nih.calculate_gradients_for_arms(embeddings_np_baseline, [arm1_nih, arm2_nih])
                                        grad1_nih = arm_grads_nih[arm1_nih]
                                        grad2_nih = arm_grads_nih[arm2_nih]
                                        alg_nih.update_matrix(grad1_nih, grad2_nih)

                                        if hasattr(alg_nih, 'info_manager') and alg_nih.info_manager is not None:
                                            arm1_wins_nih = bool(preference_nih == 1)
                                            alg_nih.info_manager.update_pairwise_probability_with_transitive(arm1_nih, arm2_nih, arm1_wins_nih)

                                        if hasattr(alg_nih, 'train_model'):
                                            try:
                                                emb1_nih_loaded = embedding_manager.load_embedding(arm1_nih)
                                                emb2_nih_loaded = embedding_manager.load_embedding(arm2_nih)
                                                emb1_np_nih = emb1_nih_loaded.cpu().numpy().reshape(1, -1) if hasattr(emb1_nih_loaded, 'cpu') else emb1_nih_loaded.reshape(1, -1)
                                                emb2_np_nih = emb2_nih_loaded.cpu().numpy().reshape(1, -1) if hasattr(emb2_nih_loaded, 'cpu') else emb2_nih_loaded.reshape(1, -1)

                                                baseline_training_data[alg_name_nih]['X1'] = np.vstack([baseline_training_data[alg_name_nih]['X1'], emb1_np_nih])
                                                baseline_training_data[alg_name_nih]['X2'] = np.vstack([baseline_training_data[alg_name_nih]['X2'], emb2_np_nih])
                                                baseline_training_data[alg_name_nih]['Y'] = np.append(baseline_training_data[alg_name_nih]['Y'], preference_nih)
                                                baseline_training_data[alg_name_nih]['query_indices'].append(query_idx)

                                                alg_nih.train_model(
                                                    baseline_training_data[alg_name_nih]['X1'],
                                                    baseline_training_data[alg_name_nih]['X2'],
                                                    baseline_training_data[alg_name_nih]['Y'],
                                                    incremental=False
                                                )
                                                alg_nih._has_trained = True
                                            except Exception as e:
                                                print(f"⚠️ [{alg_name_nih}] Trainingfailed: {e}")
                                                import traceback
                                                traceback.print_exc()

                                        winner_nih = "arm1" if preference_nih == 1 else "arm2"
                                        print(f"    [{alg_name_nih}] arm1={arm1_nih}(score={score_1_nih:.4f}) vs arm2={arm2_nih}(score={score_2_nih:.4f}) → {winner_nih}", flush=True)

                                        baseline_iteration_results[alg_name_nih] = {
                                            'arms': (arm1_nih, arm2_nih),
                                            'scores': (score_1_nih, score_2_nih),
                                            'current_value': greedy_score_nih,
                                            'best_value': baseline_best_values.get(alg_name_nih, greedy_score_nih)
                                        }

                                        del arm_grads_nih, grad1_nih, grad2_nih
                                        torch.cuda.empty_cache()
                            except Exception as e:
                                print(f"⚠️ [BaselineIteration] {alg_name_nih} failed: {e}")
                                import traceback
                                traceback.print_exc()
                                baseline_results[alg_name_nih]['values'].append(0.0)

                        for alg_name, alg in baseline_algorithms.items():
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
                                    greedy_scores_linear = alg.calculate_scores_only(embeddings_np_baseline)
                                    arm1 = torch.argmax(greedy_scores_linear).item()

                                    if not hasattr(alg, 'info_manager') or alg.info_manager is None:
                                        from information_second_term import ContextualPairwiseInformationManager, PairwiseInformationManager
                                        bayesian_alpha_linear = config.get("pohf", {}).get("bayesian_alpha", 1.0)
                                        bt_isolated_arm_mode_linear = config.get("pohf", {}).get("bt_isolated_arm_mode", "unknown_isolated")
                                        reset_info_matrix_per_query_linear = config.get("pohf", {}).get("reset_info_matrix_per_query", False)

                                        if not hasattr(alg, 'contextual_info_manager') or alg.contextual_info_manager is None:
                                            alg.contextual_info_manager = ContextualPairwiseInformationManager(
                                                len(greedy_scores_linear), use_optimized=True, bayesian_alpha=bayesian_alpha_linear,
                                                bt_isolated_arm_mode=bt_isolated_arm_mode_linear
                                            )
                                        current_input_embedding_linear = query_emb.cpu().numpy()

                                        if reset_info_matrix_per_query_linear:
                                            alg.info_manager = alg.contextual_info_manager.initialize_without_history(
                                                input_index=query_idx, input_embedding=current_input_embedding_linear
                                            )
                                        else:
                                            alg.info_manager = alg.contextual_info_manager.initialize_for_new_input(
                                                input_index=query_idx, input_embedding=current_input_embedding_linear
                                            )

                                    from IDS_TAP_parameters.py import POHF_CONFIG
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

                                personality1_baseline = domain_texts[arm1] if arm1 < len(domain_texts) else None
                                personality2_baseline = domain_texts[arm2] if arm2 < len(domain_texts) else None
                                response1_baseline, response2_baseline, score_1_baseline, score_2_baseline, score_baseline, preference_baseline, p_baseline = await generate_and_compare_async(
                                    init_instructions[arm1], init_instructions[arm2],
                                    ground_truth[1], llm_config,
                                    LLM_as_judge=LLM_as_judge, arm1_idx=arm1, arm2_idx=arm2,
                                    personality1=personality1_baseline, personality2=personality2_baseline,
                                    ground_truth_index=query_idx,
                                    counter=counter, query_idx=query_idx
                                )

                                winner_baseline = "arm1" if preference_baseline == 1 else "arm2"
                                print(f"    [{alg_name}] arm1={arm1}(score={score_1_baseline:.4f}) vs arm2={arm2}(score={score_2_baseline:.4f}) → {winner_baseline}", flush=True)

                                if alg_name == 'POHF':
                                    greedy_score = score_1_baseline
                                    baseline_results[alg_name]['values'].append(greedy_score)
                                    baseline_results[alg_name]['greedy_arm_index'] = arm1
                                    if greedy_score > baseline_best_values[alg_name]:
                                        baseline_best_values[alg_name] = greedy_score
                                        baseline_results[alg_name]['best_greedy_arm_index'] = arm1
                                elif alg_name == 'Random':
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
                                    gt_for_scoring = ground_truth[1]
                                    if isinstance(gt_for_scoring, list):
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
                                    greedy_score_linear = score_1_baseline
                                    baseline_results[alg_name]['values'].append(greedy_score_linear)
                                    baseline_results[alg_name]['greedy_arm_index'] = arm1
                                    if greedy_score_linear > baseline_best_values[alg_name]:
                                        baseline_best_values[alg_name] = greedy_score_linear
                                        baseline_results[alg_name]['best_greedy_arm_index'] = arm1

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
                                    if hasattr(alg, 'info_manager') and alg.info_manager is not None:
                                        arm1_wins_linear = (score_1_baseline >= score_2_baseline)
                                        from information_second_term import update_information_with_feedback
                                        update_information_with_feedback(alg.info_manager, arm1, arm2, arm1_wins_linear)
                                        if hasattr(alg, 'contextual_info_manager') and alg.contextual_info_manager is not None:
                                            alg.contextual_info_manager.record_comparison(arm1, arm2, arm1_wins_linear)
                                else:
                                    alg.update(arm1, arm2, preference_baseline)

                                baseline_iteration_results[alg_name] = {
                                    'arms': (arm1, arm2),
                                    'scores': (score_1_baseline, score_2_baseline),
                                    'current_value': baseline_results[alg_name]['values'][-1],
                                    'best_value': baseline_best_values[alg_name]
                                }

                                emb1_baseline = embedding_manager.load_embedding(arm1)
                                emb2_baseline = embedding_manager.load_embedding(arm2)

                                if hasattr(alg, 'train_model') and alg_name != 'Random':
                                    try:
                                        emb1_np = emb1_baseline.cpu().numpy().reshape(1, -1) if hasattr(emb1_baseline, 'cpu') else emb1_baseline.reshape(1, -1)
                                        emb2_np = emb2_baseline.cpu().numpy().reshape(1, -1) if hasattr(emb2_baseline, 'cpu') else emb2_baseline.reshape(1, -1)

                                        baseline_training_data[alg_name]['X1'] = np.vstack([baseline_training_data[alg_name]['X1'], emb1_np])
                                        baseline_training_data[alg_name]['X2'] = np.vstack([baseline_training_data[alg_name]['X2'], emb2_np])
                                        baseline_training_data[alg_name]['Y'] = np.append(baseline_training_data[alg_name]['Y'], preference_baseline)
                                        baseline_training_data[alg_name]['query_indices'].append(query_idx)

                                        from IDS_TAP_parameters.py import TRAINING_CONFIG
                                        cross_query_incremental = TRAINING_CONFIG.get("cross_query_incremental", False)
                                        _query_decay_enabled_bl = TRAINING_CONFIG.get("query_decay_enabled", False)
                                        _query_similarity_enabled_bl = TRAINING_CONFIG.get("query_similarity_enabled", False)
                                        _use_simple_weight_bl = cross_query_incremental or (not _query_decay_enabled_bl and not _query_similarity_enabled_bl)

                                        W_baseline = []
                                        for q_idx in baseline_training_data[alg_name]['query_indices']:
                                            if _use_simple_weight_bl:
                                                weight = 1.0
                                            else:
                                                weight = compute_query_weight(q_idx, query_idx, query_embeddings, TRAINING_CONFIG)
                                            W_baseline.append(weight)
                                        W_baseline = np.array(W_baseline)

                                        if cross_query_incremental:
                                            alg.train_model(
                                                baseline_training_data[alg_name]['X1'],
                                                baseline_training_data[alg_name]['X2'],
                                                baseline_training_data[alg_name]['Y'],
                                                reset_to_query_start=True,
                                                weights=W_baseline
                                            )
                                        else:
                                            alg.train_model(
                                                baseline_training_data[alg_name]['X1'],
                                                baseline_training_data[alg_name]['X2'],
                                                baseline_training_data[alg_name]['Y'],
                                                incremental=False,
                                                weights=W_baseline
                                            )
                                        alg._has_trained = True
                                    except Exception as e:
                                        progress_logger.log_error(f"{alg_name} Training failed: {e}")

                            except Exception as e:
                                print(f"⚠️ [BaselineIteration] {alg_name} failed: {e}")
                                baseline_results[alg_name]['values'].append(0.0)
                                baseline_results[alg_name]['best_values'].append(baseline_best_values[alg_name])

                        del embeddings_np_baseline
                        del embeddings_list
                        torch.cuda.empty_cache()

                    greedy_arm_index = torch.argmax(greedy_scores).item()

                    if t == 0 or (t == 3 and query_idx == 3):
                        scores_np = greedy_scores.cpu().numpy().flatten()
                        top10_indices = np.argsort(scores_np)[-10:][::-1]
                        top10_scores = scores_np[top10_indices]
                        print(f"      📊 [DEBUG] greedy_scoresDistribution: min={scores_np.min():.4f}, max={scores_np.max():.4f}, "
                              f"mean={scores_np.mean():.4f}, std={scores_np.std():.4f}")
                        print(f"      📊 [DEBUG] Top10 arms: {list(zip(top10_indices.tolist(), [f'{s:.4f}' for s in top10_scores]))}")
                        if len(scores_np) > 38:
                            print(f"      📊 [DEBUG] arm10={scores_np[10]:.6f}, arm38={scores_np[38]:.6f}, diff={scores_np[10]-scores_np[38]:.6f}")

                    if algorithm_name == "POHF-InfoGain":
                        ucb_scores = greedy_scores.clone()
                    else:
                        ucb_scores = ucb_scores_precomputed

                    arm_select1 = greedy_arm_index

                    from IDS_TAP_parameters.py import POHF_CONFIG
                    base_info_gain_scale = POHF_CONFIG.get("info_gain_scale", 1.0)
                    info_gain_normalize = POHF_CONFIG.get("info_gain_normalize", True)

                    info_gain_decay_enabled = POHF_CONFIG.get("info_gain_decay_enabled", True)
                    info_gain_decay_factor = POHF_CONFIG.get("info_gain_decay_factor", 0.995)
                    info_gain_min_scale = POHF_CONFIG.get("info_gain_min_scale", 0.1)
                    info_gain_decay_start = POHF_CONFIG.get("info_gain_decay_start", 10)
                    info_gain_decay_type = POHF_CONFIG.get("info_gain_decay_type", "exponential")

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

                    if algorithm_name == "POHF-InfoGain":
                        if not hasattr(l, 'info_manager'):
                            bayesian_alpha = config.get("pohf", {}).get("bayesian_alpha", 1.0)
                            bt_isolated_arm_mode = config.get("pohf", {}).get("bt_isolated_arm_mode", "unknown_isolated")
                            l.info_manager = PairwiseInformationManager(
                                len(greedy_scores), use_optimized=True, bayesian_alpha=bayesian_alpha,
                                bt_isolated_arm_mode=bt_isolated_arm_mode
                            )

                        available_arms = list(range(len(greedy_scores)))
                        raw_info_gains = l.info_manager.get_all_information_gains(arm_select1, available_arms)
                        if info_gain_normalize:
                            info_gains = l.info_manager.get_normalized_information_gains(arm_select1, available_arms)
                        else:
                            info_gains = raw_info_gains

                        strict_ids = config.get("pohf", {}).get("strict_ids", False)

                        if strict_ids:
                            greedy_arm_score = greedy_scores[arm_select1].item()
                            epsilon = 1e-8

                            info_ratios = {}
                            for arm_idx in available_arms:
                                if arm_idx == arm_select1:
                                    continue
                                delta = greedy_arm_score - greedy_scores[arm_idx].item()
                                ig = info_gains.get(arm_idx, epsilon)
                                if ig <= 0:
                                    ig = epsilon
                                info_ratio = (delta ** 2) / ig
                                info_ratios[arm_idx] = info_ratio

                            arm_select2 = min(info_ratios, key=info_ratios.get)

                            combined_scores = greedy_scores.clone()
                            for arm_idx, ir in info_ratios.items():
                                combined_scores[arm_idx] = -ir
                            combined_scores[arm_select1] = float('-inf')
                        else:
                            combined_scores = greedy_scores.clone()
                            for arm_idx, info_gain in info_gains.items():
                                combined_scores[arm_idx] = greedy_scores[arm_idx] + info_gain_scale * info_gain

                            combined_scores[arm_select1] = float('-inf')
                            arm_select2 = torch.argmax(combined_scores).item()

                        isolated_count = sum(1 for a in available_arms if a != arm_select1 and np.sum(l.info_manager.total_counts[a, :]) == 0)
                        connected_count = len(available_arms) - 1 - isolated_count

                        greedy_arm1 = greedy_scores[arm_select1].item()
                        greedy_arm2 = greedy_scores[arm_select2].item()
                        ig_arm2_normalized = info_gains.get(arm_select2, 0)
                        ig_arm2_raw = raw_info_gains.get(arm_select2, 0)

                        ig_values = list(info_gains.values())
                        ig_unique = len(set([round(v, 6) for v in ig_values]))
                        ig_min, ig_max = min(ig_values), max(ig_values)

                        raw_ig_values = list(raw_info_gains.values())
                        raw_ig_min, raw_ig_max = min(raw_ig_values), max(raw_ig_values)

                        greedy_np = greedy_scores.cpu().numpy().flatten()
                        greedy_without_arm1 = np.delete(greedy_np, arm_select1)
                        greedy_min, greedy_max = greedy_without_arm1.min(), greedy_without_arm1.max()

                        if strict_ids:
                            delta_arm2 = greedy_arm1 - greedy_arm2
                            info_ratio_arm2 = info_ratios.get(arm_select2, 0)
                            print(f"      🎯 [POHF-InfoGain-StrictIDS] t={t}: arm1={arm_select1}(greedy={greedy_arm1:.4f}) → arm2={arm_select2}")
                            print(f"         📊 arm2score: greedy={greedy_arm2:.4f}, Δ={delta_arm2:.4f}, ig_norm={ig_arm2_normalized:.4f}, "
                                  f"info_ratio={info_ratio_arm2:.6f}")
                            print(f"         📈 Distribution: greedy[{greedy_min:.4f}~{greedy_max:.4f}], "
                                  f"ig_norm[{ig_min:.4f}~{ig_max:.4f}], ig_raw[{raw_ig_min:.6f}~{raw_ig_max:.6f}], ig_unique={ig_unique}/{len(ig_values)}")
                            print(f"         🔗 Status: isolated={isolated_count}, connected={connected_count}, mode=StrictIDS(argmin Δ²/g)")
                        else:
                            combined_arm2 = greedy_arm2 + info_gain_scale * ig_arm2_normalized
                            print(f"      🎯 [POHF-InfoGain] t={t}: arm1={arm_select1}(greedy={greedy_arm1:.4f}) → arm2={arm_select2}")
                            print(f"         📊 arm2score: greedy={greedy_arm2:.4f}, ig_norm={ig_arm2_normalized:.4f}, ig_raw={ig_arm2_raw:.6f}, "
                                  f"combined={combined_arm2:.4f} (scale={info_gain_scale:.2f})")
                            print(f"         📈 Distribution: greedy[{greedy_min:.4f}~{greedy_max:.4f}], "
                                  f"ig_norm[{ig_min:.4f}~{ig_max:.4f}], ig_raw[{raw_ig_min:.6f}~{raw_ig_max:.6f}], ig_unique={ig_unique}/{len(ig_values)}")
                            print(f"         🔗 Status: isolated={isolated_count}, connected={connected_count}")

                        combined_scores_np = combined_scores.cpu().numpy().flatten()
                        top10_indices = np.argsort(combined_scores_np)[-10:][::-1]
                        top10_info = []
                        for arm_idx in top10_indices:
                            if arm_idx == arm_select1:
                                continue
                            is_connected = np.sum(l.info_manager.total_counts[arm_idx, :]) > 0
                            status = "C" if is_connected else "I"
                            arm_greedy = greedy_scores[arm_idx].item()
                            arm_ig_raw = raw_info_gains.get(arm_idx, 0)
                            arm_ig_norm = info_gains.get(arm_idx, 0)
                            if strict_ids:
                                arm_info_ratio = info_ratios.get(arm_idx, float('inf'))
                                top10_info.append(f"{arm_idx}({status}):g={arm_greedy:.4f},ig={arm_ig_norm:.4f},ir={arm_info_ratio:.4f}")
                            else:
                                arm_combined = combined_scores_np[arm_idx]
                                top10_info.append(f"{arm_idx}({status}):g={arm_greedy:.4f},ig_r={arm_ig_raw:.6f},ig_n={arm_ig_norm:.4f},c={arm_combined:.4f}")
                        print(f"         🏆 Top10: {' | '.join(top10_info[:10])}")
                    else:
                        ucb_scores_copy = ucb_scores.clone()
                        ucb_scores_copy[arm_select1] = float('-inf')
                        arm_select2 = torch.argmax(ucb_scores_copy).item()

                    select_idx_history += [[arm_select1, arm_select2]]

                    if algorithm_name == "POHF-InfoGain":
                        second_arm_selections.append(arm_select2)

                    llm_config = config.get("llm", {})
                    personality1 = domain_texts[arm_select1] if arm_select1 < len(domain_texts) else None
                    personality2 = domain_texts[arm_select2] if arm_select2 < len(domain_texts) else None
                    response1, response2, score_1, score_2, score, new_y, p_ = await generate_and_compare_async(
                        init_instructions[arm_select1], init_instructions[arm_select2],
                        ground_truth[1], llm_config,
                        LLM_as_judge=LLM_as_judge, arm1_idx=arm_select1, arm2_idx=arm_select2,
                        personality1=personality1, personality2=personality2,
                        ground_truth_index=query_idx,
                        counter=counter, query_idx=query_idx
                    )

                    winner = "arm1" if new_y == 1 else "arm2"
                    if LLM_as_judge:
                        print(f"  [Query {query_idx+1}][Iter {t+1}/{current_max_iter}] arm1={arm_select1} vs arm2={arm_select2} → {winner} wins (LLM judge)", flush=True)
                    else:
                        print(f"  [Query {query_idx+1}][Iter {t+1}/{current_max_iter}] arm1={arm_select1}(score={score_1:.4f}) vs arm2={arm_select2}(score={score_2:.4f}) → {winner} wins", flush=True)

                    instruction_select_history += [(init_instructions[arm_select1], score_1, init_instructions[arm_select2], score_2)]

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
                        best_instruction_over_iter += [(t, init_instructions[arm_select1], score_1, query_idx)]
                        if best_score > best_r:
                            best_index = best_arm
                            best_r = best_score
                            print(f"  🏆 New best score: {best_r:.4f} (arm={best_index})", flush=True)

                    emb1 = embedding_manager.load_embedding(arm_select1)
                    emb2 = embedding_manager.load_embedding(arm_select2)

                    x_train += [torch.cat([emb1.reshape(1,1,-1), emb2.reshape(1,1,-1)])]
                    y_train += [new_y]
                    query_indices += [query_idx]

                    from IDS_TAP_parameters.py import TRAINING_CONFIG
                    cross_query_incremental = TRAINING_CONFIG.get("cross_query_incremental", False)
                    _query_decay_enabled_main = TRAINING_CONFIG.get("query_decay_enabled", False)
                    _query_similarity_enabled_main = TRAINING_CONFIG.get("query_similarity_enabled", False)
                    _use_simple_weight_main = cross_query_incremental or (not _query_decay_enabled_main and not _query_similarity_enabled_main)

                    X1_all = []
                    X2_all = []
                    Y_all = []
                    W_all = []
                    for x_pair, y_val, q_idx in zip(x_train, y_train, query_indices):
                        x_pair_reshaped = x_pair.squeeze(1)
                        X1_all.append(x_pair_reshaped[0].cpu().numpy())
                        X2_all.append(x_pair_reshaped[1].cpu().numpy())
                        Y_all.append(y_val)
                        if _use_simple_weight_main:
                            weight = 1.0
                        else:
                            weight = compute_query_weight(q_idx, query_idx, query_embeddings, TRAINING_CONFIG)
                        W_all.append(weight)
                    X1_all = np.array(X1_all)
                    X2_all = np.array(X2_all)
                    Y_all = np.array(Y_all)
                    W_all = np.array(W_all)

                    if cross_query_incremental:
                        l.train_model(X1_all, X2_all, Y_all, reset_to_query_start=True, weights=W_all)
                    else:
                        l.train_model(X1_all, X2_all, Y_all, incremental=False, weights=W_all)
                    l._has_trained = True

                    arm_grads = l.calculate_gradients_for_arms(embeddings_np, [arm_select1, arm_select2])
                    grad1 = arm_grads[arm_select1]
                    grad2 = arm_grads[arm_select2]
                    l.update_matrix(grad1, grad2)

                    del arm_grads, grad1, grad2
                    del embeddings_np
                    torch.cuda.empty_cache()

                    if algorithm_name == "POHF-InfoGain" and hasattr(l, 'info_manager'):
                        arm1_wins = (score_1 >= score_2)
                        from information_second_term import update_information_with_feedback
                        info_gain = update_information_with_feedback(l.info_manager, arm_select1, arm_select2, arm1_wins)
                        if hasattr(l, 'contextual_info_manager') and l.contextual_info_manager is not None:
                            l.contextual_info_manager.record_comparison(arm_select1, arm_select2, arm1_wins)

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

                    if query_idx == 0 and 'POHF-InfoGain-NoHistory' in baseline_results:
                        alg_name_nih = 'POHF-InfoGain-NoHistory'
                        baseline_results[alg_name_nih]['values'].append(score_1)
                        baseline_results[alg_name_nih]['greedy_arm_index'] = greedy_arm_index
                        if score_1 > baseline_best_values.get(alg_name_nih, -np.inf):
                            baseline_best_values[alg_name_nih] = score_1
                            baseline_results[alg_name_nih]['best_greedy_arm_index'] = greedy_arm_index
                        print(f"    [POHF-InfoGain-NoHistory] Query 0: Reusing main algorithm result (greedy_arm={greedy_arm_index}, score={score_1:.4f})", flush=True)

                        baseline_iteration_results[alg_name_nih] = {
                            'arms': (arm_select1, arm_select2),
                            'scores': (score_1, score_2),
                            'current_value': score_1,
                            'best_value': baseline_best_values.get(alg_name_nih, score_1)
                        }

                if algorithm_name == "POHF-InfoGain" and hasattr(l, 'contextual_info_manager') and l.contextual_info_manager is not None:
                    l.contextual_info_manager.finalize_query()

                if 'Linear-InfoGain' in baseline_algorithms:
                    alg_linear = baseline_algorithms['Linear-InfoGain']
                    if hasattr(alg_linear, 'contextual_info_manager') and alg_linear.contextual_info_manager is not None:
                        alg_linear.contextual_info_manager.finalize_query()

                final_greedy_arm_index = greedy_arm_index if 'greedy_arm_index' in locals() else 0
                algorithm_result = {
                    "algorithm": algorithm_name,
                    "counter": counter,
                    "run_index": run_index,
                    "final_greedy_arm_index": final_greedy_arm_index,
                    "baseline_results": baseline_results if (baseline_algorithms or 'baseline_results' in locals()) else {},
                    "second_arm_selections": second_arm_selections if algorithm_name == "POHF-InfoGain" else [],
                    "total_arms": len(greedy_scores) if 'greedy_scores' in locals() else times,
                    "contextual_mode": contextual_mode_enabled,
                    "num_queries": num_queries
                }

                if not LLM_as_judge:
                    algorithm_result["best_instruction_over_iter"] = best_instruction_over_iter
                    algorithm_result["best_score"] = best_r
                    algorithm_result["best_index"] = best_index
                    algorithm_result["total_iterations"] = len(best_instruction_over_iter)

                counter_results.append(algorithm_result)

                query_key = f"query_{query_idx}"
                if query_key not in counter_persona_data["queries"]:
                    counter_persona_data["queries"][query_key] = {
                        "query_index": query_idx,
                        "query_text": current_query if 'current_query' in locals() else "",
                        "ground_truth": ground_truth_data[query_idx] if isinstance(ground_truth_data, list) and query_idx < len(ground_truth_data) else str(ground_truth_data),
                        "algorithms": {}
                    }

                if algorithm_name == "POHF-InfoGain" and final_greedy_arm_index < len(domain_texts):
                    greedy_persona = domain_texts[final_greedy_arm_index].replace("### Summary:\n", "")
                    greedy_prompt = init_instructions[final_greedy_arm_index] if final_greedy_arm_index < len(init_instructions) else ""
                    greedy_personality = domain_texts[final_greedy_arm_index] if final_greedy_arm_index < len(domain_texts) else None
                    greedy_response = await response_generator_async(greedy_prompt, llm_config, greedy_personality)

                    pohf_infogain_data = {
                        "greedy_arm_index": final_greedy_arm_index,
                        "persona_summary": greedy_persona,
                        "response": greedy_response,
                    }
                    if not LLM_as_judge:
                        pohf_infogain_data["greedy_score"] = best_r
                    counter_persona_data["queries"][query_key]["algorithms"]["POHF-InfoGain"] = pohf_infogain_data

                if baseline_algorithms and baseline_results:
                    for baseline_name, baseline_data in baseline_results.items():
                        baseline_greedy_idx = baseline_data.get('greedy_arm_index', 0)
                        if baseline_greedy_idx < len(domain_texts):
                            baseline_persona = domain_texts[baseline_greedy_idx].replace("### Summary:\n", "")
                            baseline_prompt = init_instructions[baseline_greedy_idx] if baseline_greedy_idx < len(init_instructions) else ""
                            baseline_personality = domain_texts[baseline_greedy_idx] if baseline_greedy_idx < len(domain_texts) else None
                            baseline_response = await response_generator_async(baseline_prompt, llm_config, baseline_personality)

                            baseline_algo_data = {
                                "greedy_arm_index": baseline_greedy_idx,
                                "persona_summary": baseline_persona,
                                "response": baseline_response,
                            }
                            if not LLM_as_judge:
                                if baseline_name in ['Random', 'DoubleTS']:
                                    baseline_score = baseline_data['values'][-1] if baseline_data.get('values') else None
                                else:
                                    baseline_score = baseline_best_values.get(baseline_name, None) if 'baseline_best_values' in locals() else None
                                baseline_algo_data["greedy_score"] = baseline_score

                            counter_persona_data["queries"][query_key]["algorithms"][baseline_name] = baseline_algo_data

            if not LLM_as_judge:
                progress_logger.log_Complete(best_score=best_r, greedy_arm=final_greedy_arm_index)
            else:
                progress_logger.log_Complete()

        for result in counter_results:
            all_results.append(result)

        if not LLM_as_judge and counter_results:
            main_result = next((res for res in counter_results if res.get('algorithm') == algorithm_name), None)

            if main_result and main_result.get('best_instruction_over_iter'):
                full_data = main_result['best_instruction_over_iter']
                baseline_data = main_result.get('baseline_results', {})

                q_iters = unified_training_rounds

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
                full_second_arm_selections = main_result.get('second_arm_selections', [])
                if full_second_arm_selections:
                    plot_second_arm_selection_stats(full_second_arm_selections, counter, q0_result['total_arms'], get_dataset_name(LaMP_type), num_queries=num_queries)
                print(f"   📊 [图1] Query 0 Learning curve (counter={counter}, rounds={len(q0_data)})")

                full_result_for_avg = {
                    'algorithm': algorithm_name,
                    'counter': counter,
                    'best_instruction_over_iter': full_data,
                    'baseline_results': baseline_data,
                    'total_arms': main_result.get('total_arms', times),
                    'second_arm_selections': main_result.get('second_arm_selections', []),
                    'contextual_mode': contextual_mode_enabled,
                    'num_queries': num_queries,
                    'total_iterations': len(full_data)
                }
                all_first_query_results.append({'counter': counter, 'results': [full_result_for_avg]})

                if num_queries >= 1 and contextual_mode_enabled:
                    for q_idx in range(0, num_queries):
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
                            vmax = max(all_vals)
                            norm = lambda v, vmax=vmax: v / vmax if vmax > 0 else 0.0

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
                            alg_progress = {alg: [data[q]['final_value_normalized'] for q in q_indices if q in data] for alg, data in query_final_values.items()}
                            alg_progress = {k: v for k, v in alg_progress.items() if v}

                            rand_avg_line = None
                            if 'Random' in query_final_values:
                                rand_avgs = [query_final_values['Random'].get(q, {}).get('avg_value_normalized', None) for q in q_indices]
                                rand_avgs = [v for v in rand_avgs if v is not None]
                                if rand_avgs:
                                    rand_avg_line = np.mean(rand_avgs)

                            print(f"\n   📊 [Diagnostics] Query progress details (counter={counter}):")
                            for q_idx in q_indices:
                                print(f"      Query {q_idx}:")
                                for alg_name, alg_data in query_final_values.items():
                                    if q_idx in alg_data:
                                        raw = alg_data[q_idx].get('final_value_raw', 'N/A')
                                        norm_val = alg_data[q_idx].get('final_value_normalized', 'N/A')
                                        print(f"         {alg_name}: raw={raw:.4f}, norm={norm_val:.4f}" if isinstance(raw, (int, float)) else f"         {alg_name}: {raw}")

                            q_prog_data = {'counter': counter, 'query_indices': q_indices, 'algorithms': alg_progress, 'random_avg_line': rand_avg_line}
                            plot_query_progress(q_prog_data, counter, get_dataset_name(LaMP_type))
                            print(f"   📊 [图2] Cross-Query progress plot (counter={counter})")
                            all_query_progress_data.append(q_prog_data)

        else:
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
                    greedy_personality = domain_texts[final_greedy_arm_idx] if final_greedy_arm_idx < len(domain_texts) else None
                    greedy_persona_text = greedy_personality.replace("### Summary:\n", "") if greedy_personality else None
                    cache_key = greedy_prompt + (greedy_personality or "")
                    prompt_hash = hashlib.md5(cache_key.encode()).hexdigest()
                    if prompt_hash in _response_cache:
                        greedy_response = _response_cache[prompt_hash]
                    else:
                        greedy_response = await response_generator_async(greedy_prompt, llm_config, greedy_personality)
                else:
                    greedy_prompt = "Index out of range"
                    greedy_response = "N/A"
                    greedy_persona_text = None

                counter_greedy_data["algorithms"][algorithm_name] = {
                    "greedy_arm_index": final_greedy_arm_idx,
                    "persona": greedy_persona_text,
                    "response": greedy_response
                }

                baseline_results_data = result.get('baseline_results', {})
                for baseline_name, baseline_data in baseline_results_data.items():
                    if baseline_name not in counter_greedy_data["algorithms"]:
                        final_baseline_greedy_idx = baseline_data.get('greedy_arm_index', 0)

                        if final_baseline_greedy_idx < len(init_instructions):
                            baseline_prompt = init_instructions[final_baseline_greedy_idx]
                            baseline_personality = domain_texts[final_baseline_greedy_idx] if final_baseline_greedy_idx < len(domain_texts) else None
                            baseline_persona_text = baseline_personality.replace("### Summary:\n", "") if baseline_personality else None
                            baseline_cache_key = baseline_prompt + (baseline_personality or "")
                            baseline_prompt_hash = hashlib.md5(baseline_cache_key.encode()).hexdigest()
                            if baseline_prompt_hash in _response_cache:
                                baseline_response = _response_cache[baseline_prompt_hash]
                            else:
                                baseline_response = await response_generator_async(baseline_prompt, llm_config, baseline_personality)
                        else:
                            baseline_prompt = "Index out of range"
                            baseline_response = "N/A"
                            baseline_persona_text = None

                        counter_greedy_data["algorithms"][baseline_name] = {
                            "greedy_arm_index": final_baseline_greedy_idx,
                            "persona": baseline_persona_text,
                            "response": baseline_response
                        }

            all_greedy_arm_results.append(counter_greedy_data)

            try:
                import json
                save_dir = "./final_su"
                os.makedirs(save_dir, exist_ok=True)

                json_filename = f"{get_dataset_name(LaMP_type)}_greedy_prompts.json"
                json_filepath = os.path.join(save_dir, json_filename)

                existing_results = None
                if os.path.exists(json_filepath):
                    try:
                        with open(json_filepath, 'r', encoding='utf-8') as f:
                            existing_results = json.load(f)
                    except (json.JSONDecodeError, IOError) as json_err:
                        print(f"⚠️ [Incremental save] Failed to read existing JSON: {json_err}")
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
                progress_logger.log_error(f"Incremental save failed: {save_err}")

        import json
        from datetime import datetime

        counter_persona_data["saved_at"] = datetime.now().isoformat()

        persona_save_dir = "./persona_results"
        os.makedirs(persona_save_dir, exist_ok=True)

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
            print(f"   💾 [Persona] Saving all algorithm persona data: {persona_filename}")
        except Exception as e:
            print(f"   ⚠️ [Persona] Save failed: {e}")

        embedding_manager.clear_cache()
        if os.path.exists(embedding_filename):
            pass

        if LLM_as_judge:
            _response_cache.clear()

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

    if not LLM_as_judge and len(all_results) > 1 and not single_counter_mode:
        if all_first_query_results:
            first_query_combined_results = []
            for fq_data in all_first_query_results:
                first_query_combined_results.extend(fq_data['results'])

            if first_query_combined_results:
                plot_counter_average_results(first_query_combined_results, get_dataset_name(LaMP_type))
                print(f"   📊 [Figure 1 Counter Average] 已GeneratingFirst Query Counter average plot")
        else:
            plot_counter_average_results(all_results, get_dataset_name(LaMP_type))

        if all_query_progress_data:
            plot_query_progress_counter_average(all_query_progress_data, get_dataset_name(LaMP_type))
            print(f"   📊 [Figure 2 Counter Average] Generated cross-Query progress Counter average plot")

    return all_results

def get_unique_filename(directory, base_filename):
    full_path = os.path.join(directory, base_filename)

    if not os.path.exists(full_path):
        return full_path

    name, ext = os.path.splitext(base_filename)
    counter = 1

    while True:
        new_filename = f"{name}_{counter:03d}{ext}"
        new_full_path = os.path.join(directory, new_filename)

        if not os.path.exists(new_full_path):
            return new_full_path

        counter += 1

        if counter > 999:
            import time
            timestamp_suffix = str(int(time.time()))
            new_filename = f"{name}_{timestamp_suffix}{ext}"
            new_full_path = os.path.join(directory, new_filename)
            return new_full_path

def get_plot_config_from_baseline():
    from IDS_TAP_parameters.py import BASELINE_CONFIG

    display_config = BASELINE_CONFIG.get("algorithm_display_config", {})
    plot_config = BASELINE_CONFIG.get("plot_config", {})

    colors = {}
    markers = {}

    for alg_name, config in display_config.items():
        if config.get('show_in_plots', True):
            colors[alg_name] = config.get('color', '#333333')
            markers[alg_name] = config.get('marker', 'o')

    algorithms_with_range = set(plot_config.get("show_range_for_algorithms", ['Random']))
    exclude_from_minmax = plot_config.get("exclude_from_minmax", ['Random'])

    return colors, markers, algorithms_with_range, exclude_from_minmax

def generate_minmax_counter_average_plot(algorithm_average_curves_minmax, algorithm_max_curves_minmax, algorithm_min_curves_minmax, timestamp, counter_count, lamp_type=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    if not algorithm_average_curves_minmax:
        return

    plt.figure(figsize=(14, 10))

    colors, markers, algorithms_with_range, _ = get_plot_config_from_baseline()

    for alg_name, average_curve in algorithm_average_curves_minmax.items():
        if not average_curve:
            continue

        x_values = list(range(len(average_curve)))
        upper_curve = algorithm_max_curves_minmax[alg_name]
        lower_curve = algorithm_min_curves_minmax[alg_name]
        color = colors.get(alg_name, '#333333')

        plt.fill_between(x_values, lower_curve, upper_curve,
                        color=color, alpha=0.2)

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

    plt.title(f'Counter Average Performance Curves (Min-Max Normalized)\n(Min-Max normalized scores averaged across {counter_count} counters, all algorithms show mean±1SE range)',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Min-Max Normalized Score', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')

    plt.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.02, 0.98), ncol=1)

    plt.grid(True, alpha=0.3)

    plt.xlim(0, max([len(curve) for curve in algorithm_average_curves_minmax.values()]) - 1)

    all_values = []
    for curve in algorithm_average_curves_minmax.values():
        all_values.extend(curve)

    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min

        margin = max(0.05, y_range * 0.05)
        plt.ylim(y_min - margin, y_max + margin)
    else:
        plt.ylim(0, 1.05)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()

    save_dir = "./past_average"
    os.makedirs(save_dir, exist_ok=True)

    if lamp_type is None:
        try:
            from IDS_TAP_parameters.py import PATH_CONFIG
            lamp_type = PATH_CONFIG.get('LaMP_type', 'unknown')
        except:
            lamp_type = 'unknown'

    base_filename = f"{get_filename_prefix(lamp_type)}_counter_average_curves_minmax_{timestamp}.pdf"

    filename = get_unique_filename(save_dir, base_filename)

    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    except Exception as e:
        print(f"❌ [generate_minmax_counter_average_plot] Failed to save image: {e}")

    plt.close()

def plot_counter_average_results(all_results, lamp_type=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    if not all_results:
        return

    algorithm_iteration_data = {}
    max_iterations = 0

    for result in all_results:
        counter = result['counter']
        best_instruction_over_iter = result.get('best_instruction_over_iter', [])
        baseline_results = result.get('baseline_results', {})

        algorithm_name = result.get('algorithm', 'POHF-InfoGain')
        if best_instruction_over_iter:
            if algorithm_name not in algorithm_iteration_data:
                algorithm_iteration_data[algorithm_name] = []

            main_scores = []
            for item in best_instruction_over_iter:
                if len(item) >= 3:
                    main_scores.append(item[2])
                elif isinstance(item, (int, float)):
                    main_scores.append(item)
            algorithm_iteration_data[algorithm_name].append(main_scores)
            max_iterations = max(max_iterations, len(main_scores))

        for alg_name, alg_results in baseline_results.items():
            if 'values' in alg_results and alg_results['values']:
                if alg_name not in algorithm_iteration_data:
                    algorithm_iteration_data[alg_name] = []

                baseline_scores = alg_results['values']
                algorithm_iteration_data[alg_name].append(baseline_scores)
                max_iterations = max(max_iterations, len(baseline_scores))

    max_iterations = 0
    for counter_data_list in algorithm_iteration_data.values():
        for scores in counter_data_list:
            max_iterations = max(max_iterations, len(scores))

    counter_iteration_grouped_scores = {}

    for counter_idx, result in enumerate(all_results):
        counter = result['counter']
        counter_iteration_grouped_scores[counter] = {}

        for iteration_idx in range(max_iterations):
            counter_iteration_grouped_scores[counter][iteration_idx] = []

            for alg_name, counter_data_list in algorithm_iteration_data.items():
                if alg_name != 'Random' and counter_idx < len(counter_data_list):
                    scores = counter_data_list[counter_idx]
                    if iteration_idx < len(scores):
                        counter_iteration_grouped_scores[counter][iteration_idx].append(scores[iteration_idx])

    _, _, _, exclude_from_minmax = get_plot_config_from_baseline()

    counter_normalization = {}

    for counter_idx in range(len(all_results)):
        all_scores_for_counter = []

        for alg_name, counter_data_list in algorithm_iteration_data.items():
            if alg_name not in exclude_from_minmax and counter_idx < len(counter_data_list):
                scores = counter_data_list[counter_idx]
                all_scores_for_counter.extend(scores)

        if all_scores_for_counter:
            counter_normalization[counter_idx] = {
                'max': max(all_scores_for_counter),
                'min': min(all_scores_for_counter)
            }
        else:
            counter_normalization[counter_idx] = {'max': 1.0, 'min': 0.0}

    algorithm_average_curves = {}
    algorithm_max_curves = {}
    algorithm_min_curves = {}

    for alg_name, counter_data_list in algorithm_iteration_data.items():

        normalized_curves = []

        for counter_idx, scores in enumerate(counter_data_list):
            norm_info = counter_normalization.get(counter_idx)
            normalized_scores = []
            for score in scores:
                if norm_info and norm_info['max'] > 0:
                    normalized_scores.append(score / norm_info['max'])
                else:
                    normalized_scores.append(0.0)

            normalized_curves.append(normalized_scores)

        if normalized_curves:
            max_len = max(len(curve) for curve in normalized_curves)

            average_curve = []
            upper_curve = []
            lower_curve = []

            for i in range(max_len):
                values_at_position = []
                for curve in normalized_curves:
                    if i < len(curve):
                        values_at_position.append(curve[i])

                if values_at_position:
                    n_valid = len(values_at_position)
                    mean_val = np.mean(values_at_position)
                    if n_valid > 1:
                        std_val = np.std(values_at_position, ddof=1)
                        se_val = std_val / np.sqrt(n_valid)
                    else:
                        se_val = 0.0

                    average_curve.append(mean_val)
                    upper_curve.append(mean_val + se_val)
                    lower_curve.append(mean_val - se_val)

            algorithm_average_curves[alg_name] = average_curve
            algorithm_max_curves[alg_name] = upper_curve
            algorithm_min_curves[alg_name] = lower_curve

    algorithm_average_curves_minmax = {}
    algorithm_max_curves_minmax = {}
    algorithm_min_curves_minmax = {}

    for alg_name, counter_data_list in algorithm_iteration_data.items():

        minmax_normalized_curves = []

        for counter_idx, scores in enumerate(counter_data_list):
            norm_info = counter_normalization.get(counter_idx)
            minmax_normalized_scores = []
            for score in scores:
                if norm_info and norm_info['max'] > norm_info['min']:
                    minmax_normalized_scores.append(
                        (score - norm_info['min']) / (norm_info['max'] - norm_info['min'])
                    )
                elif norm_info and norm_info['max'] == norm_info['min']:
                    minmax_normalized_scores.append(1.0)
                else:
                    minmax_normalized_scores.append(0.0)

            minmax_normalized_curves.append(minmax_normalized_scores)

        if minmax_normalized_curves:
            max_len = max(len(curve) for curve in minmax_normalized_curves)

            minmax_average_curve = []
            minmax_upper_curve = []
            minmax_lower_curve = []

            for i in range(max_len):
                values_at_position = []
                for curve in minmax_normalized_curves:
                    if i < len(curve):
                        values_at_position.append(curve[i])

                if values_at_position:
                    n_valid = len(values_at_position)
                    mean_val = np.mean(values_at_position)
                    if n_valid > 1:
                        std_val = np.std(values_at_position, ddof=1)
                        se_val = std_val / np.sqrt(n_valid)
                    else:
                        se_val = 0.0

                    minmax_average_curve.append(mean_val)
                    minmax_upper_curve.append(mean_val + se_val)
                    minmax_lower_curve.append(mean_val - se_val)

            algorithm_average_curves_minmax[alg_name] = minmax_average_curve
            algorithm_max_curves_minmax[alg_name] = minmax_upper_curve
            algorithm_min_curves_minmax[alg_name] = minmax_lower_curve
        else:
            algorithm_average_curves_minmax[alg_name] = []
            algorithm_max_curves_minmax[alg_name] = []
            algorithm_min_curves_minmax[alg_name] = []

    plt.figure(figsize=(14, 10))

    colors, markers, algorithms_with_range, _ = get_plot_config_from_baseline()

    for alg_name, average_curve in algorithm_average_curves.items():
        x_values = list(range(len(average_curve)))
        upper_curve = algorithm_max_curves[alg_name]
        lower_curve = algorithm_min_curves[alg_name]

        color = colors.get(alg_name, '#999999')

        plt.fill_between(x_values, lower_curve, upper_curve,
                        color=color, alpha=0.2)

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

    from IDS_TAP_parameters.py import DATA_CONFIG
    counter_count = DATA_CONFIG.get("counter_array_length", 40)
    plt.title(f'Counter Average Performance Curves\n(Normalized scores averaged across {counter_count} counters, all algorithms show mean±1SE range)',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Normalized Score', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')

    plt.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.02, 0.98), ncol=1)

    plt.grid(True, alpha=0.3)

    plt.xlim(0, max([len(curve) for curve in algorithm_average_curves.values()]) - 1)

    all_values = []
    for curve in algorithm_average_curves.values():
        all_values.extend(curve)

    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min

        margin = max(0.05, y_range * 0.05)
        plt.ylim(y_min - margin, y_max + margin)
    else:
        plt.ylim(0, 1.05)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "./past_average"
    os.makedirs(save_dir, exist_ok=True)

    if lamp_type is None:
        try:
            from IDS_TAP_parameters.py import PATH_CONFIG
            lamp_type = PATH_CONFIG.get('LaMP_type', 'unknown')
        except:
            lamp_type = 'unknown'

    base_filename = f"{get_filename_prefix(lamp_type)}_counter_average_curves_with_range_{timestamp}.pdf"

    filename = get_unique_filename(save_dir, base_filename)

    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    except Exception as e:
        print(f"❌ [plot_counter_average_results] Failed to save image: {e}")

    plt.close()

    generate_minmax_counter_average_plot(algorithm_average_curves_minmax, algorithm_max_curves_minmax, algorithm_min_curves_minmax, timestamp, counter_count, lamp_type)

    base_data_filename = f"{get_filename_prefix(lamp_type)}_counter_average_data_{timestamp}.json"

    data_filename = get_unique_filename(save_dir, base_data_filename)

    export_data = {}

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
                "all_iterations": {},
                "by_query": {}
            }

        for alg_name, counter_data_list in algorithm_iteration_data.items():
            if counter_idx < len(counter_data_list):
                export_data[counter_key]["all_iterations"][alg_name] = counter_data_list[counter_idx]

        best_instruction_over_iter = result.get('best_instruction_over_iter', [])
        baseline_results_data = result.get('baseline_results', {})

        if best_instruction_over_iter:
            query_grouped_main = {}
            for item in best_instruction_over_iter:
                if len(item) >= 4:
                    t, instruction, score, query_idx = item
                elif len(item) >= 3:
                    t, instruction, score = item
                    query_idx = 0
                else:
                    continue

                query_key = f"query_{query_idx}"
                if query_key not in query_grouped_main:
                    query_grouped_main[query_key] = []
                query_grouped_main[query_key].append(score)

            algorithm_name = result.get('algorithm', 'POHF-InfoGain')
            for query_key, scores in query_grouped_main.items():
                if query_key not in export_data[counter_key]["by_query"]:
                    export_data[counter_key]["by_query"][query_key] = {}
                export_data[counter_key]["by_query"][query_key][algorithm_name] = scores

            if baseline_results_data and query_grouped_main:
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

    import json
    try:
        with open(data_filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"   📄 [JSON] Saving complete data to: {data_filename}")
    except Exception as e:
        print(f"❌ [plot_counter_average_results] Failed to save JSON data: {e}")

def plot_pohf_results_with_baselines(pohf_results, baseline_results=None, save_to_file=True, show_plot=False, filename_prefix=None):
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    if not pohf_results:
        return

    pohf_x = [p[0] for p in pohf_results]
    pohf_y = [p[2] for p in pohf_results]

    plt.figure(figsize=(14, 10))

    from IDS_TAP_parameters.py import POHF_CONFIG
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

    if baseline_results:
        colors, markers, _, _ = get_plot_config_from_baseline()

        for alg_name, results in baseline_results.items():
            if 'values' in results and results['values']:
                x_baseline = list(range(len(results['values'])))
                y_baseline = results['values']

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

    plt.title('POHF vs Baseline Algorithms Performance Comparison',
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Best Score', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()

    if save_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = "./plots"
        os.makedirs(save_dir, exist_ok=True)

        if filename_prefix:
            filename = os.path.join(save_dir, f"{filename_prefix}_comparison_{timestamp}.pdf")
        else:
            filename = os.path.join(save_dir, f"pohf_baseline_comparison_{timestamp}.pdf")

        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

        if not filename_prefix:
            latest_filename = os.path.join(save_dir, "pohf_baseline_comparison_latest.pdf")
            plt.savefig(latest_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

    if show_plot:
        try:
            plt.show()
        except:
            pass
    else:
        plt.close()

def plot_pohf_results(outcome, save_to_file=True, show_plot=False, filename_prefix=None):
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    if not outcome:
        return

    x = [p[0] for p in outcome]
    y = [p[2] for p in outcome]

    plt.figure(figsize=(12, 8))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=8,
             color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=2)

    plt.xlabel("Iterations", fontsize=14, fontweight='bold')
    plt.ylabel("Reward Score", fontsize=14, fontweight='bold')
    plt.title("POHF Algorithm: Reward Evolution Over Iterations", fontsize=16, fontweight='bold', pad=20)

    max_score = max(y)
    plt.ylim(0.0, max_score * 1.1)

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)

    for xi, yi in zip(x, y):
        plt.annotate(f'{yi:.4f}', (xi, yi), textcoords="offset points",
                    xytext=(0,15), ha='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

    min_score = min(y)
    avg_score = sum(y) / len(y)

    stats_text = f"Max: {max_score:.4f}\\nMin: {min_score:.4f}\\nAvg: {avg_score:.4f}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
             fontsize=10)

    plt.tight_layout()

    if save_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = "./plots"
        os.makedirs(save_dir, exist_ok=True)

        if filename_prefix:
            filename = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.pdf")
        else:
            filename = os.path.join(save_dir, f"pohf_results_{timestamp}.pdf")

        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

        if not filename_prefix:
            latest_filename = os.path.join(save_dir, "pohf_latest_results.pdf")
            plt.savefig(latest_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

    if show_plot:
        try:
            plt.show()
        except:
            pass
    else:
        plt.close()

def plot_dual_algorithm_comparison(algorithm_results, counter, lamp_type=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 10))

    colors, markers, _, _ = get_plot_config_from_baseline()

    plotted_algorithms = set()
    plotted_baselines = set()

    for result in algorithm_results:
        algorithm_name = result['algorithm']

        if algorithm_name in plotted_algorithms:
            continue

        best_instruction_over_iter = result['best_instruction_over_iter']

        if best_instruction_over_iter and isinstance(best_instruction_over_iter[0], tuple):
            best_values = [item[2] for item in best_instruction_over_iter if len(item) >= 3]
        else:
            best_values = best_instruction_over_iter

        x_values = list(range(len(best_values)))
        color = colors.get(algorithm_name, '#999999')
        marker = markers.get(algorithm_name, 'o')

        plt.plot(x_values, best_values,
                marker=marker, linestyle='-', linewidth=3, markersize=8,
                color=color, label=algorithm_name, alpha=0.9)

        plotted_algorithms.add(algorithm_name)

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

    plt.title(f'Algorithm Performance Comparison (Counter {counter})',
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Best Score', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)

    if lamp_type is None:
        try:
            from IDS_TAP_parameters.py import PATH_CONFIG
            lamp_type = PATH_CONFIG.get('LaMP_type', 'unknown')
        except:
            lamp_type = 'unknown'

    save_dir = "./plots"
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{get_filename_prefix(lamp_type)}_algorithm_comparison_counter_{counter}.pdf")

    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def plot_second_arm_selection_stats(second_arm_selections, counter, total_arms, lamp_type=None, num_queries=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    if not second_arm_selections:
        return

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    arm_counts = Counter(second_arm_selections)
    arms = list(range(total_arms))
    frequencies = [arm_counts.get(arm, 0) for arm in arms]

    bars = ax1.bar(arms, frequencies, alpha=0.7, color='steelblue', edgecolor='navy', linewidth=0.8)
    ax1.set_xlabel('Arm Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Selection Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Second Arm Selection Frequency - All Queries (Counter {counter})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    for bar, freq in zip(bars, frequencies):
        if freq > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(freq), ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xticks(arms)
    ax1.set_xlim(-0.5, total_arms - 0.5)

    iterations = list(range(len(second_arm_selections)))
    colors = plt.cm.tab20(np.linspace(0, 1, total_arms))

    arm_colors = [colors[arm] for arm in second_arm_selections]

    scatter = ax2.scatter(iterations, second_arm_selections, c=arm_colors,
                         alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Iteration (across all queries)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Selected Arm Index', fontsize=12, fontweight='bold')
    ax2.set_title(f'Second Arm Selection Timeline - All Queries (Counter {counter})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    if num_queries is not None and num_queries > 1:
        total_iters = len(second_arm_selections)
        iters_per_query = total_iters // num_queries
        for q in range(1, num_queries):
            query_boundary = q * iters_per_query
            ax2.axvline(x=query_boundary - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax2.text(query_boundary, total_arms - 0.3, f'Q{q}', fontsize=9, color='red',
                    ha='center', va='top', fontweight='bold')
        ax2.text(iters_per_query / 2, total_arms - 0.3, 'Q0', fontsize=9, color='red',
                ha='center', va='top', fontweight='bold')

    ax2.set_yticks(arms)
    ax2.set_ylim(-0.5, total_arms - 0.5)

    total_selections = len(second_arm_selections)
    unique_arms = len(arm_counts)
    most_selected_arm = max(arm_counts, key=arm_counts.get) if arm_counts else 0
    most_selected_count = arm_counts[most_selected_arm] if arm_counts else 0

    stats_text = f"""Statistics (All Queries):
    Total Iterations: {total_selections}
    Num Queries: {num_queries if num_queries else 'N/A'}
    Unique Arms Selected: {unique_arms}/{total_arms}
    Most Selected Arm: {most_selected_arm} ({most_selected_count} times)
    Selection Rate: {most_selected_count/total_selections*100:.1f}%"""

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if lamp_type is None:
        try:
            from IDS_TAP_parameters.py import PATH_CONFIG
            lamp_type = PATH_CONFIG.get('LaMP_type', 'unknown')
        except:
            lamp_type = 'unknown'

    save_dir = "./plots"
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{get_filename_prefix(lamp_type)}_second_arm_selection_counter_{counter}.pdf")

    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def plot_query_progress(query_progress_data, counter, lamp_type=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if not query_progress_data or not query_progress_data.get('query_indices'):
        return

    query_indices = query_progress_data['query_indices']
    algorithms_data = query_progress_data['algorithms']
    random_avg_line = query_progress_data.get('random_avg_line', None)

    if not algorithms_data:
        return

    plt.figure(figsize=(12, 8))

    colors, markers, _, _ = get_plot_config_from_baseline()

    for alg_name, values in algorithms_data.items():
        if not values:
            continue

        color = colors.get(alg_name, '#999999')
        marker = markers.get(alg_name, 'o')

        plt.plot(query_indices, values,
                marker=marker, linestyle='-', linewidth=2.5, markersize=8,
                color=color, label=alg_name, alpha=0.9,
                markerfacecolor='white', markeredgewidth=2)

    if random_avg_line is not None:
        plt.axhline(y=random_avg_line, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Random Avg ({random_avg_line:.3f})')

    plt.title(f'Algorithm Progress Across Queries (Counter {counter})',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Query Index', fontsize=14, fontweight='bold')
    plt.ylabel('Final Score (Normalized)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)

    plt.xticks(query_indices)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()

    if lamp_type is None:
        try:
            from IDS_TAP_parameters.py import PATH_CONFIG
            lamp_type = PATH_CONFIG.get('LaMP_type', 'unknown')
        except:
            lamp_type = 'unknown'

    save_dir = "./plots"
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{get_filename_prefix(lamp_type)}_query_progress_counter_{counter}.pdf")

    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   📊 [Query Progress] Saving cross-Query progress plot: {filename}")

def plot_query_progress_counter_average(all_query_progress_data, lamp_type=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    if not all_query_progress_data:
        return

    all_algorithms = set()
    min_query_idx = float('inf')
    max_query_idx = -1
    random_avg_lines = []
    for data in all_query_progress_data:
        if data and 'algorithms' in data:
            all_algorithms.update(data['algorithms'].keys())
            if data.get('query_indices'):
                min_query_idx = min(min_query_idx, min(data['query_indices']))
                max_query_idx = max(max_query_idx, max(data['query_indices']))
            if data.get('random_avg_line') is not None:
                random_avg_lines.append(data['random_avg_line'])

    if not all_algorithms or max_query_idx < 0:
        return

    if min_query_idx == float('inf'):
        min_query_idx = 0

    algorithm_query_values = {alg: {q: [] for q in range(min_query_idx, max_query_idx + 1)} for alg in all_algorithms}

    for data in all_query_progress_data:
        if not data or 'algorithms' not in data:
            continue
        query_indices = data.get('query_indices', [])
        for alg_name, values in data['algorithms'].items():
            for i, q_idx in enumerate(query_indices):
                if i < len(values) and q_idx in algorithm_query_values.get(alg_name, {}):
                    algorithm_query_values[alg_name][q_idx].append(values[i])

    algorithm_average = {}
    algorithm_se_upper = {}
    algorithm_se_lower = {}

    for alg_name in all_algorithms:
        avg_values = []
        upper_values = []
        lower_values = []
        valid_queries = []

        for q_idx in range(min_query_idx, max_query_idx + 1):
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

    plt.figure(figsize=(14, 10))

    colors, markers, _, _ = get_plot_config_from_baseline()

    for alg_name, (query_indices, avg_values) in algorithm_average.items():
        color = colors.get(alg_name, '#999999')
        marker = markers.get(alg_name, 'o')
        upper = algorithm_se_upper[alg_name]
        lower = algorithm_se_lower[alg_name]

        plt.fill_between(query_indices, lower, upper, color=color, alpha=0.2)

        plt.plot(query_indices, avg_values,
                marker=marker, linestyle='-', linewidth=3, markersize=8,
                color=color, label=alg_name, alpha=0.9,
                markerfacecolor='white', markeredgewidth=2)

    if random_avg_lines:
        overall_random_avg = np.mean(random_avg_lines)
        plt.axhline(y=overall_random_avg, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Random Avg ({overall_random_avg:.3f})')

    counter_count = len(all_query_progress_data)
    plt.title(f'Query Progress Counter Average\n(Averaged across {counter_count} counters, showing mean±1SE)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Query Index', fontsize=14, fontweight='bold')
    plt.ylabel('Final Score (Normalized)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()

    if lamp_type is None:
        try:
            from IDS_TAP_parameters.py import PATH_CONFIG
            lamp_type = PATH_CONFIG.get('LaMP_type', 'unknown')
        except:
            lamp_type = 'unknown'

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "./past_average"
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{get_filename_prefix(lamp_type)}_query_progress_counter_average_{timestamp}.pdf")

    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   📊 [Query Progress Average] Saving cross-Query progress Counter Average plot: {filename}")

if __name__ == '__main__':
    from IDS_TAP_parameters.py import get_all_configs, print_config_summary

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    print_config_summary()

    config = get_all_configs()

    import asyncio
    all_results = asyncio.run(run(config=config))
