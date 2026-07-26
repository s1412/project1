import os


NETWORK_CONFIG = {
    "input_dim": None,  
    "hidden_size": 1024,  
    "depth": 1,  
    "dropout_rate": 0,  
    "activation": "GELU", 
}

TRAINING_CONFIG = {
    "learning_rate": 5e-4,
    "epochs": 100,  
    "batch_size": 4,  
    "weight_decay": 0.001,  
    "optimizer": "AdamW",
    "gradient_clip_norm": 1.0,
    "scheduler_type": "CosineAnnealingLR",
    "min_lr_ratio": 0.5,
    "adaptive_weight_decay": True,
    "min_weight_decay": 1e-3,

    "early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-3,

    "query_decay_enabled": False,
    "query_decay_gamma": 1,

    "query_similarity_enabled": False,
    "query_similarity_steepness": 6,
    "query_similarity_midpoint": 0.8,

    "cross_query_incremental": False,  

    "debug_training": True,
}

POHF_CONFIG = {
    "lambda": 0.01,
    "nu": 0.005,
    "time_factor_coeff": 0.2,
    "vrsion": "diag",
    "max_params_for_matrix": 10000,

    "cov_init_value": 1,
    "cov_init_enabled": True,

    "nu_decay_enabled": False,
    "nu_decay_factor": 1,
    "nu_min": 0.005,
    "nu_decay_start": 0,
    "nu_decay_type": "exponential",

    "info_gain_enabled": True,
    "info_gain_scale": 0.5,
    "info_gain_normalize": True,

    "info_gain_decay_enabled": True,
    "info_gain_decay_factor": 1.0,
    "info_gain_min_scale": 0.4,
    "info_gain_decay_start": 5,
    "info_gain_decay_type": "exponential",

    "bayesian_alpha": 0.7,

    "bt_isolated_arm_mode": "unknown_isolated",  
    "softmax_temperature": 4.5,

    "reset_info_matrix_per_query": False,

    "strict_ids": True,
}

DATA_CONFIG = {
    "max_len": 100,
    "times": 100,  

    "counter_array_length": 40,
    "counter_random_seed": 62,
    "max_history_items": 10,
    "embedding_normalize": True,
    "embedding_max_dim": 1024,

    "rephrase_reset_interval": 15,
    "rephrase_reset_enabled": True,

    # Domain generation mode: "rephrase" (default) or "keyword"
    "domain_generation_method": "rephase",
    "keyword_n_keywords_min": 10,       # min keywords extracted when text is short (<500 words)
    "keyword_n_keywords_max": 15,       # max keywords extracted when text is long (>1500 words)
    "keyword_combo_sizes": [2, 3],      # combination sizes used to build arms
    "keyword_random_seed": 62,          # fixed seed for combination shuffling

    "lamp_profile_threshold_small": 10,
    "lamp_profile_threshold_large": 20,

    "lamp_history_context_count_small": 5,
    "lamp_history_context_count_large": 10,

    "lamp_ranked_entries_output_count": 10,

    "lamp_total_io_count":10, 
}

PARALLEL_CONFIG = {
    "parallel_counters": 8,
    "parallel_enabled": True,
    "timeout_per_counter": 36000,
}

EXPERIMENT_CONFIG = {
    "n_init": 15,
    "total_iter": 55,
    "random_seed": 62,
    "progress_report_interval": 200,  
}

CONTEXTUAL_BANDIT_CONFIG = {
    "enabled_for_lamp": True,  

    "unified_training_rounds": 10,

    "contextual_input_dim": 2048,      
    "standard_input_dim": 1024,         

    "sigmoid_steepness": 6,          
    "sigmoid_midpoint": 0.8,            
}

DOUBLETS_CONFIG = {
    "ensemble_lambda": 0.0,        

    "weight_decay": 0.05,          
    "use_adamw": True,             
    "network": {
        "hidden_size": 256,
        "depth": 1,
        "dropout_rate": 0.1,
        "activation": None,
        "use_kaiming_init": True,
        "ensemble_count": 4,
    },
}

BASELINE_CONFIG = {
    "enabled_baselines": [
        'POHF',
        'DoubleTS',
    ],

    "algorithm_display_config": {
        'POHF': {
            'color': '#2E86AB',
            'marker': 'o',
            'label': 'POHF (UCB)',
            'show_in_plots': True,
        },
        'POHF-Random': {
            'color': '#FFA500',
            'marker': 'v',
            'label': 'POHF-Random',
            'show_in_plots': True,
        },
        'POHF-RandomPair': {
            'color': '#9B59B6',
            'marker': 'p',
            'label': 'POHF-RandomPair',
            'show_in_plots': True,
        },
        'DoubleTS': {
            'color': '#45B7D1',
            'marker': 'D',
            'label': 'DoubleTS',
            'show_in_plots': True,
        },
        'Random': {
            'color': '#FF6B6B',
            'marker': 's',
            'label': 'Random',
            'show_in_plots': True,
        },
        'POHF-InfoGain': {
            'color': '#8B2E86',
            'marker': '*',
            'label': 'POHF-InfoGain',
            'show_in_plots': True,
        },
        'POHF-InfoGain-NoHistory': {
            'color': '#8B4513',
            'marker': '^',
            'label': 'POHF-InfoGain-NoHistory',
            'show_in_plots': True,
        },
        'Linear-InfoGain': {
            'color': '#2ECC71',
            'marker': 'h',
            'label': 'Linear-InfoGain',
            'show_in_plots': True,
        },
    },

    "plot_config": {
        "show_range_for_algorithms": ['Random'],  
        "exclude_from_minmax": ['Random'],        
    },
}

API_CONFIG = {
    "embedding_api_url": "http://127.0.0.1:8400/v1/embeddings",
    "embedding_timeout": 30,
    "openai_api_key": os.environ.get("OPENROUTER_API_KEY"),
    "openai_base_url": "https://openrouter.ai/api/v1",
    "openai_model": "deepseek/deepseek-v3.2",
    "openai_temperature": 0.0,  
}

LLM_CONFIG = {
    "temperature": 0.6,
    "top_p": 1.0,
    "max_tokens": 1024,
    "use_seed": True,
    "seed": 62,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "retry_on_different_output": False,
    "max_retries": 3,  
    "retry_delay": 2.0,
    "retry_backoff": 2.0,
    "deterministic_mode": True,
    "use_cache": True,
    "cache_size": 1000,
}

ROUGE_CONFIG = {
    "a": 40,
    "b": 0.001,
    "beta_min": 5.0,
    "beta_max": 100.0,
    "use_probabilistic": True,
    "LLM_as_judge": False,
    # scoring_mode: "rouge" (default) or "bertscore"
    "scoring_mode": "bertscore",
}

BERTSCORE_CONFIG = {
    # Model used for BERTScore; roberta-large gives the best results
    "model_type": "roberta-large",
    # Device for BERTScore computation: "cuda:0", "cuda:1", "cpu", etc.
    # Use cuda:0 because CUDA_VISIBLE_DEVICES remaps the assigned GPU to cuda:0
    "device": "cuda:0",
    # Number of texts per batch when computing BERTScore
    "batch_size": 64,
    # Language hint (used when model_type is None)
    "lang": "en",
    # Bradley-Terry beta parameters (same role as ROUGE_CONFIG a/b/beta_min/beta_max)
    "a": 40,
    "b": 0.001,
    "beta_min": 5.0,
    "beta_max": 100.0,
}

DEVICE_CONFIG = {
    # Use cuda:0 because CUDA_VISIBLE_DEVICES remaps the assigned GPU to cuda:0
    "device": "cuda:0",
    "dtype": "float32",
    "cuda_launch_blocking": True,
    "clear_cache": True,
    "memory_fraction": 1,
}

PATH_CONFIG = {
            "input_address": "./ultrachat_multiturn/ultrachat_long_dialogues_with_response.json",
            "output_address": None,
            "LaMP_type": 0,
}

NEW_DATASET_CONFIG = {
    'ultrachat': {
        'input_address': './ultrachat_multiturn/ultrachat_long_dialogues_with_response.json',
        'output_address': None,
        'instruction': 'Predict the user\'s next query',
        'dataset_name': 'ultrachat',
    },
    'wildchat': {
        'input_address': './wildchat/wildchat_long_dialogues_with_response.json',
        'output_address': None,
        'instruction': 'Predict the user\'s next query',
        'dataset_name': 'wildchat',
    },
    'prefeval': {
        'input_address': './prefeval/prefeval_data.json',
        'output_address': None,
        'instruction': 'Predict the user\'s preference',
        'dataset_name': 'prefeval',
    },
}

OUTPUT_CONFIG = {
    "verbose": True, 
    "save_results": True,  
    "plot_results": True,  
    "log_level": "INFO",  
}

PERFORMANCE_CONFIG = {
    "memory_efficient": True,  
    "force_diag_threshold": 10000,  
    "embedding_batch_size": 100,  
    "progress_print_interval": 100,  
}

def get_dataset_size_from_file(input_address, lamp_type=None):
    import json

    try:
        try:
            with open(input_address, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return _get_json_array_dataset_size_from_data(data, input_address)
        except json.JSONDecodeError:
            return _get_jsonl_dataset_size(input_address)

    except Exception as e:
        print(f"Failed to read data file: {e}")
        print(" Using default dataset size1000")
        return 1000

def _get_json_array_dataset_size_from_data(data, input_address):
    if not data:
        return 0

    dataset_size = len(data)

    try:
        first_id = int(data[0]["id"])
        last_id = int(data[-1]["id"])
        print(f"📊 Dynamically getting dataset info (JSON array format):")
        print(f"   File: {input_address}")
        print(f"   Actual data count: {dataset_size}")
        print(f"   First data ID: {first_id}")
        print(f"   Last data ID: {last_id}")
        print(f"   Counterrange: 0 - {dataset_size - 1} (array index)")
    except (KeyError, IndexError):
        print(f"📊 Dynamically getting dataset info (JSON array format):")
        print(f"   File: {input_address}")
        print(f"   Actual data count: {dataset_size}")
        print(f"   Counterrange: 0 - {dataset_size - 1} (array index)")

    return dataset_size

def _get_jsonl_dataset_size(input_address):
    import json

    total_items = 0
    line_count = 0

    with open(input_address, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if isinstance(data, list):
                    total_items += len(data)
                    line_count += 1
                else:
                    total_items += 1
                    line_count += 1

    print(f"📊 Dynamically getting dataset info (JSONL format):")
    print(f"   File: {input_address}")
    print(f"   Total lines: {line_count}")
    print(f"   Actual data count: {total_items}")
    print(f"   Counterrange: 0 - {total_items - 1} (array index)")

    return total_items

def generate_counter_array(config, input_address):
    import random

    data_config = config.get("data", {})
    path_config = config.get("path", {})
    array_length = data_config.get("counter_array_length", 20)
    random_seed = data_config.get("counter_random_seed", 42)

    lamp_type = path_config.get("LaMP_type", None)

    max_dataset_size = get_dataset_size_from_file(input_address, lamp_type)

    random.seed(random_seed)

    max_counter = max_dataset_size - 1
    counter_array = random.sample(range(max_counter + 1), min(array_length, max_counter + 1))

    print(f"Generatingcounterarray:")
    print(f"   LaMP type: {lamp_type}")
    print(f"   random seed: {random_seed}")
    print(f"   array length: {len(counter_array)}")
    print(f"   value range: 0 - {max_counter}")
    print(f"   generated array: {counter_array}")

    return counter_array

def get_all_configs():
    import os

    rouge_config = ROUGE_CONFIG.copy()

    env_llm_as_judge = os.environ.get('POHF_LLM_AS_JUDGE', '').lower()
    if env_llm_as_judge in ('true', '1', 'yes'):
        rouge_config['LLM_as_judge'] = True
    elif env_llm_as_judge in ('false', '0', 'no'):
        rouge_config['LLM_as_judge'] = False

    env_scoring_mode = os.environ.get('POHF_SCORING_MODE', '').lower()
    if env_scoring_mode in ('rouge', 'bertscore'):
        rouge_config['scoring_mode'] = env_scoring_mode

    return {
        "network": NETWORK_CONFIG,
        "training": TRAINING_CONFIG,
        "pohf": POHF_CONFIG,
        "data": DATA_CONFIG,
        "experiment": EXPERIMENT_CONFIG,
        "baseline": BASELINE_CONFIG,
        "api": API_CONFIG,
        "llm": LLM_CONFIG,
        "rouge": rouge_config,
        "bertscore": BERTSCORE_CONFIG,
        "device": DEVICE_CONFIG,
        "path": PATH_CONFIG,
        "output": OUTPUT_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "parallel": PARALLEL_CONFIG,
    }

def print_config_summary():

    configs = get_all_configs()
    
    for category, config in configs.items():
        print(f"\n{category.upper()} CONFIG:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("=" * 60)

if __name__ == "__main__":
    print_config_summary()
