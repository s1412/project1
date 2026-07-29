import os
import sys
import time
import subprocess
import multiprocessing
from datetime import datetime

# 脚本所在目录（FTPERSLLM/）及其父目录（数据文件所在位置）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)

# ID_TAP.py 子进程使用 hw 环境的 Python（包含 torch/matplotlib 等依赖）
# 调度脚本本身可在 base 环境运行，子进程仍使用 hw 环境
WORKER_PYTHON = os.environ.get("WORKER_PYTHON", sys.executable)
if not os.path.exists(WORKER_PYTHON):
    WORKER_PYTHON = sys.executable  # 回退到当前 Python

# 仅在算法子进程中注入 HTTP 代理（不用 SOCKS，避免 socksio 依赖）
# 设为 None 则不注入代理
PROXY_URL = "http://127.0.0.1:7890"

LLM_AS_JUDGE = False

RUN_POHF = True
RUN_PERSONA_AGENT = False

# PersonaAgent 脚本路径
PERSONA_AGENT_SCRIPT = os.path.join(SCRIPT_DIR, "PersonaAgent", "test_time_alignment.py")


def log_print(message, gpu_id=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if gpu_id is not None:
        print(f"[{timestamp}] [GPU-{gpu_id}] {message}")
    else:
        print(f"[{timestamp}] {message}")
    sys.stdout.flush()

DATASET_COUNTER_CONFIG = {
    4: 40,
    5: 40,
    8: 40,
    9: 40,
    10: 40,
    0: 40,
    -1: 40,
    -2: 40,
}

DATASET_PARALLEL_COUNTERS_CONFIG = {
    4: 8,
    5: 8,
    8: 8,
    9: 8,
    10: 8,
    0: 8,
    -1: 8,
    -2: 8,
}

PERSONA_AGENT_PARALLEL_COUNTERS_CONFIG = {
    4: 8,
    5: 8,
    8: 8,
    9: 8,
    10: 8,
    0: 8,
    -1: 8,
    -2: 8,
}

PERSONA_AGENT_PARALLEL_PAIRS_CONFIG = {
    4: False,
    5: False,
    8: False,
    9: False,
    10: False,
    0: False,
    -1: False,
    -2: False,
}

PERSONA_AGENT_PAIR_WORKERS_CONFIG = {
    4: 1,
    5: 1,
    8: 1,
    9: 1,
    10: 1,
    0: 1,
    -1: 1,
    -2: 1,
}

PERSONA_AGENT_BATCH_SIZE = 3
PERSONA_AGENT_NUM_ITERATIONS = 1

# PersonaAgent 数据集名称映射（与 test_time_alignment.py 保持一致）
PERSONA_AGENT_DATASET_NAMES = {
    4: "lamp4", 5: "lamp5", 8: "lamp8", 9: "lamp9", 10: "lamp10",
    0: "ultrachat", -1: "wildchat", -2: "prefeval",
}

DATASET_SPECIFIC_COUNTERS = {
}

def get_dataset_config(lamp_type):
    base_path = os.path.join(BASE_DIR, 'APOHF-main')

    configs = {
        4: {
            "input_address": os.path.join(base_path, "time/LaMP_4/train/train_questions.json"),
            "output_address": os.path.join(base_path, "time/LaMP_4/train/train_outputs.json"),
            "LaMP_type": 4,
        },
        5: {
            "input_address": os.path.join(base_path, "time/LaMP_5/train/train_questions.json"),
            "output_address": os.path.join(base_path, "time/LaMP_5/train/train_outputs.json"),
            "LaMP_type": 5,
        },
        8: {
            "input_address": os.path.join(base_path, "longLaMP/abstract_generation/temporal_train.json"),
            "output_address": None,
            "LaMP_type": 8,
        },
        9: {
            "input_address": os.path.join(base_path, "longLaMP/product_review/temporal_train.json"),
            "output_address": None,
            "LaMP_type": 9,
        },
        10: {
            "input_address": os.path.join(base_path, "longLaMP/topic_writing/temporal_train.json"),
            "output_address": None,
            "LaMP_type": 10,
        },
        0: {
            "input_address": os.path.join(BASE_DIR, "ultrachat_multiturn/ultrachat_long_dialogues_with_response.json"),
            "output_address": None,
            "LaMP_type": 0,
        },
        -1: {
            "input_address": os.path.join(BASE_DIR, "wildchat/wildchat_long_dialogues_with_response.json"),
            "output_address": None,
            "LaMP_type": -1,
        },
        -2: {
            "input_address": os.path.join(BASE_DIR, "PrefEval_dataset/PrefEval_persona.json"),
            "output_address": None,
            "LaMP_type": -2,
        }
    }

    return configs.get(lamp_type)

def get_dataset_log_name(lamp_type):
    if lamp_type == 0:
        return "ultrachat"
    elif lamp_type == -1:
        return "wildchat"
    elif lamp_type == -2:
        return "prefeval"
    else:
        return f"lamp{lamp_type}"

def run_dataset_on_gpu(lamp_type, dataset_name, gpu_id, log_dir):
    log_print(f"Starting {dataset_name}", gpu_id)

    log_name = get_dataset_log_name(lamp_type)
    log_file = os.path.join(log_dir, f"{log_name}_gpu{gpu_id}.log")

    start_time = time.time()

    try:
        config = get_dataset_config(lamp_type)
        if not config:
            log_print(f"InvalidLaMP_type: {lamp_type}", gpu_id)
            return False

        if not os.path.exists(config["input_address"]):
            log_print(f"Data file not found: {config['input_address']}", gpu_id)
            return False

        counter_array_length = DATASET_COUNTER_CONFIG.get(lamp_type, 30)
        parallel_counters = DATASET_PARALLEL_COUNTERS_CONFIG.get(lamp_type, 4)
        log_print(f"  counter_array_length = {counter_array_length}, parallel_counters = {parallel_counters}", gpu_id)

        env = os.environ.copy()
        # 仅注入 HTTP 代理，清除 SOCKS 代理（hw 环境无 socksio）
        if PROXY_URL:
            env['http_proxy'] = PROXY_URL
            env['https_proxy'] = PROXY_URL
            env['HTTP_PROXY'] = PROXY_URL
            env['HTTPS_PROXY'] = PROXY_URL
        env.pop('ALL_PROXY', None)
        env.pop('all_proxy', None)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        env['POHF_LAMP_TYPE'] = str(lamp_type)
        env['POHF_INPUT_ADDRESS'] = config['input_address']
        env['POHF_OUTPUT_ADDRESS'] = config['output_address'] if config['output_address'] else ''
        specific_counters = DATASET_SPECIFIC_COUNTERS.get(lamp_type, None)
        if specific_counters:
            env['POHF_SPECIFIC_COUNTERS'] = ','.join(map(str, specific_counters))
            log_print(f"  Using specified counter list: {specific_counters}", gpu_id)
        else:
            env['POHF_COUNTER_ARRAY_LENGTH'] = str(counter_array_length)
        env['POHF_PARALLEL_COUNTERS'] = str(parallel_counters)
        env['POHF_LLM_AS_JUDGE'] = str(LLM_AS_JUDGE).lower()

        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                [WORKER_PYTHON, '-u', 'ID_TAP.py'],
                cwd=SCRIPT_DIR,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                text=True
            )

            return_code = process.wait()

        end_time = time.time()
        duration = end_time - start_time

        if return_code == 0:
            log_print(f"{dataset_name} Completed successfully! duration: {duration/60:.1f}min", gpu_id)
            return True
        else:
            log_print(f"{dataset_name} failed! error code: {return_code}", gpu_id)
            log_print(f"   View log: {log_file}", gpu_id)
            return False

    except Exception as e:
        log_print(f"{dataset_name} exception: {e}", gpu_id)
        return False

def run_datasets_on_gpu_sequential(datasets, gpu_id, log_dir):
    log_print(f"🚀 GPU {gpu_id} Starting IDS_TAP: {len(datasets)} datasets", gpu_id)

    results = []
    for lamp_type, dataset_name in datasets:
        success = run_dataset_on_gpu(lamp_type, dataset_name, gpu_id, log_dir)
        results.append((dataset_name, success))

    success_count = sum(1 for _, success in results if success)
    log_print(f"📊 GPU {gpu_id} IDS_TAP Complete: {success_count}/{len(datasets)} datasetssuccess", gpu_id)

    return results

def generate_counter_array_for_dataset(lamp_type, counter_array_length):
    import random as random_module

    from IDS_TAP_parameters import DATA_CONFIG
    random_seed = DATA_CONFIG.get("counter_random_seed", 62)

    config = get_dataset_config(lamp_type)
    if config is None:
        log_print(f"⚠️ Cannot get dataset {lamp_type}  configuration", None)
        return list(range(counter_array_length))

    input_address = config.get('input_address', '')

    from IDS_TAP_parameters import get_dataset_size_from_file
    max_dataset_size = get_dataset_size_from_file(input_address, lamp_type)

    random_module.seed(random_seed)

    max_counter = max_dataset_size - 1
    counter_array = random_module.sample(range(max_counter + 1), min(counter_array_length, max_counter + 1))

    log_print(f"  Generatingrandom counter array: seed={random_seed}, length={len(counter_array)}, range=0-{max_counter}", None)
    log_print(f"  Counter array: {counter_array}", None)

    return counter_array

def run_single_counter_subprocess(args):
    counter, cmd_args, cwd, env = args
    try:
        result = subprocess.run(
            cmd_args,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=18000
        )
        return (counter, result.returncode, result.stdout + result.stderr)
    except subprocess.TimeoutExpired:
        return (counter, -1, f"Counter {counter} timeout after 5 hours")
    except Exception as e:
        return (counter, -2, f"Counter {counter} exception: {str(e)}")

def run_persona_agent_on_dataset(lamp_type, dataset_name, gpu_id, log_dir):
    """Run PersonaAgent for a dataset using the same random counter array as IDS_TAP."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    log_print(f"Starting PersonaAgent for {dataset_name}", gpu_id)

    counter_array_length = DATASET_COUNTER_CONFIG.get(lamp_type, 40)
    parallel_counters = PERSONA_AGENT_PARALLEL_COUNTERS_CONFIG.get(lamp_type, 8)
    batch_size = PERSONA_AGENT_BATCH_SIZE
    num_iterations = PERSONA_AGENT_NUM_ITERATIONS
    dataset_name_str = PERSONA_AGENT_DATASET_NAMES.get(lamp_type, "lamp4")
    persona_dir = os.path.dirname(PERSONA_AGENT_SCRIPT)

    counter_array = generate_counter_array_for_dataset(lamp_type, counter_array_length)

    log_name = get_dataset_log_name(lamp_type)
    log_file = os.path.join(log_dir, f"{log_name}_persona_gpu{gpu_id}.log")

    env = os.environ.copy()
    # 仅注入 HTTP 代理，清除 SOCKS 代理（hw 环境无 socksio）
    if PROXY_URL:
        env['http_proxy'] = PROXY_URL
        env['https_proxy'] = PROXY_URL
        env['HTTP_PROXY'] = PROXY_URL
        env['HTTPS_PROXY'] = PROXY_URL
    env.pop('ALL_PROXY', None)
    env.pop('all_proxy', None)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Build the common command prefix
    base_cmd = [
        WORKER_PYTHON, '-u', PERSONA_AGENT_SCRIPT,
        '--dataset', dataset_name_str,
        '--batch_size', str(batch_size),
        '--iterations', str(num_iterations),
        '--quiet',
    ]
    if LLM_AS_JUDGE:
        base_cmd.append('--llm-as-judge')

    def _run_single_counter(counter):
        cmd = base_cmd + ['--counter', str(counter)]
        try:
            result = subprocess.run(
                cmd,
                cwd=persona_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=18000
            )
            return (counter, result.returncode, result.stdout + result.stderr)
        except subprocess.TimeoutExpired:
            return (counter, -1, f"Counter {counter} timeout after 5 hours\n")
        except Exception as e:
            return (counter, -2, f"Counter {counter} exception: {e}\n")

    start_time = time.time()
    success_count = 0
    fail_count = 0

    with open(log_file, 'w') as f:
        f.write(f"PersonaAgent {dataset_name} GPU{gpu_id} started\n")
        f.write(f"Counter array ({len(counter_array)} items): {counter_array}\n\n")

    try:
        # Use ThreadPoolExecutor: each thread just launches an external subprocess (I/O-bound)
        with ThreadPoolExecutor(max_workers=parallel_counters) as executor:
            futures = {executor.submit(_run_single_counter, c): c for c in counter_array}
            for future in as_completed(futures):
                counter, rc, output = future.result()
                with open(log_file, 'a') as f:
                    f.write(f"\n=== Counter {counter} (rc={rc}) ===\n")
                    f.write(output)
                if rc == 0:
                    success_count += 1
                else:
                    fail_count += 1
                log_print(
                    f"  PersonaAgent counter {counter} done (rc={rc}), "
                    f"progress: {success_count + fail_count}/{len(counter_array)}",
                    gpu_id
                )
    except Exception as e:
        log_print(f"PersonaAgent {dataset_name} exception: {e}", gpu_id)
        return False

    duration = time.time() - start_time
    log_print(
        f"PersonaAgent {dataset_name} done: {success_count}/{len(counter_array)} success, "
        f"{duration / 60:.1f}min",
        gpu_id
    )
    return success_count == len(counter_array)

def run_all_algorithms_on_gpu_sequential(datasets, gpu_id, log_dir):
    algorithms_to_run = []
    if RUN_POHF:
        algorithms_to_run.append("IDS_TAP")
    if RUN_PERSONA_AGENT:
        algorithms_to_run.append("PersonaAgent")

    log_print(f"🚀 GPU {gpu_id} Starting algorithms {algorithms_to_run}: {len(datasets)} datasets", gpu_id)

    pohf_results = []
    persona_results = []

    if RUN_POHF:
        log_print(f"📍 GPU {gpu_id} Phase 1: Running IDS_TAP", gpu_id)
        for lamp_type, dataset_name in datasets:
            success = run_dataset_on_gpu(lamp_type, dataset_name, gpu_id, log_dir)
            pohf_results.append((dataset_name, "IDS_TAP", success))

        pohf_success = sum(1 for _, _, success in pohf_results if success)
        log_print(f"📊 GPU {gpu_id} IDS_TAP Complete: {pohf_success}/{len(datasets)} datasets success", gpu_id)

    if RUN_PERSONA_AGENT:
        log_print(f"📍 GPU {gpu_id} Phase 2: Running PersonaAgent baseline", gpu_id)
        for lamp_type, dataset_name in datasets:
            success = run_persona_agent_on_dataset(lamp_type, dataset_name, gpu_id, log_dir)
            persona_results.append((dataset_name, "PersonaAgent", success))

        persona_success = sum(1 for _, _, success in persona_results if success)
        log_print(f"📊 GPU {gpu_id} PersonaAgent Complete: {persona_success}/{len(datasets)} datasets success", gpu_id)

    all_results = pohf_results + persona_results
    total_success = sum(1 for _, _, success in all_results if success)
    total_tasks = len(all_results)
    log_print(f"📊 GPU {gpu_id} All Complete: {total_success}/{total_tasks} tasks success", gpu_id)

    return all_results

def main():
    log_print("IDS_TAP Multi-GPU parallel execution script")
    log_print("=" * 80)
    log_print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("")

    log_dir = f"./parallel_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    log_print(f"📁 Log directory: {log_dir}")
    log_print("")

    gpu_dataset_mapping_parallel = {
        1: [
            (4, "LongLaMP-4 (Product Rating)"),
            # (5, "LongLaMP-5 (News Categorization)"),
            # (9, "LongLaMP-9 (Product Review)"),
        ],
        2: [
            # (5, "LongLaMP-5 (News Categorization)"),
            (10, "LongLaMP-10 (Topic Writing)"),
        ],
        3: [
            (0, "UltraChat"),
            (-2, "Prefevel"),
            # (8, "LongLaMP-8 (Abstract Writing)"),
        ],
    }

    algos = []
    if RUN_POHF:
        algos.append("IDS_TAP")
    if RUN_PERSONA_AGENT:
        algos.append("PersonaAgent")
    algo_str = " + ".join(algos) if algos else "none"

    log_print("📋 Execution plan:")
    log_print("")
    log_print(f"  🔄 GPU parallel execution (algorithms: {algo_str}):")
    for gpu_id, datasets in sorted(gpu_dataset_mapping_parallel.items()):
        dataset_names = " -> ".join([name for _, name in datasets])
        total_counters = sum(DATASET_COUNTER_CONFIG.get(lamp_type, 40) for lamp_type, _ in datasets)
        log_print(f"    GPU {gpu_id}: {dataset_names}")
        log_print(f"           Total {len(datasets)} datasets, {total_counters} counters")
        log_print(f"           Execution order: {algo_str}")
    log_print("")

    processes = []
    total_start_time = time.time()

    log_print(f"🚀 Starting GPU tasks (algorithms: {algo_str})...")

    for gpu_id, datasets in gpu_dataset_mapping_parallel.items():
        p = multiprocessing.Process(
            target=run_all_algorithms_on_gpu_sequential,
            args=(datasets, gpu_id, log_dir)
        )
        p.start()
        processes.append((gpu_id, p))
        log_print(f"  Started GPU {gpu_id} process ({algo_str})")
        time.sleep(10)

    for gpu_id, p in processes:
        p.join()
        log_print(f"  GPU {gpu_id} process Completed")

    total_time = time.time() - total_start_time

    if not LLM_AS_JUDGE:
        log_print("")
        log_print("📊 LLM_AS_JUDGE=False，Automatically generating ROUGE-L progress charts...")
        try:
            plot_output_dir = "./rougeL_conter_progress_IDS_TAP"
            os.makedirs(plot_output_dir, exist_ok=True)
            log_print(f"  📁 Output directory: {plot_output_dir}")

            plot_script = os.path.join(SCRIPT_DIR, "plot_rougescore_progress.py")
            plot_result = subprocess.run(
                [sys.executable, plot_script],
                cwd=SCRIPT_DIR,
                capture_output=True,
                text=True,
                timeout=600
            )

            if plot_result.returncode == 0:
                log_print("  ✅ Chart generationsuccess!")
                log_print(f"  Output: {plot_result.stdout[-500:] if len(plot_result.stdout) > 500 else plot_result.stdout}")
            else:
                log_print(f"  ❌ Chart generationfailed: {plot_result.stderr}")
        except Exception as e:
            log_print(f"  ⚠️ Chart generation exception: {e}")

    log_print("")
    log_print("=" * 80)
    log_print("📊 Execution summary:")
    log_print(f"  ⏱️ Total time: {total_time:.2f}seconds ({total_time/60:.1f}min, {total_time/3600:.2f}hours)")
    log_print(f"  📁 Log directory: {log_dir}")
    log_print(f"  🏁 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("")
    log_print("  📋 Datasets and algorithms executed:")
    for gpu_id, datasets in sorted(gpu_dataset_mapping_parallel.items()):
        dataset_info = " -> ".join([f"{name}" for _, name in datasets])
        log_print(f"    GPU {gpu_id}: {dataset_info}")
        log_print(f"           algorithms: {algo_str}")
    log_print("")
    log_print(f"✅ All GPU runs Complete ({algo_str})!")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
