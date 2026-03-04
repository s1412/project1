import os
import sys
import time
import subprocess
import multiprocessing
from datetime import datetime

LLM_AS_JUDGE = True

RUN_POHF = True
RUN_PERSONA_AGENT = False


def log_print(message, gpu_id=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if gpu_id is not None:
        print(f"[{timestamp}] [GPU-{gpu_id}] {message}")
    else:
        print(f"[{timestamp}] {message}")
    sys.stdout.flush()

DATASET_COUNTER_CONFIG = {
    4: 20,
    5: 20,
    8: 20,
    9: 20,
    10: 20,
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

DATASET_SPECIFIC_COUNTERS = {
}

def get_dataset_config(lamp_type):
    base_path = './APOHF-main'

    configs = {
        4: {
            "input_address": f"{base_path}/time/LaMP_4/train/train_questions.json",
            "output_address": f"{base_path}/time/LaMP_4/train/train_outputs.json",
            "LaMP_type": 4,
        },
        5: {
            "input_address": f"{base_path}/time/LaMP_5/train/train_questions.json",
            "output_address": f"{base_path}/time/LaMP_5/train/train_outputs.json",
            "LaMP_type": 5,
        },
        8: {
            "input_address": f"{base_path}/longLaMP/abstract_generation/temporal_train.json",
            "output_address": None,
            "LaMP_type": 8,
        },
        9: {
            "input_address": f"{base_path}/longLaMP/product_review/temporal_train.json",
            "output_address": None,
            "LaMP_type": 9,
        },
        10: {
            "input_address": f"{base_path}/longLaMP/topic_writing/temporal_train.json",
            "output_address": None,
            "LaMP_type": 10,
        },
        0: {
            "input_address": "./ultrachat_multiturn/ultrachat_long_dialogues_with_response.json",
            "output_address": None,
            "LaMP_type": 0,
        },
        -1: {
            "input_address": "./wildchat/wildchat_long_dialogues_with_response.json",
            "output_address": None,
            "LaMP_type": -1,
        },
        -2: {
            "input_address": "./PrefEval_dataset/PrefEval_persona.json",
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
                [sys.executable, '-u', 'IDS_TAP.py'],
                cwd='',
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

    from IDS_TAP_parameters.py import DATA_CONFIG
    random_seed = DATA_CONFIG.get("counter_random_seed", 62)

    config = get_dataset_config(lamp_type)
    if config is None:
        log_print(f"⚠️ Cannot get dataset {lamp_type}  configuration", None)
        return list(range(counter_array_length))

    input_address = config.get('input_address', '')

    from IDS_TAP_parameters.py import get_dataset_size_from_file
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
    from concurrent.futures import ProcessPoolExecutor, as_Completed

    log_print(f"🤖 Starting PersonaAgent: {dataset_name}", gpu_id)

    log_name = get_dataset_log_name(lamp_type)
    log_file = os.path.join(log_dir, f"{log_name}_personaagent_gpu{gpu_id}.log")

    start_time = time.time()

    try:
        counter_array_length = DATASET_COUNTER_CONFIG.get(lamp_type, 40)

        counter_array = generate_counter_array_for_dataset(lamp_type, counter_array_length)
        log_print(f"  PersonaAgent Usingrandom counter array: {len(counter_array)} items", gpu_id)

        parallel_counters = PERSONA_AGENT_PARALLEL_COUNTERS_CONFIG.get(lamp_type, 3)
        parallel_pairs = PERSONA_AGENT_PARALLEL_PAIRS_CONFIG.get(lamp_type, False)
        pair_workers = PERSONA_AGENT_PAIR_WORKERS_CONFIG.get(lamp_type, 5)

        log_print(f"  PersonaAgent configuration: parallel_counters={parallel_counters}, parallel_pairs={parallel_pairs}, pair_workers={pair_workers}", gpu_id)
        log_print(f"  PersonaAgent algorithm params: batch_size={PERSONA_AGENT_BATCH_SIZE}, iterations={PERSONA_AGENT_NUM_ITERATIONS}", gpu_id)

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        persona_agent_script = ''
        cwd = ''

        all_tasks = []
        for counter in counter_array:
            cmd_args = [
                sys.executable, '-u', persona_agent_script,
                '--dataset', log_name,
                '--counter', str(counter),
                '--batch_size', str(PERSONA_AGENT_BATCH_SIZE),
                '--iterations', str(PERSONA_AGENT_NUM_ITERATIONS),
                '--quiet'
            ]

            if parallel_pairs:
                cmd_args.extend(['--parallel', '--workers', str(pair_workers)])

            if LLM_AS_JUDGE:
                cmd_args.append('--llm-as-judge')

            all_tasks.append((counter, cmd_args, cwd, env))

        with open(log_file, 'w') as f:
            f.write(f"PersonaAgent configuration\n")
            f.write(f"{'='*60}\n")
            f.write(f"Random counter array: {counter_array}\n")
            f.write(f"Total {len(counter_array)}  counters\n")
            f.write(f"parallel_counters: {parallel_counters} (truly parallel)\n")
            f.write(f"parallel_pairs: {parallel_pairs}\n")
            f.write(f"pair_workers: {pair_workers}\n")
            f.write(f"batch_size (k): {PERSONA_AGENT_BATCH_SIZE}\n")
            f.write(f"iterations (E): {PERSONA_AGENT_NUM_ITERATIONS}\n")
            f.write(f"{'='*60}\n\n")

        log_print(f"  🚀 Starting {parallel_counters} items parallel process handling {len(counter_array)}  counters", gpu_id)

        Completed_count = 0
        failed_count = 0
        results_log = []

        with ProcessPoolExecutor(max_workers=parallel_counters) as executor:
            future_to_counter = {
                executor.submit(run_single_counter_subprocess, task): task[0]
                for task in all_tasks
            }

            for future in as_Completed(future_to_counter):
                counter = future_to_counter[future]
                try:
                    counter_id, return_code, output = future.result()
                    Completed_count += 1

                    if return_code == 0:
                        status = "✅"
                    else:
                        status = "❌"
                        failed_count += 1

                    results_log.append((counter_id, return_code, output))

                    if Completed_count % 5 == 0 or Completed_count == len(counter_array):
                        log_print(f"  Progress: {Completed_count}/{len(counter_array)} (failed: {failed_count})", gpu_id)

                except Exception as e:
                    failed_count += 1
                    Completed_count += 1
                    results_log.append((counter, -3, f"Future exception: {str(e)}"))
                    log_print(f"  ⚠️ Counter {counter} exception: {e}", gpu_id)

        with open(log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Execution results summary\n")
            f.write(f"{'='*60}\n")
            f.write(f"Completed: {Completed_count}, Failed: {failed_count}\n\n")

            for counter_id, return_code, output in sorted(results_log, key=lambda x: x[0]):
                status = "SUCCESS" if return_code == 0 else f"FAILED (code={return_code})"
                f.write(f"\n--- Counter {counter_id}: {status} ---\n")
                f.write(output[:5000] if len(output) > 5000 else output)
                f.write("\n")

        end_time = time.time()
        duration = end_time - start_time

        log_print(f"✅ PersonaAgent {dataset_name} Complete! duration: {duration/60:.1f}min, success: {Completed_count - failed_count}/{Completed_count}", gpu_id)
        return True

    except Exception as e:
        import traceback
        log_print(f"❌ PersonaAgent {dataset_name} exception: {e}", gpu_id)
        log_print(f"   Traceback: {traceback.format_exc()}", gpu_id)
        return False

def run_persona_agent_on_dataset_batch(lamp_type, dataset_name, gpu_id, log_dir):
    log_print(f"🤖 Starting PersonaAgent (batch mode): {dataset_name}", gpu_id)

    log_name = get_dataset_log_name(lamp_type)
    log_file = os.path.join(log_dir, f"{log_name}_personaagent_batch_gpu{gpu_id}.log")

    start_time = time.time()

    try:
        counter_array_length = DATASET_COUNTER_CONFIG.get(lamp_type, 40)

        parallel_counters = PERSONA_AGENT_PARALLEL_COUNTERS_CONFIG.get(lamp_type, 3)
        parallel_pairs = PERSONA_AGENT_PARALLEL_PAIRS_CONFIG.get(lamp_type, False)
        pair_workers = PERSONA_AGENT_PAIR_WORKERS_CONFIG.get(lamp_type, 5)

        log_print(f"  PersonaAgent batch modeconfiguration: counter_end={counter_array_length}, parallel_counters={parallel_counters}", gpu_id)

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        persona_agent_script = ''

        cmd_args = [
            sys.executable, '-u', persona_agent_script,
            '--dataset', log_name,
            '--counter-start', '0',
            '--counter-end', str(counter_array_length),
            '--parallel-counters', str(parallel_counters),
            '--batch_size', str(PERSONA_AGENT_BATCH_SIZE),
            '--iterations', str(PERSONA_AGENT_NUM_ITERATIONS),
            '--quiet'
        ]

        if parallel_pairs:
            cmd_args.extend(['--parallel', '--workers', str(pair_workers)])

        if LLM_AS_JUDGE:
            cmd_args.append('--llm-as-judge')

        log_print(f"  Command: {' '.join(cmd_args)}", gpu_id)

        with open(log_file, 'w') as f:
            f.write(f"PersonaAgent batch mode\n")
            f.write(f"Command: {' '.join(cmd_args)}\n")
            f.write(f"{'='*60}\n\n")
            f.flush()

            process = subprocess.Popen(
                cmd_args,
                cwd='',
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                text=True
            )
            return_code = process.wait()

        end_time = time.time()
        duration = end_time - start_time

        if return_code == 0:
            log_print(f"✅ PersonaAgent {dataset_name} (batch mode) Complete! duration: {duration/60:.1f}min", gpu_id)
            return True
        else:
            log_print(f"❌ PersonaAgent {dataset_name} (batch mode) failed! error code: {return_code}", gpu_id)
            return False

    except Exception as e:
        import traceback
        log_print(f"❌ PersonaAgent {dataset_name} (batch mode) exception: {e}", gpu_id)
        log_print(f"   Traceback: {traceback.format_exc()}", gpu_id)
        return False

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
        log_print(f"📊 GPU {gpu_id} IDS_TAP Complete: {pohf_success}/{len(datasets)} datasetssuccess", gpu_id)

    if RUN_PERSONA_AGENT:
        log_print(f"📍 GPU {gpu_id} Phase 2: Running PersonaAgent baseline", gpu_id)
        for lamp_type, dataset_name in datasets:
            success = run_persona_agent_on_dataset(lamp_type, dataset_name, gpu_id, log_dir)
            persona_results.append((dataset_name, "PersonaAgent", success))

        persona_success = sum(1 for _, _, success in persona_results if success)
        log_print(f"📊 GPU {gpu_id} PersonaAgent Complete: {persona_success}/{len(datasets)} datasetssuccess", gpu_id)

    all_results = pohf_results + persona_results
    total_success = sum(1 for _, _, success in all_results if success)
    total_tasks = len(all_results)
    log_print(f"📊 GPU {gpu_id} All Complete: {total_success}/{total_tasks}  tasks success", gpu_id)

    return all_results

def main():
    log_print("IDS_TAP + PersonaAgent Multi-GPU parallel execution script")
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
            (9, "LongLaMP-9 (Product Review)"),
        ],
        2: [
            (5, "LongLaMP-5 (News Categorization)"),
            (10, "LongLaMP-10 (Topic Writing)"),
        ],
        3: [
            (0, "UltraChat"),
            (-2, "Prefevel"),
            (8, "LongLaMP-8 (Abstract Writing)"),
        ],
    }

    log_print("📋 Execution plan:")
    log_print("")
    log_print("  🔄 GPU parallel execution（Each GPU runs sequentially IDS_TAP + PersonaAgent）:")
    for gpu_id, datasets in sorted(gpu_dataset_mapping_parallel.items()):
        dataset_names = " -> ".join([name for _, name in datasets])
        total_counters = sum(DATASET_COUNTER_CONFIG.get(lamp_type, 40) for lamp_type, _ in datasets)
        log_print(f"    GPU {gpu_id}: {dataset_names}")
        log_print(f"           Total {len(datasets)} datasets, {total_counters}  counters")
        log_print(f"           Execution order: IDS_TAP -> PersonaAgent")
    log_print("")

    processes = []
    total_start_time = time.time()

    log_print("🚀 Starting GPU tasks（POHF + PersonaAgent）...")

    for gpu_id, datasets in gpu_dataset_mapping_parallel.items():
        p = multiprocessing.Process(
            target=run_all_algorithms_on_gpu_sequential,
            args=(datasets, gpu_id, log_dir)
        )
        p.start()
        processes.append((gpu_id, p))
        log_print(f"  Started GPU {gpu_id} process (IDS_TAP + PersonaAgent)")
        time.sleep(10)

    for gpu_id, p in processes:
        p.join()
        log_print(f"  GPU {gpu_id} process Completed")

    total_time = time.time() - total_start_time

    if not LLM_AS_JUDGE:
        log_print("")
        log_print("📊 LLM_AS_JUDGE=False，Automatically generating ROUGE-L progress charts...")
        try:
            plot_output_dir = "./rougeL_conter_progress_withPersonaAgent"
            os.makedirs(plot_output_dir, exist_ok=True)
            log_print(f"  📁 Output directory: {plot_output_dir}")

            plot_script = "./plot_rougescore_progress.py"
            plot_result = subprocess.run(
                [sys.executable, plot_script],
                cwd='',
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
        log_print(f"           algorithms: IDS_TAP + PersonaAgent")
    log_print("")
    log_print("✅ All GPU runs Complete (IDS_TAP + PersonaAgent)!")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
