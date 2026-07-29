# IDS-TAP

This repository contains the implementation of IDS-TAP for test-time personalized language-model generation. It includes the main preference-optimization pipeline, configuration, evaluation utilities, a local embedding service, and analysis scripts.

No datasets, model checkpoints, API credentials, or generated experiment results are included.

## Environment

Python 3.10 or newer is recommended. Install the packages required by the main pipeline:

```bash
pip install torch numpy scipy scikit-learn nltk rouge-score bert-score \
  openai httpx aiohttp matplotlib tqdm sentence-transformers \
  fastapi uvicorn pydantic backpack-for-pytorch ijson \
  langchain-community faiss-cpu pandas
```

Set the API credential through an environment variable. Do not place credentials in source files:

```bash
export OPENROUTER_API_KEY="your-api-key"
export EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"
```

`EMBEDDING_MODEL` may also point to a locally available model directory.

## Data layout

Datasets are not distributed with this repository. Place them under the repository root using the paths configured in `IDS_TAP_parameters.py` and `run_all_datasets_parallel.py`. The default layout is:

```text
APOHF-main/
  time/
  longLaMP/
PrefEval_dataset/
ultrachat_multiturn/
wildchat/
```

Update `PATH_CONFIG` in `IDS_TAP_parameters.py` or the dataset mapping in `run_all_datasets_parallel.py` when using a different layout.

## Running

Start the local OpenAI-compatible embedding service:

```bash
python embedding_server.py
```

Run a single IDS-TAP experiment with the settings in `IDS_TAP_parameters.py`:

```bash
python ID_TAP.py
```

Run the configured multi-GPU experiments:

```bash
python run_all_datasets_parallel.py
```

The GPU mapping, enabled baselines, dataset selection, and experiment hyperparameters can be changed in `run_all_datasets_parallel.py` and `IDS_TAP_parameters.py`.
