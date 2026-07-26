"""
Test-Time User Preference Alignment

This module implements Algorithm 1: Test-Time User Preference Alignment
for optimizing persona based on user feedback through text gradients.

The algorithm uses:
1. RAG to retrieve relevant episodic memories (via embedding similarity)
2. LLM to generate text gradients (feedback)
3. LLM to update persona based on aggregated feedback
"""

import os
import json
import time
import random
import sys
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from rouge_score import rouge_scorer

# ============================================================================
# Result Saving Configuration
# ============================================================================

# Per-counter detailed results: persona_{dataset}_counter{counter}_all_algorithms.json
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSONA_AGENT_DETAILED_DIR = os.path.join(PROJECT_ROOT, "persona_results")

# Dataset name mapping (must match POHF.py get_dataset_name() for file merge)
DATASET_NAMES = {
    4: "lamp4",
    5: "lamp5",
    8: "lamp8",
    9: "lamp9",
    10: "lamp10",
    0: "ultrachat",
    -1: "wildchat",
    -2: "prefeval",  # Must be "prefeval" to match POHF naming convention
}


def get_dataset_type_from_identifier(dataset_identifier) -> int:
    """Convert dataset identifier to type ID."""
    if isinstance(dataset_identifier, int):
        return dataset_identifier

    name_to_type = {
        "lamp4": 4, "lamp5": 5, "lamp8": 8, "lamp9": 9, "lamp10": 10,
        "ultrachat": 0, "wildchat": -1,
        "prefeval": -2, "lamp-2": -2  # Both names map to -2
    }
    return name_to_type.get(str(dataset_identifier).lower(), None)


def get_dataset_name(dataset_type: int) -> str:
    """Get dataset name from type ID."""
    return DATASET_NAMES.get(dataset_type, f"dataset_{dataset_type}")

def log_progress(msg: str, flush: bool = True):
    """Print a timestamped progress message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=flush)

# Import LangChain for RAG retrieval
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np

# Import from load_data.py
from load_data import (
    load_template_data,
    client,  # Reuse the OpenAI client
)

# ============================================================================
# LangChain Embedding Model for RAG Retrieval
# ============================================================================

# Global LangChain embedding model (lazy initialization)
_langchain_embeddings = None

def get_langchain_embeddings():
    """Get or create the LangChain HuggingFaceEmbeddings model."""
    global _langchain_embeddings
    if _langchain_embeddings is None:
        _langchain_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    return _langchain_embeddings

# ============================================================================
# Hyperparameters
# ============================================================================

# Batch size (k): Number of relevant items to retrieve from EpisodicMemory
BATCH_SIZE = 3

# Number of optimization iterations (E)
NUM_ITERATIONS = 1

# LLM model for alignment
ALIGNMENT_MODEL = "deepseek/deepseek-v3.2"

# Maximum parallel workers for processing input/output pairs
MAX_PARALLEL_WORKERS = 5

# ============================================================================
# ROUGE-L Score Calculation (Same as POHF.py)
# ============================================================================

def compute_rouge_l_score(ground_truth: str, response: str) -> float:
    """
    Compute ROUGE-L F-measure score between ground_truth and response.

    This uses the exact same logic as POHF.py for fair comparison:
    - Uses rouge_scorer.RougeScorer with use_stemmer=True
    - Returns the rougeL.fmeasure score

    Args:
        ground_truth: The expected/reference text
        response: The generated response text

    Returns:
        ROUGE-L F-measure score (float between 0 and 1)
    """
    # Handle non-string inputs (same as POHF.py)
    if isinstance(ground_truth, list):
        ground_truth = " ".join(str(item) for item in ground_truth)
    elif not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)

    if isinstance(response, list):
        response = " ".join(str(item) for item in response)
    elif not isinstance(response, str):
        response = str(response)

    # Calculate ROUGE-L score using same method as POHF.py
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, response)
    return scores['rougeL'].fmeasure


# ============================================================================
# BERTScore Calculation
# ============================================================================

_pa_bertscore_scorer = None

def _get_pa_bertscore_scorer():
    """Lazy-init cached BERTScorer for PersonaAgent."""
    global _pa_bertscore_scorer
    if _pa_bertscore_scorer is None:
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FTPERSLLM'))
            from IDS_TAP_parameters import BERTSCORE_CONFIG
            from bert_score import BERTScorer
            model_type = BERTSCORE_CONFIG.get("model_type", "roberta-large")
            device     = BERTSCORE_CONFIG.get("device", "cuda:0")
            _pa_bertscore_scorer = BERTScorer(model_type=model_type, device=device, rescale_with_baseline=False)
        except Exception as e:
            log_progress(f"⚠️ BERTScore init failed: {e}. Falling back to ROUGE-L.")
            _pa_bertscore_scorer = None
    return _pa_bertscore_scorer

def compute_bertscore_score(ground_truth: str, response: str) -> float:
    """
    Compute BERTScore F1 between ground_truth and response.
    Falls back to ROUGE-L if bert_score is unavailable.

    Returns:
        BERTScore F1 (float between 0 and 1)
    """
    if isinstance(ground_truth, list):
        ground_truth = " ".join(str(item) for item in ground_truth)
    elif not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)

    if isinstance(response, list):
        response = " ".join(str(item) for item in response)
    elif not isinstance(response, str):
        response = str(response)

    bs = _get_pa_bertscore_scorer()
    if bs is None:
        return compute_rouge_l_score(ground_truth, response)
    try:
        P, R, F = bs.score([response], [ground_truth])
        return float(F[0])
    except Exception as e:
        log_progress(f"⚠️ BERTScore compute failed: {e}. Returning 0.")
        return 0.0


def _get_scoring_mode() -> str:
    """Return current scoring_mode from IDS_TAP_parameters (rouge or bertscore)."""
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FTPERSLLM'))
        from IDS_TAP_parameters import ROUGE_CONFIG
        return ROUGE_CONFIG.get("scoring_mode", "rouge")
    except Exception:
        return "rouge"


# ============================================================================
# Result Saving Functions
# ============================================================================

def save_persona_agent_result(
    dataset_identifier,
    counter: int,
    query: Any,
    ground_truth: Any,
    instruction: str,
    personas_list: List[str],
    responses_list: List[str],
    inputs_list: List[str],
    detailed_dir: str = PERSONA_AGENT_DETAILED_DIR,
    llm_as_judge: bool = False
) -> None:
    """
    Save PersonaAgent results to per-counter detailed file.
    File location: {detailed_dir}/persona_{dataset}_counter{counter}_all_algorithms.json

    Args:
        dataset_identifier: Dataset name or type ID
        counter: The counter/item index being processed
        query: The query text (may be list for contextual mode)
        ground_truth: The ground truth output (may be list for contextual mode)
        instruction: The task instruction
        personas_list: List of optimized personas (one per input)
        responses_list: List of generated responses (one per input)
        inputs_list: List of input texts
        detailed_dir: Directory for per-counter detailed results (default: PERSONA_AGENT_DETAILED_DIR)
        llm_as_judge: If True, skip ROUGE-L score calculation (for LLM-as-judge evaluation mode)
    """
    from datetime import datetime

    os.makedirs(detailed_dir, exist_ok=True)

    # Get dataset type and name
    dataset_type = get_dataset_type_from_identifier(dataset_identifier)
    dataset_name = get_dataset_name(dataset_type) if dataset_type is not None else str(dataset_identifier)

    # Build PersonaAgent algorithm data (POHF-compatible format)
    is_contextual = len(personas_list) > 1
    num_queries = len(personas_list)

    # Calculate scores for each response (ROUGE-L or BERTScore depending on config)
    # Skip if llm_as_judge is True (will use LLM evaluation instead)
    scoring_mode = _get_scoring_mode()
    score_label = "BERTScore" if scoring_mode == "bertscore" else "ROUGE-L"
    greedy_scores_list = []
    best_greedy_score = None

    if not llm_as_judge:
        for q_idx in range(num_queries):
            response = responses_list[q_idx] if q_idx < len(responses_list) else ""
            if isinstance(ground_truth, list) and q_idx < len(ground_truth):
                gt = ground_truth[q_idx]
            else:
                gt = ground_truth
            if scoring_mode == "bertscore":
                score = compute_bertscore_score(gt, response)
            else:
                score = compute_rouge_l_score(gt, response)
            greedy_scores_list.append(score)

        # Use the best score among all queries (or the only score for single query)
        best_greedy_score = max(greedy_scores_list) if greedy_scores_list else None
        log_progress(
            f"📊 {score_label} scores: {[f'{s:.4f}' for s in greedy_scores_list]}, best={best_greedy_score:.4f}"
            if best_greedy_score is not None else f"📊 No {score_label} scores computed"
        )
    else:
        log_progress("📊 LLM-as-judge mode: skipping score calculation")

    # PersonaAgent data for algorithms section
    if is_contextual:
        persona_agent_algo_data = {
            "greedy_arm_index": -1,  # PersonaAgent doesn't use arm selection
            "persona": personas_list[-1] if personas_list else "",  # Final optimized persona
            "response": responses_list[-1] if responses_list else "",
            "personas": personas_list,
            "responses": responses_list,
            "inputs": inputs_list,
        }
        # Only add greedy_score if not in LLM-as-judge mode
        if not llm_as_judge:
            persona_agent_algo_data["greedy_score"] = best_greedy_score
            persona_agent_algo_data["scoring_mode"] = scoring_mode
            persona_agent_algo_data["greedy_scores"] = greedy_scores_list if greedy_scores_list else None
    else:
        persona_agent_algo_data = {
            "greedy_arm_index": -1,
            "persona": personas_list[0] if personas_list else "",
            "response": responses_list[0] if responses_list else "",
        }
        # Only add greedy_score if not in LLM-as-judge mode
        if not llm_as_judge:
            persona_agent_algo_data["greedy_score"] = greedy_scores_list[0] if greedy_scores_list else None
            persona_agent_algo_data["scoring_mode"] = scoring_mode

    # ========== Save to per-counter detailed file (persona_results/persona_{dataset}_counter{counter}_all_algorithms.json) ==========
    # Add LLM suffix when llm_as_judge is True
    if llm_as_judge:
        detailed_filepath = os.path.join(detailed_dir, f"persona_{dataset_name}_counter{counter}_all_algorithms_LLM.json")
    else:
        detailed_filepath = os.path.join(detailed_dir, f"persona_{dataset_name}_counter{counter}_all_algorithms.json")

    # Build detailed per-query data (POHF-compatible format)
    queries_data = {}
    for q_idx in range(num_queries):
        query_text = inputs_list[q_idx] if q_idx < len(inputs_list) else ""
        gt_text = ground_truth[q_idx] if isinstance(ground_truth, list) and q_idx < len(ground_truth) else ground_truth

        persona_agent_query_data = {
            "greedy_arm_index": -1,
            "persona_summary": personas_list[q_idx] if q_idx < len(personas_list) else "",
            "response": responses_list[q_idx] if q_idx < len(responses_list) else "",
        }
        # Only add greedy_score if not in LLM-as-judge mode
        if not llm_as_judge:
            q_score = greedy_scores_list[q_idx] if q_idx < len(greedy_scores_list) else None
            persona_agent_query_data["greedy_score"] = q_score
            persona_agent_query_data["scoring_mode"] = scoring_mode

        queries_data[f"query_{q_idx}"] = {
            "query_index": q_idx,
            "query_text": query_text,
            "ground_truth": gt_text,
            "algorithms": {
                "PersonaAgent": persona_agent_query_data
            }
        }

    # Load existing detailed file to merge with POHF results
    existing_detailed = None
    if os.path.exists(detailed_filepath):
        try:
            with open(detailed_filepath, 'r', encoding='utf-8') as f:
                existing_detailed = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log_progress(f"⚠️ Failed to read existing detailed JSON: {e}")
            existing_detailed = None

    if existing_detailed:
        # Merge PersonaAgent results into existing POHF results
        if 'queries' in existing_detailed:
            for q_key, q_data in queries_data.items():
                if q_key in existing_detailed['queries']:
                    # Add PersonaAgent to existing query's algorithms
                    if 'algorithms' not in existing_detailed['queries'][q_key]:
                        existing_detailed['queries'][q_key]['algorithms'] = {}
                    existing_detailed['queries'][q_key]['algorithms']['PersonaAgent'] = q_data['algorithms']['PersonaAgent']
                else:
                    # Add new query entry
                    existing_detailed['queries'][q_key] = q_data
        existing_detailed['saved_at'] = datetime.now().isoformat()
        final_detailed = existing_detailed
    else:
        # Create new detailed file
        final_detailed = {
            "counter": counter,
            "lamp_type": dataset_type if dataset_type is not None else -999,
            "contextual_mode": is_contextual,
            "num_queries": num_queries,
            "queries": queries_data,
            "saved_at": datetime.now().isoformat()
        }

    # Save detailed file
    with open(detailed_filepath, 'w', encoding='utf-8') as f:
        json.dump(final_detailed, f, ensure_ascii=False, indent=2)

    log_progress(f"💾 Saved to detailed file: {detailed_filepath}")


# ============================================================================
# Prompt Templates
# ============================================================================

INITIAL_PERSONA_TEMPLATE = """You are a helpful personalized assistant.
Use the RAG tool to retrieve relevant information from user's history and preferences to answer questions.
User summary:
{semantic_memory}
STRICT RULES: when using tools, always:
1. Think step-by-step about what information you need.
2. Use the RAG tool to search for relevant user history and preferences.
3. Provide clear, concise responses based on user's personalized context.
4. Do not give explanation in the final answer."""

LOSS_GRADIENT_PROMPT = """You are a meticulous and critical evaluator of personalized AI agent responses.
Analyze the following and give the feedback on how to improve the system prompt to align with the user's preferences.
Question:
{question}
Expected Answer: {ground_truth}
Agent Response: {response}
Your feedback should focus on how to adjust the persona system prompt to tailor the agent's responses to the individual user's unique characteristics.
Make sure the feedback is concise and clear.
Tips:
1. Explain on how to improve the search keywords of tools for this user.
2. Take the user's prior interactions, preferences, and any personalization aspects into consideration.
3. Provide explicit description for user profile and preferences that is not specific to this task.
Feedback:"""

GRADIENT_UPDATE_PROMPT = """You are a prompt engineering assistant tasked with refining the personal agent system prompts for improved user preference alignment.
Current system prompt:
{current_persona}
Provided Feedback:
{aggregated_feedback}
Based on the feedback above, generate an updated system prompt that explicitly highlights the user's unique preferences.
Ensure that the prompt instructs the agent to align its responses with the user's preferences, including detailed user profile or preferences.
Please maintain a helpful and clear tone in the system prompt.
New system prompt:"""

# ============================================================================
# LLM Call Utilities
# ============================================================================

def call_llm(prompt: str, system_prompt: str = "", temperature: float = 0.4, max_retries: int = 5, label: str = "") -> str:
    """Call LLM with retry logic.

    Note: system_prompt (persona) is prepended to the user prompt instead of being
    used as a system role, to maintain consistency with naive baseline generation
    and avoid over-emphasizing persona in the system role.
    """
    base_delay = 2.0
    start_time = time.time()

    if label:
        log_progress(f"🔄 LLM call started: {label}")

    # 将 persona (system_prompt) 直接加入到 prompt 之前，不使用 system role
    # 这样可以避免过度强调 persona，与 naive baseline 保持一致
    if system_prompt:
        combined_prompt = f"""
{system_prompt}

### Task:
{prompt}

Please provide the response based on the personality description above."""
    else:
        combined_prompt = prompt

    messages = [{"role": "user", "content": combined_prompt}]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=ALIGNMENT_MODEL,
                messages=messages,
                temperature=0.4,
                timeout=60.0
            )

            if not response or not response.choices or len(response.choices) == 0:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    log_progress(f"⚠️ LLM API invalid response, retrying in {delay:.1f}s ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("LLM API returned invalid response")

            content = response.choices[0].message.content
            if content is None or not content.strip():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    log_progress(f"⚠️ LLM API empty content, retrying in {delay:.1f}s ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("LLM API returned empty content")

            elapsed = time.time() - start_time
            if label:
                log_progress(f"✅ LLM call done: {label} ({elapsed:.1f}s)")
            return content.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                log_progress(f"⚠️ LLM API error: {e}, retrying in {delay:.1f}s ({attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise e

    return ""

# ============================================================================
# Core Algorithm Functions
# ============================================================================

def retrieve_top_k_entries(
    query: str,
    episodic_memory: List[List],
    k: int = BATCH_SIZE,
    exclude_queries: List[str] = None
) -> List[Tuple[str, str, dict]]:
    """
    Use LangChain FAISS vectorstore to retrieve top-k relevant entries from EpisodicMemory.

    Args:
        query: The input query to find relevant memories for
        episodic_memory: List of [query, ground_truth, metadata]
        k: Number of entries to retrieve
        exclude_queries: List of queries to exclude (冗余参数，保留以兼容接口)

    Returns:
        List of (query, ground_truth, metadata) tuples for the top-k entries
    """
    if len(episodic_memory) == 0:
        return []

    # 注意：exclude_queries 参数是冗余的，因为 EpisodicMemory 和 test entries
    # 在 load_data.py 中已经是分开的，不会重叠。保留此参数仅为兼容现有调用接口。

    if len(episodic_memory) <= k:
        # Return all if fewer than k entries
        return [(e[0], e[1], e[2] if len(e) > 2 else {}) for e in episodic_memory]

    try:
        # Get LangChain embedding model
        embeddings = get_langchain_embeddings()

        # Build LangChain Documents from episodic_memory
        documents = []
        for i, entry in enumerate(episodic_memory):
            q = entry[0] if len(entry) > 0 else ""
            gt = entry[1] if len(entry) > 1 else ""
            meta = entry[2] if len(entry) > 2 else {}

            # Combine query and ground_truth for embedding (same as before)
            page_content = f"{q} {gt}"

            # Store original data in metadata for retrieval
            doc_metadata = {
                "original_query": q,
                "original_ground_truth": gt,
                "original_index": i,
                **meta  # Include any additional metadata
            }
            documents.append(Document(page_content=page_content, metadata=doc_metadata))

        # Build FAISS vectorstore from documents
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Retrieve top-k similar documents
        results_docs = vectorstore.similarity_search(query, k=k)

        # Convert back to expected format
        results = []
        for doc in results_docs:
            q = doc.metadata.get("original_query", "")
            gt = doc.metadata.get("original_ground_truth", "")
            # Extract original metadata (exclude our internal keys)
            meta = {k: v for k, v in doc.metadata.items()
                    if k not in ["original_query", "original_ground_truth", "original_index"]}
            results.append((q, gt, meta))

        return results

    except Exception as e:
        print(f"⚠️ LangChain FAISS retrieval error: {e}, using first {k} entries")
        return [(e[0], e[1], e[2] if len(e) > 2 else {})
                for e in episodic_memory[:k]]


def generate_agent_response(query: str, persona: str, episodic_memory: List[List] = None, k: int = BATCH_SIZE, exclude_queries: List[str] = None) -> str:
    """
    Generate response using current persona with direct LLM call.
    Optionally retrieves relevant episodic memory entries to include in context.

    Args:
        query: User query
        persona: Current persona/system prompt
        episodic_memory: Optional episodic memory for RAG context
        k: Number of relevant entries to retrieve
        exclude_queries: List of queries to exclude from retrieval (防止信息泄露)

    Returns:
        Agent's response string
    """
    # Build the enhanced query with RAG context if available
    enhanced_query = query

    if episodic_memory and len(episodic_memory) > 0:
        # Retrieve relevant entries using embedding similarity
        relevant_entries = retrieve_top_k_entries(query, episodic_memory, k, exclude_queries=exclude_queries)

        if relevant_entries:
            # Format relevant entries as context
            context_parts = []
            for i, (q, gt, meta) in enumerate(relevant_entries):
                context_parts.append(f"[Reference {i+1}]\nQuery: {q}\nResponse: {gt}")

            context_text = "\n\n".join(context_parts)

            enhanced_query = f"""Based on the following reference examples from the user's history, generate a response that matches their style and preferences.

--- Reference Examples ---
{context_text}

--- Task ---
{query}

IMPORTANT: Generate the response DIRECTLY in the user's style. 
Just write the actual content as if you are the user."""

    return call_llm(enhanced_query, system_prompt=persona, temperature=0.0)


def compute_loss_gradient(query: str, response: str, ground_truth: str) -> str:
    """
    Compute text gradient (feedback) for a single (q, r̂, r_gt) tuple.

    Args:
        query: Original query
        response: Agent's response
        ground_truth: Expected/ground truth response

    Returns:
        Text feedback/gradient
    """
    prompt = LOSS_GRADIENT_PROMPT.format(
        question=query,
        ground_truth=ground_truth,
        response=response
    )

    return call_llm(prompt, temperature=0.0)



def optimization(
    d_batch: List[Tuple[str, str, str]],
    persona: str
) -> str:
    """
    OPTIMIZATION procedure from Algorithm 1.

    Args:
        d_batch: List of (query, response, ground_truth) tuples
        persona: Current persona P

    Returns:
        Updated persona P*
    """
    opt_start = time.time()
    log_progress(f"⚡ OPTIMIZATION started with {len(d_batch)} items")

    # Step 4: Initialize empty list for loss gradients
    gradients = []

    # Steps 5-8: For each (q, r̂, r_gt) compute gradient and collect
    for idx, (query, response, ground_truth) in enumerate(d_batch):
        # Step 6: Compute gradient ∇ ← LLM_grad(q, r̂, r_gt)
        gradient = compute_loss_gradient(query, response, ground_truth)
        # Step 7: Add loss gradient/feedback ∇ to ∇̂
        gradients.append(gradient)
        log_progress(f"  📝 Gradient {idx+1}/{len(d_batch)} computed for: {query[:40]}...")

    # Step 9: Gradient update P* ← LLM_update(∇̂, P)
    aggregated_feedback = "\n\n---\n\n".join([
        f"Feedback {i+1}:\n{g}" for i, g in enumerate(gradients)
    ])

    prompt = GRADIENT_UPDATE_PROMPT.format(
        current_persona=persona,
        aggregated_feedback=aggregated_feedback
    )

    log_progress("🔧 Updating persona with aggregated feedback...")
    updated_persona = call_llm(prompt, temperature=0.0, label="persona_update")

    opt_elapsed = time.time() - opt_start
    log_progress(f"⚡ OPTIMIZATION complete ({opt_elapsed:.1f}s)")

    return updated_persona


def test_time_alignment(
    episodic_memory: List[List],
    semantic_memory: str,
    input_query: str,
    batch_size: int = BATCH_SIZE,
    num_iterations: int = NUM_ITERATIONS,
    verbose: bool = True,
    exclude_queries: List[str] = None
) -> str:
    """
    Algorithm 1: Test-Time User Preference Alignment

    Args:
        episodic_memory: User's episodic memory (list of [query, ground_truth, metadata])
        semantic_memory: User's semantic memory (high-level preferences)
        input_query: Current input query to optimize for
        batch_size: Number of items to retrieve per batch (k)
        num_iterations: Number of optimization iterations (E)
        verbose: Whether to print progress
        exclude_queries: List of queries to exclude from retrieval (防止信息泄露，排除所有测试集 inputs)

    Returns:
        Optimized persona P*
    """
    alignment_start = time.time()

    # Step 1: Initialize persona P with semantic memory
    persona = INITIAL_PERSONA_TEMPLATE.format(semantic_memory=semantic_memory)

    if verbose:
        log_progress("=" * 60)
        log_progress("🚀 Starting Test-Time User Preference Alignment")
        log_progress(f"   Batch size (k): {batch_size}")
        log_progress(f"   Iterations (E): {num_iterations}")
        log_progress(f"   EpisodicMemory entries: {len(episodic_memory)}")
        log_progress(f"   Query: {input_query[:60]}...")
        if exclude_queries:
            log_progress(f"   Excluded queries: {len(exclude_queries)} (防止信息泄露)")
        log_progress("=" * 60)

    # Steps 12-16: Main optimization loop
    for iteration in range(1, num_iterations + 1):
        iter_start = time.time()
        if verbose:
            log_progress(f"📍 Iteration {iteration}/{num_iterations} started")

        # Step 13: Obtain batch D_batch from user data D using RAG
        if verbose:
            log_progress(f"   🔍 Retrieving top-{batch_size} relevant entries...")

        retrieved_entries = retrieve_top_k_entries(
            query=input_query,
            episodic_memory=episodic_memory,
            k=batch_size,
            exclude_queries=exclude_queries
        )

        if verbose:
            log_progress(f"   ✅ Retrieved {len(retrieved_entries)} entries")

        # Step 14: Add agent responses to D_batch
        # 🔧 [修复] 生成 response 时不使用 RAG，避免信息泄露
        # 直接用 persona 作为 system prompt，只传入 query
        d_batch = []
        for idx, (query, ground_truth, _) in enumerate(retrieved_entries):
            if verbose:
                log_progress(f"   🤖 Generating response {idx+1}/{len(retrieved_entries)}: {query[:40]}...")
            # 直接调用 LLM，不使用 RAG 上下文（与最终生成保持一致）
            response = call_llm(query, system_prompt=persona, temperature=0.0)
            d_batch.append((query, response, ground_truth))

        # Step 15: P* ← OPTIMIZATION(D_batch, P)
        persona = optimization(d_batch, persona)

        iter_elapsed = time.time() - iter_start
        if verbose:
            log_progress(f"📍 Iteration {iteration}/{num_iterations} complete ({iter_elapsed:.1f}s)")

    alignment_elapsed = time.time() - alignment_start
    if verbose:
        log_progress("=" * 60)
        log_progress(f"✨ Alignment complete! Total time: {alignment_elapsed:.1f}s")
        log_progress("=" * 60)

    return persona



def _process_single_pair(
    pair_index: int,
    input_text: str,
    output_text: str,
    episodic_memory: List[List],
    semantic_memory: str,
    batch_size: int,
    num_iterations: int,
    total_pairs: int,
    verbose: bool,
    exclude_queries: List[str] = None
) -> Tuple[int, str, str, str]:
    """
    Process a single input/output pair (worker function for parallel execution).

    Returns:
        Tuple of (pair_index, persona, response, output_text)
    """
    thread_id = threading.current_thread().name
    pair_start = time.time()

    if verbose:
        log_progress(f"[{thread_id}] {'='*50}")
        log_progress(f"[{thread_id}] 📍 Processing pair {pair_index+1}/{total_pairs}")
        log_progress(f"[{thread_id}]    Input: {input_text[:60]}...")

    # Generate unique persona for this input (with verbose=False to reduce output noise)
    optimized_persona = test_time_alignment(
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        input_query=input_text,
        batch_size=batch_size,
        num_iterations=num_iterations,
        verbose=False,  # Disable verbose in parallel mode to avoid messy output
        exclude_queries=exclude_queries  # 防止信息泄露
    )

    # Generate response using the optimized persona (only query, no RAG context)
    log_progress(f"[{thread_id}] 🤖 Generating final response for pair {pair_index+1}...")
    response = call_llm(input_text, system_prompt=optimized_persona, temperature=0.0)

    pair_elapsed = time.time() - pair_start
    if verbose:
        log_progress(f"[{thread_id}] ✅ Pair {pair_index+1}/{total_pairs} complete ({pair_elapsed:.1f}s)")
        log_progress(f"[{thread_id}]    Response: {response[:60]}...")

    return (pair_index, optimized_persona, response, output_text)


def run_alignment_from_dataset(
    dataset_identifier,
    counter: int = 0,
    batch_size: int = BATCH_SIZE,
    num_iterations: int = NUM_ITERATIONS,
    verbose: bool = True,
    max_pairs: int = None,
    parallel: bool = False,
    max_workers: int = MAX_PARALLEL_WORKERS,
    auto_save: bool = True,
    detailed_dir: str = PERSONA_AGENT_DETAILED_DIR,
    llm_as_judge: bool = False
) -> Tuple[List[str], List[str], List[str]]:
    """
    Run alignment for each input/output pair from a dataset.

    For LaMP datasets (4, 5, 8, 9, 10), this function:
    1. Generates a unique persona for each input/output pair
    2. Combines each input with its corresponding persona
    3. Calls LLM to get response for each pair
    4. Auto-saves results in POHF-compatible format (if auto_save=True)

    Args:
        dataset_identifier: Dataset name or ID (e.g., "lamp4", 4, "prefeval", -2)
        counter: Item index in the dataset
        batch_size: Number of items to retrieve per batch (k)
        num_iterations: Number of optimization iterations (E)
        verbose: Whether to print progress
        max_pairs: Maximum number of input/output pairs to process (None = all)
        parallel: Whether to process pairs in parallel (default: False)
        max_workers: Maximum number of parallel workers (default: MAX_PARALLEL_WORKERS)
        auto_save: Whether to automatically save results (default: True)
        detailed_dir: Directory for per-counter detailed results (default: PERSONA_AGENT_DETAILED_DIR)
        llm_as_judge: If True, skip ROUGE-L score calculation (for LLM-as-judge evaluation mode)

    Returns:
        Tuple of (personas_list, responses_list, outputs_list)
    """
    total_start = time.time()

    # Load data from dataset
    if verbose:
        log_progress(f"📂 Loading data from dataset: {dataset_identifier}, item: {counter}")

    load_start = time.time()
    result = load_template_data(dataset_identifier, counter)
    load_elapsed = time.time() - load_start

    if verbose:
        log_progress(f"📂 Data loaded in {load_elapsed:.1f}s")

    # Check if result is in new format (with lists) or old format (single values)
    episodic_memory = result[0]
    semantic_memory = result[1]

    # Handle both old format (single input/output) and new format (lists)
    if isinstance(result[2], list):
        inputs_list = result[2]
        outputs_list = result[3]
    else:
        # Old format: single input/output
        inputs_list = [result[2]]
        outputs_list = [result[3]]

    # Limit the number of pairs to process if max_pairs is specified
    if max_pairs is not None:
        inputs_list = inputs_list[:max_pairs]
        outputs_list = outputs_list[:max_pairs]

    if verbose:
        log_progress(f"   EpisodicMemory: {len(episodic_memory)} entries")
        log_progress(f"   SemanticMemory: {len(semantic_memory)} chars")
        log_progress(f"   Number of input/output pairs to process: {len(inputs_list)}")
        if parallel:
            log_progress(f"   🚀 Parallel mode enabled with {max_workers} workers")

    if parallel and len(inputs_list) > 1:
        # Parallel processing mode
        results = [None] * len(inputs_list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {}
            for i, (input_text, output_text) in enumerate(zip(inputs_list, outputs_list)):
                future = executor.submit(
                    _process_single_pair,
                    pair_index=i,
                    input_text=input_text,
                    output_text=output_text,
                    episodic_memory=episodic_memory,
                    semantic_memory=semantic_memory,
                    batch_size=batch_size,
                    num_iterations=num_iterations,
                    total_pairs=len(inputs_list),
                    verbose=verbose,
                    exclude_queries=inputs_list  # 🔧 防止信息泄露：排除所有测试集 inputs
                )
                futures[future] = i

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    pair_idx, persona, response, output = future.result()
                    results[pair_idx] = (persona, response, output)
                except Exception as e:
                    pair_idx = futures[future]
                    print(f"⚠️ Error processing pair {pair_idx + 1}: {e}")
                    results[pair_idx] = ("Error", f"Error: {e}", outputs_list[pair_idx])

        # Unpack results maintaining original order
        personas_list = [r[0] for r in results]
        responses_list = [r[1] for r in results]
        outputs_list = [r[2] for r in results]

    else:
        # Sequential processing mode (original behavior)
        personas_list = []
        responses_list = []

        for i, (input_text, output_text) in enumerate(zip(inputs_list, outputs_list)):
            if verbose:
                print(f"\n{'='*60}")
                print(f"📍 Processing pair {i+1}/{len(inputs_list)}")
                print(f"   Input: {input_text[:100]}...")
                print(f"   Expected Output: {output_text[:100]}...")

            # Generate unique persona for this input
            optimized_persona = test_time_alignment(
                episodic_memory=episodic_memory,
                semantic_memory=semantic_memory,
                input_query=input_text,
                batch_size=batch_size,
                num_iterations=num_iterations,
                verbose=verbose,
                exclude_queries=inputs_list  # 🔧 防止信息泄露：排除所有测试集 inputs
            )
            personas_list.append(optimized_persona)

            # Generate response using the optimized persona (only query, no RAG context)
            if verbose:
                print(f"   🤖 Generating final response with optimized persona...")

            response = call_llm(input_text, system_prompt=optimized_persona, temperature=0.0)
            responses_list.append(response)

            if verbose:
                print(f"   ✅ Response: {response[:100]}...")

    # Auto-save results in POHF-compatible format
    if auto_save:
        try:
            # Determine instruction text based on dataset type
            dataset_type = get_dataset_type_from_identifier(dataset_identifier)
            instruction = ""
            if dataset_type == 4:
                instruction = "Generate a headline for the following article:"
            elif dataset_type == 5:
                instruction = "Generate a title for the following abstract:"
            elif dataset_type == 8:
                instruction = "Generate an abstract for the following paper:"
            elif dataset_type == 9:
                instruction = "Generate a product review:"
            elif dataset_type == 10:
                instruction = "Generate a post about the given topic:"
            elif dataset_type == 0:
                instruction = "Predict the next user query:"
            elif dataset_type == -1:
                instruction = "Predict the next user query:"
            elif dataset_type == -2:
                instruction = "Answer the question based on user preferences:"

            # Save results to detailed file
            save_persona_agent_result(
                dataset_identifier=dataset_identifier,
                counter=counter,
                query=inputs_list if len(inputs_list) > 1 else inputs_list[0],
                ground_truth=outputs_list if len(outputs_list) > 1 else outputs_list[0],
                instruction=instruction,
                personas_list=personas_list,
                responses_list=responses_list,
                inputs_list=inputs_list,
                detailed_dir=detailed_dir,
                llm_as_judge=llm_as_judge
            )
        except Exception as e:
            log_progress(f"⚠️ Failed to auto-save results: {e}")

    return personas_list, responses_list, outputs_list


# ============================================================================
# Cross-Counter Parallel Processing (Similar to POHF.py)
# ============================================================================

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Maximum parallel counters (can be overridden by environment variable)
MAX_PARALLEL_COUNTERS = 3


def run_single_counter_process(args) -> dict:
    """
    Process a single counter in a separate process.
    This function is called by ProcessPoolExecutor.

    Args:
        args: Tuple of (counter, run_index, total_counters, config_dict)

    Returns:
        dict: Result containing status, counter, and results
    """
    counter, run_index, total_counters, config_dict = args

    import gc

    log_progress(f"🚀 [Process {run_index + 1}/{total_counters}] Starting counter {counter}")

    try:
        # Run alignment for this counter
        personas_list, responses_list, outputs_list = run_alignment_from_dataset(
            dataset_identifier=config_dict['dataset'],
            counter=counter,
            batch_size=config_dict.get('batch_size', BATCH_SIZE),
            num_iterations=config_dict.get('num_iterations', NUM_ITERATIONS),
            verbose=config_dict.get('verbose', False),  # Reduce noise in parallel mode
            max_pairs=config_dict.get('max_pairs', None),
            parallel=config_dict.get('parallel_pairs', False),  # Inner parallelism for pairs
            max_workers=config_dict.get('max_workers', MAX_PARALLEL_WORKERS),
            auto_save=config_dict.get('auto_save', True),
            detailed_dir=config_dict.get('detailed_dir', PERSONA_AGENT_DETAILED_DIR),
            llm_as_judge=config_dict.get('llm_as_judge', False)
        )

        gc.collect()

        log_progress(f"✅ [Process {run_index + 1}/{total_counters}] Counter {counter} complete")

        return {
            'status': 'success',
            'counter': counter,
            'personas_list': personas_list,
            'responses_list': responses_list,
            'outputs_list': outputs_list
        }

    except Exception as e:
        import traceback
        log_progress(f"❌ [Process {run_index + 1}/{total_counters}] Counter {counter} failed: {e}")
        log_progress(f"   Traceback: {traceback.format_exc()}")

        gc.collect()

        return {
            'status': 'failed',
            'counter': counter,
            'error': str(e),
            'personas_list': [],
            'responses_list': [],
            'outputs_list': []
        }


def run_batch(
    dataset_identifier,
    counter_start: int = 0,
    counter_end: int = 10,
    batch_size: int = BATCH_SIZE,
    num_iterations: int = NUM_ITERATIONS,
    verbose: bool = True,
    max_pairs: int = None,
    parallel_pairs: bool = False,
    max_workers: int = MAX_PARALLEL_WORKERS,
    parallel_counters: int = MAX_PARALLEL_COUNTERS,
    auto_save: bool = True,
    detailed_dir: str = PERSONA_AGENT_DETAILED_DIR,
    llm_as_judge: bool = False
) -> List[dict]:
    """
    Run alignment for multiple counters in parallel (similar to POHF.py).

    Args:
        dataset_identifier: Dataset name or ID
        counter_start: Starting counter index (inclusive)
        counter_end: Ending counter index (exclusive)
        batch_size: Number of items to retrieve per batch (k)
        num_iterations: Number of optimization iterations (E)
        verbose: Whether to print progress
        max_pairs: Maximum number of input/output pairs to process per counter
        parallel_pairs: Whether to parallelize pairs within each counter
        max_workers: Maximum workers for pair-level parallelism
        parallel_counters: Number of counters to process in parallel
        auto_save: Whether to automatically save results
        detailed_dir: Directory for per-counter detailed results
        llm_as_judge: If True, skip ROUGE-L score calculation

    Returns:
        List of result dicts for each counter
    """
    from tqdm import tqdm

    total_start = time.time()
    counter_array = list(range(counter_start, counter_end))

    if len(counter_array) == 0:
        log_progress("⚠️ No counters to process")
        return []

    log_progress("=" * 60)
    log_progress(f"🚀 PersonaAgent Batch Processing")
    log_progress(f"   Dataset: {dataset_identifier}")
    log_progress(f"   Counters: {counter_start} to {counter_end - 1} ({len(counter_array)} total)")
    log_progress(f"   Parallel counters: {parallel_counters}")
    log_progress(f"   Batch size (k): {batch_size}")
    log_progress(f"   Iterations (E): {num_iterations}")
    log_progress("=" * 60)

    # Build config dict for each counter
    config_dict = {
        'dataset': dataset_identifier,
        'batch_size': batch_size,
        'num_iterations': num_iterations,
        'verbose': False,  # Reduce noise in parallel mode
        'max_pairs': max_pairs,
        'parallel_pairs': parallel_pairs,
        'max_workers': max_workers,
        'auto_save': auto_save,
        'detailed_dir': detailed_dir,
        'llm_as_judge': llm_as_judge
    }

    # Build task args
    task_args = [
        (counter, idx, len(counter_array), config_dict)
        for idx, counter in enumerate(counter_array)
    ]

    all_results = []
    failed_counters = []

    if parallel_counters > 1 and len(counter_array) > 1:
        # Parallel mode: use ProcessPoolExecutor
        max_procs = min(parallel_counters, len(counter_array))

        log_progress(f"🔄 Starting {max_procs} parallel processes...")

        with ProcessPoolExecutor(max_workers=max_procs) as executor:
            future_to_counter = {
                executor.submit(run_single_counter_process, args): args[0]
                for args in task_args
            }

            # Use tqdm progress bar
            with tqdm(total=len(counter_array), desc="🚀 Processing counters", unit="counter",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                for future in as_completed(future_to_counter):
                    counter = future_to_counter[future]
                    try:
                        result = future.result(timeout=3600)  # 1 hour timeout per counter

                        if result['status'] == 'success':
                            all_results.append(result)
                            pbar.set_postfix_str(f"✅ Counter {counter}")
                        else:
                            failed_counters.append(counter)
                            pbar.set_postfix_str(f"❌ Counter {counter}")

                    except Exception as e:
                        log_progress(f"❌ Counter {counter} exception: {e}")
                        failed_counters.append(counter)

                    pbar.update(1)
    else:
        # Sequential mode
        log_progress("🔄 Running in sequential mode...")

        for args in tqdm(task_args, desc="🚀 Processing counters", unit="counter"):
            result = run_single_counter_process(args)

            if result['status'] == 'success':
                all_results.append(result)
            else:
                failed_counters.append(result['counter'])

    total_elapsed = time.time() - total_start

    # Summary
    log_progress("=" * 60)
    log_progress(f"📋 BATCH PROCESSING COMPLETE")
    log_progress(f"   Total time: {total_elapsed:.1f}s")
    log_progress(f"   Successful: {len(all_results)}/{len(counter_array)}")
    if failed_counters:
        log_progress(f"   Failed counters: {failed_counters}")
    log_progress("=" * 60)

    return all_results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test-Time User Preference Alignment"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="lamp4",
        help="Dataset identifier (lamp4, lamp5, lamp8, lamp9, lamp10, ultrachat, wildchat, prefeval)"
    )
    parser.add_argument(
        "--counter", "-c",
        type=int,
        default=None,
        help="Single counter to process (use --counter-start/--counter-end for batch mode)"
    )
    parser.add_argument(
        "--counter-start", "-cs",
        type=int,
        default=0,
        help="Starting counter index for batch mode (inclusive)"
    )
    parser.add_argument(
        "--counter-end", "-ce",
        type=int,
        default=None,
        help="Ending counter index for batch mode (exclusive)"
    )
    parser.add_argument(
        "--batch_size", "-k",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size / number of items to retrieve (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--iterations", "-e",
        type=int,
        default=NUM_ITERATIONS,
        help=f"Number of optimization iterations (default: {NUM_ITERATIONS})"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--max-pairs", "-m",
        type=int,
        default=None,
        help="Maximum number of input/output pairs to process (default: all)"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Enable parallel processing of input/output pairs within each counter"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=MAX_PARALLEL_WORKERS,
        help=f"Number of parallel workers for pairs (default: {MAX_PARALLEL_WORKERS})"
    )
    parser.add_argument(
        "--parallel-counters", "-pc",
        type=int,
        default=MAX_PARALLEL_COUNTERS,
        help=f"Number of counters to process in parallel (default: {MAX_PARALLEL_COUNTERS})"
    )
    parser.add_argument(
        "--llm-as-judge", "-l",
        action="store_true",
        help="Enable LLM-as-judge mode: skip ROUGE-L score calculation"
    )

    args = parser.parse_args()

    # Determine mode: single counter or batch
    if args.counter is not None:
        # Single counter mode
        log_progress(f"🚀 Single counter mode: counter={args.counter}")

        personas_list, responses_list, outputs_list = run_alignment_from_dataset(
            dataset_identifier=args.dataset,
            counter=args.counter,
            batch_size=args.batch_size,
            num_iterations=args.iterations,
            verbose=not args.quiet,
            max_pairs=args.max_pairs,
            parallel=args.parallel,
            max_workers=args.workers,
            llm_as_judge=args.llm_as_judge
        )

        # Print final results
        print("\n" + "=" * 60)
        print("📋 RESULTS SUMMARY")
        print("=" * 60)

        for i, (persona, response, expected) in enumerate(zip(personas_list, responses_list, outputs_list)):
            print(f"\n--- Pair {i+1}/{len(personas_list)} ---")
            print(f"📝 Persona (first 200 chars): {persona[:200]}...")
            print(f"🤖 Response: {response[:200]}...")
            print(f"✅ Expected: {expected[:200]}...")

        print("\n" + "=" * 60)
        print(f"Total pairs processed: {len(personas_list)}")
        print("=" * 60)

    else:
        # Batch mode: process multiple counters
        counter_end = args.counter_end if args.counter_end is not None else args.counter_start + 10

        log_progress(f"🚀 Batch mode: counters {args.counter_start} to {counter_end - 1}")

        all_results = run_batch(
            dataset_identifier=args.dataset,
            counter_start=args.counter_start,
            counter_end=counter_end,
            batch_size=args.batch_size,
            num_iterations=args.iterations,
            verbose=not args.quiet,
            max_pairs=args.max_pairs,
            parallel_pairs=args.parallel,
            max_workers=args.workers,
            parallel_counters=args.parallel_counters,
            llm_as_judge=args.llm_as_judge
        )

        print("\n" + "=" * 60)
        print(f"📋 BATCH RESULTS: {len(all_results)} counters processed successfully")
        print("=" * 60)
