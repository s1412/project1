import json
from openai import OpenAI
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords, wordnet
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


try:
    from IDS_TAP_parameters.py import DATA_CONFIG
    LAMP_PROFILE_THRESHOLD_SMALL = DATA_CONFIG.get("lamp_profile_threshold_small", 10)
    LAMP_PROFILE_THRESHOLD_LARGE = DATA_CONFIG.get("lamp_profile_threshold_large", 20)
    LAMP_HISTORY_CONTEXT_COUNT_SMALL = DATA_CONFIG.get("lamp_history_context_count_small", 5)
    LAMP_HISTORY_CONTEXT_COUNT_LARGE = DATA_CONFIG.get("lamp_history_context_count_large", 10)
    LAMP_RANKED_ENTRIES_OUTPUT_COUNT = DATA_CONFIG.get("lamp_ranked_entries_output_count", 5)
    LAMP_TOTAL_IO_COUNT = DATA_CONFIG.get("lamp_total_io_count", 10)
except ImportError:

    LAMP_PROFILE_THRESHOLD_SMALL = 10
    LAMP_PROFILE_THRESHOLD_LARGE = 20
    LAMP_HISTORY_CONTEXT_COUNT_SMALL = 5
    LAMP_HISTORY_CONTEXT_COUNT_LARGE = 10
    LAMP_RANKED_ENTRIES_OUTPUT_COUNT = 5
    LAMP_TOTAL_IO_COUNT = 10

client = OpenAI(
    api_key="YOUR_API_KEY_HERE",
    base_url="YOUR_OPENAI_BASE_URL"
)

def load_templated_data(input_address, output_address, LaMP_type, max_len, method, times, counter, max_history_items=20):

    input_item, output = _load_single_user_data(input_address, output_address, LaMP_type, counter)

    input_id, instruction, query, history_context = input_item

    # ========== 处理UltraChat/WildChat/PrefEval数据集 ==========
    if LaMP_type in [0, -1, -2]:
        if LaMP_type == 0:
            dataset_name = "ultrachat"
        elif LaMP_type == -1:
            dataset_name = "wildchat"
        else:
            dataset_name = "prefeval"

        print(f"📂 处理{dataset_name}数据集 (counter={counter})")
        print(f"   Profile轮数: {len(history_context)} (完整保留，不截断)")

        ranked_entries = history_context
        summary = Summarization_multiturn(instruction, history_context, LaMP_type, max_len)
        synthesis = Synthesis_multiturn(history_context, LaMP_type)

        single_instruction, allVersions_summary = domain_generation(
            instruction, summary, times, LaMP_type, input_id, counter, ranked_entries
        )

        templated_input = [input_id, query, single_instruction, allVersions_summary, synthesis, ranked_entries]
        return templated_input, output

    print(f"📂 处理LaMP-{LaMP_type}数据集 (counter={counter})")
    print(f"   历史上下文条数: {len(history_context)}")
    print(f"   Input/Output对数: {len(query) if isinstance(query, list) else 1}")

    summary = Summarization_lamp(instruction, history_context, LaMP_type, max_len)
    synthesis = Synthesis_lamp(history_context, LaMP_type)


    ranked_entries_list = []

    if isinstance(query, list) and len(query) > 0:
        print(f"   🔍 为每个query计算对应的ranked_entries...")
        for q_idx, q in enumerate(query):
            q_ranked = _rank_entries_by_relevance(
                str(q),
                history_context,
                top_k=min(LAMP_RANKED_ENTRIES_OUTPUT_COUNT, len(history_context))
            )
            ranked_entries_list.append(q_ranked)
        print(f"   📋 共生成 {len(ranked_entries_list)} 组ranked_entries（每组{len(ranked_entries_list[0]) if ranked_entries_list else 0}条）")
    else:

        reference_query = str(query) if query else ""
        single_ranked = _rank_entries_by_relevance(
            reference_query,
            history_context,
            top_k=min(LAMP_RANKED_ENTRIES_OUTPUT_COUNT, len(history_context))
        )
        ranked_entries_list.append(single_ranked)
        print(f"   📋 输出ranked_entries: {len(single_ranked)}条（从{len(history_context)}条中按相关度筛选）")


    first_ranked_entries = ranked_entries_list[0] if ranked_entries_list else []

    single_instruction, allVersions_summary = domain_generation(
        instruction, summary, times, LaMP_type, input_id, counter, first_ranked_entries
    )


    templated_input = [input_id, query, single_instruction, allVersions_summary, synthesis, ranked_entries_list]

    return templated_input, output


def Summarization_multiturn(instruction, history_context, LaMP_type, max_len=500):

    dataset_name = "ultrachat" if LaMP_type == 0 else "wildchat"

    # 🔧 [修复] 限制输入长度以避免超过API的163840 token限制
    # 预估：4字符≈1 token，预留50000 tokens给prompt模板和system_prompt
    # 因此history_text最多使用约 110000 tokens ≈ 440000 字符
    MAX_HISTORY_CHARS = 400000  # 安全限制

    history_text = "\n".join([f"Turn {i+1}: {turn}" for i, turn in enumerate(history_context)])

    if len(history_text) > MAX_HISTORY_CHARS:
        print(f"⚠️  history_text过长 ({len(history_text)} chars)，截断到 {MAX_HISTORY_CHARS} chars")
        # 截断策略：保留开头和结尾部分，因为开头有用户初始意图，结尾有最新上下文
        head_chars = MAX_HISTORY_CHARS * 2 // 3  # 前2/3
        tail_chars = MAX_HISTORY_CHARS // 3      # 后1/3
        history_text = history_text[:head_chars] + "\n\n...[eliminated]  ...\n\n" + history_text[-tail_chars:]


    prompt = """
    You are an advanced profile analysis and summarization model. Based on the conversation history provided below, generate an **informative and coherent user persona summary** that reflects the user's characteristics, style, and behavior patterns.

    ### Task Instruction:
    {}

    ### Conversation History:
    {}

    ### Instructions:
    1. **CORE ANCHORING** (Most Important):
       - Identify 2-3 SPECIFIC, CONCRETE details from the conversation (e.g., specific topics discussed, specific tools/methods mentioned, specific interests revealed)
       - These concrete details MUST appear in your summary - they are the "anchors" that define this user
       - Example anchors: specific hobbies, specific professional domains, specific communication quirks

    2. **BEHAVIORAL PATTERNS**:
       - Focus on the user's **personality traits, interests, tone, communication style, preferences, goals**
       - Describe what they care about
       - Describe ho2 they ask question

    3. **SPECIFICITY OVER ABSTRACTION**:
       - Use CONCRETE language, avoid vague terms
       - Prefer "interested in Python debugging and API design" over "interested in programming"
       - Prefer "uses informal language with frequent abbreviations" over "casual communication style"

    Please provide the final user persona summary below (one paragraph, 150-300 words):
    """.format(instruction, history_text)

    system_prompt = """
    You are an advanced profile analysis and summarization model specialized in multi-turn dialogue analysis.
    Your goal is to generate a coherent, SPECIFIC, and psychologically insightful user persona summary.

    Critical Guidelines:
    - **ANCHOR ON SPECIFICS**: Always include 2-3 concrete, verifiable details from the conversation as "anchors"
    - **NO ROLE DRIFT**: Describe who the user IS based on evidence, not who they COULD become
    - **CONCRETE > ABSTRACT**: Use specific examples rather than general categories
    - Focus on: personality traits, interests, tone, communication style, preferences, goals, behavior patterns
    - Be **concise but information-rich** — aim for a natural paragraph (150-300 words)
    - Avoid generic statements; describe **HOW** and **WHY** they behave or communicate that way
    - Never include labels like "Summary:" or formatting headers
    - Maintain a neutral, descriptive tone — avoid praise or flattery
    - Use third-person perspective (e.g., "They prefer...", "This user...")
    """

    import time
    import random
    max_retries = 5
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-v3.2",
                messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}],
                temperature=0.4,
                timeout=30.0
            )

            if not response or not response.choices or len(response.choices) == 0:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️  Summarization API响应无效，等待 {delay:.1f} 秒后重试 (第 {attempt + 1}/{max_retries} 次)")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("Summarization ")

            summary = response.choices[0].message.content

            if summary is None or not summary.strip():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️   {attempt + 1}/{max_retries} ")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("API error")

            return summary.strip()

        except Exception as e:
            if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):

                combined_text = f"Instruction: {instruction}\n\nConversation: {history_text[:1000]}"
                max_summary_length = min(max_len * 2, 500)
                if len(combined_text) > max_summary_length:
                    return combined_text[:max_summary_length] + "..."
                return combined_text

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"⚠️   {attempt + 1}/{max_retries} ")
                time.sleep(delay)
            else:
                raise Exception(f"Summarization API {e}")

    raise Exception(f"Summarization API")


def Synthesis_multiturn(history_context, LaMP_type):

    dataset_name = "ultrachat" if LaMP_type == 0 else "wildchat"

    history_text = "\n".join([f"Turn {i+1}: {turn}" for i, turn in enumerate(history_context)])

    MAX_HISTORY_CHARS = 400000
    if len(history_text) > MAX_HISTORY_CHARS:
        print(f"⚠️  {MAX_HISTORY_CHARS} chars")
        head_chars = MAX_HISTORY_CHARS * 2 // 3
        tail_chars = MAX_HISTORY_CHARS // 3
        history_text = history_text[:head_chars] + "\n\n... [eliminated] ...\n\n" + history_text[-tail_chars:]

    prompt = """
    Based on the conversation history below, extract the CORE ANCHORING keywords that uniquely define this user's identity, interests, and behavior patterns.

    ### Conversation History:
    {}

    ### Extraction Guidelines:

    1. **PRIMARY ANCHORS (3-5 keywords)** - Most Important:
       - Specific topics the user discussed (e.g., "Python debugging", "Japanese cuisine", "project management")
       - Specific tools/methods/brands mentioned (e.g., "VS Code", "Scrum methodology", "Toyota")
       - Unique behavioral traits (e.g., "detailed questions", "humor", "formal tone")

    2. **SECONDARY KEYWORDS (3-5 keywords)**:
       - General interest areas (e.g., "technology", "cooking", "management")
       - Communication style indicators (e.g., "analytical", "collaborative", "direct")
       - Professional/personal context (e.g., "software developer", "team leader", "hobbyist")

    3. **SPECIFICITY RULES**:
       - Prefer SPECIFIC terms over GENERIC ones
       - Include at least 2 PROPER NOUNS or SPECIFIC TERMS if present in the conversation
       - Avoid vague terms like "various topics", "general interests"

    Output format: Comma-separated list of 6-10 keywords, ordered by importance (primary anchors first)

    Keywords:
    """.format(history_text)

    system_prompt = """
    You are a specialized keyword extraction model for user profiling.
    Extract keywords that serve as "anchors" for this user's identity - specific, concrete terms that would help distinguish this user from others.
    Return ONLY the comma-separated keywords. No explanations, no numbering, no other text.
    Prioritize: specific topics > tools/methods > behavioral traits > general categories
    """

    import time
    import random
    max_retries = 5
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-v3.2",
                messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=30.0
            )

            if not response or not response.choices or len(response.choices) == 0:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️  Synthesis API响应无效，等待 {delay:.1f} 秒后重试 (第 {attempt + 1}/{max_retries} 次)")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("Synthesis API返回无效响应结构")

            synthesis = response.choices[0].message.content

            if synthesis is None or not synthesis.strip():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️  Synthesis API返回空内容，等待 {delay:.1f} 秒后重试 (第 {attempt + 1}/{max_retries} 次)")
                    time.sleep(delay)
                    continue
                else:
                    return ""  

            return synthesis.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"⚠️  Synthesis API异常 ({e})，等待 {delay:.1f} 秒后重试 (第 {attempt + 1}/{max_retries} 次)")
                time.sleep(delay)
            else:
                print(f"⚠️  Synthesis API调用失败，返回空关键词")
                return ""

    return ""


def Summarization_lamp(instruction, history_context, LaMP_type, max_len=500):

    history_text = "\n".join([f"Entry {i+1}: {entry}" for i, entry in enumerate(history_context)])

    prompt = """
    You are an advanced profile analysis and summarization model. Based on the user's historical data provided below, generate an **informative and coherent user persona summary** that reflects the user's characteristics, style, and behavior patterns.

    ### Task Instruction:
    {}

    ### User History Data:
    {}

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

    Please provide the final user persona summary below (one paragraph, 150-300 words):
    """.format(instruction, history_text)

    system_prompt = """
    You are an advanced profile analysis and summarization model.
    Your goal is to generate a coherent, SPECIFIC, and psychologically insightful user persona summary.

    Critical Guidelines:
    - **ANCHOR ON SPECIFICS**: Always include 2-3 concrete, verifiable details from the history as "anchors"
    - Describe who the user IS based on evidence, not who they COULD become
    - **CONCRETE > ABSTRACT**: Use specific examples rather than general categories
    - Focus on: personality traits, interests, tone, communication style, preferences, goals, behavior patterns
    - Be **concise but information-rich** — aim for a natural paragraph (150-300 words)
    - Avoid generic statements; describe **HOW** and **WHY** they behave or communicate that way
    - Maintain a neutral, descriptive tone — avoid praise or flattery
    - Use third-person perspective 
    """

    import time
    import random
    max_retries = 5
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-v3.2",
                messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}],
                temperature=0.4,
                timeout=30.0
            )

            if not response or not response.choices or len(response.choices) == 0:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️ ")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("Summarization API")

            summary = response.choices[0].message.content

            if summary is None or not summary.strip():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️ ")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("Summarization API")
            print("prompt = ", prompt)
            print("summary = ", summary.strip())
            return summary.strip()

        except Exception as e:
            if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):
                combined_text = f"Instruction: {instruction}\n\nHistory: {history_text[:1000]}"
                max_summary_length = min(max_len * 2, 500)
                if len(combined_text) > max_summary_length:
                    return combined_text[:max_summary_length] + "..."
                return combined_text

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"⚠️  ")
                time.sleep(delay)
            else:
                raise Exception(f"Summarization API: {e}")

    raise Exception(f"Summarization API调用失败")


def Synthesis_lamp(history_context, LaMP_type):

    history_text = "\n".join([f"Entry {i+1}: {entry}" for i, entry in enumerate(history_context)])

    prompt = """
    Based on the user's historical data below, extract the CORE ANCHORING keywords that uniquely define this user's identity, interests, and behavior patterns.

    ### User History Data:
    {}

    ### Extraction Guidelines:

    1. **PRIMARY ANCHORS (3-5 keywords)** - Most Important:
       - Specific topics the user discussed (e.g., "machine learning", "travel photography", "economics")
       - Specific tools/methods/brands mentioned
       - Unique behavioral traits (e.g., "detailed analysis", "humor", "formal tone")

    2. **SECONDARY KEYWORDS (3-5 keywords)**:
       - General interest areas
       - Communication style indicators
       - Professional/personal context

    3. **SPECIFICITY RULES**:
       - Prefer SPECIFIC terms over GENERIC ones
       - Include at least 2 PROPER NOUNS or SPECIFIC TERMS if present
       - Avoid vague terms like "various topics", "general interests"

    Output format: Comma-separated list of 10-20 keywords, ordered by importance (primary anchors first)

    Keywords:
    """.format(history_text)

    system_prompt = """
    You are a specialized keyword extraction model for user profiling.
    Extract keywords that serve as "anchors" for this user's identity - specific, concrete terms that would help distinguish this user from others.
    Return ONLY the comma-separated keywords. No explanations, no numbering, no other text.
    Prioritize: specific topics > tools/methods > behavioral traits > general categories
    """

    import time
    import random
    max_retries = 5
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-v3.2",
                messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=30.0
            )

            if not response or not response.choices or len(response.choices) == 0:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️  ")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("Synthesis API")

            synthesis = response.choices[0].message.content

            if synthesis is None or not synthesis.strip():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"")
                    time.sleep(delay)
                    continue
                else:
                    return ""

            return synthesis.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"⚠️  Synthesis API({e})")
                time.sleep(delay)
            else:
                print(f"⚠️  Synthesis ")
                return ""

    return ""


def _load_single_user_data(input_address, output_address, LaMP_type, counter):

    # ========== 处理UltraChat/WildChat/PrefEval数据集 ==========
    if LaMP_type in [0, -1, -2]:
        return _load_single_multiturn_data(input_address, LaMP_type, counter)

    # ========== 处理LongLAMP数据（LaMP_type 8, 9, 10）==========
    elif LaMP_type in [8, 9, 10]:
        return _load_single_longlamp_data(input_address, LaMP_type, counter)

    # ========== 处理原始LaMP数据（LaMP_type 4, 5）==========
    else:
        return _load_single_lamp_data(input_address, output_address, LaMP_type, counter)


def _load_single_multiturn_data(input_address, LaMP_type, counter):

    import ijson


    if LaMP_type == 0:
        dataset_name = "ultrachat"
    elif LaMP_type == -1:
        dataset_name = "wildchat"
    else:
        dataset_name = "prefeval"

    from IDS_TAP_parameters.py import NEW_DATASET_CONFIG
    dataset_config = NEW_DATASET_CONFIG.get(dataset_name, {})
    instruction = dataset_config.get('instruction', 'Predict the user\'s next query')


    current_index = 0
    target_data = None

    with open(input_address, 'rb') as f:

        parser = ijson.items(f, 'item')
        for data in parser:
            if current_index == counter:
                target_data = data
                break
            current_index += 1

    if target_data is None:
        raise IndexError(f"Counter {counter} 超出数据集范围（共 {current_index} 条）")

    data = target_data
    input_id = data.get("id", "unknown")


    if LaMP_type == -2:
        conversation = data.get("conversation", {})
        history_context = []

        sorted_keys = sorted(conversation.keys(), key=lambda x: int(x))
        for key in sorted_keys:
            turn = conversation[key]
            user_msg = turn.get("user", "")
            assistant_msg = turn.get("assistant", "")
            turn_text = f"User: {user_msg}\nAssistant: {assistant_msg}"
            history_context.append(turn_text)

        query = data.get("question", "")
        preference = data.get("preference", "")
        explanation = data.get("explanation", "")
        persona = data.get("persona", "")

        output_text = _get_prefeval_response_with_cache(
            input_id=input_id,
            input_text=query,
            history_context=history_context,
            preference=preference,
            explanation=explanation,
            persona=persona
        )

        input_item = (input_id, instruction, query, history_context)
        output_item = (input_id, output_text)
        return input_item, output_item

    else:
        profile = data.get("profile", [])
        history_context = []

        for turn in profile:
            if isinstance(turn, dict):
                user_msg = turn.get("user", "")
                response_msg = turn.get("response", "")
                turn_text = f"User: {user_msg}\nAssistant: {response_msg}"
                history_context.append(turn_text)
            elif isinstance(turn, str):
                history_context.append(f"User: {turn}")

        output_text = data.get("output", "")
        query = ""

        input_item = (input_id, instruction, query, history_context)
        output_item = (input_id, output_text)
        return input_item, output_item


def _load_single_longlamp_data(input_address, LaMP_type, counter):
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

    if target_data is None:
        raise IndexError(f"Counter {counter} 超出数据集范围（共 {current_index} 条）")

    data = target_data

    if LaMP_type == 8:
        input_id = data.get("name", "unknown")
        instruction = "Generate an abstract for the given title using the provided items."
    elif LaMP_type == 9:
        input_id = data.get("reviewerId", "unknown")
        instruction = "Generate a product review based on the given rating and product description."
    elif LaMP_type == 10:
        input_id = data.get("author", "unknown")
        instruction = "Generate content for a reddit post based on the given topic."

    current_input = data["input"]
    current_output = data.get("output", "")

    profile_pairs = []  
    if "profile" in data and data["profile"]:
        for profile in data["profile"]:
            profile_text = _extract_profile_text(profile, LaMP_type)
            if profile_text:
                profile_pairs.append((profile_text, profile))


    all_profiles = [pair[0] for pair in profile_pairs]

    history_context, additional_profiles = _split_profile_for_lamp(all_profiles)

    history_count = len(history_context)
    additional_original_profiles = [pair[1] for pair in profile_pairs[history_count:history_count + len(additional_profiles)]]

    queries_list = [current_input]
    outputs_list = [current_output]

    for profile_item in additional_original_profiles:
        if LaMP_type == 8:
            title = profile_item.get("title", "")
            abstract = profile_item.get("abstract", "")
            query_str = title
            output_str = abstract
        elif LaMP_type == 9:
            description = profile_item.get("description", "")
            rating = profile_item.get("overall", "")
            review_text = profile_item.get("reviewText", "")
            query_str = f"Description: {description}, rating: {rating}"
            output_str = review_text
        elif LaMP_type == 10:
            summary = profile_item.get("summary", "")
            content = profile_item.get("content", "")
            query_str = summary
            output_str = content
        else:
            query_str = _extract_profile_text(profile_item, LaMP_type)
            output_str = query_str

        queries_list.append(query_str)
        outputs_list.append(output_str)

    input_item = (input_id, instruction, queries_list, history_context)
    output_item = (input_id, outputs_list)
    return input_item, output_item


def _load_single_lamp_data(input_address, output_address, LaMP_type, counter):

    import ijson

    # 流式解析输入数据
    current_index = 0
    target_data = None

    with open(input_address, 'rb') as f:
        parser = ijson.items(f, 'item')
        for data in parser:
            if current_index == counter:
                target_data = data
                break
            current_index += 1

    if target_data is None:
        raise IndexError(f"Counter {counter} ")

    data = target_data
    input_id = data["id"]
    current_input = data["input"]

    # 加载对应的输出数据
    current_output = ""
    if output_address:
        with open(output_address, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        keys = list(output_data.keys())
        output_list = output_data[keys[1]]
        for output_item_data in output_list:
            if output_item_data["id"] == input_id:
                current_output = output_item_data["output"]
                break

    # 解析instruction和query
    parts = current_input.split(":", 1)
    if len(parts) == 2:
        instruction = parts[0] + ":"
        query = parts[1]
    else:
        instruction = ""
        query = current_input


    profile_pairs = [] 
    for profile in data["profile"]:
        profile_text = _extract_profile_text(profile, LaMP_type)
        if profile_text:
            profile_pairs.append((profile_text, profile))


    all_profiles = [pair[0] for pair in profile_pairs]


    history_context, additional_profiles = _split_profile_for_lamp(all_profiles)


    history_count = len(history_context)
    additional_original_profiles = [pair[1] for pair in profile_pairs[history_count:history_count + len(additional_profiles)]]


    queries_list = [query]
    outputs_list = [current_output]


    for profile_item in additional_original_profiles:
        if LaMP_type == 4:

            text = profile_item.get("text", "")
            title = profile_item.get("title", "")
            query_str = text
            output_str = title
        elif LaMP_type == 5:

            abstract = profile_item.get("abstract", "")
            title = profile_item.get("title", "")
            query_str = abstract
            output_str = title
        else:

            query_str = _extract_profile_text(profile_item, LaMP_type)
            output_str = query_str

        queries_list.append(query_str)
        outputs_list.append(output_str)

    input_item = (input_id, instruction, queries_list, history_context)
    output_item = (input_id, outputs_list)
    return input_item, output_item


def _extract_profile_text(profile, LaMP_type):
    """
    从profile中提取文本

    Args:
        profile: profile字典
        LaMP_type: 数据集类型

    Returns:
        str: 提取的文本
    """
    parts = []

    if LaMP_type == 4:

        if "title" in profile:
            parts.append(f"Title: {profile['title']}")
        if "date" in profile:
            parts.append(f"Date: {profile['date']}")
        if "text" in profile:
            parts.append(f"Article: {profile['text']}")
    elif LaMP_type == 5:

        if "title" in profile:
            parts.append(f"Title: {profile['title']}")
        if "date" in profile:
            parts.append(f"Date: {profile['date']}")
        if "abstract" in profile:
            parts.append(f"Abstract: {profile['abstract']}")
    elif LaMP_type == 8:

        if "title" in profile:
            parts.append(f"Title: {profile['title']}")
        if "year" in profile:
            parts.append(f"Year: {profile['year']}")
        if "abstract" in profile:
            parts.append(f"Abstract: {profile['abstract']}")
    elif LaMP_type == 9:

        if "description" in profile:
            parts.append(f"Product: {profile['description']}")
        if "overall" in profile:
            parts.append(f"Rating: {profile['overall']}")
        if "summary" in profile:
            parts.append(f"Summary: {profile['summary']}")
        if "reviewText" in profile:
            parts.append(f"Review: {profile['reviewText']}")
    elif LaMP_type == 10:

        if "summary" in profile:
            parts.append(f"Topic: {profile['summary']}")
        if "content" in profile:
            parts.append(f"Content: {profile['content']}")

    if parts:
        return "\n".join(parts)
    elif "text" in profile:
        return profile["text"]
    return ""


def _split_profile_for_lamp(all_profiles):

    profile_count = len(all_profiles)

    if profile_count < LAMP_PROFILE_THRESHOLD_SMALL:

        history_count = LAMP_HISTORY_CONTEXT_COUNT_SMALL
        history_context = all_profiles[:history_count]
        additional_profiles = all_profiles[history_count:]
    elif profile_count < LAMP_PROFILE_THRESHOLD_LARGE:

        history_count = LAMP_HISTORY_CONTEXT_COUNT_LARGE
        history_context = all_profiles[:history_count]
        additional_profiles = all_profiles[history_count:]
    else:

        history_count = LAMP_HISTORY_CONTEXT_COUNT_LARGE
        max_additional = LAMP_TOTAL_IO_COUNT - 1  
        additional_profiles = all_profiles[history_count:history_count + max_additional]

    return history_context, additional_profiles


def _rank_entries_by_relevance(query, entries, top_k=None, use_rag_api=True, api_url = ""):

    if not entries:
        return []

    if top_k is None:
        top_k = len(entries)

    if use_rag_api:
        try:
            return _rank_entries_by_rag_api(query, entries, top_k, api_url)
        except Exception as e:
            print(f"⚠️ ")
            # 回退到本地模型


    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

        query_embedding = model.encode(str(query))


        entry_texts = []
        for entry in entries:
            if isinstance(entry, dict):

                entry_text = entry.get('text', str(entry))
            else:
                entry_text = str(entry)
            entry_texts.append(entry_text)  

        entry_embeddings = model.encode(entry_texts)

        similarities = [1 - cosine(query_embedding, emb) for emb in entry_embeddings]
        ranked_indices = np.argsort(similarities)[::-1]

        ranked_entries = [entries[i] for i in ranked_indices[:top_k]]
        return ranked_entries
    except Exception as e:
        print(f"⚠️ {e}")
        return entries[:top_k]


def _rank_entries_by_rag_api(query, entries, top_k, api_url, max_retries=3, retry_delay=2.0):

    import requests
    import time


    def entry_to_text(entry):
        if isinstance(entry, dict):
            return entry.get('text', str(entry))
        return str(entry)


    all_texts = [str(query)] + [entry_to_text(entry) for entry in entries]


    payload = {
        "model": "/workspace/users/zhiwei/qwen3",
        "input": all_texts
    }

    last_exception = None
    current_delay = retry_delay

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, json=payload, timeout=60)

            if response.status_code != 200:
                raise Exception(f"Embedding API返回状态码: {response.status_code}")

            result = response.json()
            if "data" not in result:
                raise Exception("Embedding API响应格式错误: 缺少'data'字段")


            embeddings = [np.array(item["embedding"]) for item in result["data"]]
            query_embedding = embeddings[0]
            entry_embeddings = embeddings[1:]


            def cosine_similarity(a, b):
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return np.dot(a, b) / (norm_a * norm_b)

            similarities = [cosine_similarity(query_embedding, emb) for emb in entry_embeddings]
            ranked_indices = np.argsort(similarities)[::-1]

            ranked_entries = [entries[i] for i in ranked_indices[:top_k]]



            return ranked_entries

        except requests.exceptions.Timeout as e:
            last_exception = e
            print(f"   ⚠️{e}")
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            print(f"   ⚠️{e}")
        except requests.exceptions.RequestException as e:
            last_exception = e
            print(f"   ⚠️  {e}")
        except Exception as e:
            last_exception = e
            print(f"   ⚠️ {e}")

        # 如果不是最后一次尝试，则等待后重试
        if attempt < max_retries - 1:
            time.sleep(current_delay)
            current_delay *= 2  # 指数退避

    # 所有重试都失败
    raise Exception(f" {last_exception}")


# PrefEval response 缓存目录
PREFEVAL_OUTPUT_DIR = "./prefeval_output"


def _get_prefeval_response_with_cache(input_id, input_text, history_context, preference, explanation, persona):

    import os


    os.makedirs(PREFEVAL_OUTPUT_DIR, exist_ok=True)


    cache_file = os.path.join(PREFEVAL_OUTPUT_DIR, f"response_{input_id}.json")


    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                response = cached_data.get("response", "")
                if response:
                    return response
        except Exception as e:
            print(f"   ⚠️ {e}")


    response = _generate_prefeval_response(input_text, history_context, preference, explanation, persona)

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
    except Exception as e:
        print(f"   ⚠️ {e}")

    return response


def _generate_prefeval_response(input_text, history_context, preference, explanation, persona):

    import time
    import random
    conversation_history = ""
    for i, turn in enumerate(history_context):
        conversation_history += f"Turn {i+1}:\n{turn}\n\n"

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

    system_prompt = "You are a helpful assistant that generates personalized responses based on user preferences and conversation history. Your responses should be natural, helpful, and aligned with the user's stated preferences."

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
                return f"Based on your preferences, here is my response to your question: {input_text}"

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"⚠️ PrefEval response API error ({e}), retrying in {delay:.1f}s ({attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"⚠️ PrefEval response generation failed after {max_retries} retries")
                return f"Based on your preferences, here is my response to your question: {input_text}"

    return f"Based on your preferences, here is my response to your question: {input_text}"




def Summarization_dependent(instruction, query, retrieved_entries, max_len):
    """using chatgpt as the generation model to generate summarization of retrieved entries
        different from that in FTPERSLLM in the way that do not train model while get immediate
        context involved"""

    if isinstance(query, str):
        query = [query]


    prompt = """
    You are an advanced profile analysis and summarization model. Based on the information provided below, generate an **informative and coherent user persona summary** that reflects the user's characteristics, style, and behavior patterns.
    ### Task Instruction:
    {}

    ### Current Context:
    {}
    
    ### The context from the user's history work:
    {}

    ### Instructions:- Focus on the user's **personality traits, interests, tone, communication style, preferences, goals, behavior patterns**.
    Please provide the final user persona summary below:
    """.format(instruction, query, retrieved_entries)


    import time
    import random
    max_retries = 5
    
    base_delay = 2.0
    system_prompt = """
    You are an advanced profile analysis and summarization model.
    Your goal is to generate a coherent, specific, and psychologically insightful user persona summary.

    Guidelines:
    - Focus on the user's **personality traits, interests, tone, communication style, preferences, goals, behavior patterns**.
    - Be **concise but information-rich** — aim for a natural paragraph that feels like a short professional psychological or stylistic profile.
    - Avoid generic statements; instead, describe **how** and **why** they behave or communicate that way.
    - Never include labels like “Summary:” or formatting headers.
    - Maintain a neutral, descriptive tone — avoid praise or flattery.
    "",
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-v3.2",
                messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}],
                temperature=0.4,
                timeout=30.0  
            )


            if not response or not response.choices or len(response.choices) == 0:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️  ")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("Summarization API")

            summary = response.choices[0].message.content

            if summary is None:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️  ")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("None")

            summary = summary.strip()
            if not summary:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("Summarization API")
            print("summary = ", summary)
            return summary

        except Exception as e:

            if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):

                query_str = ' '.join(query) if isinstance(query, list) else str(query)
                entries_text = ' '.join([str(entry) for entry in retrieved_entries[:3]])
                combined_text = f"Query: {query_str}\n\nContext: {entries_text}"


                max_summary_length = min(max_len * 2, 500)
                if len(combined_text) > max_summary_length:
                    summary = combined_text[:max_summary_length] + "..."
                else:
                    summary = combined_text

                return summary

            elif any(error_type in str(e).lower() for error_type in [
                "connection", "timeout", "network", "unreachable",
                "refused", "reset", "broken pipe", "ssl", "certificate",
                "dns", "resolve", "connect", "socket"
            ]):
                if attempt < max_retries - 1:
                    delay = base_delay * (3 ** attempt) + random.uniform(1, 5)
                    print(f"⚠️  ")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"")


            else:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️  ")
                    time.sleep(delay)
                    continue
                else:
                    raise e

    raise Exception(f"Summarization API")


def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            embeddings[word] = vector
    return embeddings

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def are_words_similar(word1, word2, glove_embeddings):
    # case 1
    if word1 == word2:
        return True
    # case 2
    synonyms_word1 = set(syn.lemma_names() for syn in wordnet.synsets(word1))
    synonyms_word2 = set(syn.lemma_names() for syn in wordnet.synsets(word2))
    if len(synonyms_word1 & synonyms_word2) > 0:
        return True
    # case 3
    if word1 in glove_embeddings and word2 in glove_embeddings:
        vec1 = glove_embeddings[word1]
        vec2 = glove_embeddings[word2]
        if euclidean_distance(vec1, vec2) < 4:
            return True
    return False


def get_filtered_words(texts, stop_words, min_idf=1.5):
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(texts)
    idf_values = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    filtered_words = sorted(
        [word for word, idf in idf_values.items() if word not in stop_words and idf >= min_idf],
        key=lambda word: -idf_values[word]
    )
    return filtered_words  

def Synthesis_dependent(query, history_context):

    stop_words = set(stopwords.words("english"))


    if isinstance(query, str):
        query_list = [query]
    elif isinstance(query, list):
        query_list = [str(q) for q in query] 
    else:
        query_list = [str(query)]


    if isinstance(history_context, str):
        history_list = [history_context]
    elif isinstance(history_context, list):
        history_list = [str(h) for h in history_context] 
    else:
        history_list = [str(history_context)]

    query_filtered = get_filtered_words(query_list, stop_words)
    history_filtered = get_filtered_words(history_list, stop_words)

    glove_embeddings = load_glove_embeddings('./glove.42B.300d.txt')  

    candidate_words = set()

    for word1 in query_filtered:
        for word2 in history_filtered:
            if are_words_similar(word1, word2, glove_embeddings):
                candidate_words.add(word2)  

    listed_candidate_words = list(candidate_words)

    """fullfill the interaction bettween chatgpt"""
    prompt = f"""
    You are an advanced text analysis model. Please extract the 20 most important and relevant keywords from the following lists:

    ### Key words in Current Context:
    {query_filtered}

    ### Key words in Retrieved History Context:
    {history_filtered}

    ### Relevant key words from Retrieved History Context:
    {listed_candidate_words}

    ### Instructions:
    - Identify the top 10 most important and relevant keywords from the lists above.
    - Focus on the keywords that are most critical for the topic and overall understanding.
    - Provide the 10 most important keywords in a list format only.
    "",

    Please list the top 10 keywords below:
    """


    import time
    import random
    max_retries = 5
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
            model="deepseek/deepseek-v3.2",   
            messages=[{"role": "system", "content": "You are an expert in text summarization."},
                        {"role": "user", "content": prompt}],
            temperature=0.4,
            timeout=30.0  
            )


            if not response or not response.choices or len(response.choices) == 0:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️  ")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("Synthesis API返回无效响应结构")

            synthesis = response.choices[0].message.content
            if synthesis is None:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️  ")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("Synthesis API")

            synthesis = synthesis.strip()
            if not synthesis:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️ ")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("Synthesis API")

            return synthesis

        except Exception as e:

            if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):

                query_str = str(query)
                history_str = ' '.join([str(entry) for entry in history_context[:5]])


                import re
                from collections import Counter


                combined_text = f"{query_str} {history_str}".lower()
                words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)


                stop_words_set = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
                filtered_words = [word for word in words if word not in stop_words_set and len(word) > 3]


                word_counts = Counter(filtered_words)
                top_keywords = [word for word, _ in word_counts.most_common(20)]

                synthesis = ', '.join(top_keywords)
                return synthesis


            elif any(error_type in str(e).lower() for error_type in [
                "connection", "timeout", "network", "unreachable",
                "refused", "reset", "broken pipe", "ssl", "certificate",
                "dns", "resolve", "connect", "socket"
            ]):
                if attempt < max_retries - 1:
                    delay = base_delay * (3 ** attempt) + random.uniform(1, 5)
                    print(f"⚠️  Synthesis")
                    time.sleep(delay)
                    continue
                else:

                    query_str = str(query)
                    history_str = ' '.join([str(entry) for entry in history_context[:5]])
                    import re
                    from collections import Counter
                    combined_text = f"{query_str} {history_str}".lower()
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
                    stop_words_set = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
                    filtered_words = [word for word in words if word not in stop_words_set and len(word) > 3]
                    word_counts = Counter(filtered_words)
                    top_keywords = [word for word, _ in word_counts.most_common(20)]
                    synthesis = ', '.join(top_keywords)
                    return synthesis

            else:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️ ...")
                    time.sleep(delay)
                    continue
                else:

                    raise e


    raise Exception(f"Synthesis API调")


def get_rephrase(text, max_length=None, lamp_type=None):

    import time
    import random


    text = text.strip()


    if not text:
        return "No summary available for this user."


    if max_length:
        length_constraint = (
            f"- **CRITICAL LENGTH LIMIT**: Your output MUST be approximately {max_length} characters or less. "
            f"This is a STRICT requirement. Do NOT exceed {max_length} characters under any circumstances.\n"
        )
    else:
        length_constraint = "- The output should be close to that of the original text (no longer than the original one).\n"

    max_retries = 5  
    base_delay = 2.0  


    is_multiturn_dialogue = lamp_type in [0, -1]

    for attempt in range(max_retries):
        try:
            if is_multiturn_dialogue:

                system_content = (
                    "You are an advanced language model specialized in *balanced rephrasing* of personal profiles. "
                    "Your task is to rewrite a user persona summary with a SLIGHTLY DIFFERENT perspective or emphasis. "
                    "CRITICAL BALANCE: Explore different angles while PRESERVING the user's core identity and specific details. "
                    f"IMPORTANT: Keep the output concise, approximately {max_length} characters or less.\n\n"
                    "Key principles:\n"
                    "1. MUST preserve 2-3 specific concrete details (names, topics, specific experiences)\n"
                    "2. May emphasize different personality aspects, but stay within reasonable variation\n"
                    "3. Do NOT create a completely different person - this is the SAME user seen from a different angle\n"
                )

                user_content = (
                    "Rewrite the following user persona summary with a SLIGHTLY DIFFERENT perspective:\n\n"
                    f"{text}\n\n"
                    "### STRICT Constraints:\n"
                    "- **PRESERVE CORE ANCHORS** (MANDATORY): Keep 2-3 specific concrete details from the original (e.g., specific topics, tools, experiences)\n"
                    "- **BOUNDED VARIATION**: Change emphasis or perspective, but the person should still be recognizable as the same user\n"
                    "- **NO EXTREME SHIFTS**: Do NOT flip from 'analytical' to 'emotional' or vice versa in adjacent versions\n"
                    "- Focus on: personality traits, interests, communication style, preferences, goals, behavior patterns\n"
                    "- Use a third-person perspective\n"
                    f"{length_constraint}"
                    "- Do not make up personal information \n"
                    "- Output only the rewritten description, no other text:"
                )


                temperature = 0.3
                presence_penalty = 0.25
                frequency_penalty = 0.25

            else:

                system_content = (
                    "You are an advanced language model specialized in *balanced rephrasing* of personal profiles. "
                    "Your task is to rewrite a user persona summary with a SLIGHTLY DIFFERENT perspective or emphasis. "
                    "CRITICAL BALANCE: Explore different angles while PRESERVING the user's core identity and specific details. "
                    f"IMPORTANT: Keep the output concise, approximately {max_length} characters or less.\n\n"
                    "Key principles:\n"
                    "1. MUST preserve 2-3 specific concrete details (names, topics, specific experiences)\n"
                    "2. May emphasize different personality aspects, but stay within reasonable variation\n"
                    "3. Do NOT create a completely different person - this is the SAME user seen from a different angle\n"
                )

                user_content = (
                    "Rewrite the following user persona summary with a SLIGHTLY DIFFERENT perspective:\n\n"
                    f"{text}\n\n"
                    "### STRICT Constraints:\n"
                    "- **PRESERVE CORE ANCHORS** (MANDATORY): Keep 2-3 specific concrete details from the original (e.g., specific topics, tools, experiences)\n"
                    "- **BOUNDED VARIATION**: Change emphasis or perspective, but the person should still be recognizable as the same user\n"
                    "- **NO EXTREME SHIFTS**: Do NOT flip from 'analytical' to 'emotional' or vice versa in adjacent versions\n"
                    "- Focus on: personality traits, interests, communication style, preferences, goals, behavior patterns\n"
                    "- Use a third-person perspective\n"
                    f"{length_constraint}"
                    "- Do not make up personal information\n"
                    "- Output only the rewritten description, no other text:"
                )


                temperature = 0.3
                presence_penalty = 0.25
                frequency_penalty = 0.25


            response = client.chat.completions.create(
                model="deepseek/deepseek-v3.2",  
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                top_p = 1.0,
                timeout=30.0,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )


            if response is None:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"Rephrase API")


            if not hasattr(response, 'choices') or not response.choices:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"Rephrase API")


            if not hasattr(response.choices[0], 'message') or not response.choices[0].message:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"Rephrase API")

            content = response.choices[0].message.content
            if content is None or content.strip() == "":
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"Rephrase API")

            result = content.strip()


            if max_length and len(result) > max_length:
                print(f"⚠️ [get_rephrase] ")
                result = result[:max_length - 3] + "..."

            return result

        except Exception as e:
            if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):
                return text  

            elif "rate_limit" in str(e).lower() or "429" in str(e):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)  
                    print(f"⚠️  ")
                    time.sleep(delay)
                    continue
                else:
                    return text


            elif any(error_type in str(e).lower() for error_type in [
                "connection", "timeout", "network", "unreachable",
                "refused", "reset", "broken pipe", "ssl", "certificate",
                "dns", "resolve", "connect", "socket"
            ]):
                if attempt < max_retries - 1:

                    delay = base_delay * (3 ** attempt) + random.uniform(1, 5)  
                    print(f"⚠️  ")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"Rephrase API")


            else:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️  ")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"Rephrase API")

    raise Exception(f"Rephrase API")

def get_rephrase_with_context(history_context, original_summary, previous_rephrase, max_length=None, lamp_type=None):
    import time as time_module
    import random


    if isinstance(history_context, list):
        formatted_entries = []
        for i, item in enumerate(history_context[:10], 1):
            if isinstance(item, dict):
              
                text = item.get('text', item.get('content', str(item)))
            else:
                text = str(item)
            formatted_entries.append(f"[{i}] {text}")
        history_context_str = "\n\n".join(formatted_entries)
    else:
        history_context_str = str(history_context)


    if not history_context_str or not history_context_str.strip():
        return previous_rephrase if previous_rephrase else "No summary available for this user."

    if not original_summary or not original_summary.strip():
        original_summary = "No reference summary available."

    if not previous_rephrase or not previous_rephrase.strip():
        previous_rephrase = original_summary

    ref_length = len(original_summary)
    if max_length is None:
        max_length = ref_length * 2  

    length_constraint = (
        f"- **CRITICAL LENGTH LIMIT**: Your output MUST be approximately {ref_length} characters. "
        f"NEVER exceed {max_length} characters. This is a STRICT requirement.\n"
    )

    max_retries = 5
    base_delay = 2.0

    is_multiturn_dialogue = lamp_type in [0, -1]

    for attempt in range(max_retries):
        try:
            if is_multiturn_dialogue:

                system_content = (
                    "You are an advanced language model specialized in creating user persona summaries from conversation history. "
                    "Your task is to RE-GENERATE a fresh user persona summary directly from the RAW HISTORY CONTEXT. "
                    "This is a RESET operation - treat the history context as if you are seeing it for the first time. "
                    f"CRITICAL: Keep output approximately {ref_length} characters, NEVER exceed {max_length} characters. "
                    "\n\n"
                    "You will be given: "
                    "1) RAW HISTORY CONTEXT - the user's original conversation, generate summary DIRECTLY from this "
                    "2) REFERENCE FORMAT - use this as a template for format, length, and writing style "
                )

                user_content = (
                    "RE-GENERATE a user persona summary based on the following:\n\n"
                    f"### RAW HISTORY CONTEXT (user's original conversation - generate summary DIRECTLY from this):\n{history_context_str}\n\n"
                    f"### REFERENCE FORMAT (use as template for format and length):\n{original_summary}\n\n"
                    "Generate a FRESH persona summary that:\n"
                    "1. **DIRECTLY DERIVED**: Extract characteristics DIRECTLY from RAW HISTORY CONTEXT (not from previous versions)\n"
                    "2. **PRESERVE CORE ANCHORS**: Include 2-3 SPECIFIC concrete details from the history (names, topics, specific behaviors)\n"
                    "3. **MAINTAINS FORMAT**: Follow the structure and style of REFERENCE FORMAT\n"
                    f"4. **LENGTH CONSTRAINT**: {length_constraint}\n"
                    "- Focus on: personality traits, interests, communication style, preferences, goals, behavior patterns\n"
                    "- Use a third-person perspective (They/This user...)\n"
                    "- Do not make up personal information (e.g., name)\n"
                    "- Output only the persona summary, no other text:\n"
                )


                temperature = 0.4
                presence_penalty = 0.2
                frequency_penalty = 0.2

            else:

                system_content = (
                    "You are an advanced language model specialized in creating user persona summaries from user history data. "
                    "Your task is to RE-GENERATE a fresh user persona summary directly from the RAW HISTORY CONTEXT. "
                    "This is a RESET operation - treat the history context as if you are seeing it for the first time. "
                    f"CRITICAL: Keep output approximately {ref_length} characters, NEVER exceed {max_length} characters. "
                    "\n\n"
                    "You will be given: "
                    "1) RAW HISTORY CONTEXT - the user's original writings, generate summary DIRECTLY from this "
                    "2) REFERENCE FORMAT - use this as a template for format, length, and writing style "
                )

                user_content = (
                    "RE-GENERATE a user persona summary based on the following:\n\n"
                    f"### RAW HISTORY CONTEXT (user's original writings - generate summary DIRECTLY from this):\n{history_context_str}\n\n"
                    f"### REFERENCE FORMAT (use as template for format and length):\n{original_summary}\n\n"
                    "### Task:\n"
                    "Generate a FRESH persona summary that:\n"
                    "1. **DIRECTLY DERIVED**: Extract characteristics DIRECTLY from RAW HISTORY CONTEXT (not from previous versions)\n"
                    "2. **PRESERVE CORE ANCHORS**: Include 2-3 SPECIFIC concrete details from the history (names, topics, specific behaviors)\n"
                    "3. **MAINTAINS FORMAT**: Follow the structure and style of REFERENCE FORMAT\n"
                    f"4. **LENGTH CONSTRAINT**: {length_constraint}"
                    "- Focus on: personality traits, interests, communication style, preferences, goals, behavior patterns\n"
                    "- Use a third-person perspective (They/This user...)\n"
                    "- Do not make up personal information (e.g., name)\n"
                    "- Output only the persona summary, no other text:\n"
                )


                temperature = 0.4
                presence_penalty = 0.2
                frequency_penalty = 0.2

            response = client.chat.completions.create(
                model="deepseek/deepseek-v3.2",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                top_p=0.95,
                timeout=30.0,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

            if response is None:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time_module.sleep(delay)
                    continue
                else:
                    raise Exception(f"Rephrase with context API返回None")

            if not hasattr(response, 'choices') or not response.choices:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time_module.sleep(delay)
                    continue
                else:
                    raise Exception(f"Rephrase with context API返回格式异常")

            content = response.choices[0].message.content
            if content is None or content.strip() == "":
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time_module.sleep(delay)
                    continue
                else:
                    raise Exception(f"Rephrase with context API返回内容为空")

            result = content.strip()

            if max_length and len(result) > max_length:
                print(f"⚠️ [get_rephrase_with_context] 输出长度 ({len(result)}) 超过限制 ({max_length})，进行截断")
                result = result[:max_length - 3] + "..."

            return result

        except Exception as e:
            if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):
                return previous_rephrase

            elif "rate_limit" in str(e).lower() or "429" in str(e):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️  ")
                    time_module.sleep(delay)
                    continue
                else:
                    return previous_rephrase

            elif any(error_type in str(e).lower() for error_type in [
                "connection", "timeout", "network", "unreachable"
            ]):
                if attempt < max_retries - 1:
                    delay = base_delay * (3 ** attempt) + random.uniform(1, 5)
                    time_module.sleep(delay)
                    continue
                else:
                    raise Exception(f"Rephrase with context API")

            else:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    time_module.sleep(delay)
                    continue
                else:
                    raise Exception(f"Rephrase with context API")

    raise Exception(f"Rephrase with context API")

def domain_generation(instruction, summary, time, lamp_type=None, input_id=None, counter=None, history_context=None):

    import os
    import json
    from datetime import datetime

    def save_incremental_data(filepath, data):
        """增量保存数据到文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"⚠️ 保存文件失败: {e}")
            return False

    # 去除首尾空白字符
    summary = summary.strip() if summary else ""
    instruction = instruction.strip() if instruction else ""

    # 🔧 检查输入参数的有效性 - 这些情况不应该发生，如果发生说明上游有问题
    if not summary:

        summary = "No summary available for this user."

    if not instruction:

        instruction = "Generate a personalized recommendation:"


    single_instruction = instruction


    storage_dir = "./rephase_data"
    os.makedirs(storage_dir, exist_ok=True)

    if lamp_type == 0:
        dataset_prefix = "ultrachat"
    elif lamp_type == -1:
        dataset_prefix = "wildchat"
    else:
        dataset_prefix = f"LaMP{lamp_type}"


    counter_str = f"_counter{counter}" if counter is not None else ""
    filename = f"{dataset_prefix}_times{time}_{input_id}{counter_str}.json"
    filepath = os.path.join(storage_dir, filename)


    existing_files = []
    if os.path.exists(storage_dir):
        for existing_file in os.listdir(storage_dir):

            counter_pattern = f"_counter{counter}" if counter is not None else ""
            expected_suffix = f"_{input_id}{counter_pattern}.json"


            expected_patterns = [f"{dataset_prefix}_times"]  #

            if lamp_type == 0:
                expected_patterns.append("LaMP0_times")
            elif lamp_type == -1:
                expected_patterns.append("LaMP-1_times")


            matches_pattern = any(existing_file.startswith(pattern) for pattern in expected_patterns)

            if matches_pattern and existing_file.endswith(expected_suffix):

                try:

                    parts = existing_file.split('_')
                    times_part = None
                    for part in parts:
                        if part.startswith('times'):
                            times_part = part
                            break

                    if times_part:
                        existing_times = int(times_part.replace('times', ''))
                        existing_files.append({
                            'filename': existing_file,
                            'filepath': os.path.join(storage_dir, existing_file),
                            'times': existing_times
                        })
                except Exception as parse_e:
                    print(f"⚠️ [rephrase_with_history] {parse_e}")
                    continue

    def validate_file_completeness(filepath, expected_times):

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                stored_data = json.load(f)

            # 检查必要字段
            if 'summaries' not in stored_data:
                return False, 

            stored_summaries = stored_data.get('summaries', [])

            # 检查数量是否匹配目标times
            if len(stored_summaries) != expected_times:
                return False, f""

            # 检查每个summary是否为空或无效
            for i, summary in enumerate(stored_summaries):
                if not summary or not isinstance(summary, str) or len(summary.strip()) == 0:
                    return False, f""

            # 检查文件是否有其他必要的元数据
            required_fields = ['instruction', 'original_summary', 'lamp_type', 'times', 'created_at']
            for field in required_fields:
                if field not in stored_data:
                    return False, f"{field}"

            return True, ""

        except json.JSONDecodeError as json_e:
            print(f"⚠️  {json_e}")
            return False, "JSON格式错误"
        except Exception as e:
            print(f"⚠️  {e}")
            return False, f" {e}"


    filepath_candidates = [filepath] 

    if lamp_type == 0:
        old_filename = f"LaMP0_times{time}_{input_id}{counter_str}.json"
        filepath_candidates.append(os.path.join(storage_dir, old_filename))
    elif lamp_type == -1:
        old_filename = f"LaMP-1_times{time}_{input_id}{counter_str}.json"
        filepath_candidates.append(os.path.join(storage_dir, old_filename))


    matched_filepath = None
    for candidate_path in filepath_candidates:
        if os.path.exists(candidate_path):
            matched_filepath = candidate_path
            break

    if matched_filepath:
        is_valid, reason = validate_file_completeness(matched_filepath, time)
        if is_valid:
            try:
                with open(matched_filepath, 'r', encoding='utf-8') as f:
                    stored_data = json.load(f)
                    stored_summaries = stored_data.get('summaries', [])

                print(f"✅ ")
                return single_instruction, stored_summaries

            except Exception as e:
                print(f"⚠️ ")
        else:
            print(f"⚠️ ")
            print(f"   ")
            # 不删除文件，而是作为基础文件处理
            try:
                with open(matched_filepath, 'r', encoding='utf-8') as f:
                    stored_data = json.load(f)
                    existing_summaries = stored_data.get('summaries', [])

                # 直接使用这个不完整文件作为基础
                allVersions_summary = existing_summaries.copy()
                current_count = len(allVersions_summary)
                print(f"")


                if current_count >= time:
                    print(f"✅")
                    return single_instruction, allVersions_summary[:time]


                for i in range(current_count, time):
                    try:
                        base_summary = allVersions_summary[-1]
                        print(f"   ")
                        rephrased_summary = get_rephrase(base_summary, lamp_type=lamp_type)

                        if rephrased_summary and rephrased_summary.strip() != base_summary.strip():
                            allVersions_summary.append(rephrased_summary)
                            print(f"   ")
                        else:
                            print(f"   ")
                            allVersions_summary.append(base_summary)


                        current_data = {
                            "instruction": single_instruction,
                            "original_summary": summary,
                            "summaries": allVersions_summary.copy(),
                            "lamp_type": lamp_type,
                            "times": time,  #
                            "input_id": input_id,
                            "counter": counter,
                            "created_at": datetime.now().isoformat(),
                            "generation_method": "chain_rephrase",
                            "current_progress": len(allVersions_summary)  
                        }


                        target_filename = f"{dataset_prefix}_times{time}_{input_id}{counter_str}.json"
                        target_filepath = os.path.join(storage_dir, target_filename)

                        if save_incremental_data(target_filepath, current_data):
                            print(f"   💾  {len(allVersions_summary)}/{time})")

                    except Exception as e:
                        print(f"   ❌  {e}")
                        allVersions_summary.append(allVersions_summary[-1])  

                print(f"🎉 ")
                return single_instruction, allVersions_summary

            except Exception as e:
                print(f"⚠️ ")



    base_file = None
    if existing_files:
        existing_files.sort(key=lambda x: x['times'], reverse=True)  
        base_file = existing_files[0]  

    allVersions_summary = []
    current_count = 0

    if base_file:

        is_valid, reason = validate_file_completeness(base_file['filepath'], base_file['times'])
        if is_valid:
            try:
                with open(base_file['filepath'], 'r', encoding='utf-8') as f:
                    stored_data = json.load(f)
                    existing_summaries = stored_data.get('summaries', [])

                if base_file['times'] >= time:

                    print(f"✅ ")
                    return single_instruction, existing_summaries[:time]
                else:

                    allVersions_summary = existing_summaries.copy()
                    current_count = len(allVersions_summary)
                    print(f"📂 ")

            except Exception as e:
                print(f"⚠️ {e}")

                allVersions_summary = [summary]  
                current_count = 1
        else:
            print(f"⚠️  {reason}")
            print(f"   ")

            try:
                with open(base_file['filepath'], 'r', encoding='utf-8') as f:
                    stored_data = json.load(f)
                    existing_summaries = stored_data.get('summaries', [])

                allVersions_summary = existing_summaries.copy()
                current_count = len(allVersions_summary)
                print(f"📂 使用不完整基础文件，已有 {current_count} 个版本，需要继续生成 {time - current_count} 个版本")

            except Exception as e:
                print(f"⚠️ 读取不完整基础文件失败: {e}")

                allVersions_summary = [summary]  
                current_count = 1
    else:

        allVersions_summary = [summary]  
        current_count = 1


    from IDS_TAP_parameters.py import DATA_CONFIG
    reset_interval = DATA_CONFIG.get("rephrase_reset_interval", 40)
    reset_enabled = DATA_CONFIG.get("rephrase_reset_enabled", True)

    original_summary_for_ref = allVersions_summary[0] if allVersions_summary else summary
    ref_length = len(original_summary_for_ref)
    max_length = ref_length * 2  


    for i in range(current_count, time):
        try:
            base_summary = allVersions_summary[-1]  

            is_reset_point = reset_enabled and (i > 0) and (i % reset_interval == 0)

            if is_reset_point and history_context is not None:

                original_summary = allVersions_summary[0]  
                rephrased_summary = get_rephrase_with_context(history_context, original_summary, base_summary, max_length=max_length, lamp_type=lamp_type)
            elif is_reset_point and history_context is None:

                rephrased_summary = get_rephrase(base_summary, max_length=max_length, lamp_type=lamp_type)
            else:

                rephrased_summary = get_rephrase(base_summary, max_length=max_length, lamp_type=lamp_type)


            if rephrased_summary and rephrased_summary.strip() != base_summary.strip():
                allVersions_summary.append(rephrased_summary)
            else:
                allVersions_summary.append(base_summary)


            current_data = {
                'lamp_type': lamp_type,
                'input_id': input_id,
                'counter': counter,
                'times': time,  
                'original_summary': summary,
                'summaries': allVersions_summary,
                'created_time': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'current_progress': len(allVersions_summary)  
            }

            target_filename = f"{dataset_prefix}_times{time}_{input_id}{counter_str}.json"
            target_filepath = os.path.join(storage_dir, target_filename)

            if save_incremental_data(target_filepath, current_data):
                print(f"   💾 {len(allVersions_summary)}/{time}")
            else:
                print(f"   ⚠️ ")

        except Exception as e:
            print(f"❌ {e}")
            raise e

    while len(allVersions_summary) < time:
        allVersions_summary.append(allVersions_summary[-1])  

    print(f"🎉" )
    return single_instruction, allVersions_summary
