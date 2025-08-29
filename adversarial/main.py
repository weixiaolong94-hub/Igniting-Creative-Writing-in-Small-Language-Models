import json
import random
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union


from utils import process_queries, data_process


# --- Configuration ---
API_CALL_DELAY = 0.2 
ADVERSARIAL_BATCH_SIZE = 4
SELF_REFLECTION_BATCH_SIZE = 4


def call_llm_batch(prompt_list: List[str], input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calls the LLM API (process_queries) for a batch of prompts.
    Parses the list of expected JSON string responses, which might be out of order.
    Returns a list of parsed dictionaries (including request_id) or error dictionaries.
    """
    print("-" * 20)
    print(f"Calling LLM API (Batch of {len(prompt_list)}) for: {prompt_list[0][:60]}...") 
    results = []
    if not prompt_list:
        return []

    for i, data in enumerate(input_data_list):
        if "request_id" not in data or not data["request_id"]: # Check for empty too
             data["request_id"] = str(uuid.uuid4())
             print(f"Debug: Assigned new request_id {data['request_id']} to input item {i}")

    id_to_data_map = {d["request_id"]: d for d in input_data_list if d.get("request_id")}

    try:
        response_dicts = data_process(process_queries(input_data_list, prompt_list))

        if not isinstance(response_dicts, list):
            print(f"Error: data_process did not return a list. Got: {response_dicts}")
            return [{"error": "API processed data is not a list", "request_id": d.get("request_id", "unknown")} for d in input_data_list]

        if len(response_dicts) != len(prompt_list):
            print(f"Warning: API returned {len(response_dicts)} results for {len(prompt_list)} prompts.")
            if abs(len(response_dicts) - len(prompt_list)) > 0:
                print("Attempting to match responses to requests via request_id...")

        processed_requests = set()
        unmatched_responses = []

        for i, res_dict in enumerate(response_dicts):
            if not isinstance(res_dict, dict):
                 print(f"Error: Response item {i} is not a dictionary: {res_dict}")
                 original_request_id_by_index = input_data_list[i].get("request_id","unknown_at_index_"+str(i)) if i < len(input_data_list) else f"unknown_index_{i}"
                 results.append({"error": "Response item not a dict", "raw_response": str(res_dict), "guessed_request_id": original_request_id_by_index})
                 continue

            req_id = res_dict.get("request_id")

            if req_id and req_id in id_to_data_map:
                results.append(res_dict)
                processed_requests.add(req_id)
            else:
                 print(f"Warning: Response item {i} has missing or unmatched request_id ('{req_id}'). Raw: {str(res_dict)[:100]}...")
                 if len(response_dicts) == len(prompt_list) and i < len(input_data_list):
                      original_request_id_by_index = input_data_list[i].get("request_id", "unknown_at_index_"+str(i))
                      res_dict["request_id"] = original_request_id_by_index # Force assign ID
                      res_dict["warning"] = "Request ID assigned by index fallback"
                      results.append(res_dict)
                      processed_requests.add(original_request_id_by_index)
                      print(f"  Fallback: Assigned response {i} to request_id {original_request_id_by_index} based on index.")
                 else:
                      unmatched_responses.append(res_dict)


        for req_id, data in id_to_data_map.items():
            if req_id not in processed_requests:
                 print(f"Error: No matching response found for request_id '{req_id}'. Input query: {data.get('query', 'N/A')[:50]}...")
                 results.append({"error": "No matching response received from API", "request_id": req_id})

        if unmatched_responses:
             print(f"Error: {len(unmatched_responses)} responses could not be matched to any request.")
             for um_res in unmatched_responses:
                  results.append({"error": "Unmatched response", "raw_response": um_res, "request_id": um_res.get("request_id", "unknown_unmatched")})


    except NameError as ne:
         print(f"Error: A required function ('process_queries' or 'data_process') is not defined: {ne}")
         results = [{"error": "API function not defined", "request_id": d.get("request_id", "unknown")} for d in input_data_list]
    except Exception as api_err:
        print(f"Error: API call or data processing raised an exception: {api_err}")
        results = [{"error": f"API call/processing exception: {api_err}", "request_id": d.get("request_id", "unknown")} for d in input_data_list]

    if prompt_list: 
        print(f"Waiting {API_CALL_DELAY} seconds...")
        time.sleep(API_CALL_DELAY)

    print(f"Parsed LLM Outputs (Batch): {len(results)} items processed (incl. errors).")
    print("-" * 20)
    return results



GENERATOR_PROMPT_TEMPLATE = """
# 角色: “伪”祝福语构造大师
# 核心任务: 根据用户请求（Query）和当前的“坏祝福语构造策略”，生成一条**看起来合理、但实际上存在某种缺陷**的祝福语。同时，解释你是**如何故意制造**这些缺陷的。请在最终的JSON输出中包含传入的`request_id`。
# 背景: 你需要生成有挑战性的“坏”样本来训练判别器。
# 输入:
# * 用户请求 (Query): ```{query}```
# * 优质祝福语示例 (可选): ```{good_response_example}```
# * 坏祝福语构造策略 (S_G): ```{current_bad_strategies}```
# * 请求ID (Request ID): ```{request_id}```
# 任务指示:
# 1. 理解 Query 的核心需求。
# 2. 遵循 坏祝福语构造策略 (S_G)，在生成的祝福语中**故意引入**缺陷。
# 3. 生成“坏”祝福语 (Bad Response)。
# 4. 生成“缺陷说明” (Explanation of Flaws)，解释你是如何依据策略故意制造这些缺陷的。
# 5. 将结果格式化为 JSON，并包含原始的 `request_id`。
# 输出格式 (JSON):
# ```json
# {{
#   "request_id": "{request_id}",
#   "bad_response": "生成的、带有隐藏缺陷的祝福语文本...",
#   "explanation_of_flaws": "解释如何依据策略故意制造这些缺陷的。"
# }}
# ```
"""

DETECTOR_PROMPT_TEMPLATE = """
# 角色: 祝福语质量评估员
# 核心任务: 根据用户请求（Query）和当前的“祝福语评估标准”，判断给定的祝福语是“好”的还是“坏”的，并提供详细的评估解释。
#          **如果判断为 'Bad'**，请基于你发现的缺陷，提出一条对**生成器 (Generator) 的“坏祝福语构造策略 (S_G)”** 的**优化建议**。这个建议应当帮助生成器制造更难被发现的同类缺陷。
#          请在最终的JSON输出中包含传入的`request_id`以及可能的策略建议。
# 背景: 你需要评估祝福语的质量，并帮助对抗训练。
# 输入:
# * 用户请求 (Query): ```{query}```
# * 待评估祝福语 (Response to Evaluate): ```{response_to_evaluate}```
# * 祝福语评估标准 (S_D): ```{current_evaluation_strategies}```
# * 请求ID (Request ID): ```{request_id}```
# 任务指示:
# 1. 理解 Query 的核心需求。
# 2. 依据 Evaluation Strategies (S_D)，审查 Response to Evaluate。
# 3. 做出判断：“Good”或“Bad”。
# 4. 生成“评估解释” (Evaluation Explanation)。
# 5. **如果 verdict 为 "Bad"**: 思考如何让 Generator 下次能生成更难被你发现的、同类型的坏祝福语，并生成 `suggested_for_G` (针对 S_G 的优化建议，简洁地说明如何调整策略以隐藏缺陷)。
# 6. `suggested_for_G` 应是具体的行动建议，而非原因分析。
# 7. 将结果格式化为 JSON，包含 `request_id`, `verdict`, `evaluation_explanation` 以及可能的 `suggested_for_G`。
# 输出格式 (JSON):
# ```json
# {{
#   "request_id": "{request_id}",
#   "verdict": "判断结果 (Good / Bad)",
#   "evaluation_explanation": "详细的评估解释，说明为什么判断为Good或Bad。",
#   "suggested_for_G": "如果verdict是Bad，这里是给Generator策略S_G的优化建议，否则此字段不出现或为空"
# }}
# ```
"""

REFLECTOR_PROMPT_TEMPLATE = """
# 角色: 评估过程纠错与分析师
# 核心任务: 当“祝福语质量评估员”(Detector)的判断与“真实标签”(Ground Truth Label)不符时，分析其错误原因，生成“反思反馈”，并据此提出一条对**判别器 (Detector) 的“祝福语评估标准 (S_D)”** 的**优化建议**。
# 背景: 你是评估流程中的质量监控环节，旨在帮助 Detector 改进评估标准 S_D。
# 输入:
# * 用户请求 (Query): ```{query}```
# * 被评估的祝福语 (Evaluated Response): ```{evaluated_response}```
# * 评估员的判断 (Detector Verdict): ```{detector_verdict}```
# * 评估员的解释 (Detector Explanation): ```{detector_explanation}```
# * 真实标签 (Ground Truth Label): ```{ground_truth_label}```
# * 请求ID (Request ID): ```{request_id}```
# 任务指示:
# 1. 比较 Detector Verdict 和 Ground Truth Label。
# 2. **仅当两者不一致时**:
#    a. 分析错误根源并生成“反思反馈” 。
#    b. 基于错误分析，生成对 Detector 策略 S_D 的优化建议 `suggested_for_D`。这个建议应旨在**改进当前的S_D**，使其能正确处理此类错误。
#    c. `suggested_for_D` 应是具体的策略调整建议，而非仅仅重复错误。
# 3. 将结果格式化为 JSON，包含 `request_id` 以及可能的 `reflection_feedback` 和 `suggested_for_D`。
# 输出格式 (JSON):
# ```json
# {{
#   "request_id": "{request_id}",
#   "reflection_feedback": "反思反馈文本...", // 仅在判断错误时生成
#   "suggested_for_D": "基于反思提出的对Detector策略S_D的优化建议..." // 仅在判断错误时生成
# }}
# ```
# * 如果判断正确，则输出: {{"request_id": "{request_id}"}}
"""

STRATEGY_OPTIMIZER_PROMPT_TEMPLATE = """
# 角色: 策略优化师
# 核心任务: 基于提供的反馈，优化现有的策略集（S_G 或 S_D），使其更有效、更清晰、无冗余。
# 背景: 你接收当前的策略和一个具体的优化反馈（可能来自Detector的建议、Reflector的建议或Generator的失败解释），需要整合这些信息来改进策略。
# 输入:
# * 当前策略 (Current Strategy Set): ```{current_strategy}```
# * 优化反馈 (Optimization Feedback): ```{feedback}```
# * 请求ID (Request ID): ```{request_id}```
# 任务指示:
# 1. 理解 当前策略 的规则。
# 2. 理解 优化反馈 的核心改进点。
# 3. **修改** 当前策略:
#    * **整合** 反馈中的有效信息。
#    * **改进** 现有规则，使其更精确或更有效。
#    * **合并** 相似或重叠的规则。
#    * **删除** 无效或与反馈冲突的规则。
#    * **添加** 反馈中提出的、但当前策略未覆盖的新规则（如果适用）。
#    * 保持策略规则的简洁和可操作性。
#    * 最终的优化策略不能超过10条。
# 4. 输出优化后的完整策略集。
# 5. 将结果格式化为 JSON。
# 输出格式 (JSON):
# ```json
# {{
#   "request_id": "{request_id}",
#   "optimized_strategy": "优化后的完整策略文本，规则间用换行符分隔..."
# }}
# ```
"""

# --- Core Functions ---

def run_generator_batch(batch_queries: List[str], S_G: str, good_example: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Runs generator in batch, returns results and a map from request_id to original query."""
    prompts = []
    inputs_list = []
    request_id_to_query_map = {}
    for query in batch_queries:
        request_id = str(uuid.uuid4())
        inputs_list.append({"request_id": request_id, "query": query})
        prompt_str = GENERATOR_PROMPT_TEMPLATE.format(
            query=query,
            good_response_example=good_example if good_example else "无",
            current_bad_strategies=S_G,
            request_id=request_id
        )
        prompts.append(prompt_str)
        request_id_to_query_map[request_id] = query 

    batch_results = call_llm_batch(prompts, inputs_list)
    return batch_results, request_id_to_query_map

def run_detector_batch(batch_data: List[Dict[str, Any]], S_D: str, self_reflect=False) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Runs detector in batch.
    batch_data items should contain: query, response_to_evaluate (or response if self_reflect).
    Returns results and a map from request_id to original input item.
    """
    prompts = []
    inputs_list = []
    request_id_to_input_map = {}

    key_response = "response" if self_reflect else "response_to_evaluate"

    for item in batch_data:
        request_id = str(uuid.uuid4())
        if 'query' not in item or key_response not in item:
            print(f"Warning: Skipping item in detector batch due to missing keys ('query' or '{key_response}'): {item}")
            continue

        inputs_list.append({"request_id": request_id, "query": item['query']})
        prompt_str = DETECTOR_PROMPT_TEMPLATE.format(
            query=item['query'],
            response_to_evaluate=item[key_response],
            current_evaluation_strategies=S_D, 
            request_id=request_id
        )
        prompts.append(prompt_str)
        request_id_to_input_map[request_id] = item 

    batch_results = call_llm_batch(prompts, inputs_list)
    return batch_results, request_id_to_input_map


def run_reflector_batch(reflection_batch_inputs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Runs reflector in batch, returns results and a map from request_id to original input item."""
    prompts = []
    inputs_list = []
    request_id_to_input_map = {}
    if not reflection_batch_inputs:
        return [], {}

    for item in reflection_batch_inputs:
        request_id = item.get("request_id")
        if not request_id:
             request_id = str(uuid.uuid4())
             item["request_id"] = request_id
             print(f"Warning: Generated missing request_id for reflector input: {request_id}")

        required_keys = ['query', 'evaluated_response', 'detector_verdict', 'detector_explanation', 'ground_truth_label', 'request_id']
        if not all(key in item for key in required_keys):
            print(f"Error: Missing required keys for reflector prompt formatting in item with ID {request_id}. Skipping. Keys present: {list(item.keys())}")
            continue

        inputs_list.append({"request_id": request_id, "query": item['query']})
        prompt_str = REFLECTOR_PROMPT_TEMPLATE.format(
             query=item['query'],
             evaluated_response=item['evaluated_response'],
             detector_verdict=item['detector_verdict'],
             detector_explanation=item['detector_explanation'],
             ground_truth_label=item['ground_truth_label'],
             request_id=request_id
        )
        prompts.append(prompt_str)
        request_id_to_input_map[request_id] = item

    batch_results = call_llm_batch(prompts, inputs_list)
    return batch_results, request_id_to_input_map


def optimize_strategy(current_strategy: str, feedback: str, strategy_name: str) -> str:
    """
    Uses an LLM call to optimize the given strategy string based on feedback.
    Returns the optimized strategy string, or the original if optimization fails.
    """
    print(f"*** Optimizing {strategy_name} Strategy...")
    print(f"    Feedback: {feedback[:150]}...") 

    request_id = str(uuid.uuid4())
    optimizer_input = {"request_id": request_id} 
    optimizer_prompt = STRATEGY_OPTIMIZER_PROMPT_TEMPLATE.format(
        current_strategy=current_strategy,
        feedback=feedback,
        request_id=request_id
    )


    optimizer_results = call_llm_batch([optimizer_prompt], [optimizer_input])

    if not optimizer_results:
        print(f"Error: Strategy Optimizer ({strategy_name}, ID: {request_id}) returned no result.")
        return current_strategy 

    result = optimizer_results[0] 

    if "error" in result or "optimized_strategy" not in result:
        print(f"Error: Strategy Optimizer ({strategy_name}, ID: {request_id}) failed or gave invalid output: {result.get('error', 'Missing optimized_strategy key')}")
        return current_strategy 

    new_strategy = result.get("optimized_strategy")
    if isinstance(new_strategy, str) and new_strategy.strip():
        print(f"*** {strategy_name} Strategy Optimized Successfully (ID: {request_id}).")
        if len(new_strategy) < 0.5 * len(current_strategy):
             print(f"Warning: Optimized strategy is significantly shorter. Check quality.")
        if new_strategy == current_strategy:
             print(f"Note: Optimizer returned the same strategy for {strategy_name}.")

        return new_strategy.strip()
    else:
        print(f"Error: Strategy Optimizer ({strategy_name}, ID: {request_id}) returned empty or invalid strategy: {new_strategy}")
        return current_strategy 


def run_llm_gan_blessing_eval(queries: List[str], known_data: List[Dict[str, str]], num_iterations: int):
    """
    Runs the LLM-GAN process using batched LLM calls and strategy optimization.
    Handles potentially unordered results using request_ids.
    Uses LLM-based strategy optimization instead of simple appending.
    """
    S_G = """
    - 生成非常通用、适用于多种场合的祝福，避免个性化。
    - 使用至少一个听起来略显陈旧或过时的祝福短语。
    - 故意忽略用户请求中的某个次要细节。
    - 保持语句极其简短，缺乏具体情感表达。
    - 采用与请求场景不太匹配的语气（例如对亲近朋友过于正式）。
    """.strip()

    S_D = """
    - 检查祝福语是否直接回应了用户请求的核心目的（如生日、康复、祝贺等）。
    - 评估祝福语是否包含真诚的情感，而非仅仅是客套话。
    - 检查祝福语的用词和语气是否适合发送者与接收者的关系以及祝福场合。
    - 识别并标记过于泛滥或缺乏新意的陈词滥调。
    - 确保祝福语没有明显的语法错误或逻辑不通之处。
    - 判断祝福语是否体现了对接收者或特定事件的关怀，而不仅仅是模板。
    """.strip()

    print("=" * 30)
    print("开始 LLM-GAN 祝福语评估训练 (带策略优化)")
    print("=" * 30)

    all_queries = list(queries) 
    all_known_data = list(known_data)

    for i in range(num_iterations):
        print(f"\n--- 第 {i+1}/{num_iterations} 轮 ---")

        print("\n>>> 1. 对抗阶段 (Batched)")
        if not all_queries:
             print("无可用 Query，跳过对抗阶段。")
             continue # Skip if no queries left

        actual_adv_batch_size = min(ADVERSARIAL_BATCH_SIZE, len(all_queries))
        query_batch = random.sample(queries, actual_adv_batch_size) if len(queries) >= actual_adv_batch_size else list(queries)

        print(f"选择 {len(query_batch)} 个 Query 进行对抗批处理...")

        generator_results, gen_id_to_query_map = run_generator_batch(query_batch, S_G)


        detector_batch_inputs = []

        detector_input_ref_to_gen_result_map = {}

        valid_gen_count = 0
        for gen_result in generator_results:
            gen_request_id = gen_result.get("request_id")
            if "error" in gen_result or not gen_request_id:
                print(f"  对抗: Generator (ID: {gen_request_id or '未知'}) 失败或无ID: {gen_result.get('error', '无ID/错误')}")
                continue

            bad_response = gen_result.get("bad_response")
            explanation_of_flaws = gen_result.get("explanation_of_flaws")
            original_query = gen_id_to_query_map.get(gen_request_id) # Look up query using the ID from the result

            if not bad_response or not explanation_of_flaws:
                 print(f"  对抗: Generator (ID: {gen_request_id}) 输出字段不完整 (bad_response or explanation missing). 跳过。")
                 continue
            if not original_query:
                print(f"  对抗: Generator (ID: {gen_request_id}) 无法关联回原始 Query。跳过。")
                continue

            valid_gen_count += 1
            detector_input_item = {
                "query": original_query,
                "response_to_evaluate": bad_response,
            }
            detector_batch_inputs.append(detector_input_item)
            detector_input_ref_to_gen_result_map[id(detector_input_item)] = gen_result

        print(f"  对抗: Generator 批处理完成，{valid_gen_count} 个有效输出进入 Detector。")

        if not detector_batch_inputs:
             print("  对抗: 无有效 Generator 输出，跳过 Detector 运行和策略优化。")
             continue

        detector_results, detector_id_to_input_map = run_detector_batch(detector_batch_inputs, S_D)

        print("  对抗: 处理 Detector 结果并优化策略...")
        processed_detector_results_map = {res.get("request_id"): res for res in detector_results if res.get("request_id")}

        detector_inputs_processed_count = 0
        s_g_optimizations = 0
        s_d_optimizations = 0

        for det_request_id, det_input_item_ref in detector_id_to_input_map.items():
             detector_inputs_processed_count +=1
             detector_result = processed_detector_results_map.get(det_request_id)

             if not detector_result:
                  print(f"    对抗优化: 未找到 Detector 结果 for request_id {det_request_id}")
                  continue
             if "error" in detector_result:
                  print(f"    对抗优化: Detector (ID: {det_request_id}) 失败: {detector_result['error']}")
                  continue

             detector_verdict = detector_result.get("verdict")
             generator_result = detector_input_ref_to_gen_result_map.get(id(det_input_item_ref))

             if not generator_result:
                 print(f"    对抗优化: 内部错误 - 无法关联 Detector 结果 (ID: {det_request_id}) 到 Generator 结果。")
                 continue

             if detector_verdict == "Bad":
                 feedback_for_G = detector_result.get("suggested_for_G")
                 if feedback_for_G and isinstance(feedback_for_G, str) and feedback_for_G.strip():
                     print(f"    对抗优化: Detector (ID: {det_request_id}) 判定为 Bad。优化 S_G...")
                     S_G = optimize_strategy(S_G, feedback_for_G, "Generator (S_G)")
                     s_g_optimizations += 1
                 else:
                      print(f"    对抗优化: Detector (ID: {det_request_id}) 判定为 Bad，但未提供有效 S_G 建议。")

             elif detector_verdict == "Good":
                 feedback_for_D = generator_result.get("explanation_of_flaws")
                 if feedback_for_D and isinstance(feedback_for_D, str) and feedback_for_D.strip():
                     print(f"    对抗优化: Detector (ID: {det_request_id}) 判定为 Good (错误)。优化 S_D...")
                     S_D = optimize_strategy(S_D, feedback_for_D, "Detector (S_D)")
                     s_d_optimizations += 1
                 else:
                      print(f"    对抗优化: Detector (ID: {det_request_id}) 判定错误，但未找到 Generator 的解释以优化 S_D。")
             else:
                  print(f"    对抗优化: Detector (ID: {det_request_id}) 返回无效 verdict: {detector_verdict}")

        print(f"  对抗: 完成策略优化 ({s_g_optimizations} S_G 优化, {s_d_optimizations} S_D 优化 / {detector_inputs_processed_count} 项)")


        print("\n>>> 2. 自反思阶段 (Batched with Optimization)")
        if not all_known_data:
            print(f"无可用已知数据，跳过自反思阶段。")
            continue

        actual_reflect_batch_size = min(SELF_REFLECTION_BATCH_SIZE, len(all_known_data))
        batch_examples = random.sample(all_known_data, actual_reflect_batch_size)
        print(f"选择 {len(batch_examples)} 个已知样本进行批处理...")

        detector_batch_results, detector_id_to_input_map = run_detector_batch(batch_examples, S_D, self_reflect=True)

        reflector_inputs = []
        processed_detector_ids = set()
        correct_count = 0
        error_count = 0
        missing_id_count = 0

        for result in detector_batch_results:
            request_id = result.get("request_id")
            original_example = None

            if request_id and request_id != "unknown" and request_id in detector_id_to_input_map:
                 original_example = detector_id_to_input_map[request_id]
            else:
                 print(f"Warning: Detector result missing valid/matching request_id ('{request_id}') in self-reflection. Result: {result}")
                 missing_id_count += 1
                 error_count += 1
                 continue 

            if request_id in processed_detector_ids:
                 print(f"Warning: Duplicate request_id '{request_id}' encountered in self-reflection detector batch results. Skipping.")
                 continue
            processed_detector_ids.add(request_id)

            if "error" in result:
                print(f"  自反思: Detector (ID: {request_id}) 失败: {result['error']}")
                error_count += 1
                continue


            detector_verdict = result.get("verdict")
            detector_explanation = result.get("evaluation_explanation", "N/A") 
            ground_truth = original_example.get('label') 

            if ground_truth is None:
                 print(f"Error: Original example for request_id '{request_id}' is missing 'label'. Skipping.")
                 error_count += 1
                 continue

            if detector_verdict not in ["Good", "Bad"]:
                 print(f"  自反思: Detector (ID: {request_id}) 返回无效 verdict: {detector_verdict}")
                 error_count += 1
                 continue

            if detector_verdict != ground_truth:
                print(f"  自反思: Detector (ID: {request_id}) 判断错误 (判为 {detector_verdict}, 实为 {ground_truth})。准备反思...")
                reflector_inputs.append({
                    "query": original_example['query'],
                    "evaluated_response": original_example['response'], 
                    "detector_verdict": detector_verdict,
                    "detector_explanation": detector_explanation,
                    "ground_truth_label": ground_truth,
                    "request_id": request_id 
                })
            else:
                correct_count += 1

        print(f"  自反思 Detector 批处理完成: {correct_count} 正确, {len(reflector_inputs)} 错误准备反思, {error_count} 处理错误.")
        if missing_id_count > 0:
             print(f"  Warning: {missing_id_count} detector results had ID issues and were skipped.")

        if reflector_inputs:
            print(f"  运行 Reflector 批处理 ({len(reflector_inputs)} 项)...")
            reflector_batch_results, reflector_id_map = run_reflector_batch(reflector_inputs)

            s_d_optimizations_reflection = 0
            processed_reflector_ids = set()
            for result in reflector_batch_results:
                 request_id = result.get("request_id") 

                 if not request_id or request_id == "unknown":
                      print(f"Warning: Reflector result missing valid request_id. Cannot apply feedback. Result: {result}")
                      continue
                 if request_id in processed_reflector_ids:
                     print(f"Warning: Duplicate request_id '{request_id}' in reflector results. Skipping.")
                     continue
                 processed_reflector_ids.add(request_id)

                 if "error" in result:
                      print(f"    Reflector (ID: {request_id}) 失败: {result['error']}")
                      continue

                 reflection_feedback = result.get("reflection_feedback")
                 feedback_for_D = result.get("suggested_for_D")

                 if reflection_feedback and feedback_for_D and isinstance(feedback_for_D, str) and feedback_for_D.strip():
                      print(f"    Reflector (ID: {request_id}) 提供反馈。优化 Detector 策略 S_D...")
                      S_D = optimize_strategy(S_D, feedback_for_D, "Detector (S_D)")
                      s_d_optimizations_reflection += 1
                 elif reflection_feedback and not feedback_for_D:
                      print(f"    Reflector (ID: {request_id}) 提供反馈但无 S_D 建议。")


            print(f"  自反思: 从 {s_d_optimizations_reflection} 条有效反思中优化了 Detector 策略 S_D。")
        else:
            print("  自反思: 无需运行 Reflector。")


        print(f"\n--- 第 {i+1} 轮结束 ---")
        print(f"当前策略状态:")
        print(f"  S_G (Generator - Bad Strategies):\n{S_G}")
        print(f"  S_D (Detector - Evaluation Strategies):\n{S_D}")

    print("\n" + "=" * 30)
    print("LLM-GAN 训练完成 (带策略优化)")
    print("=" * 30)
    print("最终 Generator 策略 (S_G):")
    print(S_G if S_G else "无")
    print("\n最终 Detector 策略 (S_D):")
    print(S_D if S_D else "无")

if __name__ == "__main__":
    sample_queries = [
        "给晋升的同事写个祝福语", "祝贺邻居乔迁之喜", "给长辈祝寿"
    ]
    sample_known_data = [
        {"query": "给生病住院的朋友写一句祝福", "response": "好好休息，安心养病，祝你早日康复！", "label": "Good"},
        {"query": "祝贺朋友新店开业", "response": "生意兴隆！", "label": "Bad"}, 
        {"query": "给过生日的奶奶写祝福", "response": "奶奶，祝您生日快乐，身体健康，笑口常开，我们都爱您！", "label": "Good"},
        {"query": "给结婚的新人祝福", "response": "祝你们新婚快乐，百年好合。", "label": "Good"},
        {"query": "给找到新工作的同学祝福", "response": "听说你换工作了，加油。", "label": "Bad"}, 
        {"query": "给考试前的妹妹打气", "response": "放轻松，好好考，我相信你一定能行！祝你取得好成绩！", "label": "Good"},
        {"query": "祝贺客户项目成功", "response": "合作愉快，再创佳绩！", "label": "Good"}, 
        {"query": "给远方的朋友寄去思念", "response": "天气凉了，注意身体。", "label": "Bad"}, 
    ] * 2 

    run_llm_gan_blessing_eval(
        queries=sample_queries,
        known_data=sample_known_data,
        num_iterations=5 
    )