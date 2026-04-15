import os
import json
import re
import time
from typing import Dict

# Import Pipeline and client
try:
    from rubrics_pipeline_mode import RubricPipeline, llm_client
except ImportError:
    print("❌ Error: 'rubrics_pipeline_mode.py' not found. Please ensure the file is in the same directory.")
    exit(1)

# ================= CONFIGURATION =================

# 
START_INDEX = 0
END_INDEX = 150

# File path configuration
QUERY_FILE = "data/conversations_all.txt"
EVIDENCE_DIR = "evidence"
OUTPUT_DIR = "ablation_results"  

# Define the experimental mode to be run
# no_router: Stage 1 use original Query + Authoritative domain name whitelist search
# no_audit = no_refinement（pass Step 4）
ABLATION_MODES = [
    "full",
    "no_router",
    "no_atomic",
    "no_intent",
    "no_audit",
]

# When API token/credit is exhausted, immediately stop the entire batch processing to avoid wasteful calls
QUOTA_ERROR_KEYWORDS = (
    "insufficient",
    "quota",
    "rate limit",
    "too many requests",
    "429",
    "credit",
    "billing",
    "exceeded your current quota",
)

API_CALL_ERROR_KEYWORDS = (
    "error code",
    "model_not_found",
    "not_found_error",
    "does not exist or you do not have access",
    "unauthorized",
    "authentication",
    "forbidden",
    "invalid api key",
    "api key",
    "permission",
    "access denied",
) + QUOTA_ERROR_KEYWORDS


def _is_quota_error_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(k in lowered for k in QUOTA_ERROR_KEYWORDS)


def _is_api_call_error_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(k in lowered for k in API_CALL_ERROR_KEYWORDS)


def _trace_has_quota_error(obj) -> bool:
    if isinstance(obj, dict):
        return any(_trace_has_quota_error(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_trace_has_quota_error(v) for v in obj)
    if isinstance(obj, str):
        return _is_quota_error_text(obj)
    return False


def _collect_api_error_texts(obj, found=None):
    if found is None:
        found = []
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_api_error_texts(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _collect_api_error_texts(v, found)
    elif isinstance(obj, str):
        if _is_api_call_error_text(obj):
            found.append(obj)
    return found


def _extract_message_from_error_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(r"[\"']message[\"']\s*:\s*[\"'](.+?)[\"']", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_rubric_count(obj) -> int:
    if not isinstance(obj, dict):
        return 0
    raw = obj.get("generated_rubrics")
    if isinstance(raw, dict):
        rubrics = raw.get("rubrics", [])
    elif isinstance(raw, list):
        rubrics = raw
    else:
        rubrics = []
    if not isinstance(rubrics, list):
        return 0
    count = 0
    for r in rubrics:
        if isinstance(r, dict) and str(r.get("criterion", "")).strip():
            count += 1
    return count


def _existing_result_is_valid(path: str) -> bool:
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return _extract_rubric_count(obj) > 0
    except Exception:
        return False

# ================= HELPER FUNCTIONS =================

def load_queries(file_path: str) -> Dict[int, str]:
    queries = {}
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: Query file {file_path} not found.")
        return queries
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r'^(\d+)\s+User:\s+(.*)', line)
            if match:
                queries[int(match.group(1))] = match.group(2).strip()
    return queries

# ================= MAIN RUNNER =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"🚀 Starting Ablation Studies Batch")
    print(f"📊 Range: {START_INDEX} -> {END_INDEX}")
    print(f"🧪 Modes: {ABLATION_MODES}")

    if not llm_client:
        print("\n❌ Error: llm_client initialization failed! Please check the API Key Settings.")
        return

    all_queries = load_queries(QUERY_FILE)
    print(f"📂 Loaded {len(all_queries)} queries.")

    for idx in range(START_INDEX, END_INDEX + 1):
        if idx not in all_queries: continue
            
        user_query = all_queries[idx]
        print(f"\n==========================================")
        print(f"🔬 Processing ID {idx} [{idx - START_INDEX + 1}/{END_INDEX - START_INDEX + 1}]: {user_query[:50]}...")
        
        # Load Evidence: full with evidence/，no_router  with  evidence_no_router/
        full_evidence_path = os.path.join(EVIDENCE_DIR, f"conversation_{idx}.json")
        no_router_evidence_path = os.path.join("evidence_no_router", f"conversation_{idx}.json")
        
        full_evidence = {}
        no_router_evidence = {}
        has_full = os.path.exists(full_evidence_path)
        has_no_router = os.path.exists(no_router_evidence_path)
        
        if has_full:
            try:
                with open(full_evidence_path, 'r', encoding='utf-8') as f:
                    full_evidence = json.load(f)
            except: pass
        if has_no_router:
            try:
                with open(no_router_evidence_path, 'r', encoding='utf-8') as f:
                    no_router_evidence = json.load(f)
            except: pass

        for mode in ABLATION_MODES:
            mode_dir = os.path.join(OUTPUT_DIR, mode)
            if not os.path.exists(mode_dir):
                os.makedirs(mode_dir)
            
            # file rubric_{ID}.json
            output_filename = os.path.join(mode_dir, f"rubric_{idx}.json")
            
            if _existing_result_is_valid(output_filename):
                print(f"   ⏭️  Mode: [{mode}] already exists with non-empty rubrics. Skipping.")
                continue
            if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                print(f"   🔁 Mode: [{mode}] existing file is empty/invalid. Retrying generation.")

            print(f"   👉 Running Mode: [{mode}]")
            
            # Data source selection
            if mode == "no_router":
                current_evidence = no_router_evidence
                if not has_no_router:
                    print(f"      ❌ Skipping: Missing no_router evidence. Run: python batch_run_no_router.py")
                    continue
            else:
                current_evidence = full_evidence
                if not has_full:
                    print(f"      ❌ Skipping: Missing full RAG evidence.")
                    continue

            try:
                start_time = time.time()
                #  Pipeline
                pipeline = RubricPipeline(
                    query=user_query, 
                    evidence_data=current_evidence, 
                    ablation_mode=mode
                )
                
                trace = pipeline.execute()

                if _trace_has_quota_error(trace):
                    print("      🛑 Detected API quota/token exhaustion in pipeline trace.")
                    print("      🛑 Stopping batch now to avoid empty/invalid outputs.")
                    return
                
                final_rubrics = trace.get("step_4_final_rubrics")
                if not final_rubrics:
                    final_rubrics = trace.get("step_3_draft_rubrics")
                count = len(final_rubrics.get("rubrics", [])) if final_rubrics and "rubrics" in final_rubrics else 0

                # Print messages for API call failures if count is 0 
                api_errors = _collect_api_error_texts(trace)
                if count == 0 and api_errors:
                    print("      🛑 Detected API call failure with 0 generated criteria.")
                    printed = set()
                    for err in api_errors:
                        msg = _extract_message_from_error_text(err)
                        if msg and msg not in printed:
                            print(f"      🛑 message: {msg}")
                            printed.add(msg)
                    print("      🛑 Stopping batch now.")
                    return
                
                result_record = {
                    "id": idx,
                    "query": user_query,
                    "mode": mode,
                    "generated_rubrics": final_rubrics,
                    "full_trace": trace
                }
                
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(result_record, f, indent=2, ensure_ascii=False)
                    
                elapsed = time.time() - start_time
                print(f"      ✅ Success ({elapsed:.1f}s). Criteria: {count}. Saved to /{mode}/rubric_{idx}.json")

            except Exception as e:
                err_text = str(e)
                if _is_api_call_error_text(err_text):
                    print("      🛑 API call failure detected.")
                    print(f"      🛑 message: {_extract_message_from_error_text(err_text)}")
                    print("      🛑 Stopping batch now.")
                    return
                if _is_quota_error_text(str(e)):
                    print(f"      🛑 API quota/token exhaustion detected: {e}")
                    print("      🛑 Stopping batch now.")
                    return
                print(f"      ❌ Pipeline Error: {e}")

    print("\n🎉 Ablation Study Batch Complete!")

if __name__ == "__main__":
    main()
