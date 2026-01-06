import os
import json
import re
import glob
from tqdm import tqdm
from rubrics_pipeline import RubricPipeline

# ================= CONFIGURATION =================

# 1. 范围设置
START_INDEX = 0
END_INDEX = 300

# 2. 文件路径
QUERY_FILE_PATH = "conversations_all.txt"
EVIDENCE_DIR = "evidence"
OUTPUT_FILE = "final_generated_rubrics.jsonl" 
INDIVIDUAL_DIR = "rubrics_individual"

# ================= HELPER FUNCTIONS =================

def load_user_queries(file_path: str) -> dict:
    """Load queries from file"""
    queries = {}
    if not os.path.exists(file_path): return queries
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            match = re.match(r'^(\d+)\s+User:\s+(.*)', line, re.IGNORECASE)
            if match: queries[int(match.group(1))] = match.group(2)
    return queries

def get_processed_indices(output_file: str) -> set:
    """Check existing progress"""
    processed = set()
    if not os.path.exists(output_file): return processed
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if 'index' in data: processed.add(data['index'])
                except: continue
    except: pass
    return processed

def validate_rubric_item(item):
    """Helper to validate a single rubric dictionary"""
    if not isinstance(item, dict): return None
    if "criterion" not in item: return None
    
    # Ensure points are int
    if "points" in item:
        try:
            val = str(item["points"]).lower()
            if "inf" in val:
                item["points"] = 10 if "-" not in val else -10
            else:
                item["points"] = int(float(item["points"]))
            item["points"] = max(-10, min(10, item["points"]))
        except:
            item["points"] = 0
            
    # Ensure axis is valid
    valid_axes = ["accuracy", "completeness", "context_awareness", "communication_quality", "instruction_following"]
    if item.get("axis") not in valid_axes:
        item["axis"] = "accuracy"
        
    return item

def extract_rubrics_from_trace(trace_data: dict) -> list:
    """
    Robust extraction that handles:
    1. Dict: {"rubrics": [...]}
    2. List: [...]
    3. Nested List: [{"rubrics": [...]}]  <-- Fixing your specific issue here
    """
    raw_data = None
    
    # Try Step 4 first
    s4 = trace_data.get("step_4_final_rubrics")
    if s4: raw_data = s4
    
    # Fallback to Step 3
    if not raw_data:
        s3 = trace_data.get("step_3_draft_rubrics")
        if s3: raw_data = s3
        
    if not raw_data: return []

    # Flatten logic
    candidates = []
    
    if isinstance(raw_data, dict):
        if "rubrics" in raw_data:
            candidates = raw_data["rubrics"]
            
    elif isinstance(raw_data, list):
        # Check if it's a list of rubrics OR a list containing a dict with rubrics
        if len(raw_data) > 0 and isinstance(raw_data[0], dict) and "rubrics" in raw_data[0]:
            # Handle the [{"rubrics": [...]}] case
            candidates = raw_data[0]["rubrics"]
        else:
            # Assume it's just [{}, {}, {}]
            candidates = raw_data

    # Final Validation
    final_list = []
    if isinstance(candidates, list):
        for item in candidates:
            valid_item = validate_rubric_item(item)
            if valid_item:
                final_list.append(valid_item)
                
    return final_list

# ================= MAIN LOGIC =================

def main():
    print(f"🚀 Starting Batch Process (Range: {START_INDEX} -> {END_INDEX})...")
    
    if not os.path.exists(INDIVIDUAL_DIR):
        os.makedirs(INDIVIDUAL_DIR)

    queries_map = load_user_queries(QUERY_FILE_PATH)
    processed_indices = get_processed_indices(OUTPUT_FILE)

    evidence_files = glob.glob(os.path.join(EVIDENCE_DIR, "conversation_*.json"))
    evidence_files.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
    
    print(f"📂 Total Evidence Files: {len(evidence_files)}")
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        
        for ev_file in tqdm(evidence_files, desc="Generating"):
            try:
                filename = os.path.basename(ev_file)
                match = re.search(r'conversation_(\d+)\.json', filename)
                if not match: continue
                idx = int(match.group(1))
                
                if not (START_INDEX <= idx <= END_INDEX): continue 
                if idx in processed_indices: continue

                user_query = queries_map.get(idx)
                if not user_query: continue
                
                with open(ev_file, 'r', encoding='utf-8') as f:
                    evidence_data = json.load(f)
                
                # RUN PIPELINE
                pipeline = RubricPipeline(user_query, evidence_data)
                full_trace = pipeline.execute()
                
                # EXTRACT (Now with flattening logic)
                final_rubrics_list = extract_rubrics_from_trace(full_trace)
                
                # OUTPUT - SUMMARY
                summary_record = {
                    "index": idx,
                    "original_query": user_query,
                    "generated_rubrics": final_rubrics_list
                }
                
                # OUTPUT - FULL TRACE
                full_record = {
                    "index": idx,
                    "original_query": user_query,
                    "generated_rubrics": final_rubrics_list,
                    "raw_pipeline_output": full_trace
                }
                
                f_out.write(json.dumps(summary_record, ensure_ascii=False) + "\n")
                f_out.flush()
                
                ind_file = os.path.join(INDIVIDUAL_DIR, f"rubric_{idx}.json")
                with open(ind_file, 'w', encoding='utf-8') as f_ind:
                    json.dump(full_record, f_ind, indent=2, ensure_ascii=False)
                
                processed_indices.add(idx)
                
            except Exception as e:
                # print(f"❌ Error file {ev_file}: {e}")
                continue

    print(f"\n✅ All Done!")

if __name__ == "__main__":
    main()