import os
import re
import json
import asyncio
import time
from rag_pipeline import MedicalRAGPipeline

# Input and output configuration
INPUT_FILE = 'conversations_all.txt'
OUTPUT_DIR = 'evidence'

def parse_conversations(file_path):
    """
    parde conversations_all.txt file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    conversations = {}
    current_index = None
    current_buffer = []

    # Regular Expression: Match the numeric index at the beginning of the line (ignore the tag)
    # Matching Logic:
    # 1. Optional
    # 2. Capture the digital ID
    # 3. Must follow the User closely:
    pattern = re.compile(r"^(?:\\s*)?(\d+)\s+User:(.*)")

    for line in lines:
        line = line.strip()
        if not line: continue

        match = pattern.match(line)
        if match:
            # save existing index
            if current_index is not None:
                conversations[current_index] = "\n".join(current_buffer).strip()
            
            # start new
            current_index = match.group(1)
            content_start = match.group(2).strip()
            current_buffer = [f"User: {content_start}"]
        else:
            if current_index is not None:
                current_buffer.append(line)

    # save
    if current_index is not None:
        conversations[current_index] = "\n".join(current_buffer).strip()

    print(f"✅ Complete：Find {len(conversations)} conversations.")
    return conversations

async def process_batch():
    # 1. Initialize Pipeline (use Groq + Tavily)
    print("🚀 Initialize Pipeline (API + Tavily)...")
    # pipeline = MedicalRAGPipeline(provider="groq", model_name="llama-3.3-70b-versatile")
    
    pipeline = MedicalRAGPipeline(provider="cerebras", model_name="llama3.3-70b")
    
    # 2. Parse
    conversations = parse_conversations(INPUT_FILE)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 3. Processing
    total = len(conversations)
    processed_count = 0

    # By Index (0, 1, 2...)
    sorted_ids = sorted(conversations.keys(), key=lambda x: int(x))

    for idx in sorted_ids:
        text = conversations[idx]
        output_filename = os.path.join(OUTPUT_DIR, f"conversation_{idx}.json")

        # [Resume from Breakpoint] Check if the file already exists
        if os.path.exists(output_filename):
            print(f"⏭️  [Skipping] Conversation {idx} exits。")
            continue

        print(f"\n[{processed_count + 1}/{total}] is processing Conversation {idx}...")
        print(f"📝 Summary: {text[:60]}...")

        try:
            #  Pipeline
            start_time = time.time()
            result = await pipeline.run(text)
            duration = time.time() - start_time

            # Save
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Completed! (time-consuming {duration:.2f}s) -> saved to {output_filename}")

        except Exception as e:
            print(f"❌ Conversation {idx} Failed: {e}")
            # Record error logs without interrupting the entire program
            with open("error_log.txt", "a") as log:
                log.write(f"ID {idx} Failed: {str(e)}\n")
        
        processed_count += 1
        
        # [Rate Limit guard] 
        await asyncio.sleep(1.5)

    print("\n🎉 Completed！")

if __name__ == "__main__":
    asyncio.run(process_batch())
