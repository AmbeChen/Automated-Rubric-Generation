import os
import re
import json
import asyncio
import time
from rag_pipeline import MedicalRAGPipeline

# 输入和输出配置
INPUT_FILE = 'conversations_all.txt'
OUTPUT_DIR = 'evidence'

def parse_conversations(file_path):
    """
    解析 conversations_all.txt 文件。
    处理格式：'0 User: ...' 或 '53 User: ...' 以及多行对话。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    conversations = {}
    current_index = None
    current_buffer = []

    # 正则表达式：匹配行首的数字索引 (忽略 标记)
    # 匹配逻辑：
    # 1. 可选的 
    # 2. 捕获数字 ID
    # 3. 必须紧跟 User:
    pattern = re.compile(r"^(?:\\s*)?(\d+)\s+User:(.*)")

    for line in lines:
        line = line.strip()
        if not line: continue

        match = pattern.match(line)
        if match:
            # 如果之前有正在处理的对话，先保存
            if current_index is not None:
                conversations[current_index] = "\n".join(current_buffer).strip()
            
            # 开始新的一条
            current_index = match.group(1)
            content_start = match.group(2).strip()
            current_buffer = [f"User: {content_start}"]
        else:
            # 如果不是新开头，说明是上一条对话的续行（例如 source 引用或 Assistant 回复）
            if current_index is not None:
                current_buffer.append(line)

    # 保存最后一条
    if current_index is not None:
        conversations[current_index] = "\n".join(current_buffer).strip()

    print(f"✅ 解析完成：共找到 {len(conversations)} 条对话。")
    return conversations

async def process_batch():
    # 1. 初始化 Pipeline (使用 Groq + Tavily)
    print("🚀 初始化 Pipeline (API + Tavily)...")
    # pipeline = MedicalRAGPipeline(provider="groq", model_name="llama-3.3-70b-versatile")
    
    pipeline = MedicalRAGPipeline(provider="cerebras", model_name="llama3.3-70b")
    
    # 2. 解析文件
    conversations = parse_conversations(INPUT_FILE)
    
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 3. 遍历处理
    total = len(conversations)
    processed_count = 0

    # 按 Index 数字排序处理 (0, 1, 2...)
    sorted_ids = sorted(conversations.keys(), key=lambda x: int(x))

    for idx in sorted_ids:
        text = conversations[idx]
        output_filename = os.path.join(OUTPUT_DIR, f"conversation_{idx}.json")

        # [断点续传] 检查文件是否已存在
        if os.path.exists(output_filename):
            print(f"⏭️  [Skipping] Conversation {idx} 已存在。")
            continue

        print(f"\n[{processed_count + 1}/{total}] 正在处理 Conversation {idx}...")
        print(f"📝 内容摘要: {text[:60]}...")

        try:
            # 调用 Pipeline
            start_time = time.time()
            result = await pipeline.run(text)
            duration = time.time() - start_time

            # 保存结果
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 完成 (耗时 {duration:.2f}s) -> 保存至 {output_filename}")

        except Exception as e:
            print(f"❌ 处理 Conversation {idx} 失败: {e}")
            # 记录错误日志，但不中断整个程序
            with open("error_log.txt", "a") as log:
                log.write(f"ID {idx} Failed: {str(e)}\n")
        
        processed_count += 1
        
        # [Rate Limit 保护] 
        # 免费版 Groq 和 Tavily 都有速率限制，建议每处理完一条休息 1-2 秒
        await asyncio.sleep(1.5)

    print("\n🎉 全部任务处理完成！")

if __name__ == "__main__":
    asyncio.run(process_batch())