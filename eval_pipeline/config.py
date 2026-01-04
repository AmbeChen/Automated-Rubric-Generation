import os

# ================= API CONFIG =================

# 1. 确定供应商 (默认为 cerebras)
API_PROVIDER = os.getenv("LLM_PROVIDER", "cerebras").lower()

# 2. 根据供应商获取对应的密钥 (这样就不会混淆了)
if API_PROVIDER == "groq":
    # 优先找 GROQ_API_KEY，找不到再找 OPENAI_API_KEY
    API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    BASE_URL = "https://api.groq.com/openai/v1"
elif API_PROVIDER == "cerebras":
    # 优先找 CEREBRAS_API_KEY
    API_KEY = os.getenv("CEREBRAS_API_KEY") or os.getenv("OPENAI_API_KEY")
    BASE_URL = "https://api.cerebras.ai/v1"
else:
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = "https://api.openai.com/v1"

# 导出给 judge_utils.py 使用
OPENAI_API_KEY = API_KEY 

# 3. 模型配置
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3.1-8b")

# ================= 其余配置保持不变 =================
TIMEOUT = 60.0
TEMPERATURE = 0.2
N_REPEAT = 3          
DO_ORDER_SWAP = True  
START_INDEX = int(os.getenv("START_INDEX", 0))
END_INDEX = int(os.getenv("END_INDEX", 10**9))
OUTPUT_DIR = "outputs"
CACHE_DIR = f"{OUTPUT_DIR}/judge_cache"
DEBUG_DIR = f"{OUTPUT_DIR}/debug_traces"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# 调试用：确保运行时知道自己在用哪个 Key (运行时删掉或注释掉这行)
# print(f"DEBUG: Using {API_PROVIDER} with URL {BASE_URL} and Key starting with {str(API_KEY)[:5]}")