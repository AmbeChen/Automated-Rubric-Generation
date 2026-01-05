import os
from openai import OpenAI

def get_client(provider="groq"):
    """
    provider: 'groq' or 'cerebras'
    """
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        base_url = "https://api.groq.com/openai/v1"
    elif provider == "cerebras":
        api_key = os.getenv("CEREBRAS_API_KEY")
        base_url = "https://api.cerebras.ai/v1"
    else:
        raise ValueError("Unknown provider")

    if not api_key:
        raise RuntimeError(f"{provider.upper()}_API_KEY not set")

    return OpenAI(api_key=api_key, base_url=base_url)
