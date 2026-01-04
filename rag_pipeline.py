import os
import asyncio
from typing import List, Dict, Literal
from dotenv import load_dotenv

# --- Tavily Search ---
from tavily import TavilyClient
import trafilatura

# LangChain 组件
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 引入你原始的数据结构
from data_schema import EvidenceBlock

load_dotenv()

class MedicalRAGPipeline:
    def __init__(self, provider: Literal["openai", "groq", "cerebras"] = "openai", model_name: str = None):
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            raise ValueError("⚠️ 缺少 TAVILY_API_KEY，请在 .env 文件中配置！")
        self.tavily = TavilyClient(api_key=tavily_key)
        
        self.provider = provider
        self.model_name = model_name
        
        # 初始化两个 LLM
        self.smart_llm = self._initialize_llm(type="smart") # 70B or GPT-4o
        self.fast_llm = self._initialize_llm(type="fast")   # 8B or GPT-3.5/4o-mini
        
        print(f"🔧 Pipeline Initialized.")
        print(f"   🧠 Smart Model (Router/Synth): {self.smart_llm.model_name}")
        print(f"   ⚡ Fast Model (Reranker): {self.fast_llm.model_name}")

    def _initialize_llm(self, type="smart"):
        temperature = 0.1
        
        if self.provider == "openai":
            # Smart用GPT-4o, Fast用GPT-4o-mini
            model = (self.model_name or "gpt-4o") if type == "smart" else "gpt-4o-mini"
            return ChatOpenAI(model=model, temperature=temperature)
            
        elif self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            # Smart用 70B, Fast用 8B (Instant)
            # 注意：Groq 的 8B 模型通常叫 llama-3.1-8b-instant
            if type == "smart":
                model = self.model_name or "llama-3.3-70b-versatile"
            else:
                model = "llama-3.1-8b-instant" 
                
            return ChatGroq(model=model, temperature=temperature, api_key=api_key)
            
        elif self.provider == "cerebras":
            api_key = os.getenv("CEREBRAS_API_KEY")
            # Cerebras 目前主要推 70B，8B 也可以用 llama3.1-8b
            model = (self.model_name or "llama-3.3-70b") if type == "smart" else "llama3.1-8b"
            return ChatOpenAI(base_url="https://api.cerebras.ai/v1", api_key=api_key, model=model, temperature=temperature)
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        

    async def step_1_router_and_query(self, user_query: str):
        print(f"--- [Step 1] Routing & Generating Queries... ---")
        
        # 扩展的 Source 列表 + {{ }} 转义修复
        system_prompt = """
        You are an expert Medical Research Assistant. 
        Analyze the user's query and decide which authoritative sources are needed.
        
        **Available Domains:**
        1. **Guidelines**: CDC (site:cdc.gov), WHO (site:who.int), NICE (site:nice.org.uk), Merck Manuals (site:merckmanuals.com)
        2. **Drugs**: Drugs.com (site:drugs.com), BNF (site:bnf.nice.org.uk)
        3. **Patient Ed**: Mayo Clinic (site:mayoclinic.org), Cleveland Clinic (site:clevelandclinic.org), NHS (site:nhs.uk)
        4. **Research**: PubMed (site:ncbi.nlm.nih.gov)
        
        Task:
        1. Identify the Intent.
        2. Generate **3 to 5 specific search queries** combining medical terms with the most relevant sites.
        
        IMPORTANT: Output ONLY valid JSON.
        Example format:
        {{
            "intent": "string",
            "queries": ["query1", "query2", "query3"]
        }}
        """
        
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{query}")])
        chain = prompt | self.smart_llm | JsonOutputParser()
        try:
            return await chain.ainvoke({"query": user_query})
        except Exception as e:
            print(f"   [Router Error] {e}. Using default query.")
            return {"intent": "General", "queries": [user_query]}

    def step_2_search_web(self, queries: List[str], max_results_per_query=2):
        """
        Step 2: Search - 每个 Query 找 2 个 (High Diversity Pool)
        """
        print(f"--- [Step 2] Searching Web with Tavily... ---")
        raw_results = []
        
        for q in queries:
            print(f"   -> Searching: {q}")
            try:
                response = self.tavily.search(query=q, search_depth="basic", max_results=max_results_per_query)
                results = response.get('results', [])
                print(f"      Found {len(results)} results.")
                
                for r in results:
                    raw_results.append({
                        "query": q,
                        "title": r.get('title', 'No Title'),
                        "href": r.get('url', ''),
                        "snippet": r.get('content', '') 
                    })
            except Exception as e:
                print(f"   [Error] Search failed for {q}: {e}")
        
        # 去重
        seen_urls = set()
        unique_results = []
        for r in raw_results:
            if r['href'] and r['href'] not in seen_urls:
                unique_results.append(r)
                seen_urls.add(r['href'])
                
        print(f"   -> Total unique candidates: {len(unique_results)}")
        return unique_results

    async def step_3_rerank_results(self, user_query: str, candidates: List[Dict], top_k=5):
        """
        Step 3: Reranker (带有自动补位机制，防止 8B 模型选不够)
        """
        print(f"--- [Step 3] Reranking Candidates (Top {top_k})... ---")
        
        if len(candidates) <= top_k:
            return candidates

        candidate_text = ""
        for i, item in enumerate(candidates):
            candidate_text += f"[{i}] URL: {item['href']}\nTitle: {item['title']}\nSnippet: {item['snippet']}\n\n"

        system_prompt = """
        You are a Medical Search Ranker. 
        Select the **Top {top_k}** most relevant results for the user's query.
        
        Criteria:
        1. **Authority**: Prioritize Guidelines (CDC, WHO, NICE, Merck) & Patient Sites (Mayo, NHS).
        2. **Relevance**: Look for specific criteria, dosages, or protocols.
        3. **Diversity**: Mix Guidelines and Patient-Friendly sources.

        User Query: {user_query}

        Candidates:
        {candidate_text}

        Output ONLY a JSON list of indices. Example: [0, 4, 2]
        """

        prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
        
        # 使用 Fast LLM (8B)
        chain = prompt | self.fast_llm | JsonOutputParser()
        
        try:
            indices = await chain.ainvoke({
                "user_query": user_query,
                "candidate_text": candidate_text,
                "top_k": top_k
            })
            
            if isinstance(indices, dict): indices = list(indices.values())[0]
            if not isinstance(indices, list): indices = []
            
            # 1. 清洗索引 (去重、去非法值)
            valid_indices = []
            for i in indices:
                if isinstance(i, int) and 0 <= i < len(candidates):
                    if i not in valid_indices:
                        valid_indices.append(i)
            
            print(f"   -> LLM selected {len(valid_indices)} indices: {valid_indices}")

            # 2. 【关键修复】自动补位 (Backfill)
            # 如果 LLM 选的不够 k 个，从原始列表中按顺序补齐
            if len(valid_indices) < top_k:
                print(f"   -> [Auto-Fix] LLM picked fewer than {top_k}. Backfilling from original list...")
                for i in range(len(candidates)):
                    if i not in valid_indices:
                        valid_indices.append(i)
                    if len(valid_indices) == top_k:
                        break
            
            # 3. 截取最终结果
            selected_results = [candidates[i] for i in valid_indices]
            return selected_results
            
        except Exception as e:
            print(f"   [Rerank Error] {e}. Fallback to first {top_k}.")
            return candidates[:top_k]

    async def step_4_scrape_content(self, selected_results: List[Dict]):
        """
        Step 4: Scrape - 只抓取 Rerank 后的 Top 5
        """
        print(f"--- [Step 4] Extracting Content... ---")
        scraped_data = []
        
        for item in selected_results:
            content = f"SOURCE: {item['title']} ({item['href']})\n"
            
            downloaded = trafilatura.fetch_url(item['href'])
            if downloaded:
                full_text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
                if full_text:
                     content += f"FULL CONTENT:\n{full_text[:5000]}\n\n"
                else:
                     content += f"SNIPPET:\n{item['snippet']}\n\n"
            else:
                content += f"SNIPPET:\n{item['snippet']}\n\n"
            
            scraped_data.append(content)
                
        return "\n".join(scraped_data)

    async def step_5_synthesize_evidence(self, original_query: str, intent: str, queries_used: List[str], raw_text: str) -> EvidenceBlock:
        """
        Step 5: Synthesize - 【恢复 V1 逻辑与数据结构】
        """
        print(f"--- [Step 5] Synthesizing Evidence Block... ---")
        
        # 使用你定义的 EvidenceBlock 确保输出格式一模一样
        parser = JsonOutputParser(pydantic_object=EvidenceBlock)
        format_instructions = parser.get_format_instructions()

        # 这里使用 V1 风格的 Prompt，强调 evidence_sources 和 synthesis 字段
        system_prompt = """
        You are a Medical Evidence Evaluator. 
        Your goal is to create a structured "Evidence Block" strictly following the provided JSON schema.
        
        Input Context:
        1. User Query: {query}
        2. Scraped Text from Web: {raw_text}
        
        **Instructions:**
        1. **Check for Conflicts**: Does the text show differences between sources? Record this in the 'synthesis' section.
        2. **Extract Facts (evidence_sources)**: 
           - Populate the 'evidence_sources' list.
           - For each source, extract the 'key_excerpt'.
           - Pull out recommendations, specific numbers in treatment or schedules related to conversation.
           - **Important**: If you find tables (e.g., vaccine schedules), SUMMARIZE them into clear sentences within the 'key_excerpt'. Do not say "See Table".
        3. **Red Flags**: Identify any safety warnings for the 'synthesis' -> 'red_flags' list.
        4. **Source Attribution**: Ensure every entry has a valid URL.
        
        **Output Instruction:**
        - Output ONLY valid JSON.
        - Do not include conversational text.
        
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
        chain = prompt | self.smart_llm | parser
        
        try:
            result = await chain.ainvoke({
                "query": original_query,
                "raw_text": raw_text,
                "format_instructions": format_instructions
            })
            
            # 手动填充元数据
            result['search_queries_used'] = queries_used
            if 'intent_category' not in result:
                result['intent_category'] = intent
            return result
            
        except Exception as e:
            print(f"   [Synthesis Error] {e}. Retrying...")
            # Fallback 保持 V1 结构
            return {
                "intent_category": intent,
                "search_queries_used": queries_used,
                "evidence_sources": [],
                "synthesis": {"consensus": "Error", "contention": str(e), "red_flags": [], "regional_context": None}
            }

    async def run(self, conversation_text: str):
        try:
            # 1. Router
            route_plan = await self.step_1_router_and_query(conversation_text)
            if isinstance(route_plan, list): route_plan = route_plan[0] 
            
            intent = route_plan.get('intent', 'General')
            queries = route_plan.get('queries', [conversation_text])
            
            # 2. Search
            search_results = self.step_2_search_web(queries)
            if not search_results: return {"error": "No results"}
            
            # 3. Rerank (Top 5)
            top_results = await self.step_3_rerank_results(conversation_text, search_results, top_k=5)

            # 4. Scrape
            full_text = await self.step_4_scrape_content(top_results)

            # 5. Synthesize (V1 Format)
            final_json = await self.step_5_synthesize_evidence(conversation_text, intent, queries, full_text)
            return final_json
        except Exception as e:
            print(f"Pipeline Error: {e}")
            return {"error": str(e)}