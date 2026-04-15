import os
import json
import re
import time
from typing import Dict, Any, Literal, List
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIGURATION =================
DEFAULT_PROVIDER: Literal["groq", "cerebras"] = "groq"


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(round(len(text) / 4)))


def _usage_get(usage_obj, key: str):
    if usage_obj is None:
        return None
    if isinstance(usage_obj, dict):
        return usage_obj.get(key)
    return getattr(usage_obj, key, None)

# ================= UNIVERSAL CLIENT (保持不变) =================
class UniversalLLMClient:
    def __init__(self, provider: str = DEFAULT_PROVIDER):
        self.provider = provider.lower()
        self.client = None
        self.model = ""
        self.last_call_meta = {}
        self.call_history = []
        try:
            if self.provider == "groq":
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                self.client = Groq(api_key=api_key)
                self.model = "llama-3.3-70b-versatile"
            elif self.provider == "cerebras":
                from cerebras.cloud.sdk import Cerebras
                api_key = os.getenv("CEREBRAS_API_KEY")
                self.client = Cerebras(api_key=api_key)
                self.model = "llama3.3-70b"
        except Exception as e:
            print(f"Client Init Error: {e}")

    def reset_metrics(self):
        self.last_call_meta = {}
        self.call_history = []

    def create_completion(self, system_prompt: str, user_prompt: str) -> str:
        start = time.perf_counter()
        est_prompt_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
        try:
            resp = self.client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            usage = getattr(resp, "usage", None)
            prompt_tokens = _usage_get(usage, "prompt_tokens")
            completion_tokens = _usage_get(usage, "completion_tokens")
            total_tokens = _usage_get(usage, "total_tokens")
            if prompt_tokens is None:
                prompt_tokens = est_prompt_tokens
            if completion_tokens is None:
                completion_tokens = _estimate_tokens(content or "")
            if total_tokens is None:
                total_tokens = prompt_tokens + completion_tokens
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            meta = {
                "provider": self.provider,
                "model": self.model,
                "latency_ms": elapsed_ms,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(total_tokens),
                "usage_source": "api" if usage is not None else "estimated",
                "error": None,
            }
            self.last_call_meta = meta
            self.call_history.append(meta)
            return content
        except Exception as e:
            err_text = str(e)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            meta = {
                "provider": self.provider,
                "model": self.model,
                "latency_ms": elapsed_ms,
                "prompt_tokens": int(est_prompt_tokens),
                "completion_tokens": 0,
                "total_tokens": int(est_prompt_tokens),
                "usage_source": "estimated",
                "error": err_text,
            }
            self.last_call_meta = meta
            self.call_history.append(meta)
            return json.dumps({"error": err_text})

try:
    llm_client = UniversalLLMClient()
except:
    llm_client = None


def clean_json_output(text: str) -> Dict:
    text = text.replace("Infinity", "100").replace("-Infinity", "-100")
    try:
        return json.loads(text)
    except:
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        return {}

# ================= MODIFIED PIPELINE FOR ABLATION =================

class RubricPipeline:
    def __init__(self, query: str, evidence_data: Dict, ablation_mode: str = "full"):
        """
        ablation_mode options:
        - "full": Complete process
        - "no_router": No intelligent routing (caller passes evidence_no_router data, pipeline runs normally)
        - "no_atomic": Skip atomic fact decomposition (1.1/1.2), synthesis uses Raw web retrieval results + intent
        - "no_intent": Skip intent analysis (Step 2), synthesis only inputs atomic extraction results
        - "no_audit" / "no_refinement": Skip Step 4 audit/refinement
        """
        self.query = query
        self.evidence_data = evidence_data
        self.ablation_mode = ablation_mode
        self.evidence_text = ""
        
        self.debug_trace = {
            "step_1_1_atomic_facts": {},
            "step_1_2_filtered_facts": {},
            "step_2_interaction_intent": {},
            "step_3_draft_rubrics": {},
            "step_4_final_rubrics": {},
            "metrics": {
                "steps": {},
                "llm_calls": []
            }
        }

    def _run_step(self, step_name: str, fn):
        calls_before = len(getattr(llm_client, "call_history", [])) if llm_client else 0
        t0 = time.perf_counter()
        fn()
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        calls_after = len(getattr(llm_client, "call_history", [])) if llm_client else calls_before
        new_calls = []
        if llm_client and hasattr(llm_client, "call_history") and calls_after > calls_before:
            new_calls = llm_client.call_history[calls_before:calls_after]

        prompt_tokens = sum(int(c.get("prompt_tokens", 0)) for c in new_calls if isinstance(c, dict))
        completion_tokens = sum(int(c.get("completion_tokens", 0)) for c in new_calls if isinstance(c, dict))
        total_tokens = sum(int(c.get("total_tokens", 0)) for c in new_calls if isinstance(c, dict))
        errors = [c.get("error") for c in new_calls if isinstance(c, dict) and c.get("error")]

        self.debug_trace["metrics"]["steps"][step_name] = {
            "latency_ms": elapsed_ms,
            "llm_calls": len(new_calls),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "errors": errors,
        }
        for c in new_calls:
            cc = dict(c)
            cc["step"] = step_name
            self.debug_trace["metrics"]["llm_calls"].append(cc)

    def step_0_preprocess(self):
        synthesis = self.evidence_data.get("synthesis", {})
        sources = self.evidence_data.get("evidence_sources", [])
        
        parts = []
        if synthesis.get("consensus"):
            parts.append(f"Primary Consensus: {synthesis['consensus']}")
        if synthesis.get("red_flags"):
            parts.append(f"Critical Warnings: {json.dumps(synthesis['red_flags'])}")
        
        for idx, src in enumerate(sources):
            excerpt = src.get("key_excerpt")
            if excerpt:
                parts.append(f"Source {idx+1} Data: {excerpt}")
        
        self.evidence_text = "\n".join(parts)

    def step_1_1_atomic_extraction(self):
        """Track 1: Atomic Extraction (Abstract Definition)"""
        
        prompt = """You are a Medical Data Analyst. 
        Task: Decompose the provided text into a comprehensive list of Atomic Facts.
        
        **DEFINITION OF 'ATOMIC FACT' (EXTRACT ALL CATEGORIES):**
        1. **Qualitative Statements**: Definitions, descriptions, mechanisms, characteristics, or procedural steps.
        2. **Quantitative Data**: Specific numbers, measurements, timeframes, dosages, or frequencies.
        3. **Conditional Logic**: "If X, then Y" statements or dependency rules.
        
        **Instructions:**
        - Do not omit or miss information; extract raw information segments.
        - Deconstruct complex sentences into single, standalone premises.
        
        Output JSON:
        {
            "positive_atomic_facts": ["Fact statement 1", "Fact statement 2"...],
            "negative_constraints": ["Explicit prohibitions", "Contraindications"...],
            "safety_red_flags": ["Emergency warnings", "Critical alerts"...]
        }"""
        
        res = llm_client.create_completion(prompt, f"Medical Evidence:\n{self.evidence_text}")
        self.debug_trace["step_1_1_atomic_facts"] = clean_json_output(res)

    def step_1_2_filtering(self):
        """Track 1: Relevance Filtering (Abstract Semantic Matching)"""
        
        prompt = """You are a Medical Context Filter. 
        Task: Filter the Atomic Facts to retain only those RELEVANT to the User Query.
        
        **FILTERING LOGIC:**
        1. **Direct Alignment**: Keep facts that directly address the user's specific questions or stated symptoms.
        2. **Contextual Necessity**: Keep definitions or background info that is necessary to understand the answer.
        3. **Semantic Relevance**: Discard facts that pertain to completely different medical conditions, demographics, or treatments NOT implied by the User Query.
        4. **Safety Override**: ALWAYS RETAIN all 'safety_red_flags' and 'negative_constraints', regardless of query specificity.
        
        Output JSON: {"relevant_positive_facts": [], "relevant_negative_constraints": [], "relevant_red_flags": []}"""
        
        inp = f"User Query: {self.query}\nRaw Facts: {json.dumps(self.debug_trace['step_1_1_atomic_facts'])}"
        res = llm_client.create_completion(prompt, inp)
        self.debug_trace["step_1_2_filtered_facts"] = clean_json_output(res)

    def step_2_intent_analysis(self):
        """Track 2: Intent Analysis (Abstract Requirements)"""
        
        prompt = """Medical Interaction Analyst. Analyze User Query for implicit requirements.
        
        1. **User Persona**: Identify the user's likely knowledge level and emotional state based on their phrasing.
        2. **Missing Context**: Identify specific variables (e.g., demographics, history, severity) that are medically necessary to provide a safe answer but are absent in the query.
        3. **Tone**: Determine the appropriate communication style.
        
        Output JSON: {"user_persona": "...", "missing_context_questions": ["Question 1", "Question 2"], "tone": "..."}"""
        
        res = llm_client.create_completion(prompt, f"Query: {self.query}\nEvidence Context: {self.evidence_text}")
        self.debug_trace["step_2_interaction_intent"] = clean_json_output(res)

    def step_3_synthesis(self):
        """Track 3: Synthesis (Holistic, Goal-Oriented Generation)
        
        Input varies by ablation_mode:
        - full: filtered_facts + intent
        - no_atomic: raw evidence_text + intent (无 atomic facts)
        - no_intent: filtered_facts only (no intent)
        """
        if self.ablation_mode == "no_atomic":
            # 使用 Raw 网页检索结果 + intent，无 atomic facts
            prompt = """You are a Senior Medical AI Evaluator.
        
        **YOUR GOAL:**
        Design a comprehensive, reliable evaluation rubric (set of criteria) to grade an AI's response to the User Query: "{user_query}".
        
        **INPUT DATA:**
        1. **Raw Web Retrieval Results**: Raw text from web search (consensus, excerpts). Extract key medical facts, treatments, safety warnings.
        2. **User Intent**: The user's persona, missing context needs, and required tone.
        
        **GENERATION STRATEGY:**
        - Identify: symptoms, treatments, dosages, red flags, contraindications from the raw text.
        - Convert each critical piece into a rubric criterion.
        - **Safety First**: Any safety warning MUST have a corresponding high-stakes criterion.
        
        **HARD CONSTRAINTS (Format & Axes):**
        1. **Score Range**: Integers from **-10 to 10**.
        2. **Allowed Axes**: accuracy, completeness, context_awareness, communication_quality, instruction_following
        3. Output strictly valid JSON: {"rubrics": [{"criterion": "...", "axis": "...", "points": ...}, ...]}
        """
            inp = f"""
        [Raw Web Retrieval Evidence]:\n{self.evidence_text}

        [User Interaction Analysis]: {json.dumps(self.debug_trace['step_2_interaction_intent'])}
        """
        else:
            prompt = """You are a Senior Medical AI Evaluator.
        
        **YOUR GOAL:**
        Design a comprehensive, reliable evaluation rubric (set of criteria) to grade an AI's response to the User Query: "{user_query}".
        
        **INPUT DATA:**
        1. **Medical Evidence**: A list of verified atomic facts (Symptoms, Treatments, Red Flags).
        2. **User Intent** (if provided): The user's persona, missing context needs, and required tone.
        
        **CONSOLIDATION STRATEGY (Cluster & Enumerate):**
        - You must **GROUP** related concepts
        - You could summerize but Do NOT miss important information 
        
        **GENERATION STRATEGY (Holistic Coverage):**
        - Do not just check for facts. Think: "What makes a perfect answer?" and "What makes a dangerous answer?"
        - **Maximize Coverage**: Ensure that EVERY relevant aspect of the evidence is converted into a criterion.
        - **Granularity**: If the evidence lists specific items, the rubric MUST require them specifically.
        - **Safety First**: Any Red Flag or Contraindication MUST have a corresponding high-stakes criterion.
        
        **HARD CONSTRAINTS (Format & Axes):**
        1. **Score Range**: Integers from **-10 to 10**.
           - Use high magnitude (±8-10) for Safety/Critical Accuracy.
           - Use medium magnitude (±4-7) for Completeness/Context.
           - Use low magnitude (±1-3) for Minor Details/Tone.
        2. **Allowed Axes**: accuracy, completeness, context_awareness, communication_quality, instruction_following
        3. Output strictly valid JSON: {"rubrics": [{"criterion": "...", "axis": "...", "points": ...}, ...]}
        
        Example: "Correctly identifies the recommended dosage of **500mg**." (Accuracy, 8)
        """
            if self.ablation_mode == "no_intent":
                inp = f"""
        [Verified Evidence Facts]: {json.dumps(self.debug_trace['step_1_2_filtered_facts'])}
        [User Intent]: (Skipped in ablation - not provided)
        """
            else:
                inp = f"""
        [Verified Evidence Facts]: {json.dumps(self.debug_trace['step_1_2_filtered_facts'])}
        [User Interaction Analysis]: {json.dumps(self.debug_trace['step_2_interaction_intent'])}
        """
        
        prompt = prompt.replace("{user_query}", self.query)
        res = llm_client.create_completion(prompt, inp)
        self.debug_trace["step_3_draft_rubrics"] = clean_json_output(res)

    def step_4_refinement(self):
        """Track 4: Refinement (Audit & Supplement Strategy)"""
        
        prompt = """You are a Senior Medical Lead Auditor.
        Your task is to REVIEW the Draft Rubrics and **FILL ANY GAPS** by supplementing, filtering, merging to Finalize a **Complete, Reliable, and Concise** evaluation rubric set to grade an AI's medical response.
        
        **INPUTS:**
        1. **User Query**: "{user_query}"
        2. **Source Truth**: The list of Filtered Atomic Facts & Identified User Intent.
        3. **Draft Rubrics**: The current set of criteria generated.

        **YOUR AUDIT PROCEDURE:**

        **PHASE 1: GAP ANALYSIS & SUPPLEMENTATION (CRITICAL)**
        - **Scan the Source Truth**: Look at every Symptom, Treatment, Red Flag, and Context Question listed.
        - **Check Coverage**: Does the Draft Rubrics list cover this item?
        - **ACTION**: If a key fact (e.g., a specific drug name, a specific symptom, or a Safety Warning) is **MISSING** in the Draft, you MUST **GENERATE A NEW CRITERION** for it and add it to the list.
        - *Rule*: It is better to have an extra criterion than to miss a critical medical fact.

        **PHASE 2: QUALITY CONTROL (Filtering)**
        - **Relevance**: Remove criteria that do not answer the User Query.
        - **Hallucination**: Remove criteria NOT supported by the Source Truth.
        - **Axis Compliance**: Ensure strictly 5 axes: [accuracy, completeness, context_awareness, communication_quality, instruction_following].
        - **Negative Check**: Ensure at least one Negative Criterion (penalty) exists.
        
        **PHASE 3: SMART CONSOLIDATION (Merging)*
        - **Directive**: Identify fragmented criteria that validate the same concept (e.g., lists of symptoms, treatments, or questions) and **MERGE** them into a single composite criterion.
        - **Constraint**: The final output must contain **NO MORE THAN 20 CRITERIA**.
        - **Action**: If the list is too long, you must **MERGE** related criteria.
        - **Rule**: When merging, you must **retain the specific keywords** (entities/numbers) in the new description to maintain evaluative rigor.
        - **Exception**: Do **NOT** merge distinct Safety Red Flags (keep them separate for visibility) or distinct Negative Constraints.

        **OUTPUT FORMAT RULES:**
        - **JSON ONLY**: Return a single JSON object. Do not output lists like `[...]` at the top level.
        - **Structure**: `{"rubrics": [{"criterion": "...", "axis": "...", "points": ...}, ...]}`
        """
        
        prompt = prompt.replace("{user_query}", self.query)
        
        # Pass appropriate Source Truth based on ablation_mode
        if self.ablation_mode == "no_atomic":
            # no_atomic: use raw evidence as "facts", keep intent
            facts_block = self.evidence_text
            intent_block = json.dumps(self.debug_trace['step_2_interaction_intent'])
        elif self.ablation_mode == "no_intent":
            facts_block = json.dumps(self.debug_trace['step_1_2_filtered_facts'])
            intent_block = json.dumps({"user_persona": "(skipped)", "missing_context_questions": [], "tone": "professional"})
        else:
            facts_block = json.dumps(self.debug_trace['step_1_2_filtered_facts'])
            intent_block = json.dumps(self.debug_trace['step_2_interaction_intent'])
        
        inp = f"""
        [Source Truth - Facts]: {facts_block}
        [Source Truth - Intent]: {intent_block}
        [Draft Rubrics]: {json.dumps(self.debug_trace['step_3_draft_rubrics'])}
        """
        
        res = llm_client.create_completion(prompt, inp)
        self.debug_trace["step_4_final_rubrics"] = clean_json_output(res)

    def execute(self):
        if not llm_client: return {"error": "No Client"}
        if hasattr(llm_client, "reset_metrics"):
            llm_client.reset_metrics()

        self._run_step("step_0_preprocess", self.step_0_preprocess)
        if not self.evidence_text.strip(): 
            # If it's w/o RAG and no simulated data is passed, it will error here, so the external caller must ensure evidence_data is not empty
            pass
        
        # no_router: Only data source differs (evidence_no_router), process remains the same
        skip_refinement = self.ablation_mode in ("no_audit", "no_refinement")
        
        if self.ablation_mode == "no_atomic":
            # 跳过 atomic extraction + filtering；保留 intent；synthesis 用 raw evidence + intent
            self._run_step("step_2_intent_analysis", self.step_2_intent_analysis)
            self._run_step("step_3_synthesis", self.step_3_synthesis)
        else:
            self._run_step("step_1_1_atomic_extraction", self.step_1_1_atomic_extraction)
            self._run_step("step_1_2_filtering", self.step_1_2_filtering)
            if self.ablation_mode != "no_intent":
                self._run_step("step_2_intent_analysis", self.step_2_intent_analysis)
            else:
                self.debug_trace["step_2_interaction_intent"] = {}
                self.debug_trace["metrics"]["steps"]["step_2_intent_analysis"] = {
                    "latency_ms": 0.0,
                    "llm_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "errors": [],
                    "skipped": True,
                }
            self._run_step("step_3_synthesis", self.step_3_synthesis)
        
        if skip_refinement:
            # 用 step3 的输出作为最终结果
            self.debug_trace["step_4_final_rubrics"] = self.debug_trace.get("step_3_draft_rubrics", {})
            self.debug_trace["metrics"]["steps"]["step_4_refinement"] = {
                "latency_ms": 0.0,
                "llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "errors": [],
                "skipped": True,
            }
        else:
            self._run_step("step_4_refinement", self.step_4_refinement)
        
        return self.debug_trace
