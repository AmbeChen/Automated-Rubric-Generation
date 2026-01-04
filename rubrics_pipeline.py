import os
import json
import re
from typing import Dict, Any, Literal
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIGURATION =================
DEFAULT_PROVIDER: Literal["groq", "cerebras"] = "cerebras"

# ================= UNIVERSAL CLIENT =================
class UniversalLLMClient:
    def __init__(self, provider: str = DEFAULT_PROVIDER):
        self.provider = provider.lower()
        self.client = None
        self.model = ""
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

    def create_completion(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return resp.choices[0].message.content
        except Exception as e:
            return json.dumps({"error": str(e)})

try:
    llm_client = UniversalLLMClient()
except:
    llm_client = None


def clean_json_output(text: str) -> Dict:
    """Enhanced JSON Parser (Handles Infinity/NaN)"""
    # Preprocessing: Fix illegal JSON values sometimes output by LLMS
    text = text.replace("Infinity", "100") # prevent float('inf') 
    text = text.replace("-Infinity", "-100")
    
    try:
        return json.loads(text)
    except:
        # 1. Extraction of Markdown 
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        
        # 2. Extraction of curly braces
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        
        return {}

# ================= ABSTRACT PIPELINE CLASS =================

class RubricPipeline:
    def __init__(self, query: str, evidence_data: Dict):
        self.query = query
        self.evidence_data = evidence_data
        self.evidence_text = ""
        
        self.debug_trace = {
            "step_1_1_atomic_facts": {},
            "step_1_2_filtered_facts": {},
            "step_2_interaction_intent": {},
            "step_3_draft_rubrics": {},
            "step_4_final_rubrics": {}
        }

    def step_0_preprocess(self):
        """Pre-processing: Clean inputs"""
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
        
        # Prompt
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
        """Track 3: Synthesis 2.0 (Holistic, Goal-Oriented Generation)"""
        
        prompt = """You are a Senior Medical AI Evaluator.
        
        **YOUR GOAL:**
        Design a comprehensive, reliable evaluation rubric (set of criteria) to grade an AI's response to the User Query: "{user_query}".
        
        **INPUT DATA:**
        1. **Medical Evidence**: A list of verified atomic facts (Symptoms, Treatments, Red Flags).
        2. **User Intent**: The user's persona, missing context needs, and required tone.
        
        **CONSOLIDATION STRATEGY (Cluster & Enumerate):**
        - You must **GROUP** related concepts
        - You could summerize but Do NOT miss important information 
        
        **GENERATION STRATEGY (Holistic Coverage):**
        - Do not just check for facts. Think: "What makes a perfect answer?" and "What makes a dangerous answer?"
        - **Maximize Coverage**: Ensure that EVERY relevant aspect of the evidence (Medical Facts, Safety Warnings, Contextual Questions) is converted into a criterion.
        - **Granularity**: If the evidence lists specific items (e.g., specific drugs or symptoms), the rubric MUST require them specifically. Vague criteria are useless.
        - **Safety First**: Any Red Flag or Contraindication in the evidence MUST have a corresponding high-stakes criterion.
        
        **HARD CONSTRAINTS (Format & Axes):**
        1. **Score Range**: Integers from **-10 to 10**.
           - Use high magnitude (±8-10) for Safety/Critical Accuracy.
           - Use medium magnitude (±4-7) for Completeness/Context.
           - Use low magnitude (±1-3) for Minor Details/Tone.
        2. **Allowed Axes** (Assign the most logical one):
           - **accuracy**: Factual correctness and Safety violations (Negative).
           - **completeness**: Coverage of required topics (Positive).
           - **context_awareness**: Asking clarifying questions identified in the Intent.
           - **communication_quality**: Tone, empathy, clarity.
           - **instruction_following**: Formatting or constraints.

        **FORMAT CONSTRAINTS:**
        - Total Rubrics: Aim for **comprehensive coverage**, but try to organize them into **under 20 criteria** by using effective clustering.
        - Output strictly valid JSON.
        
        Example Criterion Style:
        - "Correctly identifies the recommended dosage of **500mg**." (Accuracy, 8)
        - "Mentions all key symptoms: **Fever, Rash, and Nausea**." (Completeness, 7)
        - "Explicitly warns against **alcohol use**." (Completeness, 10)
        """
        
        # Inject queries to enhance context awareness
        prompt = prompt.replace("{user_query}", self.query)
        
        inp = f"""
        [Verified Evidence Facts]: {json.dumps(self.debug_trace['step_1_2_filtered_facts'])}
        [User Interaction Analysis]: {json.dumps(self.debug_trace['step_2_interaction_intent'])}
        """
        
        res = llm_client.create_completion(prompt, inp)
        self.debug_trace["step_3_draft_rubrics"] = clean_json_output(res)

    def step_4_refinement(self):
        """Track 4: Refinement 2.0 (Audit & Supplement Strategy)"""
        
        # An Auditor used for deleting (filtering) and adding (completing) ===
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
        
        # Pass the structured facts of Step 1.2 to Step 4 again
        inp = f"""
        [Source Truth - Facts]: {json.dumps(self.debug_trace['step_1_2_filtered_facts'])}
        [Source Truth - Intent]: {json.dumps(self.debug_trace['step_2_interaction_intent'])}
        [Draft Rubrics]: {json.dumps(self.debug_trace['step_3_draft_rubrics'])}
        """
        
        res = llm_client.create_completion(prompt, inp)
        self.debug_trace["step_4_final_rubrics"] = clean_json_output(res)

    def execute(self):
        if not llm_client: return {"error": "No Client"}
        self.step_0_preprocess()
        if not self.evidence_text.strip(): return {"error": "Empty Evidence"}
        
        self.step_1_1_atomic_extraction()
        self.step_1_2_filtering()
        self.step_2_intent_analysis()
        self.step_3_synthesis()
        self.step_4_refinement()
        return self.debug_trace
