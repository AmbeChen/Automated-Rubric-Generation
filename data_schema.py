from typing import List, Optional
from pydantic import BaseModel, Field

class SourceEntry(BaseModel):
    source_name: str = Field(..., description="Name of the source (e.g., CDC, WHO, PubMed)")
    url: str = Field(..., description="URL of the source")
    relevance: str = Field(..., description="High/Medium/Low")
    key_excerpt: str = Field(..., description="The specific text snippet relevant to the query")
    conflict_note: Optional[str] = Field(None, description="Does this source conflict with others?")

class SynthesizedEvidence(BaseModel):
    consensus: str = Field(..., description="What do most sources agree on?")
    contention: Optional[str] = Field(None, description="Where do sources disagree?")
    red_flags: List[str] = Field(default_factory=list, description="Safety warnings or immediate referral criteria")
    regional_context: Optional[str] = Field(None, description="Context specific to location (e.g., Kenya vs USA)")

class EvidenceBlock(BaseModel):
    """
    Final Output Object to be passed to Rubric Generator
    """
    intent_category: str = Field(..., description="Category: Guideline, Safety, or Fact-Check")
    search_queries_used: List[str]
    evidence_sources: List[SourceEntry]
    synthesis: SynthesizedEvidence