"""Core data models for the multi-image safety dataset."""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class HarmCategory(str, Enum):
    """12-class harm taxonomy."""
    CSEA = "CSEA"            # Child Sexual Exploitation & Abuse
    WMD = "WMD"              # Weapons of Mass Destruction / CBRNE
    VIOLENCE = "VIOLENCE"    # Violence & Violent Crime
    SELF_HARM = "SELF_HARM"  # Self-Harm & Suicide
    SEXUAL = "SEXUAL"        # Sexual Crime & Non-Consensual Content
    HATE = "HATE"            # Hate Speech / Discrimination / Harassment
    CRIME = "CRIME"          # Non-Violent Crime & Malicious Activity
    PRIVACY = "PRIVACY"      # Privacy Violation
    MISINFO = "MISINFO"      # Misinformation / Disinformation
    IP = "IP"                # Intellectual Property Violation
    REGULATED = "REGULATED"  # Regulated & Sensitive Content
    ADVICE = "ADVICE"        # Unqualified Professional Advice


class Pattern(str, Enum):
    """Four core harm patterns."""
    A = "A"  # Compositional covert harm: both images benign, combination harmful
    B = "B"  # Direct single-image harm: at least one image inherently harmful
    C = "C"  # Text-induced harm: prompt guides benign images to harmful interpretation
    D = "D"  # Contextualized harm: life scenario + dangerous intent reasoning


class ImageSource(str, Enum):
    """Image acquisition method."""
    GENERATED = "generated"          # T2I generation (SD/FLUX)
    LAION_RETRIEVAL = "laion_retrieval"  # CLIP retrieval from LAION-5B
    EXISTING_DATASET = "existing_dataset"  # From existing safety datasets
    MIXED = "mixed"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SourcePath(int, Enum):
    """Data construction path."""
    PATH1 = 1  # KG concept pairs
    PATH2 = 2  # Prompt decomposition
    PATH3 = 3  # Dataset expansion
    PATH4 = 4  # Scenario construction
    PATH5 = 5  # Embedding pair matching
    PATH6 = 6  # TAG+KG fusion


class SampleMetadata(BaseModel):
    """Metadata for each sample."""
    pattern: Pattern
    source_path: SourcePath
    image_source: ImageSource
    image1_description: str = ""
    image2_description: str = ""
    individual_safety_scores: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    compositional_harm_score: float = 0.0
    covertness_score: int = Field(default=1, ge=1, le=5)
    difficulty: Difficulty = Difficulty.MEDIUM
    reasoning_chain: str = ""
    vlm_dimension: str = ""  # Cross-cutting VLM dimension


class Conversation(BaseModel):
    """A single turn in the conversation."""
    from_: str = Field(alias="from")
    value: str

    class Config:
        populate_by_name = True


class Sample(BaseModel):
    """Complete sample in MIS-compatible format."""
    id: int
    conversations: list[Conversation]
    image: list[str]  # ["images/{id}/image1.png", "images/{id}/image2.png"]
    category: str
    sub_category: str = ""
    metadata: SampleMetadata

    @classmethod
    def create(
        cls,
        sample_id: int,
        text_prompt: str,
        safety_cot_response: str,
        image1_path: str,
        image2_path: str,
        category: HarmCategory,
        sub_category: str,
        metadata: SampleMetadata,
    ) -> "Sample":
        """Factory method to create a sample with proper conversation format."""
        return cls(
            id=sample_id,
            conversations=[
                Conversation(**{"from": "human", "value": f"<image>\n<image>\n{text_prompt}"}),
                Conversation(**{"from": "gpt", "value": safety_cot_response}),
            ],
            image=[image1_path, image2_path],
            category=category.value,
            sub_category=sub_category,
            metadata=metadata,
        )

    def to_mis_format(self) -> dict:
        """Export to MIS-compatible dict (without metadata for training)."""
        return {
            "id": self.id,
            "conversations": [c.model_dump(by_alias=True) for c in self.conversations],
            "image": self.image,
            "category": self.category,
            "sub_category": self.sub_category,
        }

    def to_full_dict(self) -> dict:
        """Export with full metadata for analysis."""
        d = self.to_mis_format()
        d["metadata"] = self.metadata.model_dump()
        return d


class ImagePair(BaseModel):
    """An image pair before being assembled into a full sample."""
    image1_path: str
    image2_path: str
    image1_desc: str = ""
    image2_desc: str = ""
    category: HarmCategory
    sub_category: str = ""
    pattern: Pattern
    source_path: SourcePath
    text_prompt: str = ""
    reasoning: str = ""
    covertness_score: int = 1


class QualityMetrics(BaseModel):
    """Quality metrics for a sample or batch."""
    total_samples: int = 0
    passed_safety_check: int = 0
    passed_composition_verify: int = 0
    passed_dedup: int = 0
    category_distribution: dict[str, int] = Field(default_factory=dict)
    pattern_distribution: dict[str, int] = Field(default_factory=dict)
    difficulty_distribution: dict[str, int] = Field(default_factory=dict)
    avg_covertness_score: float = 0.0
