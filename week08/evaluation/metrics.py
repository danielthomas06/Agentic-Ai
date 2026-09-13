from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationMetrics:
    test_name: str
    success: bool
    latency_seconds: float
    iterations: int
    agents_executed: list[str]
    routing_correct: bool
    answer_length: int

    required_concepts: list[str]
    concepts_found: list[str]
    quality_score: float

    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "success": self.success,
            "latency_seconds": round(
                self.latency_seconds,
                3,
            ),
            "iterations": self.iterations,
            "agents_executed": self.agents_executed,
            "routing_correct": self.routing_correct,
            "answer_length": self.answer_length,
            "required_concepts": self.required_concepts,
            "concepts_found": self.concepts_found,
            "quality_score": self.quality_score,
            "error": self.error,
        }