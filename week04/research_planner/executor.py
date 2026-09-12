from dataclasses import dataclass, field
from typing import Any

import requests

import sys
from pathlib import Path

WEEK3_ROOT = (
    Path(__file__).resolve().parents[2]
    / "week03"
    / "research_agent_v2"
)

sys.path.insert(0, str(WEEK3_ROOT))

from ingestion.prepare import prepare_document
from retrieval.keyword_search import KeywordRetriever
from retrieval.vector_store import VectorStore
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import Reranker

MAX_RETRIES = 2
MAX_STEPS = 10

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

KNOWLEDGE_FILE = (
    "week03/research_agent_v2/knowledge/robotics.txt"
)


@dataclass
class ExecutionState:
    results: dict[int, Any] = field(default_factory=dict)
    status: dict[int, str] = field(default_factory=dict)

    def initialize(self, plan: Plan):

        for step in plan.steps:
            self.status[step.id] = "pending"

    def can_execute(self, step: PlanStep) -> bool:

        return all(
            self.status.get(dep) == "completed"
            for dep in step.depends_on
        )

    def mark_running(self, step_id: int):
        self.status[step_id] = "running"

    def mark_completed(
        self,
        step_id: int,
        result: Any,
    ):
        self.status[step_id] = "completed"
        self.results[step_id] = result

    def mark_failed(
        self,
        step_id: int,
        error: str,
    ):
        self.status[step_id] = "failed"

        self.results[step_id] = {
            "error": error
        }


class Executor:

    def __init__(self):

        documents = prepare_document(
            KNOWLEDGE_FILE
        )

        keyword_retriever = KeywordRetriever(
            documents
        )

        vector_store = VectorStore()

        self.retriever = HybridRetriever(
            vector_store=vector_store,
            keyword_retriever=keyword_retriever,
        )

        self.reranker = Reranker()

    def ask_llm(self, prompt: str) -> str:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()

        return response.json()[
            "message"
        ]["content"]

    def research(
        self,
        task: str,
    ) -> dict:

        candidates = self.retriever.search(
            task,
            top_k=10,
            candidate_k=10,
        )

        results = self.reranker.rerank(
            task,
            candidates,
            top_k=5,
        )

        context = []

        for result in results:

            metadata = result["metadata"]

            source = metadata.get(
                "source",
                "unknown",
            )

            page = metadata.get("page")

            if page:
                citation = (
                    f"{source}, page {page}"
                )
            else:
                citation = source

            context.append(
                f"[Source: {citation}]\n"
                f"{result['text']}"
            )

        return {
            "task": task,
            "context": "\n\n".join(context),
        }

    def execute_step(
        self,
        step: PlanStep,
        state: ExecutionState,
    ) -> Any:

        print(
            f"\n[EXECUTOR] Step {step.id}"
        )

        print(
            f"[EXECUTOR] {step.task}"
        )

        dependencies = {
            dep: state.results[dep]
            for dep in step.depends_on
        }

        if step.tool == "research":

            return self.research(
                step.task
            )

        if step.tool == "analysis":

            research_context = "\n\n".join(
                str(result)
                for result in dependencies.values()
            )

            prompt = f"""
You are an expert research analyst.

Task:
{step.task}

Research results:
{research_context}

Analyze the research results carefully.

Do not invent information.
Clearly distinguish facts from conclusions.
"""

            return self.ask_llm(prompt)

        if step.tool == "final":

            previous_results = "\n\n".join(
                str(result)
                for result in dependencies.values()
            )

            prompt = f"""
You are a research assistant.

Prepare the final answer for the original research task.

Use the following research and analysis:

{previous_results}

Requirements:
- Be accurate.
- Do not invent facts.
- Explain the reasoning clearly.
- Include source references when available.
"""

            return self.ask_llm(prompt)

        raise ValueError(
            f"Unknown tool: {step.tool}"
        )

    def run(
        self,
        plan: Plan,
    ) -> ExecutionState:

        state = ExecutionState()

        state.initialize(plan)

        while True:

            pending_steps = [
                step
                for step in plan.steps
                if state.status[step.id]
                == "pending"
            ]

            if not pending_steps:
                break

            progress = False

            for step in pending_steps:

                if not state.can_execute(step):
                    continue

                progress = True

                state.mark_running(
                    step.id
                )

                try:

                    result = self.execute_step(
                        step,
                        state,
                    )

                    state.mark_completed(
                        step.id,
                        result,
                    )

                except Exception as exc:

                    state.mark_failed(
                        step.id,
                        str(exc),
                    )

                    raise

            if not progress:

                raise RuntimeError(
                    "No executable steps remain. "
                    "Check plan dependencies."
                )

        return state