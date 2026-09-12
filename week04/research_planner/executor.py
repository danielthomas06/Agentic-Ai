from dataclasses import dataclass, field
from typing import Any
import json
import requests
import sys
from pathlib import Path

from models import Plan, PlanStep


# ============================================================
# WEEK 3 IMPORTS
# ============================================================

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


# ============================================================
# CONFIGURATION
# ============================================================

MAX_RETRIES = 2
MAX_STEPS = 10

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

KNOWLEDGE_FILE = (
    "week03/research_agent_v2/knowledge/robotics.txt"
)


# ============================================================
# EXECUTION STATE
# ============================================================

@dataclass
class ExecutionState:
    results: dict[int, Any] = field(default_factory=dict)
    status: dict[int, str] = field(default_factory=dict)
    errors: dict[int, str] = field(default_factory=dict)
    attempts: dict[int, int] = field(default_factory=dict)

    def initialize(self, plan: Plan):
        for step in plan.steps:
            self.status[step.id] = "pending"
            self.attempts[step.id] = 0

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
        self.errors[step_id] = error

        self.results[step_id] = {
            "error": error
        }

    def reset_for_retry(self, step_id: int):
        self.status[step_id] = "pending"


# ============================================================
# EXECUTOR
# ============================================================

class Executor:

    def __init__(self):

        print("[EXECUTOR] Loading knowledge base...")

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

        print("[EXECUTOR] Ready.")

    # ========================================================
    # LLM
    # ========================================================

    def ask_llm(
        self,
        prompt: str,
        timeout: int = 180,
    ) -> str:

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
            timeout=timeout,
        )

        response.raise_for_status()

        return response.json()[
            "message"
        ]["content"]

    # ========================================================
    # RESEARCH
    # ========================================================

    def research(
        self,
        task: str,
    ) -> dict:

        print("[RESEARCH] Retrieving documents...")

        candidates = self.retriever.search(
            task,
            top_k=8,
            candidate_k=8,
        )

        results = self.reranker.rerank(
            task,
            candidates,
            top_k=4,
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

    # ========================================================
    # COMPACT DEPENDENCY CONTEXT
    # ========================================================

    def build_dependency_context(
        self,
        step: PlanStep,
        state: ExecutionState,
    ) -> str:

        sections = []

        for dep_id in step.depends_on:

            result = state.results.get(dep_id)

            if result is None:
                continue

            sections.append(
                f"RESULT FROM STEP {dep_id}:\n"
                f"{result}"
            )

        return "\n\n".join(sections)

    # ========================================================
    # STEP EXECUTION
    # ========================================================

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

        # ----------------------------------------------------
        # RESEARCH
        # ----------------------------------------------------

        if step.tool == "research":

            return self.research(
                step.task
            )

        # ----------------------------------------------------
        # ANALYSIS
        # ----------------------------------------------------

        if step.tool == "analysis":

            research_context = (
                self.build_dependency_context(
                    step,
                    state,
                )
            )

            prompt = f"""
You are an expert research analyst.

Task:
{step.task}

Relevant previous results:
{research_context}

Analyze only the information provided.

Requirements:
- Do not invent information.
- Clearly distinguish facts from conclusions.
- Compare the evidence where appropriate.
- Keep the answer concise.
"""

            return self.ask_llm(prompt)

        # ----------------------------------------------------
        # FINAL
        # ----------------------------------------------------

        if step.tool == "final":

            previous_results = (
                self.build_dependency_context(
                    step,
                    state,
                )
            )

            prompt = f"""
You are a research assistant.

Prepare the final answer for this research task:

{step.task}

Use the following previous results:

{previous_results}

Requirements:
- Be accurate.
- Do not invent facts.
- Clearly explain the conclusion.
- Include source references when available.
"""

            return self.ask_llm(prompt)

        raise ValueError(
            f"Unknown tool: {step.tool}"
        )

    # ========================================================
    # RETRY
    # ========================================================

    def execute_with_retry(
        self,
        step: PlanStep,
        state: ExecutionState,
    ) -> Any:

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

            state.attempts[step.id] = attempt

            print(
                f"[EXECUTOR] Attempt "
                f"{attempt}/{MAX_RETRIES}"
            )

            try:

                result = self.execute_step(
                    step,
                    state,
                )

                print(
                    f"[EXECUTOR] Step "
                    f"{step.id} completed."
                )

                return result

            except Exception as exc:

                error = str(exc)

                print(
                    f"[EXECUTOR] Step "
                    f"{step.id} failed: {error}"
                )

                state.errors[step.id] = error

                if attempt < MAX_RETRIES:

                    print(
                        "[EXECUTOR] Retrying..."
                    )

                else:

                    print(
                        "[EXECUTOR] Retries exhausted."
                    )

        raise RuntimeError(
            f"Step {step.id} failed after "
            f"{MAX_RETRIES} attempts."
        )

    # ========================================================
    # REPLANNING
    # ========================================================

    def replan_step(
        self,
        step: PlanStep,
        state: ExecutionState,
    ) -> PlanStep:

        error = state.errors.get(
            step.id,
            "Unknown failure",
        )

        prompt = f"""
You are a recovery planning agent.

A research agent attempted the following task:

Task:
{step.task}

Tool:
{step.tool}

It failed with:

{error}

Create ONE replacement task that can accomplish
the same objective using a different or simpler approach.

Rules:
- Keep the same step ID.
- Do not repeat the exact failed approach.
- Prefer the research tool for information retrieval.
- Prefer analysis for comparing existing results.
- Prefer final for final synthesis.
- Preserve the original dependencies.
- Return ONLY valid JSON.

JSON format:

{{
    "id": {step.id},
    "task": "replacement task",
    "tool": "{step.tool}",
    "depends_on": {json.dumps(step.depends_on)}
}}
"""

        print(
            f"[REPLANNER] Replanning step {step.id}..."
        )

        response = self.ask_llm(
            prompt,
            timeout=180,
        )

        data = json.loads(response)

        replacement = PlanStep.model_validate(
            data
        )

        print(
            f"[REPLANNER] Replacement task:"
        )

        print(
            f"[REPLANNER] {replacement.task}"
        )

        return replacement

    # ========================================================
    # MAIN EXECUTION LOOP
    # ========================================================

    def run(
        self,
        plan: Plan,
    ) -> ExecutionState:

        state = ExecutionState()

        state.initialize(plan)

        execution_count = 0

        while True:

            execution_count += 1

            if execution_count > MAX_STEPS:
                raise RuntimeError(
                    "Maximum execution steps exceeded."
                )

            pending_steps = [
                step
                for step in plan.steps
                if state.status.get(step.id)
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

                    result = (
                        self.execute_with_retry(
                            step,
                            state,
                        )
                    )

                    state.mark_completed(
                        step.id,
                        result,
                    )

                except Exception as exc:

                    # ----------------------------------------
                    # FAILED AFTER RETRIES
                    # ----------------------------------------

                    state.mark_failed(
                        step.id,
                        str(exc),
                    )

                    print(
                        f"[EXECUTOR] "
                        f"Step {step.id} requires replanning."
                    )

                    # ----------------------------------------
                    # REPLAN
                    # ----------------------------------------

                    try:

                        replacement = (
                            self.replan_step(
                                step,
                                state,
                            )
                        )

                        # Replace the failed task
                        # with the new task.
                        step.task = replacement.task
                        step.tool = replacement.tool
                        step.depends_on = (
                            replacement.depends_on
                        )

                        state.reset_for_retry(
                            step.id
                        )

                        print(
                            f"[EXECUTOR] "
                            f"Step {step.id} replanned."
                        )

                    except Exception as replan_error:

                        state.mark_failed(
                            step.id,
                            f"Replanning failed: "
                            f"{replan_error}",
                        )

                        print(
                            "[EXECUTOR] "
                            "Replanning failed."
                        )

                        raise RuntimeError(
                            f"Step {step.id} could "
                            f"not be recovered."
                        )

            if not progress:

                unresolved = [
                    step.id
                    for step in plan.steps
                    if state.status.get(step.id)
                    == "pending"
                ]

                raise RuntimeError(
                    "No executable steps remain. "
                    f"Unresolved steps: {unresolved}. "
                    "Check plan dependencies."
                )

        return state