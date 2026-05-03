"""
Memory Retrieval Module for AMA-Agent

Multi-stage retrieval pipeline (Fig. 8 (B)):
  Stage 1 — Async similarity-based top-k node retrieval via embedding engine.
  Stage 2 — LLM sufficiency judgment (CHUNK_SUFFICIENCY_JUDGMENT_PROMPT_TEMPLATE):
             SUFFICIENT  → return immediately.
             NEED_GRAPH  → parse the spec, retrieve adjacent/range/individual turns.
             NEED_CODE   → generate and execute a Python search script.
  Stage 3 — Synthesize all gathered evidence into a final context string.
"""

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

from .prompt import CHUNK_SUFFICIENCY_JUDGMENT_PROMPT_TEMPLATE
from .utils import (
    _extract_chunks,
    _format_chunks,
    _retrieve_graph_turns,
    _run_keyword_search,
    _similarity_retrieve,
    truncate_trajectory_text,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════


def memory_retrieve(
    memory: Dict[str, Any],
    question: str,
    call_llm_func: Callable,
    top_k: int = 5,
    embed_engine: Optional[Any] = None,
    max_context_length: int = 114688,
) -> str:
    """
    Retrieve relevant context from memory for answering a question.

    Pipeline:
      1. Async similarity retrieval → top_k seed chunks stored in ``mem``.
      2. CHUNK_SUFFICIENCY_JUDGMENT_PROMPT_TEMPLATE → SUFFICIENT / NEED_GRAPH / NEED_CODE.
         - SUFFICIENT : return immediately.
         - NEED_CODE  : generate & execute a Python search script via
                        CODE_GENERATION_PROMPT_TEMPLATE.
         - NEED_GRAPH : parse the adjacency / range / index spec from the
                        response and extend ``mem`` with the requested turns.

    Args:
        memory:         Memory dict with keys state_mem, causal_graph, text_mem, embed_mem.
        question:       Question to answer.
        call_llm_func:  Synchronous LLM call: (prompt: str) -> (_, response: str).
        top_k:          Number of top chunks for similarity retrieval (default: 5).
        embed_engine:   EmbeddingEngine whose base_url/port identifies the server;
                        falls back to BM25 when None.

    Returns:
        Context string ready to prepend to the answering prompt.
    """
    state_mem = memory.get("state_mem", "")
    text_mem = memory.get("text_mem", {})
    embed_mem = memory.get("embed_mem")

    state_mem_str = str(state_mem) if state_mem else ""
    task = text_mem.get("task", "")
    trajectory_data = text_mem.get("trajectory_data", {})
    trajectory = trajectory_data.get("trajectory", [])

    # ── Stage 1: Similarity retrieval (top_k) ────────────────────────────────────────
    # embed_engine is a synchronous callable — call it directly, no event loop.
    seed_indices = _similarity_retrieve(
        question, trajectory, embed_engine, embed_mem, top_k
    )

    # Internal accumulated memory — extended as additional turns are retrieved.
    mem: List[Dict[str, Any]] = _extract_chunks(trajectory, seed_indices)
    # Track seen turns from Stage 1 onwards to prevent duplicates across all stages.
    existing_turns: set = {c["turn"] for c in mem}

    # ── Stage 2: Sufficiency judgment loop (up to 3 iterations) ──────────────
    # Each NEED_GRAPH response extends ``mem`` with the requested turns and the
    # judgment is re-run with the enriched context.  NEED_CODE or exhausting
    # all iterations breaks out of the loop.
    _MAX_ITERS = 3
    _TIMEOUT = 60.0  # seconds
    extra_evidence: str = ""
    sufficiency_response: str = ""
    _need_code = False
    _deadline = time.monotonic() + _TIMEOUT

    for _iter in range(_MAX_ITERS):
        if time.monotonic() >= _deadline:
            break
        _chunks_str = _format_chunks(mem)
        # Guard against oversized sufficiency prompts: cap the chunks portion.
        _overhead = len(question) + 4096  # template boilerplate + answer budget
        _chunks_budget = max(4096, max_context_length - _overhead)
        if len(_chunks_str) > _chunks_budget:
            _chunks_str = truncate_trajectory_text(_chunks_str, _chunks_budget)
        sufficiency_prompt = CHUNK_SUFFICIENCY_JUDGMENT_PROMPT_TEMPLATE.format(
            query=question,
            retrieved_chunks=_chunks_str,
        )
        _, sufficiency_response = call_llm_func(sufficiency_prompt)
        resp_upper = (sufficiency_response or "").upper()

        # SUFFICIENT — can answer immediately.
        if "SUFFICIENT" in resp_upper and "NEED_" not in resp_upper:
            return _synthesize(
                state_mem_str=state_mem_str,
                task=task,
                chunks=mem,
                extra_evidence="",
                max_context_length=max_context_length,
            )

        if "NEED_CODE" in resp_upper:
            _need_code = True
            break

        # NEED_GRAPH — retrieve the requested turns and loop.
        new_chunks = _retrieve_graph_turns(trajectory, sufficiency_response or "")
        new_unique = [c for c in new_chunks if c["turn"] not in existing_turns]
        if not new_unique:
            break  # Nothing new to add; stop looping.
        mem.extend(new_unique)
        existing_turns.update(c["turn"] for c in new_unique)

    if _need_code:
        # ── Stage 3a: Code-generation search with retry (up to 3 attempts) ───
        # Each failed attempt feeds its error back to the next code generation.
        _prev_error = ""
        _code_deadline = time.monotonic() + _TIMEOUT
        for _ in range(_MAX_ITERS):
            if time.monotonic() >= _code_deadline:
                break
            _remaining = max(1.0, _code_deadline - time.monotonic())
            extra_evidence = _run_keyword_search(
                trajectory_data=trajectory_data,
                question=question,
                task=task,
                call_llm_func=call_llm_func,
                previous_error=_prev_error,
                timeout=_remaining,
            )
            if not (
                extra_evidence.startswith("error:") or "Traceback" in extra_evidence
            ):
                break
            _prev_error = extra_evidence  # feed error to next iteration

    # ── Stage 4: Synthesize all evidence ─────────────────────────────────────
    return _synthesize(
        state_mem_str=state_mem_str,
        task=task,
        chunks=mem,
        extra_evidence=extra_evidence,
        max_context_length=max_context_length,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Context / output builders
# ═══════════════════════════════════════════════════════════════════════════════


def _synthesize(
    state_mem_str: str,
    task: str,
    chunks: List[Dict[str, Any]],
    extra_evidence: str,
    max_context_length: int = 114688,
) -> str:
    """Stage 4: assemble all gathered evidence into the final context string."""
    all_chunks = list(chunks) + _extract_chunks_from_extra_evidence(extra_evidence)

    # Deduplicate by turn, then sort ascending.
    seen: Dict[int, Dict[str, Any]] = {}
    for c in all_chunks:
        turn = c.get("turn")
        if not isinstance(turn, int):
            continue
        if turn not in seen:
            seen[turn] = c
            continue
        # Keep the richer chunk when duplicate turns appear.
        old = seen[turn]
        old_len = len(str(old.get("action", ""))) + len(str(old.get("observation", "")))
        new_len = len(str(c.get("action", ""))) + len(str(c.get("observation", "")))
        if new_len > old_len:
            seen[turn] = c
    sorted_chunks = sorted(seen.values(), key=lambda x: x["turn"])

    # Cap state_mem to at most 30% of the budget so Evidence has sufficient room.
    _state_mem_budget = max(4096, int(max_context_length * 0.30))
    if len(state_mem_str) > _state_mem_budget:
        state_mem_str = truncate_trajectory_text(state_mem_str, _state_mem_budget)

    parts = [
        f"# State Memory\n{state_mem_str}",
        f"# Task\n{task}",
        f"# Evidence\n{_format_chunks(sorted_chunks)}",
    ]

    context = "\n\n".join(parts)
    return truncate_trajectory_text(context, max_context_length)


def _extract_chunks_from_extra_evidence(extra_evidence: str) -> List[Dict[str, Any]]:
    """Parse turn chunks from keyword-search output."""
    if not extra_evidence:
        return []

    text = extra_evidence.strip()
    chunks: List[Dict[str, Any]] = []

    # 1) Prefer JSON output from keyword-search script (most common path).
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        payload = None

    if payload is not None:

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                turn_raw = node.get("turn", node.get("turn_idx"))
                try:
                    turn = int(turn_raw)
                except (TypeError, ValueError):
                    turn = None
                if turn is not None:
                    chunks.append(
                        {
                            "turn": turn,
                            "action": str(node.get("action", "")),
                            "observation": str(node.get("observation", "")),
                        }
                    )
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(payload)
        if chunks:
            return chunks

    # 2) Fallback: parse already-formatted plain text.
    pattern = re.compile(
        r"Turn\s+(-?\d+):\s*\n\s*Action:\s*(.*?)\n\s*Observation:\s*(.*?)(?=\n\s*Turn\s+-?\d+:|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        chunks.append(
            {
                "turn": int(match.group(1)),
                "action": match.group(2).strip(),
                "observation": match.group(3).strip()[:500],
            }
        )

    return chunks
