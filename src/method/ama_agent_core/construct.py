"""
Memory Construction Module for AMA-Agent

This module handles the construction of state memory from trajectory data.
It processes trajectory text into different turns and embeds them for retrieval.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from .prompt import CAUSAL_PROMPT_TEMPLATE, COMPRESS_PROMPT_TEMPLATE
from .utils import extract_state_memory_from_response, truncate_trajectory_text


def construct_state_memory(
    trajectory_text: str,
    task: str = "",
    call_llm_func: Optional[Callable] = None,
    chunk_size: int = 2048,
    session_size: int = 16384,
    embed_engine: Optional[Callable] = None,
    causal: bool = False,
) -> Dict[str, Any]:
    """
    Construct state memory from trajectory text.

    Process:
    1. Parse trajectory text into different turns
    2. Compress trajectory into state memory using LLM
    3. Optionally embed each turn for retrieval
    4. Optionally extract causal graph (if causal=True)

    Args:
        trajectory_text: String-formatted trajectory text with turns
        task: Task description
        call_llm_func: Function for LLM interaction
        chunk_size: Maximum size for each chunk (default: 8192)
        embed_engine: Optional embedding function for turn-level embeddings
        causal: If True, also extract causal relationships to build a causal
                graph alongside the state memory (default: False)

    Returns:
        Dictionary containing:
            - state_mem: Compressed state memory string
            - causal_graph: Extracted causal relationships (if causal=True, else None)
            - text_mem: Original trajectory data
            - embed_mem: Turn-level embeddings (if embed_engine provided, else None)
            - trajectory: Parsed trajectory list
    """
    # Parse trajectory text into turns
    trajectory = _parse_trajectory_text(trajectory_text)

    # Build text memory
    trajectory_data = {"trajectory": trajectory, "task": task, "episode_id": "episode"}
    text_mem = {
        "task": task,
        "trajectory_text": trajectory_text,
        "trajectory_data": trajectory_data,
        "episode_id": "episode",
        "num_turns": len(trajectory),
    }

    # Build state memory (and optionally causal graph)
    if causal:
        state_mem, causal_graph = _process_trajectory_causal(
            trajectory_text=trajectory_text,
            task=task,
            session_size=session_size,
            call_llm_func=call_llm_func,
        )
    else:
        state_mem = _process_trajectory(
            trajectory_text=trajectory_text,
            task=task,
            session_size=session_size,
            call_llm_func=call_llm_func,
        )
        causal_graph = None

    # Build turn-level embeddings only if embed_engine is provided
    embed_mem = _build_turn_embeddings(
        trajectory=trajectory,
        embed_engine=embed_engine,
        min_chunk_size=chunk_size,
    )

    return {
        "state_mem": state_mem,
        "causal_graph": causal_graph,
        "text_mem": text_mem,
        "embed_mem": embed_mem,
        "trajectory": trajectory,
    }


def _parse_trajectory_text(trajectory_text: str) -> List[Dict[str, Any]]:
    """
    Parse trajectory text into list of turn dictionaries.

    Expected format:
        Turn 0:
          Action: ...
          Observation: ...
        Turn 1:
          Action: ...
          Observation: ...

    Multi-line action/observation values are supported: lines that follow an
    Action/Observation header and do not match any other recognized header are
    appended to the current field.

    Args:
        trajectory_text: Formatted trajectory text

    Returns:
        List of turn dictionaries with keys: turn_idx, action, observation
    """
    trajectory = []
    lines = trajectory_text.strip().split("\n")

    current_turn: Dict[str, Any] = {}
    current_field: Optional[str] = None

    for line in lines:
        stripped = line.strip()

        # Match "Turn X:" or "Step X:" header
        if (
            stripped.startswith("Turn ") or stripped.startswith("Step ")
        ) and ":" in stripped:
            if current_turn:
                trajectory.append(current_turn)
            try:
                turn_num = int(stripped.split(":")[0].split()[-1])
                current_turn = {"turn_idx": turn_num}
            except (ValueError, IndexError):
                current_turn = {}
            current_field = None

        elif stripped.startswith("Action:"):
            current_turn["action"] = stripped[7:].strip()
            current_field = "action"

        elif stripped.startswith("Observation:"):
            current_turn["observation"] = stripped[12:].strip()
            current_field = "observation"

        elif current_field and stripped and current_turn:
            # Continuation of a multi-line action or observation
            current_turn[current_field] = (
                current_turn.get(current_field, "") + "\n" + stripped
            )

    if current_turn:
        trajectory.append(current_turn)

    return trajectory


def _build_trajectory_chunks(text: str, chunk_size: int) -> List[str]:
    """Split trajectory text into chunks of at most chunk_size characters."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _process_trajectory(
    trajectory_text: str,
    task: str,
    session_size: int,
    call_llm_func: Optional[Callable],
) -> Optional[str]:
    """
    Process trajectory text to build state memory.

    Fits in one session → single LLM call.
    Exceeds session_size → sessions processed concurrently (Phase 1),
    partial state memories concatenated then capped at session_size * 4 (Phase 2).
    """
    if not call_llm_func:
        return None

    total_chars = len(trajectory_text)

    # Single-session path
    if total_chars <= session_size:
        compress_prompt = COMPRESS_PROMPT_TEMPLATE.format(
            task=task, trajectory_text=trajectory_text, previous_state_text=""
        )
        _, llm_response = call_llm_func(compress_prompt)
        if llm_response:
            return extract_state_memory_from_response(llm_response)
        return None

    # Multi-session path
    chunks = _build_trajectory_chunks(trajectory_text, session_size)

    # Phase 1: process all sessions concurrently (each independently)
    def _compress_chunk(chunk_text: str) -> Optional[str]:
        prompt = COMPRESS_PROMPT_TEMPLATE.format(
            task=task, trajectory_text=chunk_text, previous_state_text=""
        )
        _, response = call_llm_func(prompt)
        return extract_state_memory_from_response(response) if response else None

    with ThreadPoolExecutor() as executor:
        chunk_states: List[Optional[str]] = list(executor.map(_compress_chunk, chunks))

    # Phase 2: merge partial state memories, capped to avoid downstream context overflows.
    valid_states = [s for s in chunk_states if s]
    if not valid_states:
        return None
    merged = "\n\n".join(valid_states)
    return merged


def _build_turn_embeddings(
    trajectory: List[Dict[str, Any]],
    embed_engine: Optional[Callable],
    min_chunk_size: int = 2048,
) -> Optional[Dict[str, Any]]:
    """
    Build turn-level embeddings for retrieval.

    Turns are grouped into chunks of at least _MIN_EMBED_CHUNK_SIZE characters
    before embedding.  A turn whose text already exceeds the minimum is embedded
    alone.  Grouped turns share the resulting embedding vector so the retrieval
    interface (parallel flat lists) stays unchanged.

    Args:
        trajectory: List of turns
        embed_engine: Synchronous embedding function, or None to skip embedding

    Returns:
        Dictionary containing embeddings/turn_texts/turn_indices,
        or None if embed_engine is not provided
    """
    if embed_engine is None:
        return None

    # Build per-turn texts
    turns_data: List[tuple] = []
    for turn in trajectory:
        turn_idx = turn.get("turn_idx", 0)
        action = turn.get("action", "")
        observation = turn.get("observation", "")
        turn_text = f"Turn {turn_idx}: Action={action}, Observation={observation}"
        turns_data.append((turn_idx, turn_text))

    # Group turns into chunks with minimum size _MIN_EMBED_CHUNK_SIZE
    # Each chunk: (chunk_text_to_embed, [turn_indices_in_chunk])
    chunks: List[tuple] = []
    buf_text = ""
    buf_indices: List[int] = []

    for turn_idx, turn_text in turns_data:
        if len(turn_text) >= min_chunk_size:
            # Flush accumulated buffer first
            if buf_indices:
                chunks.append((buf_text, buf_indices))
                buf_text = ""
                buf_indices = []
            # Large turn is its own chunk
            chunks.append((turn_text, [turn_idx]))
        else:
            buf_text = (
                (buf_text + "\n" + turn_text).lstrip("\n") if buf_text else turn_text
            )
            buf_indices.append(turn_idx)
            if len(buf_text) >= min_chunk_size:
                chunks.append((buf_text, buf_indices))
                buf_text = ""
                buf_indices = []

    if buf_indices:
        chunks.append((buf_text, buf_indices))

    # Embed one text per chunk in parallel
    chunk_texts = [c[0] for c in chunks]
    with ThreadPoolExecutor() as executor:
        chunk_embeddings = list(executor.map(embed_engine, chunk_texts))

    # Expand back to per-turn flat lists (grouped turns share the same embedding)
    turn_texts_out: List[str] = []
    turn_indices_out: List[int] = []
    embeddings_out: List[Any] = []

    tidx_to_text = {tidx: txt for tidx, txt in turns_data}
    for (_, turn_idxs), emb in zip(chunks, chunk_embeddings):
        for tidx in turn_idxs:
            turn_texts_out.append(tidx_to_text[tidx])
            turn_indices_out.append(tidx)
            embeddings_out.append(emb)

    return {
        "embeddings": embeddings_out,
        "turn_texts": turn_texts_out,
        "turn_indices": turn_indices_out,
    }


def _extract_causal_graph_from_response(
    llm_response: str,
) -> Optional[List[Dict[str, Any]]]:
    """
    Extract the causal graph JSON array from LLM response after **CAUSAL_GRAPH** marker.

    Args:
        llm_response: Raw LLM response text

    Returns:
        Parsed list of causal relationship dicts, or None if extraction fails
    """
    if not llm_response:
        return None

    marker = "**CAUSAL_GRAPH**"
    pos = llm_response.find(marker)
    if pos == -1:
        pos = llm_response.upper().find(marker)
    if pos == -1:
        return None

    after_marker = llm_response[pos + len(marker) :].strip()

    # Find the JSON array
    import re

    json_match = re.search(r"(\[.*?\])", after_marker, re.DOTALL)
    if not json_match:
        return None

    try:
        return json.loads(json_match.group(1))
    except json.JSONDecodeError:
        return None


def _process_trajectory_causal(
    trajectory_text: str,
    task: str,
    session_size: int,
    call_llm_func: Optional[Callable],
) -> tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """
    Process trajectory to extract both state memory and causal graph.

    Args:
        trajectory: Parsed trajectory list
        trajectory_text: Formatted trajectory text
        task: Task description
        chunk_size: Maximum size for each chunk
        call_llm_func: LLM function

    Returns:
        Tuple of (state_mem, causal_graph) where causal_graph is a list of
        causal relationship dicts
    """
    if not call_llm_func:
        return None, None

    total_chars = len(trajectory_text)
    accumulated_state = ""
    all_causal_edges: List[Dict[str, Any]] = []

    chunks = (
        [trajectory_text]
        if total_chars <= session_size
        else _build_trajectory_chunks(trajectory_text, session_size)
    )

    for chunk_text in chunks:
        previous_state_text = (
            f"Previous State Memory:\n{accumulated_state}" if accumulated_state else ""
        )

        causal_prompt = CAUSAL_PROMPT_TEMPLATE.format(
            task=task,
            trajectory_text=chunk_text,
            previous_state_text=previous_state_text,
        )

        _, llm_response = call_llm_func(causal_prompt)

        if llm_response:
            # Extract state memory
            chunk_state = extract_state_memory_from_response(llm_response)
            if chunk_state:
                accumulated_state = chunk_state

            # Extract causal graph
            causal_edges = _extract_causal_graph_from_response(llm_response)
            if causal_edges:
                all_causal_edges.extend(causal_edges)

    state_mem = accumulated_state if accumulated_state else None
    causal_graph = all_causal_edges if all_causal_edges else None
    return state_mem, causal_graph
