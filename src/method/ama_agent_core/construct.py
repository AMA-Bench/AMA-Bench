"""
Memory Construction Module for AMA-Agent

This module handles the construction of state memory from trajectory data.
It processes trajectory text into different turns and embeds them for retrieval.
"""
import json
from typing import Dict, List, Any, Optional, Callable
from .utils import extract_state_memory_from_response, truncate_trajectory_text
from .prompt import COMPRESS_PROMPT_TEMPLATE, CAUSAL_PROMPT_TEMPLATE


def construct_state_memory(
    trajectory_text: str,
    task: str = "",
    call_llm_func: Optional[Callable] = None,
    chunk_size: int = 8192,
    embed_engine: Optional[Callable] = None,
    causal: bool = False
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
    trajectory_data = {
        'trajectory': trajectory,
        'task': task,
        'episode_id': 'episode'
    }
    text_mem = {
        'task': task,
        'trajectory_text': trajectory_text,
        'trajectory_data': trajectory_data,
        'episode_id': 'episode',
        'num_turns': len(trajectory)
    }

    total_chars = len(trajectory_text)

    # Build state memory (and optionally causal graph)
    if causal:
        state_mem, causal_graph = _process_trajectory_causal(
            trajectory=trajectory,
            trajectory_text=trajectory_text,
            task=task,
            chunk_size=chunk_size,
            call_llm_func=call_llm_func
        )
    else:
        state_mem = _process_trajectory(
            trajectory=trajectory,
            trajectory_text=trajectory_text,
            task=task,
            chunk_size=chunk_size,
            call_llm_func=call_llm_func
        )
        causal_graph = None

    # Build turn-level embeddings only if embed_engine is provided
    embed_mem = None
    if embed_engine is not None:
        embed_mem = _build_turn_embeddings(
            trajectory=trajectory,
            embed_engine=embed_engine
        )

    return {
        'state_mem': state_mem,
        'causal_graph': causal_graph,
        'text_mem': text_mem,
        'embed_mem': embed_mem,
        'trajectory': trajectory
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

    Args:
        trajectory_text: Formatted trajectory text

    Returns:
        List of turn dictionaries with keys: turn_idx, action, observation
    """
    trajectory = []
    lines = trajectory_text.strip().split('\n')

    current_turn = {}
    for line in lines:
        line = line.strip()

        # Match "Turn X:" pattern
        if line.startswith('Turn ') and ':' in line:
            if current_turn:
                trajectory.append(current_turn)
            # Extract turn number
            try:
                turn_num = int(line.replace('Turn ', '').replace(':', '').strip())
                current_turn = {'turn_idx': turn_num}
            except ValueError:
                current_turn = {}

        # Match "Action: ..." pattern
        elif line.startswith('Action:'):
            current_turn['action'] = line[7:].strip()  # Remove 'Action:' prefix

        # Match "Observation: ..." pattern
        elif line.startswith('Observation:'):
            current_turn['observation'] = line[12:].strip()  # Remove 'Observation:' prefix

    # Add last turn
    if current_turn:
        trajectory.append(current_turn)

    return trajectory


def _process_trajectory(
    trajectory: List[Dict[str, Any]],
    trajectory_text: str,
    task: str,
    chunk_size: int,
    call_llm_func: Optional[Callable]
) -> Optional[str]:
    """
    Process trajectory with optional chunking to build state memory.

    Args:
        trajectory: Parsed trajectory list
        trajectory_text: Formatted trajectory text
        task: Task description
        chunk_size: Maximum size for each chunk
        call_llm_func: Async LLM function

    Returns:
        Compressed state memory string or None if failed
    """
    if not call_llm_func:
        return None

    total_chars = len(trajectory_text)

    # Single chunk processing
    if total_chars <= chunk_size:
        compress_prompt = COMPRESS_PROMPT_TEMPLATE.format(
            task=task,
            trajectory_text=trajectory_text,
            previous_state_text=""
        )

        _, llm_response = call_llm_func(compress_prompt)

        if llm_response:
            state_mem = extract_state_memory_from_response(llm_response)
            if state_mem:
                return state_mem

        return None

    # Multi-chunk processing

    # Split into chunks by turns
    chunks = []
    current_chunk = []
    current_length = 0

    for turn in trajectory:
        turn_text = _format_single_turn(turn)
        turn_length = len(turn_text)

        if current_length + turn_length > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0

        current_chunk.append(turn)
        current_length += turn_length

    if current_chunk:
        chunks.append(current_chunk)
    # Process chunks sequentially, accumulating state
    accumulated_state = ""

    for i, chunk in enumerate(chunks):
        chunk_text = _format_chunk_for_llm(chunk)

        # Include previous state in prompt
        previous_state_text = f"Previous State Memory:\n{accumulated_state}" if accumulated_state else ""

        compress_prompt = COMPRESS_PROMPT_TEMPLATE.format(
            task=task,
            trajectory_text=chunk_text,
            previous_state_text=previous_state_text
        )

        _, llm_response = call_llm_func(compress_prompt)

        if llm_response:
            chunk_state = extract_state_memory_from_response(llm_response)
            if chunk_state:
                accumulated_state = chunk_state

    return accumulated_state if accumulated_state else None


def _format_single_turn(turn: Dict[str, Any]) -> str:
    """Format single turn into readable text."""
    turn_idx = turn.get('turn_idx', 0)
    action = turn.get('action', '')
    observation = turn.get('observation', '')
    return f"Turn {turn_idx}:\n  Action: {action}\n  Observation: {observation[:200]}...\n"


def _format_chunk_for_llm(chunk: List[Dict[str, Any]]) -> str:
    """Format chunk of turns into readable text for LLM."""
    lines = []
    for turn in chunk:
        turn_idx = turn.get('turn_idx', 0)
        action = turn.get('action', '')
        observation = turn.get('observation', '')
        lines.append(f"Turn {turn_idx}:")
        lines.append(f"  Action: {action}")
        lines.append(f"  Observation: {observation[:200]}...")
    return "\n".join(lines)


def _build_turn_embeddings(
    trajectory: List[Dict[str, Any]],
    embed_engine: Optional[Callable]
) -> Optional[Dict[str, Any]]:
    """
    Build turn-level embeddings for retrieval.

    Args:
        trajectory: List of turns
        embed_engine: Synchronous embedding function, or None to skip embedding

    Returns:
        Dictionary containing embeddings/turn_texts/turn_indices,
        or None if embed_engine is not provided
    """
    if embed_engine is None:
        return None

    turn_texts = []
    turn_indices = []

    for turn in trajectory:
        turn_idx = turn.get('turn_idx', 0)
        action = turn.get('action', '')
        observation = turn.get('observation', '')[:500]  # Truncate long observations
        turn_text = f"Turn {turn_idx}: Action={action}, Observation={observation}"
        turn_texts.append(turn_text)
        turn_indices.append(turn_idx)

    embeddings = [embed_engine(t) for t in turn_texts]

    return {
        'embeddings': embeddings,
        'turn_texts': turn_texts,
        'turn_indices': turn_indices
    }


def _extract_causal_graph_from_response(llm_response: str) -> Optional[List[Dict[str, Any]]]:
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

    after_marker = llm_response[pos + len(marker):].strip()

    # Find the JSON array
    import re
    json_match = re.search(r'(\[.*?\])', after_marker, re.DOTALL)
    if not json_match:
        return None

    try:
        return json.loads(json_match.group(1))
    except json.JSONDecodeError:
        return None


def _process_trajectory_causal(
    trajectory: List[Dict[str, Any]],
    trajectory_text: str,
    task: str,
    chunk_size: int,
    call_llm_func: Optional[Callable]
) -> tuple:
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

    # Determine chunks
    if total_chars <= chunk_size:
        chunks = [trajectory]
    else:
        chunks = []
        current_chunk: List[Dict[str, Any]] = []
        current_length = 0
        for turn in trajectory:
            turn_text = _format_single_turn(turn)
            turn_length = len(turn_text)
            if current_length + turn_length > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_length = 0
            current_chunk.append(turn)
            current_length += turn_length
        if current_chunk:
            chunks.append(current_chunk)


    for i, chunk in enumerate(chunks):
        chunk_text = trajectory_text if len(chunks) == 1 else _format_chunk_for_llm(chunk)
        previous_state_text = f"Previous State Memory:\n{accumulated_state}" if accumulated_state else ""

        causal_prompt = CAUSAL_PROMPT_TEMPLATE.format(
            task=task,
            trajectory_text=chunk_text,
            previous_state_text=previous_state_text
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
