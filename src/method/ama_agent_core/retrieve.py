"""
Memory Retrieval Module for AMA-Agent

This module handles the retrieval of relevant information from memory.
It uses a two-stage approach:
1. Retrieve top_k=5 chunks using relevance scoring
2. Use LLM to judge if the information is sufficient
"""
import json
from typing import Dict, List, Any, Optional, Callable
from .utils import retrieve_with_qwen
from .prompt import CHECK_STATE_MEM_PROMPT_TEMPLATE


def memory_retrieve(
    memory: Dict[str, Any],
    question: str,
    call_llm_func: Callable,
    top_k: int = 5
) -> str:
    """
    Retrieve relevant context from memory for answering a question.

    Two-stage retrieval process:
    1. Stage 1: Check if state memory is sufficient
    2. Stage 2: Retrieve top_k chunks and judge sufficiency with LLM

    Args:
        memory: Memory dictionary containing state_mem, text_mem, embed_mem
        question: Question to answer
        call_llm_func: Async LLM function
        top_k: Number of top chunks to retrieve (default: 5)

    Returns:
        Retrieved context as string
    """
    state_mem = memory.get('state_mem', '')
    text_mem = memory.get('text_mem', {})

    # Stage 1: Check if state memory has sufficient information

    state_mem_str = str(state_mem) if state_mem else ""
    task = text_mem.get('task', '')

    check_prompt = CHECK_STATE_MEM_PROMPT_TEMPLATE.format(
        state_mem_str=state_mem_str,
        query=question
    )

    _, check_response = call_llm_func(check_prompt)

    need_retrieval = False
    if check_response and "NEED_RETRIEVAL" in check_response.upper():
        need_retrieval = True


    # If state memory is sufficient, return it
    if not need_retrieval:
        context = f"""# State Memory
{state_mem_str}

# Task
{task}"""
        return context

    # Stage 2: Retrieve top_k chunks and judge sufficiency


    # Get top_k chunks using Qwen relevance scoring
    keywords_info, relevant_turn_indices = retrieve_with_qwen(
        question, text_mem, call_llm_func
    )


    # Extract the relevant chunks
    relevant_chunks = _extract_chunks(
        trajectory=text_mem['trajectory_data']['trajectory'],
        turn_indices=relevant_turn_indices
    )

    # Judge if retrieved chunks are sufficient
    context = _judge_sufficiency_and_build_context(
        question=question,
        state_mem=state_mem_str,
        task=task,
        chunks=relevant_chunks,
        call_llm_func=call_llm_func
    )

    return context



def _extract_chunks(
    trajectory: List[Dict[str, Any]],
    turn_indices: List[int]
) -> List[Dict[str, Any]]:
    """
    Extract chunks from trajectory based on turn indices.

    Args:
        trajectory: Full trajectory list
        turn_indices: List of turn indices to extract

    Returns:
        List of relevant turn dictionaries
    """
    chunks = []

    for turn in trajectory:
        turn_idx = turn.get('turn_idx', -1)
        if turn_idx in turn_indices:
            chunks.append({
                'turn': turn_idx,
                'action': turn.get('action', ''),
                'observation': turn.get('observation', '')
            })

    # Sort by turn index
    chunks.sort(key=lambda x: x['turn'])

    return chunks


def _judge_sufficiency_and_build_context(
    question: str,
    state_mem: str,
    task: str,
    chunks: List[Dict[str, Any]],
    call_llm_func: Callable
) -> str:
    """
    Judge if retrieved chunks are sufficient and build final context.

    Args:
        question: User question
        state_mem: State memory string
        task: Task description
        chunks: Retrieved chunks
        call_llm_func: Async LLM function

    Returns:
        Final context string
    """
    # Format chunks for display
    chunks_text = _format_chunks(chunks)

    # Ask LLM to judge sufficiency
    sufficiency_prompt = f"""Given the question and retrieved information, determine if the information is SUFFICIENT to answer the question.

Question: {question}

State Memory:
{state_mem}

Retrieved Information (Top {len(chunks)} chunks):
{chunks_text}

Analyze:
1. Does the retrieved information directly answer the question?
2. Are there any missing details needed to provide a complete answer?
3. Is the information clear and unambiguous?

Respond with:
- SUFFICIENT: [brief explanation] - if you can answer the question with this information
- INSUFFICIENT: [what's missing] - if you need more information

Your judgment:"""

    _, llm_response = call_llm_func(sufficiency_prompt)

    is_sufficient = False
    if llm_response and "SUFFICIENT" in llm_response.upper() and "INSUFFICIENT" not in llm_response.upper():
        is_sufficient = True


    # Build final context
    context = f"""# State Memory
{state_mem}

# Task
{task}

# Retrieved Relevant Information ({len(chunks)} chunks)
{chunks_text}

# Sufficiency Assessment
{llm_response}"""

    return context


def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Format chunks into readable text.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Formatted string
    """
    if not chunks:
        return "No chunks retrieved."

    lines = []
    for chunk in chunks:
        turn = chunk.get('turn', 0)
        action = chunk.get('action', '')
        observation = chunk.get('observation', '')[:500]  # Truncate long observations

        lines.append(f"Turn {turn}:")
        lines.append(f"  Action: {action}")
        lines.append(f"  Observation: {observation}")
        lines.append("")  # Empty line for readability

    return "\n".join(lines)
