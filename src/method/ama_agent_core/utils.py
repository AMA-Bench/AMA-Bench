"""
Utility functions for memory agent
"""
import json
import asyncio
import os
import signal
import subprocess
import shutil
import tempfile
import re
import math
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from collections import Counter
import ray


def load(file_path: str) -> Dict[str, Any]:
    """
    Load trajectory data from JSON file.

    Args:
        file_path: Path to JSON file (e.g., tw_out_batch/coin_collector/coin_collector_0.json)

    Returns:
        Dictionary containing:
            - trajectory: List of trajectory turns with turn_idx, action, observation
            - task: Task description string
            - episode_id: Episode identifier
            - task_type: Type of task
            - state: Success or failure state
            - num_turns: Total number of turns
            - qa_pairs: List of question-answer pairs (if available)
            - state_snapshots: State snapshots at each turn (if available)
            - events: Important events during trajectory (if available)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {
        'trajectory': data.get('trajectory', []),
        'task': data.get('task', ''),
        'episode_id': data.get('episode_id', ''),
        'task_type': data.get('task_type', ''),
        'state': data.get('state', ''),
        'fail_reason': data.get('fail_reason', ''),
        'num_turns': data.get('num_turns', 0),
        'total_tokens': data.get('total_tokens', 0),
        'qa_pairs': data.get('qa_pairs', []),
        'state_snapshots': data.get('state_snapshots', []),
        'events': data.get('events', []),
    }

    return result


def _ensure_ray_initialized() -> None:
    """
    Ensure Ray is initialized.
    """
    if ray.is_initialized():
        return
    
    init_kwargs = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "logging_level": "ERROR",
    }

    num_cpus_env = os.getenv("RAY_NUM_CPUS")
    if num_cpus_env:
        try:
            num_cpus = float(num_cpus_env)
            if num_cpus > 0:
                init_kwargs["num_cpus"] = num_cpus
        except (ValueError, TypeError):
            pass

    ray_tmp_dir = "/tmp/verl_ray"
    ray_spill_dir = "/tmp/verl_spill"
    os.makedirs(ray_tmp_dir, exist_ok=True)
    os.makedirs(ray_spill_dir, exist_ok=True)
    
    init_kwargs["_temp_dir"] = ray_tmp_dir
    spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
    init_kwargs["_system_config"] = {
        "object_spilling_config": json.dumps(spilling_conf)
    }
    
    ray.init(**init_kwargs)




async def _run_python_script(
    script: str,
    timeout: float = 40.0
) -> str:
    """
    Execute Python script in isolated environment with timeout.
    
    Args:
        script: Python script content to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Script output as string, or "timeout" if execution exceeds timeout
    """
    os.makedirs("tmp", exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="mem_exec_", dir="tmp")
    script_path = os.path.join(tmpdir, "script.py")
    stdout_path = os.path.join(tmpdir, "stdout.txt")
    stderr_path = os.path.join(tmpdir, "stderr.txt")

    proc = None
    stdout_file = None
    stderr_file = None
    result = "timeout"

    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        env = os.environ.copy()

        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        venv_python = os.path.join(workspace_root, "pettingllms_venv/bin/python")
        python_executable = venv_python if os.path.exists(venv_python) else "python"

        stdout_file = open(stdout_path, "wb")
        stderr_file = open(stderr_path, "wb")
        proc = await asyncio.create_subprocess_exec(
            python_executable,
            script_path,
            stdout=stdout_file,
            stderr=stderr_file,
            cwd=tmpdir,
            env=env,
            start_new_session=True,
        )

        await asyncio.wait_for(proc.wait(), timeout=timeout)

        stdout_file.close()
        stderr_file.close()
        stdout_file = None
        stderr_file = None

        with open(stdout_path, "rb") as f_out:
            out_bytes = f_out.read()
        with open(stderr_path, "rb") as f_err:
            err_bytes = f_err.read()

        stdout_str = out_bytes.decode(errors="replace")
        stderr_str = err_bytes.decode(errors="replace")

        if stderr_str.strip():
            result = f"error: {stderr_str}\n\nSTDOUT:\n{stdout_str}"
        else:
            result = stdout_str
        
    except asyncio.TimeoutError:
        if proc and proc.pid:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        result = "timeout"

    finally:
        if stdout_file and not stdout_file.closed:
            stdout_file.close()
        if stderr_file and not stderr_file.closed:
            stderr_file.close()

        if proc and proc.returncode is None:
            if proc.pid:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.kill()
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
            if os.path.exists(tmpdir):
                subprocess.run(['rm', '-rf', tmpdir], timeout=5, capture_output=True)
    
    return result


def get_ray_worker_cls(num_workers=180):
    """
    Get or create the Ray worker class for MemAgent operations.
    
    Returns a Ray remote actor class that can execute Python scripts.
    
    Args:
        num_workers: Number of workers to create (used for CPU allocation)
    
    Returns:
        Ray remote actor class
    """
    _ensure_ray_initialized()

    cache_key = f"_cls_{num_workers}"
    if hasattr(get_ray_worker_cls, cache_key):
        return getattr(get_ray_worker_cls, cache_key)

    try:
        import multiprocessing
        total_cpus = multiprocessing.cpu_count()
        cpus_per_worker = min(4.0, (total_cpus * 0.6) / num_workers)
        print(f"Ray worker resource allocation: total_cpus={total_cpus}, num_workers={num_workers}, "
              f"cpus_per_worker={cpus_per_worker:.3f}")
    except Exception:
        cpus_per_worker = 0.001

    @ray.remote(num_cpus=cpus_per_worker, max_concurrency=2000)
    class _RayWorker:
        def __init__(self, idx):
            if isinstance(idx, (int, float)):
                self.idx = int(idx)
            elif isinstance(idx, str) and re.fullmatch(r"\s*-?\d+\s*", idx):
                self.idx = int(idx)
            else:
                self.idx = 0

        def get_idx(self):
            return self.idx

        async def run(
            self,
            script: str,
            timeout: float = 40.0,
        ) -> str:
            """
            Execute Python script and return output.
            
            Args:
                script: Python script to execute
                timeout: Execution timeout
                
            Returns:
                Script execution output as string
            """
            return await _run_python_script(
                script=script,
                timeout=timeout,
            )

    RayWorker = _RayWorker
    cache_key = f"_cls_{num_workers}"
    setattr(get_ray_worker_cls, cache_key, RayWorker)
    return RayWorker


# ============================================================================
# Retrieval Helper Functions
# ============================================================================


def cosine_similarity(vec1, vec2) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector (list or numpy array)
        vec2: Second vector (list or numpy array)

    Returns:
        Cosine similarity score
    """
    if isinstance(vec1, list):
        vec1 = [float(x) for x in vec1]
        vec2 = [float(x) for x in vec2]

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
    else:
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# ============================================================================
# Retrieval Functions
# ============================================================================

def retrieve_with_qwen(
    query: str,
    text_mem: Dict[str, Any],
    call_llm_func
) -> Tuple[Dict[str, Any], List[int]]:
    """
    Retrieve relevant chunks using Qwen3-4B for relevance scoring.

    This method:
    1. Chunks the trajectory into segments
    2. Uses Qwen3-4B to score each chunk's relevance to the query
    3. Returns top 5 most relevant chunks

    Args:
        query: Query string
        text_mem: Text memory containing trajectory data
        call_llm_func: Async LLM call function (should use qwen3-4b)

    Returns:
        Tuple of (keywords_info, relevant_turn_indices)
    """
    trajectory = text_mem['trajectory_data']['trajectory']

    # Group trajectory into chunks (e.g., every 3-5 turns as one chunk)
    chunk_size = 5
    chunks = []
    for i in range(0, len(trajectory), chunk_size):
        chunk_turns = trajectory[i:i+chunk_size]
        chunks.append(chunk_turns)


    # Score each chunk using Qwen3-4B
    chunk_scores = []
    for chunk_idx, chunk_turns in enumerate(chunks):
        # Format chunk text
        chunk_text_parts = []
        turn_indices = []
        for turn in chunk_turns:
            turn_idx = turn.get('turn_idx', 0)
            action = turn.get('action', '')
            observation = turn.get('observation', '')[:300]  # Truncate long observations
            chunk_text_parts.append(f"Turn {turn_idx}: Action={action}, Observation={observation}")
            turn_indices.append(turn_idx)

        chunk_text = "\n".join(chunk_text_parts)

        # Create relevance scoring prompt
        relevance_prompt = f"""Rate the relevance of the following trajectory chunk to the query on a scale of 0-10.

Query: {query}

Trajectory Chunk:
{chunk_text}

Consider:
- Does this chunk contain information directly answering the query?
- Are there relevant actions, observations, or state changes?
- How closely do the events relate to what the query is asking?

Respond with ONLY a single number from 0 to 10, where:
- 0 = completely irrelevant
- 5 = somewhat relevant
- 10 = highly relevant

Score:"""

        try:
            _, score_response = call_llm_func(relevance_prompt)

            # Parse score from response
            score_match = re.search(r'(\d+(?:\.\d+)?)', score_response.strip())
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(10.0, score))  # Clamp to [0, 10]
            else:
                score = 0.0
        except Exception as e:
            score = 0.0

        chunk_scores.append((chunk_idx, turn_indices, score))

    # Sort by score and get top 5 chunks
    chunk_scores.sort(key=lambda x: x[2], reverse=True)
    top_k = min(5, len(chunk_scores))

    # Extract all turn indices from top chunks
    relevant_turn_indices = []
    for i in range(top_k):
        chunk_idx, turn_indices, score = chunk_scores[i]
        if score > 0:
            relevant_turn_indices.extend(turn_indices)

    # Remove duplicates and sort
    relevant_turn_indices = sorted(list(set(relevant_turn_indices)))

    # Extract keywords from query for metadata
    keywords = re.findall(r'\w+', query.lower())
    keywords_info = {
        "keywords": keywords[:5],
        "search_mode": "qwen_relevance",
        "method": "qwen3-4b",
        "num_chunks": len(chunks),
        "top_chunks": top_k
    }


    return keywords_info, relevant_turn_indices


async def retrieve_with_llm(
    query: str,
    state_mem: Dict[str, Any],
    text_mem: Dict[str, Any],
    call_llm_func
) -> Tuple[Dict[str, Any], List[int]]:
    """
    Retrieve using LLM to extract keywords.

    Args:
        query: Query string
        state_mem: State memory
        text_mem: Text memory containing trajectory data
        call_llm_func: Async LLM call function

    Returns:
        Tuple of (keywords_info, relevant_turn_indices)
    """
    from .tool import traj_find

    state_mem_str = json.dumps(state_mem, indent=2)

    keyword_prompt = f"""Given the query, extract relevant keywords and search criteria.

Query: {query}

State Memory Summary:
{state_mem_str}

Extract:
1. Key entities, objects, or actions mentioned
2. Time-related information (turn numbers, ranges)
3. Specific events or patterns to look for

Format as JSON:
{{
  "keywords": ["keyword1", "keyword2"],
  "turn_range": {{"start": 1, "end": 5}} or null,
  "search_mode": "keyword" or "action" or "entity"
}}

Only output the JSON:"""

    _, keyword_response = await call_llm_func(keyword_prompt)

    keywords_info = {}
    if keyword_response:
        try:
            keyword_clean = keyword_response.strip()
            if keyword_clean.startswith("```json"):
                keyword_clean = keyword_clean[7:]
            if keyword_clean.startswith("```"):
                keyword_clean = keyword_clean[3:]
            if keyword_clean.endswith("```"):
                keyword_clean = keyword_clean[:-3]
            keyword_clean = keyword_clean.strip()
            keywords_info = json.loads(keyword_clean)
            keywords_info['method'] = 'llm'
        except:
            keywords_info = {"keywords": [query], "search_mode": "keyword", "method": "llm"}

    trajectory_text_json = json.dumps(text_mem['trajectory_data'])
    keywords = keywords_info.get('keywords', [query])
    relevant_turn_indices = []

    for keyword in keywords:
        indices = traj_find(trajectory_text_json, keyword, mode=keywords_info.get('search_mode', 'keyword'))
        relevant_turn_indices.extend(indices)

    relevant_turn_indices = sorted(list(set(relevant_turn_indices)))[:5]

    return keywords_info, relevant_turn_indices


async def retrieve_with_embed(
    query: str,
    text_mem: Dict[str, Any],
    embed_mem: Dict[str, Any],
    embed_engine
) -> Tuple[Dict[str, Any], List[int]]:
    """
    Retrieve using embedding-based similarity.

    Args:
        query: Query string
        text_mem: Text memory containing trajectory data
        embed_mem: Embedding memory
        embed_engine: Embedding function

    Returns:
        Tuple of (keywords_info, relevant_turn_indices)
    """
    if asyncio.iscoroutinefunction(embed_engine):
        query_embedding = await embed_engine(query)
    else:
        query_embedding = embed_engine(query)

    turn_embeddings = embed_mem['embeddings']
    turn_texts = embed_mem['turn_texts']

    similarities = []
    trajectory = text_mem['trajectory_data']['trajectory']

    for i, turn_emb in enumerate(turn_embeddings):
        similarity = cosine_similarity(query_embedding, turn_emb)
        turn_idx = trajectory[i].get('turn_idx', i)
        similarities.append((turn_idx, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = min(5, len(similarities))
    relevant_turn_indices = [similarities[i][0] for i in range(top_k)]

    query_tokens = tokenize(query)
    keywords_info = {
        "keywords": query_tokens[:5],
        "search_mode": "embed",
        "method": "embed"
    }

    return keywords_info, relevant_turn_indices


def fallback_retrieve(query: str, trajectory_text_json: str) -> str:
    """
    Fallback retrieval using simple keyword search.

    Args:
        query: Query string
        trajectory_text_json: JSON string of trajectory data

    Returns:
        Retrieved text
    """
    from .tool import traj_find, traj_get

    indices = traj_find(trajectory_text_json, query, mode="keyword")
    if indices:
        return traj_get(trajectory_text_json, span={'indices': indices})
    return ""


def extract_state_memory_from_response(llm_response: str) -> Optional[str]:
    """Extract state memory content from LLM response after **STATE_MEMORY** marker.
    
    Args:
        llm_response: The LLM response text
        
    Returns:
        The content after **STATE_MEMORY** marker, or None if marker not found
    """
    if not llm_response:
        return None
    
    # Look for **STATE_MEMORY** marker
    marker = "**STATE_MEMORY**"
    marker_pos = llm_response.find(marker)
    
    if marker_pos == -1:
        # Try case-insensitive search
        marker_pos = llm_response.upper().find(marker)
        if marker_pos == -1:
            return None
    
    # Extract everything after the marker
    state_mem = llm_response[marker_pos + len(marker):].strip()
    
    return state_mem if state_mem else None


def extract_code_from_response(llm_response: str) -> str:
    """Extract Python code from LLM response by removing think tags and extracting code blocks.
    
    Supports multiple formats:
    1. **CODE**: ```python ... ``` (preferred format)
    2. ```python ... ``` (legacy format)
    3. ``` ... ``` (fallback format)
    """
    if not llm_response:
        return ""

    llm_response_clean = llm_response.strip()

    # Remove <think>...</think> blocks
    llm_response_clean = re.sub(r'<think>.*?</think>', '', llm_response_clean, flags=re.DOTALL)
    llm_response_clean = llm_response_clean.strip()

    # Try to extract code from **CODE**: ```python ... ``` format (preferred)
    code_marker_match = re.search(r'\*\*CODE\*\*:?\s*```python\s*\n(.*?)\n```', llm_response_clean, re.DOTALL | re.IGNORECASE)
    if code_marker_match:
        return code_marker_match.group(1).strip()

    # Try to extract code from ```python ... ``` blocks (legacy)
    python_code_match = re.search(r'```python\s*\n(.*?)\n```', llm_response_clean, re.DOTALL)
    if python_code_match:
        return python_code_match.group(1).strip()

    # Try to extract from ``` ... ``` blocks without language specifier (fallback)
    code_match = re.search(r'```\s*\n(.*?)\n```', llm_response_clean, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # If no code blocks found, return the entire cleaned response
    # (assume the entire response is code)
    return llm_response_clean


def truncate_trajectory_text(trajectory_text: str, max_length: int) -> str:
    """Truncate trajectory text to fit within max_length.

    Uses head-tail truncation strategy: keeps beginning and end of the text.

    Args:
        trajectory_text: The trajectory text to truncate
        max_length: Maximum length in characters

    Returns:
        Truncated text
    """
    if len(trajectory_text) <= max_length:
        return trajectory_text

    # Use 70% for head, 30% for tail
    head_length = int(max_length * 0.7)
    tail_length = max_length - head_length

    truncated = trajectory_text[:head_length] + "\n...\n" + trajectory_text[-tail_length:]
    return truncated

