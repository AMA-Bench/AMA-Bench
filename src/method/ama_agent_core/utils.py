"""
Utility functions for memory agent
"""

import asyncio
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import ray

from utils.evaluation_metrics import tokenize


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

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {
        "trajectory": data.get("trajectory", []),
        "task": data.get("task", ""),
        "episode_id": data.get("episode_id", ""),
        "task_type": data.get("task_type", ""),
        "state": data.get("state", ""),
        "fail_reason": data.get("fail_reason", ""),
        "num_turns": data.get("num_turns", 0),
        "total_tokens": data.get("total_tokens", 0),
        "qa_pairs": data.get("qa_pairs", []),
        "state_snapshots": data.get("state_snapshots", []),
        "events": data.get("events", []),
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
    spilling_conf = {
        "type": "filesystem",
        "params": {"directory_path": [ray_spill_dir]},
    }
    init_kwargs["_system_config"] = {
        "object_spilling_config": json.dumps(spilling_conf)
    }

    ray.init(**init_kwargs)


async def _run_python_script(script: str, timeout: float = 40.0) -> str:
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

        workspace_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
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
                subprocess.run(["rm", "-rf", tmpdir], timeout=5, capture_output=True)

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
        print(
            f"Ray worker resource allocation: total_cpus={total_cpus}, num_workers={num_workers}, "
            f"cpus_per_worker={cpus_per_worker:.3f}"
        )
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


async def retrieve_with_llm(
    query: str, state_mem: Dict[str, Any], text_mem: Dict[str, Any], call_llm_func
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
            keywords_info["method"] = "llm"
        except:
            keywords_info = {
                "keywords": [query],
                "search_mode": "keyword",
                "method": "llm",
            }

    trajectory_text_json = json.dumps(text_mem["trajectory_data"])
    keywords = keywords_info.get("keywords", [query])
    relevant_turn_indices = []

    for keyword in keywords:
        indices = traj_find(
            trajectory_text_json,
            keyword,
            mode=keywords_info.get("search_mode", "keyword"),
        )
        relevant_turn_indices.extend(indices)

    relevant_turn_indices = sorted(list(set(relevant_turn_indices)))[:5]

    return keywords_info, relevant_turn_indices


async def retrieve_with_embed(
    query: str, text_mem: Dict[str, Any], embed_mem: Dict[str, Any], embed_engine
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

    turn_embeddings = embed_mem["embeddings"]
    turn_texts = embed_mem["turn_texts"]

    similarities = []
    trajectory = text_mem["trajectory_data"]["trajectory"]

    for i, turn_emb in enumerate(turn_embeddings):
        similarity = cosine_similarity(query_embedding, turn_emb)
        turn_idx = trajectory[i].get("turn_idx", i)
        similarities.append((turn_idx, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = min(5, len(similarities))
    relevant_turn_indices = [similarities[i][0] for i in range(top_k)]

    query_tokens = tokenize(query)
    keywords_info = {
        "keywords": query_tokens[:5],
        "search_mode": "embed",
        "method": "embed",
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
        return traj_get(trajectory_text_json, span={"indices": indices})
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
    state_mem = llm_response[marker_pos + len(marker) :].strip()

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
    llm_response_clean = re.sub(
        r"<think>.*?</think>", "", llm_response_clean, flags=re.DOTALL
    )
    llm_response_clean = llm_response_clean.strip()

    # Try to extract code from **CODE**: ```python ... ``` format (preferred)
    code_marker_match = re.search(
        r"\*\*CODE\*\*:?\s*```python\s*\n(.*?)\n```",
        llm_response_clean,
        re.DOTALL | re.IGNORECASE,
    )
    if code_marker_match:
        return code_marker_match.group(1).strip()

    # Try to extract code from ```python ... ``` blocks (legacy)
    python_code_match = re.search(
        r"```python\s*\n(.*?)\n```", llm_response_clean, re.DOTALL
    )
    if python_code_match:
        return python_code_match.group(1).strip()

    # Try to extract from ``` ... ``` blocks without language specifier (fallback)
    code_match = re.search(r"```\s*\n(.*?)\n```", llm_response_clean, re.DOTALL)
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

    truncated = (
        trajectory_text[:head_length] + "\n...\n" + trajectory_text[-tail_length:]
    )
    return truncated


# ============================================================================
# Trajectory chunk helpers
# ============================================================================


def _extract_chunks(
    trajectory: List[Dict[str, Any]],
    turn_indices: List[int],
) -> List[Dict[str, Any]]:
    """Extract and sort turn dicts for the given indices."""
    index_set = set(turn_indices)
    chunks = [
        {
            "turn": t.get("turn_idx", -1),
            "action": t.get("action", ""),
            "observation": t.get("observation", ""),
        }
        for t in trajectory
        if t.get("turn_idx", -1) in index_set
    ]
    chunks.sort(key=lambda x: x["turn"])
    return chunks


_MAX_OBS_CHARS = 1500  # max observation chars per turn in formatted output


def _format_chunks(
    chunks: List[Dict[str, Any]], max_obs_chars: int = _MAX_OBS_CHARS
) -> str:
    """Format a list of chunk dicts into a readable string."""
    if not chunks:
        return "No chunks retrieved."
    lines = []
    for chunk in chunks:
        turn = chunk.get("turn", 0)
        action = chunk.get("action", "")
        observation = str(chunk.get("observation", ""))
        if len(observation) > max_obs_chars:
            observation = observation[:max_obs_chars] + "...[truncated]"
        lines.extend(
            [
                f"Turn {turn}:",
                f"  Action: {action}",
                f"  Observation: {observation}",
                "",
            ]
        )
    return "\n".join(lines)


# ============================================================================
# Similarity retrieval
# ============================================================================


async def _async_similarity_retrieve(
    question: str,
    trajectory: List[Dict[str, Any]],
    embed_engine: Optional[Any],
    embed_mem: Optional[Dict[str, Any]],
    top_k: int = 5,
) -> List[int]:
    """
    Async similarity retrieval.  Query embedding and stored turn vectors are
    handled concurrently via run_in_executor.

    Priority:
      1. embed_engine + pre-built embed_mem  → parallel query embed + stored turn vectors
      2. neither / error                     → BM25 keyword fallback
    """
    if not trajectory:
        return []

    if embed_engine is not None:
        try:
            loop = asyncio.get_event_loop()
            if embed_mem is not None:
                query_emb = await loop.run_in_executor(None, embed_engine, question)
                sims = [
                    (turn_idx, cosine_similarity(query_emb, emb))
                    for turn_idx, emb in zip(
                        embed_mem["turn_indices"], embed_mem["embeddings"]
                    )
                ]
                sims.sort(key=lambda x: x[1], reverse=True)
                return [idx for idx, _ in sims[:top_k]]
        except Exception:
            pass  # fall through to BM25 on any error

    return _bm25_retrieve(question, trajectory, top_k=top_k)


def _bm25_retrieve(
    question: str,
    trajectory: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[int]:
    """Simple BM25-style keyword fallback retrieval."""
    query_tokens = set(question.lower().split())
    scores: List[tuple] = []

    for turn in trajectory:
        turn_idx = turn.get("turn_idx", 0)
        doc_text = f"{turn.get('action', '')} {turn.get('observation', '')}".lower()
        doc_tokens = doc_text.split()
        tf = Counter(doc_tokens)
        n = len(doc_tokens) or 1
        score = sum(tf.get(tok, 0) / n for tok in query_tokens)
        scores.append((turn_idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    non_zero = [idx for idx, s in scores if s > 0]
    return (non_zero if non_zero else [s[0] for s in scores])[:top_k]


def _similarity_retrieve(
    question: str,
    trajectory: List[Dict[str, Any]],
    embed_engine: Optional[Any],
    embed_mem: Optional[Dict[str, Any]],
    top_k: int = 5,
) -> List[int]:
    """
    Synchronous similarity retrieval — calls embed_engine directly (no event
    loop required).  Falls back to BM25 when embed_engine / embed_mem are
    absent or the embedding call raises.
    """
    if not trajectory:
        return []

    if embed_engine is not None and embed_mem is not None:
        try:
            query_emb = embed_engine(question)
            sims = [
                (turn_idx, cosine_similarity(query_emb, emb))
                for turn_idx, emb in zip(
                    embed_mem["turn_indices"], embed_mem["embeddings"]
                )
            ]
            sims.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, _ in sims[:top_k]]
        except Exception:
            pass  # fall through to BM25

    return _bm25_retrieve(question, trajectory, top_k=top_k)


# ============================================================================
# NEED_GRAPH: adjacency / range / index turn retrieval
# ============================================================================


def _retrieve_graph_turns(
    trajectory: List[Dict[str, Any]],
    response: str,
) -> List[Dict[str, Any]]:
    """
    Parse a NEED_GRAPH response and retrieve the requested turns.

    Supported spec formats (one or more NEED_GRAPH lines, comma-separated):

      Adjacent turns
        NEED_GRAPH: turn_5 before=2 after=1
        NEED_GRAPH: turn_8 before=3 after=0, turn_15 before=0 after=2

      Ranges
        NEED_GRAPH: turns 5 to 10
        NEED_GRAPH: turns 3 to 8, turns 15 to 20

      Individual indices
        NEED_GRAPH: turns 3, 7, 12, 18
    """
    turn_map: Dict[int, Dict[str, Any]] = {t.get("turn_idx", -1): t for t in trajectory}
    all_indices = sorted(turn_map.keys())
    requested: set = set()

    for line in response.splitlines():
        stripped = line.strip()
        if not re.match(r"NEED_GRAPH\s*:", stripped, re.IGNORECASE):
            continue

        spec = re.sub(r"^NEED_GRAPH\s*:\s*", "", stripped, flags=re.IGNORECASE).strip()

        for clause in spec.split(","):
            clause = clause.strip()
            if not clause:
                continue

            # Format A: turn_5 before=2 after=1
            m = re.match(
                r"turn[_\s](\d+)(?:\s+before=(\d+))?(?:\s+after=(\d+))?",
                clause,
                re.IGNORECASE,
            )
            if m:
                center = int(m.group(1))
                before = int(m.group(2)) if m.group(2) else 0
                after = int(m.group(3)) if m.group(3) else 0
                for idx in all_indices:
                    if center - before <= idx <= center + after:
                        requested.add(idx)
                continue

            # Format B: turns X to Y
            m = re.match(r"turns?\s+(\d+)\s+to\s+(\d+)", clause, re.IGNORECASE)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                for idx in all_indices:
                    if lo <= idx <= hi:
                        requested.add(idx)
                continue

            # Format C: turns X, Y, Z
            clause_clean = re.sub(r"^turns?\s*", "", clause, flags=re.IGNORECASE)
            for n in re.findall(r"\d+", clause_clean):
                idx = int(n)
                if idx in turn_map:
                    requested.add(idx)

    return _extract_chunks(trajectory, list(requested))


# ============================================================================
# NEED_CODE: keyword / code-execution search
# ============================================================================


def _run_keyword_search(
    trajectory_data: Dict[str, Any],
    question: str,
    task: str,
    call_llm_func: Callable,
    previous_error: str = "",
    timeout: float = 30.0,
) -> str:
    """
    LLM generates a Python script that operates on the trajectory; the script is
    executed in an isolated subprocess for precise keyword matching and
    statistical aggregation.

    Args:
        previous_error: Error output from a prior failed attempt; when non-empty
                        it is appended to the code-generation prompt so the LLM
                        can fix the mistake.
    """
    from .prompt import CODE_GENERATION_PROMPT_TEMPLATE

    trajectory = trajectory_data.get("trajectory", [])
    sample_chunks = [
        {
            "turn": t.get("turn_idx", i),
            "action": t.get("action", ""),
            "observation": t.get("observation", "")[:100],
        }
        for i, t in enumerate(trajectory[:3])
    ]
    code_prompt = CODE_GENERATION_PROMPT_TEMPLATE.format(
        query=question,
        task=task,
        trajectory_sample=_format_chunks(sample_chunks),
    )
    if previous_error:
        # Truncate to avoid blowing up context length (errors can contain full trajectory JSON)
        _MAX_ERROR_CHARS = 3000
        truncated_error = (
            previous_error
            if len(previous_error) <= _MAX_ERROR_CHARS
            else previous_error[:_MAX_ERROR_CHARS] + "\n...[truncated]"
        )
        code_prompt += (
            "\n\n**Previous attempt failed with the following error — fix it in your new code:**\n"
            f"```\n{truncated_error}\n```\n"
        )
    _, code_response = call_llm_func(code_prompt)
    code = extract_code_from_response(code_response)

    if not code:
        return "Keyword search: no code was generated."

    traj_json_str = json.dumps(trajectory_data)
    full_script = (
        f"trajectory_json = {repr(traj_json_str)}\n\n"
        f"{code}\n\n"
        "if 'result' in dir():\n"
        "    import json as _json\n"
        "    print(_json.dumps(result) if not isinstance(result, str) else result)"
    )

    os.makedirs("tmp", exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="mem_exec_", dir="tmp")
    script_path = os.path.join(tmpdir, "script.py")
    result = "Keyword search: empty result."
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(full_script)

        workspace_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        venv_python = os.path.join(workspace_root, "ama_venv", "bin", "python")
        python_executable = (
            venv_python if os.path.exists(venv_python) else sys.executable
        )

        proc = subprocess.run(
            [python_executable, script_path],
            capture_output=True,
            timeout=timeout,
            cwd=tmpdir,
        )
        stderr_str = proc.stderr.decode(errors="replace")
        stdout_str = proc.stdout.decode(errors="replace")
        if stderr_str.strip():
            # Cap error output so it never bloats the next code-gen prompt
            _MAX_STDERR = 2000
            _MAX_STDOUT_ERR = 1000
            _s = (
                stderr_str
                if len(stderr_str) <= _MAX_STDERR
                else stderr_str[:_MAX_STDERR] + "\n...[truncated]"
            )
            _o = (
                stdout_str
                if len(stdout_str) <= _MAX_STDOUT_ERR
                else stdout_str[:_MAX_STDOUT_ERR] + "\n...[truncated]"
            )
            result = f"error: {_s}\n\nSTDOUT:\n{_o}"
        else:
            result = stdout_str if stdout_str else "Keyword search: empty result."
    except subprocess.TimeoutExpired:
        result = "timeout"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return result
