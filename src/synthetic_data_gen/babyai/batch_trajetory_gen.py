"""
BabyAI Batch Trajectory Generation with Difficulty Levels

Usage:
python batch_trajetory_gen.py 
    --difficulty hard_large 
    --random_ratio 0     
    --observation_format natural     
    --traj_per_bin 2     
    --output_dir babyai_out_batch
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import random
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tiktoken

# ---------------------------------------------------------------------------
# MiniGrid / BabyAI grid encoding -> natural language
# Each cell in obs["image"] is (OBJECT_IDX, COLOR_IDX, STATE_IDX).
# Try to use minigrid's constants; fallback to standard encoding.
# ---------------------------------------------------------------------------
try:
    from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

    _IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
    _IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
    _IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
except ImportError:
    print("Warning: minigrid.core.constants not found, using default constants")
    _IDX_TO_OBJECT = {
        0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door",
        5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava", 10: "agent",
    }
    _IDX_TO_COLOR = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}
    _IDX_TO_STATE = {0: "open", 1: "closed", 2: "locked"}


ACTION_ID_TO_NAME = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}


@dataclass
class DifficultyConfig:
    """Configuration for a difficulty level"""
    env_ids: List[str]
    max_steps: int


@dataclass
class TokenBin:
    """Token bin for categorizing trajectories by length"""
    name: str
    min_tokens: int
    max_tokens: int
    count: int = 0
    target: int = 10


TOKEN_BINS = [
    TokenBin("8K", 4097, 8192, target=10),
    TokenBin("16K", 8193, 16384, target=10),
]


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def count_trajectory_tokens(trajectory: List[Dict[str, Any]], task: str) -> int:
    """Count total tokens in a trajectory including task and all observations/actions."""
    total_text = task + "\n"
    for step in trajectory:
        total_text += f"At turn {step['turn_idx']}:\n"
        total_text += f"The agent's observation: {step['observation']}\n"
        total_text += f"The action: {step['action']}\n"
    return count_tokens(total_text)


def get_difficulty_levels() -> List[Tuple[str, DifficultyConfig]]:
    """
    Return a list of difficulty configurations for BabyAI environments,
    ordered from easy to hard.
    """
    return [
        ("easy", DifficultyConfig(
            env_ids=["BabyAI-GoToLocal-v0", "BabyAI-GoToObj-v0",
                     "BabyAI-GoToRedBall-v0", "BabyAI-GoToRedBallGrey-v0"],
            max_steps=128,
        )),
        ("medium", DifficultyConfig(
            env_ids=["BabyAI-PutNextLocal-v0", "BabyAI-PickupLoc-v0",
                     "BabyAI-GoToSeq-v0", "BabyAI-Synth-v0"],
            max_steps=256,
        )),
        ("medium_hard", DifficultyConfig(
            env_ids=["BabyAI-GoToImpUnlock-v0", "BabyAI-UnblockPickup-v0",
                     "BabyAI-Unlock-v0", "BabyAI-KeyCorridor-v0"],
            max_steps=384,
        )),
        ("hard", DifficultyConfig(
            env_ids=["BabyAI-SynthLoc-v0", "BabyAI-SynthSeq-v0",
                     "BabyAI-ActionObjDoor-v0", "BabyAI-FindObjS7-v0",
                     "BabyAI-UnlockPickup-v0", "BabyAI-BlockedUnlockPickup-v0"],
            max_steps=1024,  # Increased to allow longer trajectories
        )),
        ("hard_large", DifficultyConfig(
            # Larger environments for generating longer trajectories
            env_ids=["BabyAI-FindObjS7-v0", "BabyAI-GoToObjMazeS7-v0",
                     "BabyAI-PutNextS7N4-v0", "BabyAI-GoToLocalS8N4-v0",
                     "BabyAI-GoToLocalS8N5-v0", "BabyAI-MoveTwoAcrossS8N9-v0"],
            max_steps=2048,  # Even more steps for large environments
        )),
        ("very_hard", DifficultyConfig(
            env_ids=["BabyAI-MiniBossLevel-v0", "BabyAI-BossLevel-v0",
                     "BabyAI-BossLevelNoUnlock-v0"],
            max_steps=1024,
        )),
    ]


def find_token_bin(token_count: int, bins: List[TokenBin]) -> Optional[TokenBin]:
    """Find which token bin a trajectory belongs to."""
    for bin_obj in bins:
        if bin_obj.min_tokens <= token_count <= bin_obj.max_tokens:
            return bin_obj
    return None


def are_all_bins_full(bins: List[TokenBin], bin_counts: Dict[str, int]) -> bool:
    return all(bin_counts.get(b.name, 0) >= b.target for b in bins)


_DIRECTION_TO_STR = {0: "east", 1: "south", 2: "west", 3: "north"}


def _decode_image_to_natural_language(image: Any, direction: int) -> str:
    """
    Decode MiniGrid/BabyAI obs['image'] (H,W,3) of (obj, color, state) into
    natural language. Uses only the observation array, no env access.
    """
    arr = np.asarray(image, dtype=np.int32)
    if arr.ndim != 3:
        return "You see the surrounding area."
    # Support (H,W,3) or (3,H,W)
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] != 3:
        return "You see the surrounding area."

    H, W, _ = arr.shape
    items: List[str] = []
    has_wall = False
    has_floor = False

    for i in range(H):
        for j in range(W):
            o, c, s = int(arr[i, j, 0]), int(arr[i, j, 1]), int(arr[i, j, 2])
            obj = _IDX_TO_OBJECT.get(o, "object")
            if obj in ("unseen", "empty"):
                continue
            if obj == "wall":
                has_wall = True
                continue
            if obj == "floor":
                has_floor = True
                continue
            if obj == "agent":
                continue

            color = _IDX_TO_COLOR.get(c, "")
            color = (color + " ") if color else ""
            if obj == "door":
                state = _IDX_TO_STATE.get(s, "closed")
                items.append(f"a {color}{obj} ({state})".strip())
            else:
                items.append(f"a {color}{obj}".strip())

    if not items and not has_wall and not has_floor:
        return "You see nothing."

    parts = [f"{n}x {desc}" if n > 1 else desc for desc, n in Counter(items).items()]

    view = "In your view: " + ", ".join(parts) + "."
    if has_wall:
        view += " Walls border the area."
    if has_floor and not (parts or has_wall):
        view = "In your view: floor and walls."
    dir_str = _DIRECTION_TO_STR.get(direction, "unknown")
    view += f" You are facing {dir_str}."
    return view


def _decode_image_to_grid(image: Any, direction: int) -> str:
    """
    Decode MiniGrid/BabyAI obs['image'] into a grid representation.
    """
    arr = np.asarray(image, dtype=np.int32)
    if arr.ndim != 3:
        return "Grid unavailable."
    # Support (H,W,3) or (3,H,W)
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] != 3:
        return "Grid unavailable."

    H, W, _ = arr.shape
    grid_lines = []
    dir_str = _DIRECTION_TO_STR.get(direction, "unknown")
    
    for i in range(H):
        row = []
        for j in range(W):
            o, c, s = int(arr[i, j, 0]), int(arr[i, j, 1]), int(arr[i, j, 2])
            obj = _IDX_TO_OBJECT.get(o, "?")
            if obj == "unseen":
                row.append("?")
            elif obj == "empty":
                row.append(".")
            elif obj == "wall":
                row.append("#")
            elif obj == "floor":
                row.append(".")
            elif obj == "agent":
                row.append("A")
            elif obj == "door":
                state = _IDX_TO_STATE.get(s, "closed")
                row.append("D" if state == "open" else "d")
            elif obj == "key":
                row.append("K")
            elif obj == "ball":
                row.append("B")
            elif obj == "box":
                row.append("X")
            elif obj == "goal":
                row.append("G")
            else:
                row.append("?")
        grid_lines.append("".join(row))
    
    grid_str = "\n".join(grid_lines)
    return f"Grid view (H={H}, W={W}):\n{grid_str}\nYou are facing {dir_str}."


def _decode_image_to_detailed(image: Any, direction: int) -> str:
    """
    Decode MiniGrid/BabyAI obs['image'] into detailed natural language with positions.
    """
    arr = np.asarray(image, dtype=np.int32)
    if arr.ndim != 3:
        return "You see the surrounding area."
    # Support (H,W,3) or (3,H,W)
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] != 3:
        return "You see the surrounding area."

    H, W, _ = arr.shape
    items_by_pos: List[str] = []
    agent_pos = None

    for i in range(H):
        for j in range(W):
            o, c, s = int(arr[i, j, 0]), int(arr[i, j, 1]), int(arr[i, j, 2])
            obj = _IDX_TO_OBJECT.get(o, "object")
            if obj in ("unseen", "empty", "floor"):
                continue
            if obj == "wall":
                items_by_pos.append(f"Wall at position ({i}, {j})")
                continue
            if obj == "agent":
                agent_pos = (i, j)
                continue

            color = _IDX_TO_COLOR.get(c, "")
            color = (color + " ") if color else ""
            if obj == "door":
                state = _IDX_TO_STATE.get(s, "closed")
                items_by_pos.append(f"{color}{obj} ({state}) at position ({i}, {j})".strip())
            else:
                items_by_pos.append(f"{color}{obj} at position ({i}, {j})".strip())

    parts = []
    if agent_pos:
        parts.append(f"Agent position: {agent_pos}")
    if items_by_pos:
        parts.append("Objects in view: " + "; ".join(items_by_pos))
    else:
        parts.append("No objects visible in the immediate view.")
    
    dir_str = _DIRECTION_TO_STR.get(direction, "unknown")
    parts.append(f"Facing direction: {dir_str}")
    
    return "\n".join(parts)


def format_observation(obs: Any, mission: str = "", observation_format: str = "natural") -> str:
    """
    Format observation as human-readable text.
    Converts the grid (obs['image']) to text; keeps mission and direction.
    
    Args:
        obs: Observation dictionary
        mission: Mission string
        observation_format: Format type - "natural", "grid", or "detailed"
    """
    parts: List[str] = []

    m = mission or (obs.get("mission", "") if isinstance(obs, dict) else "")
    if m:
        parts.append(f"Mission: {m}")

    direction = int(obs.get("direction", 0)) if isinstance(obs, dict) else 0

    if isinstance(obs, dict) and "image" in obs:
        try:
            if observation_format == "grid":
                view = _decode_image_to_grid(obs["image"], direction)
            elif observation_format == "detailed":
                view = _decode_image_to_detailed(obs["image"], direction)
            else:  # default to "natural"
                view = _decode_image_to_natural_language(obs["image"], direction)
            parts.append(view)
        except Exception:
            dir_str = _DIRECTION_TO_STR.get(direction, "unknown")
            parts.append(f"You are facing {dir_str}.")
    else:
        dir_str = _DIRECTION_TO_STR.get(direction, "unknown")
        parts.append(f"You are facing {dir_str}.")

    return "\n".join(parts) if parts else "Current state of the environment."


def _try_make_bot_with_module(module_name: str, env: Any) -> Optional[Any]:
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None

    candidates = []
    for attr in ["BabyAIBot", "Bot", "BabyAI_Bot", "ExpertBot", "Expert"]:
        if hasattr(mod, attr):
            candidates.append(getattr(mod, attr))

    for cls in candidates:
        try:
            bot = cls(env)
            return bot
        except TypeError:
            try:
                bot = cls(env.unwrapped)
                return bot
            except Exception:
                continue
        except Exception:
            continue

    return None


def _discover_and_make_babyai_bot(env: Any) -> Any:
    """
    Robust import across different package layouts
    We try common paths first, then scan minigrid package for a module containing Bot class
    """
    common_modules = [
        "minigrid.utils.baby_ai_bot",
        "minigrid.utils.babyai_bot",
        "minigrid.babyai.bot",
        "minigrid.bot",
        "babyai.bot",
    ]

    for m in common_modules:
        bot = _try_make_bot_with_module(m, env)
        if bot is not None:
            return bot

    try:
        import minigrid  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import minigrid, please install with: pip install minigrid"
        ) from e

    for _, modname, _ in pkgutil.walk_packages(minigrid.__path__, minigrid.__name__ + "."):
        name_low = modname.lower()
        if "bot" not in name_low:
            continue
        if "baby" not in name_low and "ai" not in name_low:
            continue
        bot = _try_make_bot_with_module(modname, env)
        if bot is not None:
            return bot

    raise RuntimeError(
        "Could not locate BabyAI expert bot in your installed packages. "
        "Try upgrading minigrid, or install the original babyai repo."
    )


def _random_action_fn(random_ratio: float = 1.0) -> Callable[[Any, Any], int]:
    """
    Return a pure random action function.
    
    Args:
        random_ratio: Probability of taking a random action (1.0 = pure random)
    """
    valid_actions = [0, 1, 2, 3, 4, 5, 6]  # left, right, forward, pickup, drop, toggle, done
    
    def action_fn(_obs: Any, _prev_action: Optional[int]) -> int:
        return random.choice(valid_actions)
    
    return action_fn


def _bot_action_fn(bot: Any, random_ratio: float = 0.0, use_pure_random: bool = False) -> Callable[[Any, Any], int]:
    """
    Return (obs, prev_action) -> action_id. BabyAIBot uses replan(prev_action);
    other bots use act(obs) / get_action(obs). prev_action is int or None.
    
    Args:
        bot: The bot instance (ignored if use_pure_random=True)
        random_ratio: Probability of taking a random action instead of bot action (0.0-1.0)
        use_pure_random: If True, return a pure random action function (ignores bot)
    """
    if use_pure_random:
        return _random_action_fn()
    
    def _extract_action(action: Any) -> int:
        """Extract action from tuple/list if needed."""
        if isinstance(action, (tuple, list)):
            return int(action[0])
        return int(action)
    
    # Valid actions: left, right, forward, pickup, drop, toggle, done
    valid_actions = [0, 1, 2, 3, 4, 5, 6]
    
    # BabyAIBot.replan(prev_action) must be called after env.reset(); it does not use obs.
    if hasattr(bot, "replan") and callable(getattr(bot, "replan")):
        fn = getattr(bot, "replan")
        def action_fn(_obs: Any, prev_action: Optional[int]) -> int:
            if random.random() < random_ratio:
                return random.choice(valid_actions)
            return _extract_action(fn(prev_action))
        return action_fn

    for method_name in ["get_action", "act", "step", "predict"]:
        if hasattr(bot, method_name) and callable(getattr(bot, method_name)):
            fn = getattr(bot, method_name)
            # Use a closure to capture fn, random_ratio, valid_actions, and _extract_action
            def make_action_fn(bot_fn: Callable, ratio: float, actions: List[int], extract_fn: Callable) -> Callable:
                def action_fn(obs: Any, _prev_action: Optional[int]) -> int:
                    if random.random() < ratio:
                        return random.choice(actions)
                    return extract_fn(bot_fn(obs))
                return action_fn
            return make_action_fn(fn, random_ratio, valid_actions, _extract_action)

    raise RuntimeError("Bot has no get_action, act, step, predict, or replan")


def generate_qa_pairs(
    trajectory: List[Dict[str, Any]],
    task: str,
    task_type: str,
    difficulty_name: str = "",
    seed: Optional[int] = None,
    **kwargs: Any,
) -> List[Dict[str, str]]:
    """
    Generate QA pairs for an episode. Override this function to add your own logic.

    Expected item format (aligned with TextWorld / benchmark):
        {"question": str, "answer": str, "type": str, "sub_type": str}
    Optional: "step" (int) to indicate which turn the QA refers to.

    Args:
        trajectory: list of {turn_idx, action, observation}
        task: mission string
        task_type: env_id (e.g. BabyAI-GoToLocal-v0)
        difficulty_name: difficulty level name
        seed: Random seed for QA generation reproducibility
        **kwargs: extra context (e.g. state_snapshots) when you extend

    Returns:
        List of QA dicts; default implementation returns [].
    """
    # Try to import and use the actual QA generator
    try:
        from babyai_qa_generator import generate_qa_for_trajectory
        trajectory_data = {
            "trajectory": trajectory,
            "task": task,
            "task_type": task_type,
        }
        return generate_qa_for_trajectory(trajectory_data, target_count=12, seed=seed)
    except ImportError:
        # If import fails, return empty list (backward compatibility)
        return []


def run_single_episode(
    env: Any,
    env_id: str,
    difficulty_name: str,
    seed: int,
    max_steps: int,
    episode_num: int,
    random_ratio: float = 0.0,
    observation_format: str = "natural",
    use_pure_random: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run a single episode with the expert bot and return trajectory in the target format.
    Bot is created after env.reset() so BabyAIBot can read env.unwrapped.instrs.

    Args:
        env: The environment instance
        env_id: Environment ID
        difficulty_name: Difficulty level name
        seed: Random seed
        max_steps: Maximum number of steps
        episode_num: Episode number
        random_ratio: Probability of taking random action (0.0-1.0)
        observation_format: Observation format ("natural", "grid", or "detailed")

    Returns:
        Episode dict with task, trajectory, and metadata, or None if failed
    """
    try:
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        obs, _ = env.reset(seed=seed)
        mission = obs.get("mission", "") if isinstance(obs, dict) else ""
        if use_pure_random:
            action_fn = _random_action_fn()
        else:
            bot = _discover_and_make_babyai_bot(env)
            action_fn = _bot_action_fn(bot, random_ratio=random_ratio, use_pure_random=False)

        trajectory: List[Dict[str, Any]] = []
        total_reward = 0.0
        prev_action: Optional[int] = None
        for t in range(max_steps):
            try:
                # Add timeout protection for action_fn call using threading
                action_result = [None]
                action_exception = [None]
                
                def action_wrapper():
                    try:
                        action_result[0] = action_fn(obs, prev_action)
                    except Exception as e:
                        action_exception[0] = e
                
                action_thread = threading.Thread(target=action_wrapper)
                action_thread.daemon = True
                action_thread.start()
                action_thread.join(timeout=5.0)  # 5 second timeout
                
                if action_thread.is_alive():
                    # Thread is still running, timeout occurred
                    print(f"  Warning: action_fn timed out at step {t} in {env_id}, returning None")
                    return None
                
                if action_exception[0] is not None:
                    raise action_exception[0]
                
                action_id = action_result[0]
                if action_id is None:
                    return None
            except Exception as e:
                return None

            action_name = ACTION_ID_TO_NAME.get(action_id, str(action_id))
            obs_next, reward, terminated, truncated, _ = env.step(action_id)
            total_reward += float(reward) if reward is not None else 0.0
            trajectory.append({
                "turn_idx": t,
                "action": action_name,
                "observation": format_observation(obs, mission, observation_format=observation_format),
            })

            obs = obs_next
            prev_action = action_id

            if terminated or truncated:
                break
        success = bool(terminated and total_reward > 0)
        episode_id = f"babyai_{difficulty_name}_{env_id}_{seed}_{episode_num}"
        total_tokens = count_trajectory_tokens(trajectory, mission)

        result = {
            "episode_id": episode_id,
            "task": mission,
            "task_type": env_id,
            "difficulty": difficulty_name,
            "state": "success" if success else "fail",
            "fail_reason": "" if success else ("truncated" if truncated else "reached_max_steps"),
            "success": success,
            "num_turns": len(trajectory),
            "total_tokens": total_tokens,
            "trajectory": trajectory,
            "qa_pairs": generate_qa_pairs(trajectory, mission, env_id, difficulty_name, seed=seed),
            "summary": {},
            "summary_scores": {},
            "source": "babyai",
            "random_ratio": random_ratio,
            "observation_format": observation_format,
        }
        return result

    except Exception as e:
        print(f"  Error in episode {episode_num}: {e}")
        return None


def run_single_episode_standalone(
    env_id: str,
    difficulty_name: str,
    seed: int,
    max_steps: int,
    episode_num: int,
    random_ratio: float = 0.0,
    observation_format: str = "natural",
    use_pure_random: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run a single episode with its own environment instance.
    
    Args:
        env_id: Environment ID
        difficulty_name: Difficulty level name
        seed: Random seed
        max_steps: Maximum number of steps
        episode_num: Episode number
        random_ratio: Probability of taking random action (0.0-1.0)
        observation_format: Observation format ("natural", "grid", or "detailed")
    
    Returns:
        Episode dict with task, trajectory, and metadata, or None if failed
    """
    try:
        import gymnasium as gym  # type: ignore
        import minigrid.envs.babyai  # noqa: F401  # register BabyAI envs
    except Exception as e:
        print(f"  Error importing dependencies in episode {episode_num}: {e}")
        return None
    
    env = None
    try:
        env = gym.make(env_id)
        result = run_single_episode(
            env=env,
            env_id=env_id,
            difficulty_name=difficulty_name,
            seed=seed,
            max_steps=max_steps,
            episode_num=episode_num,
            random_ratio=random_ratio,
            observation_format=observation_format,
            use_pure_random=use_pure_random,
        )
        return result
    except Exception as e:
        print(f"  Error in episode {episode_num}: {e}")
        return None
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def generate_trajectories_for_bin(
    difficulty_name: str,
    cfg: DifficultyConfig,
    random_ratio: float,
    observation_format: str,
    target_bin: TokenBin,
    output_dir: Path,
    base_seed: int,
    traj_per_bin: int,
    include_failures: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate trajectories for a specific token bin.
    Continues generating until we have enough trajectories in the target bin.
    Uses adaptive parameters for larger bins to generate longer trajectories.
    
    Args:
        include_failures: If True, accept failed trajectories (they're often longer)
    """
    episodes = []
    seed_counter = base_seed
    env_idx = 0
    max_attempts = traj_per_bin * 300  # Allow more attempts for larger bins
    consecutive_failures = 0
    consecutive_wrong_bin = 0
    MAX_CONSECUTIVE_FAILURES = 100
    MAX_CONSECUTIVE_WRONG_BIN = 300  # Increased for larger bins
    
    # Adaptive parameters for larger bins (16K+)
    is_large_bin = target_bin.min_tokens >= 8193
    bin_size_multiplier = {
        "8K": 1.5,
        "16K": 3.0,
        "32K": 4.0,
        "64K": 6.0,
        "128K": 8.0,
    }
    multiplier = bin_size_multiplier.get(target_bin.name, 1.0)
    adaptive_max_steps = int(cfg.max_steps * multiplier)
    
    # For larger bins, use high random ratio and larger environments
    if is_large_bin:
        adaptive_random_ratio = 0.9
        # Try to use hard_large config if available
        for name, large_cfg in get_difficulty_levels():
            if name == "hard_large":
                cfg = large_cfg
                adaptive_max_steps = int(cfg.max_steps * (multiplier / 2.5))
                break
    else:
        adaptive_random_ratio = min(0.8, random_ratio + (multiplier-1) * 0.5)
    
    print(f"  Generating for bin {target_bin.name} ({target_bin.min_tokens:,}-{target_bin.max_tokens:,} tokens)...")
    print(f"    Using adaptive max_steps: {adaptive_max_steps}, random_ratio: {adaptive_random_ratio:.2f}")
    if is_large_bin:
        print(f"    Using high random ratio and larger environments")
    if include_failures:
        print(f"    Accepting both success and failure trajectories")
    
    for attempt in range(1, max_attempts + 1):
        if len(episodes) >= traj_per_bin:
            break
        
        env_id = cfg.env_ids[env_idx]
        env_idx = (env_idx + 1) % len(cfg.env_ids)
        
        episode = run_single_episode_standalone(
            env_id, difficulty_name, seed_counter, adaptive_max_steps, len(episodes),
            random_ratio=adaptive_random_ratio,
            observation_format=observation_format,
            use_pure_random=is_large_bin,
        )
        seed_counter += 1
        
        if episode is None:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"    ⚠ Stopping: {consecutive_failures} consecutive failures")
                break
            continue
        
        consecutive_failures = 0
        token_count = episode["total_tokens"]
        token_bin = find_token_bin(token_count, [target_bin])
        
        # Accept episodes that fall into the target bin
        # If include_failures is True, accept both success and failure
        # If False, only accept success (but failures are often longer, so useful for large bins)
        if token_bin and token_bin.name == target_bin.name:
            if include_failures or episode["success"]:
                episodes.append(episode)
                consecutive_wrong_bin = 0  # Reset counter on success
                status_icon = "✓" if episode["success"] else "✗"
                print(f"    {status_icon} [{len(episodes)}/{traj_per_bin}] {episode['state']} | "
                      f"Turns: {episode['num_turns']} | Tokens: {token_count:,} ({target_bin.name})")
            else:
                # Episode in right bin but failed, and we don't accept failures
                consecutive_wrong_bin += 1
                if attempt % 20 == 0:
                    print(f"    ... Attempt {attempt}: episode failed (tokens {token_count:,} in bin {target_bin.name})")
        else:
            # Token count out of target bin range, skip
            consecutive_wrong_bin += 1
            if consecutive_wrong_bin >= MAX_CONSECUTIVE_WRONG_BIN:
                print(f"    ⚠ Stopping: {consecutive_wrong_bin} consecutive episodes not in target bin {target_bin.name}")
                print(f"    Last token count: {token_count:,} (target: {target_bin.min_tokens:,}-{target_bin.max_tokens:,})")
                break
            if attempt % 20 == 0:
                print(f"    ... Attempt {attempt}: tokens {token_count:,} not in bin {target_bin.name} (need {target_bin.min_tokens:,}-{target_bin.max_tokens:,})")
    
    if len(episodes) < traj_per_bin:
        print(f"    ⚠ Warning: Only generated {len(episodes)}/{traj_per_bin} trajectories for bin {target_bin.name}")
    
    return episodes


def generate_trajectories_for_config(
    difficulty_name: str,
    cfg: DifficultyConfig,
    random_ratio: float,
    observation_format: str,
    output_dir: Path,
    base_seed: int,
    traj_per_bin: int,
    bins: List[TokenBin],
) -> List[Dict[str, Any]]:
    """
    Generate trajectories for a specific configuration (difficulty, random_ratio, observation_format).
    Generates trajectories for each token bin (8K, 16K, 32K, 64K, 128K).
    """
    # Create filename prefix with parameter abbreviations
    diff_abbr = {"easy": "e", "medium": "m", "medium_hard": "mh"}.get(difficulty_name, difficulty_name[:2])
    ratio_abbr = f"r{int(random_ratio * 10)}"  # r0, r1, r2 for 0.0, 0.1, 0.2
    format_abbr = {"natural": "nat", "grid": "grd"}.get(observation_format, observation_format[:3])
    
    print(f"\n{'='*60}")
    print(f"Configuration: {difficulty_name} | Random: {random_ratio} | Format: {observation_format}")
    print(f"File prefix: {diff_abbr}_{ratio_abbr}_{format_abbr}")
    print(f"Trajectories per bin: {traj_per_bin}")
    print(f"{'='*60}\n")

    all_episodes = []
    current_seed = base_seed
    
    # Generate trajectories for each bin
    for bin_obj in bins:
        include_failures = bin_obj.min_tokens >= 8193  # 16K+ accept failures
        bin_episodes = generate_trajectories_for_bin(
            difficulty_name=difficulty_name,
            cfg=cfg,
            random_ratio=random_ratio,
            observation_format=observation_format,
            target_bin=bin_obj,
            output_dir=output_dir,
            base_seed=current_seed,
            traj_per_bin=traj_per_bin,
            include_failures=include_failures,
        )
        
        # Save each episode to a separate file
        for idx, episode in enumerate(bin_episodes):
            filename = f"{diff_abbr}_{ratio_abbr}_{format_abbr}_{bin_obj.name}_{idx}.json"
            output_file = output_dir / filename
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(episode, f, ensure_ascii=False, indent=2)
        
        all_episodes.extend(bin_episodes)
        current_seed += len(bin_episodes) * 10  # Increment seed for next bin
        
        print(f"  ✓ Bin {bin_obj.name}: {len(bin_episodes)}/{traj_per_bin} trajectories generated\n")
    
    print(f"  Completed: {len(all_episodes)} total trajectories across {len(bins)} bins\n")
    return all_episodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BabyAI trajectories for specified configuration across 5 token bins (8K-128K)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="babyai_out_batch",
        help="Output directory for generated trajectories"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        required=True,
        choices=["easy", "medium", "medium_hard", "hard", "very_hard", "hard_large"],
        help="Difficulty level: easy, medium, medium_hard, hard, very_hard, or hard_large"
    )
    parser.add_argument(
        "--random_ratio",
        type=float,
        required=True,
        help="Probability of taking random action (0.0-1.0, e.g., 0.0, 0.1, 0.2)"
    )
    parser.add_argument(
        "--observation_format",
        type=str,
        required=True,
        choices=["natural", "grid"],
        help="Observation format: 'natural' or 'grid'"
    )
    parser.add_argument(
        "--traj_per_bin",
        type=int,
        required=True,
        help="Number of trajectories to generate per token bin"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for trajectory generation (default: 42)"
    )
    args = parser.parse_args()

    # Validate random_ratio
    if not 0.0 <= args.random_ratio <= 1.0:
        parser.error("--random_ratio must be between 0.0 and 1.0")

    # Set global random seeds for reproducibility
    base_seed = args.seed
    random.seed(base_seed)
    np.random.seed(base_seed)

    output_base_dir = Path(args.output_dir).resolve()
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Get difficulty configuration
    difficulty_levels = get_difficulty_levels()
    difficulty_config = None
    for name, cfg in difficulty_levels:
        if name == args.difficulty:
            difficulty_config = (name, cfg)
            break
    
    if difficulty_config is None:
        print(f"Error: Difficulty level '{args.difficulty}' not found")
        return

    difficulty_name, cfg = difficulty_config

    # Use 5 token bins (8K, 16K, 32K, 64K, 128K)
    bins = [TokenBin(b.name, b.min_tokens, b.max_tokens, 0, args.traj_per_bin) for b in TOKEN_BINS]

    print(f"\n{'='*60}")
    print("CONFIGURATION")
    print(f"{'='*60}")
    print(f"Difficulty: {difficulty_name}")
    print(f"Random ratio: {args.random_ratio}")
    print(f"Observation format: {args.observation_format}")
    print(f"Trajectories per bin: {args.traj_per_bin}")
    print(f"Base seed: {base_seed}")
    print(f"Token bins: {[b.name for b in bins]}")
    print(f"Total trajectories: {len(bins) * args.traj_per_bin}")
    print(f"{'='*60}\n")

    # Generate trajectories for the specified configuration
    all_episodes = generate_trajectories_for_config(
        difficulty_name=difficulty_name,
        cfg=cfg,
        random_ratio=args.random_ratio,
        observation_format=args.observation_format,
        output_dir=output_base_dir,
        base_seed=base_seed,
        traj_per_bin=args.traj_per_bin,
        bins=bins,
    )

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total trajectories generated: {len(all_episodes)}")
    print(f"Expected: {len(bins) * args.traj_per_bin}")
    
    if all_episodes:
        success_count = sum(1 for e in all_episodes if e["state"] == "success")
        fail_count = len(all_episodes) - success_count
        success_rate = success_count / len(all_episodes) * 100
        avg_tokens = sum(e["total_tokens"] for e in all_episodes) / len(all_episodes)
        print(f"Success: {success_count} | Failed: {fail_count} | Success rate: {success_rate:.1f}%")
        print(f"Average tokens: {avg_tokens:,.0f}")
        
        # Token bins distribution
        bin_counts_summary = defaultdict(int)
        for episode in all_episodes:
            bin_obj = find_token_bin(episode["total_tokens"], bins)
            if bin_obj:
                bin_counts_summary[bin_obj.name] += 1
        print(f"\nToken bins distribution:")
        for bin_obj in bins:
            count = bin_counts_summary.get(bin_obj.name, 0)
            print(f"  {bin_obj.name}: {count}/{args.traj_per_bin}")
    
    print(f"\nOutput directory: {output_base_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
