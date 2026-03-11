import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import textworld
import tiktoken

from textworld_label_generator import TextWorldStateTracker, TextWorldQAGenerator


@dataclass
class DifficultyConfig:
    nb_rooms: int
    nb_objects: int
    quest_length: int
    min_depth: int
    max_depth: int
    min_breadth: int
    max_breadth: int


def parse_kv_args(argv: List[str]) -> Dict[str, str]:
    args: Dict[str, str] = {}
    for token in argv[1:]:
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if k:
            args[k] = v
    return args


def to_int(value: Optional[str], default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def count_trajectory_tokens(trajectory: List[Dict[str, Any]], task: str) -> int:
    """Count total tokens in a trajectory including task and all observations/actions."""
    total_text = task + "\n"
    for step in trajectory:
        total_text += f"Action: {step['action']}\n"
        total_text += f"Observation: {step['observation']}\n"
    return count_tokens(total_text)


def pick_preset(name: str) -> DifficultyConfig:
    key = (name or "").strip().lower()
    if key in {"easy", "e"}:
        return DifficultyConfig(
            nb_rooms=5,
            nb_objects=8,
            quest_length=4,
            min_depth=2,
            max_depth=3,
            min_breadth=1,
            max_breadth=1,
        )
    if key in {"medium", "m"}:
        return DifficultyConfig(
            nb_rooms=9,
            nb_objects=16,
            quest_length=8,
            min_depth=3,
            max_depth=5,
            min_breadth=1,
            max_breadth=2,
        )
    if key in {"hard", "h"}:
        return DifficultyConfig(
            nb_rooms=15,
            nb_objects=28,
            quest_length=14,
            min_depth=5,
            max_depth=9,
            min_breadth=2,
            max_breadth=3,
        )
    return pick_preset("medium")


def build_game(out_dir: Path, cfg: DifficultyConfig, seed: int) -> Tuple[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    options = textworld.GameOptions()
    options.seeds = seed

    options.nb_rooms = cfg.nb_rooms
    options.nb_objects = cfg.nb_objects

    options.quest_length = cfg.quest_length
    options.chaining.min_length = cfg.quest_length
    options.chaining.max_length = cfg.quest_length

    options.chaining.min_depth = cfg.min_depth
    options.chaining.max_depth = cfg.max_depth
    options.chaining.min_breadth = cfg.min_breadth
    options.chaining.max_breadth = cfg.max_breadth

    # Set the output path in options instead of passing it to make()
    options.path = str(out_dir)
    game_file, game = textworld.make(options)
    return game_file, game


def run_episode(
    env: Any,
    max_steps: int,
    seed: int,
    game_file: str,
    episode_id: str,
    policy: str = "random_admissible",
    mode: str = "validate",
) -> Dict[str, Any]:
    """
    Run a single episode and return trajectory in alfworld-compatible format.

    Args:
        mode: "validate" generates 10 QA pairs, "train" generates 50 QA pairs

    Returns:
        Episode dictionary with format:
        {
            "episode_id": str,
            "task": str,
            "task_type": str,
            "game_file": str,
            "state": str,
            "fail_reason": str,
            "num_turns": int,
            "total_tokens": int,
            "trajectory": [{"turn_idx": int, "action": str, "observation": str, "action_space": list}],
            "state_snapshots": list,
            "events": list,
            "qa_pairs": list
        }
    """
    rng = random.Random(seed)

    obs = env.reset()
    trajectory: List[Dict[str, Any]] = []

    # Extract task description from objective
    task = obs.get("objective", obs.get("description", ""))

    # Initialize StateTracker and QAGenerator
    tracker = TextWorldStateTracker()
    qa_generator = TextWorldQAGenerator(tracker, [], task)  # Empty trajectory for now

    done = False
    t = 0
    total_reward = 0.0

    while (not done) and (t < max_steps):
        admissible = obs.get("admissible_commands", [])
        policy_cmds = obs.get("policy_commands", [])

        # Use optimal policy commands if available, otherwise fall back to random
        if policy == "random_admissible" and isinstance(policy_cmds, list) and len(policy_cmds) > 0:
            action = policy_cmds[0]  # Get the first optimal action
        elif policy == "random_admissible" and isinstance(admissible, list) and len(admissible) > 0:
            action = rng.choice(admissible)
        else:
            action = "look"

        obs, reward, done = env.step(action)
        total_reward += reward

        # Extract observation text
        obs_text = obs.get("feedback", obs.get("description", ""))

        # Get updated action space after taking the action
        action_space = obs.get("admissible_commands", [])

        # Get facts for state tracking
        facts = obs.get("facts", [])
        inventory_text = obs.get("inventory", "")
        location = obs.get("location", "")

        # Update state tracker
        tracker.update(
            t=t,
            observation=obs_text,
            action=action,
            facts=facts,
            inventory_text=inventory_text,
            location=location,
            admissible_commands=action_space
        )

        # Add current action space to observation
        if action_space:
            action_space_str = ", ".join(action_space)
            obs_text += f"\n\nThe current available actions are: {action_space_str}"

        trajectory.append(
            {
                "turn_idx": t,
                "action": action,
                "observation": obs_text,
                #"action_space": action_space,
            }
        )
        
        # Generate question candidates for this step
        qa_generator.maybe_add_per_step(t)

        t += 1

    # Determine success based on reward and done status
    # In TextWorld, winning typically gives positive reward
    success = bool(done and total_reward > 0)

    # Count tokens
    total_tokens = count_trajectory_tokens(trajectory, task)

    # Update qa_generator trajectory and finalize QA pairs based on mode
    qa_generator.trajectory = trajectory
    target_qa_count = 10 if mode == "validate" else 50
    qa_pairs = qa_generator.generate_all(target_count=target_qa_count)

    # Create episode data in alfworld format
    episode = {
        "episode_id": episode_id,
        "task": task,
        "task_type": tracker.inferred_task_type or "textworld",
        "game_file": game_file,
        "state": "success" if success else "fail",
        "fail_reason": "" if success else ("reached_max_steps" if t >= max_steps else "unknown"),
        "num_turns": len(trajectory),
        "total_tokens": total_tokens,
        "trajectory": trajectory,
        "state_snapshots": tracker.step_snapshots,
        "events": tracker.events,
        "qa_pairs": qa_pairs,
    }

    return episode


def main() -> None:
    kv = parse_kv_args(sys.argv)

    difficulty = kv.get("difficulty", "medium")
    preset = pick_preset(difficulty)

    cfg = DifficultyConfig(
        nb_rooms=to_int(kv.get("rooms"), preset.nb_rooms),
        nb_objects=to_int(kv.get("objects"), preset.nb_objects),
        quest_length=to_int(kv.get("quest"), preset.quest_length),
        min_depth=to_int(kv.get("min_depth"), preset.min_depth),
        max_depth=to_int(kv.get("max_depth"), preset.max_depth),
        min_breadth=to_int(kv.get("min_breadth"), preset.min_breadth),
        max_breadth=to_int(kv.get("max_breadth"), preset.max_breadth),
    )

    seed = to_int(kv.get("seed"), 1234)
    max_steps = to_int(kv.get("max_steps"), 200)
    mode = kv.get("mode", "validate")

    out_dir = Path(kv.get("out_dir", "tw_out")).resolve()
    # Change output file extension to .jsonl for alfworld compatibility
    log_path = Path(kv.get("log", str(out_dir / "trajectory.jsonl"))).resolve()

    game_file, _ = build_game(out_dir=out_dir, cfg=cfg, seed=seed)

    request_infos = textworld.EnvInfos(
        description=True,
        inventory=True,
        objective=True,
        admissible_commands=True,
        policy_commands=True,
        facts=True,  # Enable facts for state tracking
    )

    env = textworld.start(game_file, request_infos=request_infos)

    try:
        # Generate episode_id from difficulty and seed
        episode_id = f"textworld_{difficulty}_{seed}"
        episode = run_episode(
            env=env,
            max_steps=max_steps,
            seed=seed,
            game_file=str(game_file),
            episode_id=episode_id,
            policy="random_admissible",
            mode=mode
        )
    finally:
        try:
            env.close()
        except Exception:
            pass

    # Save in JSONL format (one JSON object per line, like alfworld)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf8") as f:
        f.write(json.dumps(episode, ensure_ascii=False) + "\n")

    # Print summary
    print(f"Episode: {episode_id}")
    print(f"State: {episode['state']}")
    print(f"Turns: {episode['num_turns']}")
    print(f"Tokens: {episode['total_tokens']}")
    print(f"Saved to: {log_path}")


if __name__ == "__main__":
    main()
