import asyncio
import json
import os
import random
import shutil
import sys
import tempfile
from collections import defaultdict
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


@dataclass
class TokenBin:
    name: str
    min_tokens: int
    max_tokens: int
    count: int = 0
    target: int = 10


# Token bins definition
TOKEN_BINS = [
    TokenBin("4K", 0, 4096, target=10),
    TokenBin("8K", 4097, 8192, target=10),
    TokenBin("16K", 8193, 16384, target=10),
    TokenBin("32K", 16385, 32768, target=10),
    TokenBin("64K", 32769, 65536, target=10),
    TokenBin("128K", 65537, 131072, target=10),
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


def get_difficulty_levels(game_type: str) -> List[Tuple[str, DifficultyConfig]]:
    """
    Return a list of difficulty configurations for each game type,
    ordered from easy to hard.
    """
    if game_type == "coin_collector":
        return [
            ("easy", DifficultyConfig(6, 8, 5, 2, 4, 1, 2)),
            ("medium", DifficultyConfig(10, 12, 7, 3, 5, 1, 2)),
            ("medium_hard", DifficultyConfig(15, 20, 10, 4, 7, 2, 3)),
            ("hard", DifficultyConfig(20, 28, 15, 5, 9, 2, 3)),
            ("very_hard", DifficultyConfig(24, 32, 18, 6, 10, 2, 3)),
            ("extreme", DifficultyConfig(28, 38, 22, 7, 12, 3, 4)),
            ("ultra", DifficultyConfig(35, 48, 28, 8, 14, 3, 4)),
            ("mega", DifficultyConfig(42, 58, 35, 10, 18, 4, 5)),
        ]
    elif game_type == "cooking":
        return [
            ("easy", DifficultyConfig(5, 10, 5, 2, 4, 1, 2)),
            ("medium", DifficultyConfig(8, 15, 8, 3, 5, 1, 2)),
            ("medium_hard", DifficultyConfig(12, 22, 12, 4, 7, 2, 3)),
            ("hard", DifficultyConfig(18, 30, 18, 6, 10, 2, 3)),
            ("very_hard", DifficultyConfig(22, 36, 22, 7, 12, 2, 3)),
            ("extreme", DifficultyConfig(28, 45, 28, 9, 15, 3, 4)),
            ("ultra", DifficultyConfig(35, 55, 35, 11, 18, 3, 4)),
            ("mega", DifficultyConfig(42, 68, 45, 14, 22, 4, 5)),
        ]
    elif game_type == "treasure_hunter":
        return [
            ("easy", DifficultyConfig(7, 10, 5, 2, 4, 1, 2)),
            ("medium", DifficultyConfig(12, 18, 9, 3, 6, 1, 2)),
            ("medium_hard", DifficultyConfig(18, 26, 14, 5, 8, 2, 3)),
            ("hard", DifficultyConfig(25, 35, 20, 7, 12, 2, 3)),
            ("very_hard", DifficultyConfig(30, 42, 24, 8, 14, 2, 3)),
            ("extreme", DifficultyConfig(36, 52, 30, 10, 17, 3, 4)),
            ("ultra", DifficultyConfig(45, 65, 38, 12, 21, 3, 4)),
            ("mega", DifficultyConfig(55, 80, 48, 15, 26, 4, 5)),
        ]
    else:
        raise ValueError(f"Unknown game type: {game_type}")


def expand_observation_verbose(
    obs_text: str,
    t: int,
    location: str,
    inventory_text: str,
    action_space: List[str],
    facts: List[Any],
    objective: str,
    verbosity: str = "high",
) -> str:
    """
    Expand observation with verbose state information to increase token count.
    
    Args:
        verbosity: "low" (minimal), "medium" (moderate), "high" (very detailed)
    """
    expanded = []
    
    if verbosity == "low":
        expanded.append(obs_text)
        if action_space:
            action_summary = ", ".join(action_space[:10])
            if len(action_space) > 10:
                action_summary += f", ... ({len(action_space)} total)"
            expanded.append(f"\nAvailable actions: {action_summary}")
        return "\n".join(expanded)
    
    if verbosity in ["medium", "high"]:
        expanded.append(f"=== Step {t} - Environment State ===")
    
    expanded.append(f"\n[Observation at Turn {t}]")
    expanded.append(obs_text)
    
    if objective and verbosity in ["medium", "high"]:
        expanded.append(f"\n[Current Task]")
        expanded.append(f"Objective: {objective}")
    
    if action_space:
        expanded.append(f"\n[Available Actions]")
        if verbosity == "high":
            expanded.append(f"The agent has {len(action_space)} possible actions available at this step.")
            expanded.append("These actions represent all valid commands the agent can execute given the current state:")
            for i, action in enumerate(action_space):
                expanded.append(f"  {i+1}. {action}")
        else:
            expanded.append(f"Total available: {len(action_space)} actions")
            for i, action in enumerate(action_space[:20]):
                expanded.append(f"  {i+1}. {action}")
            if len(action_space) > 20:
                expanded.append(f"  ... and {len(action_space) - 20} more actions")
    
    return "\n".join(expanded)


def build_game(
    out_dir: Path,
    cfg: DifficultyConfig,
    seed: int,
    game_type: str
) -> Tuple[str, Any]:
    """Build a TextWorld game with specific configuration."""
    out_dir.mkdir(parents=True, exist_ok=True)

    options = textworld.GameOptions()
    options.seeds = seed

    options.nb_rooms = cfg.nb_rooms
    options.nb_objects = cfg.nb_objects

    options.quest_length = cfg.quest_length
    # Allow some flexibility in quest length
    options.chaining.min_length = max(1, cfg.quest_length - 2)
    options.chaining.max_length = cfg.quest_length + 2

    options.chaining.min_depth = cfg.min_depth
    options.chaining.max_depth = cfg.max_depth
    options.chaining.min_breadth = cfg.min_breadth
    options.chaining.max_breadth = cfg.max_breadth

    # Customize game type
    if game_type == "coin_collector":
        options.grammar.theme = "house"
        options.grammar.include_adj = False
    elif game_type == "cooking":
        options.grammar.theme = "house"
        options.grammar.include_adj = True
    elif game_type == "treasure_hunter":
        options.grammar.theme = "house"
        options.grammar.include_adj = True

    options.path = str(out_dir)
    game_file, game = textworld.make(options)
    return game_file, game


async def run_episode(
    env: Any,
    max_steps: int,
    seed: int,
    game_file: str,
    episode_id: str,
    game_type: str,
    difficulty_name: str = "medium",
    mode: str = "validate",
) -> Dict[str, Any]:
    """Run a single episode and return trajectory.
    
    Args:
        difficulty_name: Difficulty level name, used to determine verbosity
        mode: "validate" generates 10 QA pairs, "train" generates 50 QA pairs
    """
    verbosity_map = {
        "easy": "low",
        "medium": "low",
        "medium_hard": "medium",
        "hard": "medium",
        "very_hard": "high",
        "extreme": "high",
        "ultra": "high",
        "mega": "high",
    }
    verbosity_level = verbosity_map.get(difficulty_name, "medium")
    rng = random.Random(seed)

    obs = await asyncio.to_thread(env.reset)
    trajectory: List[Dict[str, Any]] = []

    task = obs.get("objective", obs.get("description", ""))
    initial_facts = obs.get("facts", [])

    # Initialize StateTracker and QAGenerator with initial facts
    tracker = TextWorldStateTracker(initial_facts=initial_facts, task=task)
    qa_generator = TextWorldQAGenerator(tracker, [], task)

    done = False
    t = 0
    total_reward = 0.0

    while (not done) and (t < max_steps):
        admissible = obs.get("admissible_commands", [])
        if isinstance(admissible, list) and len(admissible) > 0:
            action = rng.choice(admissible)
        else:
            action = "look"

        obs, reward, done = await asyncio.to_thread(env.step, action)
        total_reward += reward

        obs_text = obs.get("feedback", obs.get("description", ""))
        action_space = obs.get("admissible_commands", [])

        facts = obs.get("facts", [])
        inventory_text = obs.get("inventory", "")
        location = obs.get("location", "")
        objective = obs.get("objective", task)

        tracker.update(
            t=t,
            observation=obs_text,
            action=action,
            facts=facts,
            inventory_text=inventory_text,
            location=location,
            admissible_commands=action_space
        )

        verbose_obs = expand_observation_verbose(
            obs_text=obs_text,
            t=t,
            location=location,
            inventory_text=inventory_text,
            action_space=action_space,
            facts=facts,
            objective=objective,
            verbosity=verbosity_level,
        )

        trajectory.append(
            {
                "turn_idx": t,
                "action": action,
                "observation": verbose_obs,
            }
        )

        qa_generator.maybe_add_per_step(t)

        t += 1

    success = bool(done and total_reward > 0)
    total_tokens = count_trajectory_tokens(trajectory, task)

    qa_generator.trajectory = trajectory
    target_qa_count = 10 if mode == "validate" else 50
    qa_pairs = qa_generator.generate_all(target_count=target_qa_count)

    episode = {
        "episode_id": episode_id,
        "task": task,
        "task_type": tracker.inferred_task_type or game_type,
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


def find_token_bin(token_count: int, bins: List[TokenBin]) -> Optional[TokenBin]:
    """Find which token bin a trajectory belongs to."""
    for bin_obj in bins:
        if bin_obj.min_tokens <= token_count <= bin_obj.max_tokens:
            return bin_obj
    return None


async def test_difficulty_level_async(
    cfg: DifficultyConfig,
    game_type: str,
    seed: int,
    request_infos: Any,
    bins: List[TokenBin],
    bin_counts: Dict[str, int],
    mode: str = "validate",
    test_samples: int = 3,
    min_avg_tokens: int = 8000,
) -> Tuple[bool, float, int]:
    """
    Test a difficulty level with a few samples to see if it produces long enough trajectories.

    Args:
        cfg: Difficulty configuration
        game_type: Type of game
        seed: Starting seed for tests
        request_infos: TextWorld environment info settings
        bins: Token bins for classification
        bin_counts: Current bin counts
        mode: Generation mode
        test_samples: Number of test samples to generate (default 3)
        min_avg_tokens: Minimum average tokens to consider difficulty acceptable (default 8000)

    Returns:
        (should_skip, avg_tokens, acceptable_count):
            - should_skip: True if should skip this difficulty and increase
            - avg_tokens: Average token count from test samples
            - acceptable_count: Number of samples that would be accepted
    """
    test_tokens = []
    acceptable_count = 0

    print(f"  Testing difficulty with {test_samples} samples...")

    test_seeds = [seed + i for i in range(test_samples)]
    episodes = await generate_trajectories_batch_async(
        game_type=game_type,
        difficulty_name="test",
        cfg=cfg,
        seeds=test_seeds,
        request_infos=request_infos,
        mode=mode,
        max_concurrent=test_samples,
    )

    for episode in episodes:
        token_count = episode["total_tokens"]
        test_tokens.append(token_count)

        token_bin = find_token_bin(token_count, bins)
        if token_bin and bin_counts[token_bin.name] < 10:
            acceptable_count += 1

    if not test_tokens:
        return False, 0.0, 0

    avg_tokens = sum(test_tokens) / len(test_tokens)
    should_skip = avg_tokens < min_avg_tokens or acceptable_count == 0

    print(f"    Test results: avg_tokens={avg_tokens:.0f}, acceptable={acceptable_count}/{len(test_tokens)}, "
          f"decision={'SKIP (too short)' if should_skip else 'CONTINUE (good length)'}")

    return should_skip, avg_tokens, acceptable_count


async def generate_single_episode_async(
    game_type: str,
    difficulty_name: str,
    cfg: DifficultyConfig,
    seed: int,
    request_infos: Any,
    mode: str = "validate",
) -> Optional[Dict[str, Any]]:
    """Generate a single episode: build game and run rollout together.
    
    Args:
        game_type: Type of game
        difficulty_name: Difficulty level name
        cfg: Difficulty configuration
        seed: Seed for this episode
        request_infos: TextWorld environment info settings
        mode: Generation mode
    
    Returns:
        Generated episode or None if failed
    """
    temp_dir = None
    env = None
    
    try:
        temp_dir = tempfile.mkdtemp()
        game_dir = Path(temp_dir) / f"game_{seed}"
        
        game_file, _ = await asyncio.to_thread(
            build_game,
            out_dir=game_dir,
            cfg=cfg,
            seed=seed,
            game_type=game_type
        )
        
        env = textworld.start(game_file, request_infos=request_infos)
        
        base_steps = 100 + cfg.quest_length * 15
        if difficulty_name in ["mega", "ultra"]:
            max_steps = min(base_steps, 2500)
        elif difficulty_name in ["extreme", "very_hard"]:
            max_steps = min(base_steps, 1500)
        else:
            max_steps = min(base_steps, 1000)
        
        episode_id = f"{game_type}_{difficulty_name}_{seed}"
        
        episode = await run_episode(
            env=env,
            max_steps=max_steps,
            seed=seed,
            game_file=str(game_file),
            episode_id=episode_id,
            game_type=game_type,
            difficulty_name=difficulty_name,
            mode=mode,
        )
        
        return episode
        
    except Exception as e:
        return None
        
    finally:
        if env:
            try:
                env.close()
            except Exception:
                pass
        
        if temp_dir:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


async def generate_trajectories_batch_async(
    game_type: str,
    difficulty_name: str,
    cfg: DifficultyConfig,
    seeds: List[int],
    request_infos: Any,
    mode: str = "validate",
    max_concurrent: int = 50,
) -> List[Dict[str, Any]]:
    """Generate multiple trajectories concurrently.
    Each episode builds its game and runs rollout together.
    
    Args:
        game_type: Type of game
        difficulty_name: Difficulty level name
        cfg: Difficulty configuration
        seeds: List of seeds for episode generation
        request_infos: TextWorld environment info settings
        mode: Generation mode
        max_concurrent: Maximum number of episodes to run concurrently
    
    Returns:
        List of generated episodes (excluding None/failed episodes)
    """
    tasks = [
        generate_single_episode_async(
            game_type=game_type,
            difficulty_name=difficulty_name,
            cfg=cfg,
            seed=seed,
            request_infos=request_infos,
            mode=mode,
        )
        for seed in seeds
    ]
    
    results = []
    for i in range(0, len(tasks), max_concurrent):
        batch_tasks = tasks[i:i + max_concurrent]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                continue
            elif result is not None:
                results.append(result)
    
    return results


def choose_difficulty_for_bins(
    bin_counts: Dict[str, int],
    difficulty_levels: List[Tuple[str, DifficultyConfig]],
) -> Tuple[int, str]:
    """
    Choose difficulty based on which bins need filling with variance.
    
    Strategy (with bias towards harder difficulties for larger bins):
    - 4K, 8K bins: easy or medium (quick success = short)
    - 16K bin: medium or medium_hard
    - 32K bin: medium_hard or hard
    - 64K bin: very_hard or extreme (high variance, longer episodes)
    - 128K bin: extreme, ultra, or mega (maximum variance and length)
    
    Returns:
        (difficulty_idx, reason)
    """
    import random
    
    unfilled_bins = [(name, 10 - count) for name, count in bin_counts.items() if count < 10]
    
    if not unfilled_bins:
        return 0, "all_full"
    
    unfilled_bins.sort(key=lambda x: x[1], reverse=True)
    most_needed_bin = unfilled_bins[0][0]
    
    bin_to_difficulty_options = {
        "4K": [(0, 0.7), (1, 0.3)],
        "8K": [(0, 0.6), (1, 0.4)],
        "16K": [(1, 0.6), (2, 0.4)],
        "32K": [(2, 0.5), (3, 0.5)],
        "64K": [(4, 0.4), (5, 0.4), (6, 0.2)],
        "128K": [(5, 0.2), (6, 0.4), (7, 0.4)],
    }
    
    difficulty_options = bin_to_difficulty_options.get(most_needed_bin, [(1, 1.0)])
    difficulty_indices = [idx for idx, _ in difficulty_options]
    weights = [weight for _, weight in difficulty_options]
    
    target_idx = random.choices(difficulty_indices, weights=weights)[0]
    target_idx = min(target_idx, len(difficulty_levels) - 1)
    
    difficulty_name = difficulty_levels[target_idx][0]
    
    return target_idx, f"targeting_{most_needed_bin}_needs_{10-bin_counts[most_needed_bin]}_more_using_{difficulty_name}"


def generate_trajectories_for_game_type(
    game_type: str,
    target_count: int,
    output_dir: Path,
    base_seed: int = 1000,
    mode: str = "validate",
) -> List[Dict[str, Any]]:
    """
    Generate trajectories for a specific game type with bin-targeted difficulty selection.

    Args:
        game_type: One of 'coin_collector', 'cooking', 'treasure_hunter'
        target_count: Number of trajectories to generate (60)
        output_dir: Directory to save game files
        base_seed: Starting seed value
        mode: "validate" generates 10 QA pairs per trajectory, "train" generates 50

    Returns:
        List of generated episodes
    """
    print(f"\n{'='*60}")
    print(f"Generating {target_count} trajectories for {game_type}")
    print(f"{'='*60}\n")

    episodes = []
    difficulty_levels = get_difficulty_levels(game_type)

    bins = [TokenBin(b.name, b.min_tokens, b.max_tokens, 0, b.target) for b in TOKEN_BINS]
    bin_counts = {b.name: 0 for b in bins}

    game_output_dir = output_dir / game_type
    game_output_dir.mkdir(parents=True, exist_ok=True)

    seed = base_seed
    generated_count = 0

    request_infos = textworld.EnvInfos(
        description=True,
        inventory=True,
        objective=True,
        admissible_commands=True,
        policy_commands=True,
        facts=True,
    )

    batch_size = 50
    max_concurrent = 20
    attempts_per_difficulty = {}
    max_attempts_per_bin = 30
    
    while generated_count < target_count:
        difficulty_idx, reason = choose_difficulty_for_bins(bin_counts, difficulty_levels)
        difficulty_name, cfg = difficulty_levels[difficulty_idx]
        
        if difficulty_name not in attempts_per_difficulty:
            attempts_per_difficulty[difficulty_name] = 0
            print(f"\n→ Selected difficulty: {difficulty_name} ({reason})")
            print(f"   Current bin status: {dict(bin_counts)}\n")
        
        if attempts_per_difficulty[difficulty_name] >= max_attempts_per_bin:
            print(f"⚠ Max attempts reached for {difficulty_name}, forcing next batch...")
        
        batch_seeds = list(range(seed, seed + batch_size))
        attempts_per_difficulty[difficulty_name] += batch_size
        
        try:
            print(f"Generating batch of {batch_size} episodes at {difficulty_name} (seeds {seed}-{seed+batch_size-1})...")
            batch_episodes = asyncio.run(generate_trajectories_batch_async(
                game_type=game_type,
                difficulty_name=difficulty_name,
                cfg=cfg,
                seeds=batch_seeds,
                request_infos=request_infos,
                mode=mode,
                max_concurrent=max_concurrent,
            ))
            
            for episode in batch_episodes:
                token_count = episode["total_tokens"]
                token_bin = find_token_bin(token_count, bins)
                
                if token_bin and bin_counts[token_bin.name] < 10:
                    episodes.append(episode)
                    bin_counts[token_bin.name] += 1
                    generated_count += 1
                    
                    output_file = game_output_dir / f"{game_type}_{generated_count-1}.json"
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(episode, f, ensure_ascii=False, indent=2)
                    
                    print(f"✓ Generated {generated_count}/{target_count}: "
                          f"{episode['episode_id']} | "
                          f"{episode['state']} | "
                          f"Turns: {episode['num_turns']} | "
                          f"Tokens: {token_count:,} ({token_bin.name}) | "
                          f"Saved: {output_file.name}")
                    
                    if generated_count % 10 == 0:
                        print(f"\n  Bin Status: {dict(bin_counts)}\n")
                    
                    if generated_count >= target_count:
                        break
                elif token_bin and bin_counts[token_bin.name] >= 10:
                    print(f"✗ Skipped (bin {token_bin.name} is full: {bin_counts[token_bin.name]}/10, tokens: {token_count:,})")
                elif not token_bin:
                    print(f"✗ Skipped (tokens {token_count:,} out of range)")
                    
        except Exception as e:
            print(f"✗ Error generating batch (seeds {seed}-{seed+batch_size-1}): {e}")
        
        seed += batch_size
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"Completed {game_type}: {generated_count} trajectories")
    print(f"Final bin distribution:")
    for bin_name, count in sorted(bin_counts.items()):
        print(f"  {bin_name}: {count} trajectories")
    print(f"{'='*60}\n")
    
    return episodes


def main():
    """Main entry point for batch generation."""
    import sys
    
    # Parse command line arguments
    mode = "validate"  # default
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("mode="):
                mode = arg.split("=", 1)[1]
    
    output_base_dir = Path("tw_out_batch").resolve()
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Game types and their targets
    game_configs = [
        ("coin_collector", 60, 1000),
        ("cooking", 60, 2000),
        ("treasure_hunter", 60, 3000),
    ]
    
    all_episodes = []
    
    for game_type, target_count, base_seed in game_configs:
        episodes = generate_trajectories_for_game_type(
            game_type=game_type,
            target_count=target_count,
            output_dir=output_base_dir,
            base_seed=base_seed,
            mode=mode,
        )
        all_episodes.extend(episodes)
    
    # Save all trajectories to a single JSONL file for easy loading
    output_file = output_base_dir / "all_trajectories.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for episode in all_episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")
    
    # Note: Individual JSON files have already been saved during generation
    print(f"\nAll trajectories also saved to: {output_file}")
    
    # Generate summary statistics
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total trajectories generated: {len(all_episodes)}")
    
    # Count by game type
    print("\nBy game type:")
    for game_type in ["coin_collector", "cooking", "treasure_hunter"]:
        count = sum(1 for e in all_episodes if e["task_type"] == game_type)
        print(f"  {game_type}: {count}")
    
    # Count by token bins
    print("\nBy token bins (across all game types):")
    overall_bin_counts = defaultdict(int)
    for episode in all_episodes:
        token_count = episode["total_tokens"]
        bin_obj = find_token_bin(token_count, TOKEN_BINS)
        if bin_obj:
            overall_bin_counts[bin_obj.name] += 1
    
    for bin_name in ["4K", "8K", "16K", "32K", "64K", "128K"]:
        count = overall_bin_counts[bin_name]
        print(f"  {bin_name}: {count}")
    
    # Success rate
    success_count = sum(1 for e in all_episodes if e["state"] == "success")
    success_rate = (success_count / len(all_episodes) * 100) if all_episodes else 0
    print(f"\nSuccess rate: {success_rate:.1f}% ({success_count}/{len(all_episodes)})")
    
    # Average tokens and turns
    avg_tokens = sum(e["total_tokens"] for e in all_episodes) / len(all_episodes) if all_episodes else 0
    avg_turns = sum(e["num_turns"] for e in all_episodes) / len(all_episodes) if all_episodes else 0
    print(f"Average tokens: {avg_tokens:,.0f}")
    print(f"Average turns: {avg_turns:.1f}")
    
    print(f"\nOutput saved to: {output_base_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
