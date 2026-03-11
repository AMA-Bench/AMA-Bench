# Synthetic Data Generation

This directory contains two primary game environments for generating synthetic trajectories for the AMA-Bench:

1. **BabyAI** - Grid-based navigation and object manipulation tasks
2. **TextWorld** - Text-based interactive fiction games (cooking, coin_collector, treasure_hunter)

## Data Synthesis Pipeline

The synthetic data generation follows a systematic multi-stage pipeline:

### 1. Trajectory Generation
Game episodes are generated using either random or optimal policies. Each trajectory consists of a sequence of observations and actions executed by the agent in the environment.

### 2. Observation Enhancement
- **BabyAI**: Raw grid observations (2D matrices) are converted to natural language descriptions
- **TextWorld**: Observations are augmented with detailed state information based on difficulty level (low/medium/high verbosity)

### 3. State Tracking
A state tracker monitors and records:
- Inventory changes
- Object locations and attributes
- Room transitions
- Container states (open/closed/locked)
- Key events (taking, cooking, unlocking, eating, etc.)

### 4. Question-Answer Generation
Multi-type QA pairs are automatically generated to evaluate memory capabilities:
- **Type A**: Temporal information (when, what happened at step X)
- **Type B**: Causal reasoning (preconditions, state dependencies)
- **Type C**: State inference (inventory changes, hidden states)
- **Type D**: Trajectory summarization

### 5. Token-Based Binning
Trajectories are categorized into token bins based on total length:
- **4K-8K bins**: Easy/medium difficulty → shorter trajectories
- **16K bin**: Medium/medium-hard difficulty
- **32K bin**: Medium-hard/hard difficulty
- **64K bin**: Very-hard/extreme difficulty
- **128K bin**: Extreme/ultra/mega difficulty → longest trajectories

The generation process adaptively selects difficulty levels to fill each bin uniformly (target: 10 trajectories per bin per game type).

---

# BabyAI Trajectory Generation

Generate game trajectories from BabyAI for the Agentmemory benchmark. Observations are converted to **natural language** (no 2D matrices). QA pairs are automatically generated during dataset conversion using `babyai_qa_generator.py`.

## 1. Generate

Generate trajectories for a specific configuration across 5 token bins (8K, 16K, 32K, 64K, 128K):

```bash
cd memory_data_generation/babyai
python batch_trajetory_gen.py \
    --difficulty hard_large \
    --random_ratio 0 \
    --observation_format natural \
    --traj_per_bin 10 \
    --output_dir babyai_out_batch \
    --seed 42
```

**Parameters:**
- `--difficulty`: Difficulty level - `easy`, `medium`, `medium_hard`, `hard`, `very_hard`, or `hard_large`
- `--random_ratio`: Probability of taking random action (0.0-1.0, e.g., 0.0, 0.1, 0.2)
- `--observation_format`: Observation format - `natural` or `grid`
- `--traj_per_bin`: Number of trajectories to generate per token bin
- `--output_dir`: Output directory (default: `babyai_out_batch`)
- `--seed`: Base random seed for trajectory generation (default: `42`)

**Reproducibility:**
- All random operations (environment, actions, QA generation) use seeds for reproducibility
- Each episode uses `base_seed + episode_num` as its seed
- QA generation uses deterministic seeds derived from episode_id if not explicitly provided

**Output:**
- Individual JSON files: `babyai_out_batch/<prefix>_<bin>_<idx>.json`
  - Example: `ha_r0_nat_8K_0.json`, `ha_r0_nat_8K_1.json`, `ha_r0_nat_16K_0.json`, etc.
  - Prefix format: `<difficulty_abbr>_r<random_ratio>_<format_abbr>`
  - Each file contains one complete trajectory


**Note:** For larger bins (16K+), the script automatically uses larger environments and higher random ratios to generate longer trajectories. Failed trajectories are accepted for larger bins as they're often longer.

## 2. Analyze

Analyze generated trajectories:

```bash
python analyze_trajectories.py --input babyai_out_batch
```

- Can analyze either a directory of JSON files or a JSONL file
- Stats: by `task_type`, `difficulty`, token bin (8K-128K), success rate
- Optional: `--min-per-bin 10` for requirement checks

## 3. Convert to dataset (with automatic QA generation)

The conversion script automatically generates QA pairs if they are missing or empty. **Default paths are automatically adjusted** based on where you run the script from.

From **project root** (recommended):

```bash
python memory_data_generation/babyai/convert_to_dataset.py
```

From `memory_data_generation/babyai/`:

```bash
python convert_to_dataset.py
```

- **Auto-detects paths**: Default input/output paths adjust automatically based on current directory
  - From project root: `memory_data_generation/babyai/babyai_out_batch/` → `dataset/babyai/`
  - From babyai dir: `babyai_out_batch/` → `../../dataset/babyai/`
- Reads all JSON files from input directory, writes `dataset/babyai/task_babyai_1.json`, etc., with `source: "babyai"`.
- **Automatically generates QA pairs** 
- Use `--target-qa-count N` to change the number of QA pairs (default: 12).
- Use `--seed N` to set random seed for QA generation reproducibility (default: 42).
- Use `--no-auto-qa` to disable automatic QA generation and use existing QA pairs only.
- Use `--input` and `--output` to override default paths if needed.

## QA pairs

QA pairs are automatically generated during conversion using `babyai_qa_generator.py`. The generator creates questions across 4 categories, including **multi-hop questions** that require information from multiple steps:

- **A**: Temporal Information (A1-A4: various temporal questions)
- **B**: State Dependency (B1-B3: state-action dependency questions)
- **C**: State Update (C1-C2: inventory and object state changes)
- **D**: State Summary (D1-D2: trajectory summary questions)

**Default distribution per trajectory:**
- A + B + C = 10 questions
- D = 2 questions
- **Total = 12 questions**

Each QA pair has the format (aligned with TextWorld):

```python
{"question": str, "answer": str, "type": str, "sub_type": str}
```

The `sub_type` field indicates the specific subtype (e.g., "A1", "A2", "B1", "D1", "D2").

Multi-hop questions require examining evidence from multiple steps in the trajectory, making them more challenging for memory evaluation.

### Customizing QA generation

To customize QA generation, you have two options:

1. **Modify `babyai_qa_generator.py`**: Edit the `BabyAIQAGenerator` class to add new question types or modify existing ones.

2. **Override in `batch_trajetory_gen.py`**: The `generate_qa_pairs()` function in `batch_trajetory_gen.py` currently returns an empty list. You can override it to generate QA pairs during trajectory generation instead of during conversion:

```python
def generate_qa_pairs(trajectory, task, task_type, difficulty_name, **kwargs):
    # Your custom QA generation logic
    return [{"question": "...", "answer": "...", "type": "A", "sub_type": "A1"}]
```

If QA pairs are already present in the trajectory data, the conversion script will skip automatic generation (unless you use `--no-auto-qa` to force regeneration).

### Manual QA generation

To generate QA pairs for existing trajectories without converting to dataset:

```bash
# Generate QA for all JSON files in a directory
python generate_qa_batch.py --input_dir babyai_out_batch --target_qa_count 12 --seed 42

# Generate QA for a JSONL file
python generate_qa_batch.py --input_dir babyai_out_batch --file_pattern "*.jsonl" --target_qa_count 12 --seed 42
```

**Parameters:**
- `--input_dir`: Directory containing trajectory JSON/JSONL files
- `--target_qa_count`: Target number of QA pairs per trajectory
- `--file_pattern`: File pattern to match (default: `*.json`, use `*.jsonl` for JSONL files)
- `--seed`: Random seed for QA generation reproducibility (default: 42, uses episode_id hash if not provided)

---

# TextWorld Trajectory Generation

Generate interactive fiction game trajectories from TextWorld. The system supports three game types with configurable difficulty levels and automatically generates QA pairs during trajectory generation.

## 1. Generate Trajectories

### Batch Generation (Recommended)

Generate trajectories for all three game types with automatic token bin distribution:

```bash
cd src/synthetic_data_gen/textworld
python batch_generate_trajectories.py mode=validate
```

**Game Types:**
- `coin_collector`: Navigation and object collection (60 trajectories, seeds 1000-1999)
- `cooking`: Recipe-based cooking tasks (60 trajectories, seeds 2000-2999)
- `treasure_hunter`: Exploration and treasure finding (60 trajectories, seeds 3000-3999)

**Modes:**
- `validate`: Generates 10 QA pairs per trajectory (default)
- `train`: Generates 50 QA pairs per trajectory

**Output:**
- Individual JSON files: `tw_out_batch/<game_type>/<game_type>_<idx>.json`
- Combined JSONL: `tw_out_batch/all_trajectories.jsonl`

**Difficulty Levels:**
Each game type has 8 difficulty levels (easy → mega), with increasing:
- Number of rooms
- Number of objects
- Quest length
- Quest depth/breadth

**Token Bins:**
Same as BabyAI (4K, 8K, 16K, 32K, 64K, 128K), with target of 10 trajectories per bin per game type.

### Single Trajectory Generation

Generate a single trajectory with specific parameters:

```bash
python generate_trajectory.py difficulty=hard seed=1234 max_steps=200 mode=validate
```

**Parameters:**
- `difficulty`: easy, medium, hard (default: medium)
- `seed`: Random seed for reproducibility
- `max_steps`: Maximum steps per episode (default: 200)
- `mode`: validate (10 QA) or train (50 QA)
- `out_dir`: Output directory (default: tw_out)
- `log`: Output trajectory file path

**Custom Configuration:**
You can override specific difficulty parameters:
- `rooms=N`: Number of rooms
- `objects=N`: Number of objects
- `quest=N`: Quest length
- `min_depth=N`, `max_depth=N`: Quest depth range
- `min_breadth=N`, `max_breadth=N`: Quest breadth range

## 2. Analyze Trajectories

Analyze generated trajectories to verify token distribution:

```bash
python analyze_trajectories.py --input tw_out_batch
```

**Features:**
- Token bin statistics (4K-128K)
- Success rate per game type
- Average trajectory length
- Difficulty distribution

## 3. QA Generation

QA pairs are automatically generated during trajectory generation using `textworld_label_generator.py`. The generator tracks game state using facts from the TextWorld environment and creates questions across multiple categories.

**QA Categories:**

- **Type A**: Temporal queries
  - A1-A4: "What happened at step X?", "When did event Y occur?"

- **Type B**: Causal reasoning
  - B1: Multi-condition preconditions
  - B2: State dependency analysis

- **Type C**: State inference
  - Hidden state tracking (objects in closed containers)
  - Inventory memory

- **Type D**: Trajectory summarization
  - Overall task completion
  - Key events sequence

**State Tracking:**
The `TextWorldStateTracker` uses environment facts to track:
- Object locations (in room, in inventory, in/on containers)
- Object attributes (edible, cooked, sliced, etc.)
- Container states (open, closed, locked)
- Room transitions
- Inventory changes

**Example QA Format:**
```json
{
  "question": "At step 5, what are ALL the preconditions that must be satisfied to execute 'put apple on table'?",
  "answer": "1. The apple must be in the agent's inventory; 2. The target location must be accessible; 3. The table must exist and be reachable; 4. The action must be in the available action space",
  "type": "B",
  "sub_type": "B1"
}
```

## 4. Observation Verbosity

TextWorld trajectories use adaptive verbosity based on difficulty to reach target token counts:

- **Low** (easy/medium): Minimal observation + limited action space
- **Medium** (medium_hard/hard): Structured state information
- **High** (very_hard/extreme/ultra/mega): Detailed state enumeration with full action space

This ensures longer trajectories for higher token bins while maintaining informativeness.

## Implementation Details

### Concurrent Generation
The batch generator uses async/await for parallel trajectory generation:
- Maximum 50 trajectories queued concurrently
- Processes in batches of 20 concurrent episodes
- Each episode builds its game and runs rollout independently

### Adaptive Difficulty Selection
The system automatically selects difficulty based on which bins need filling:
- Tracks current bin counts
- Probabilistically selects difficulty levels targeting unfilled bins
- Example: For 128K bin, uses extreme (20%), ultra (40%), mega (40%)

### Reproducibility
All random operations are seeded:
- Game generation uses explicit seeds
- Action selection uses per-episode seeds
- QA generation uses deterministic seeds from episode_id

---

# Directory Structure

```
synthetic_data_gen/
├── README.md                        # This file
├── babyai/                          # BabyAI trajectory generation
│   ├── batch_trajetory_gen.py      # Main: batch generation script
│   ├── babyai_qa_generator.py      # QA pair generator for BabyAI
│   ├── convert_to_dataset.py       # Convert trajectories to dataset format
│   ├── generate_qa_batch.py        # Batch QA generation for existing trajectories
│   ├── analyze_trajectories.py     # Analyze generated trajectories
│   └── add_qa_answer_ids.py        # Add relevant_turn_indices to QA pairs
└── textworld/                       # TextWorld trajectory generation
    ├── batch_generate_trajectories.py    # Main: batch generation script
    ├── generate_trajectory.py            # Single trajectory generation
    ├── textworld_label_generator.py      # QA pair generator for TextWorld
    ├── textworld_utils.py                # Utility functions
    ├── textworld_facts_analyzer.py       # Facts-based state tracking
    ├── convert_jsonl_to_json.py          # Convert JSONL to individual JSON files
    └── analyze_trajectories.py           # Analyze generated trajectories
```

## Core Scripts

### BabyAI

- **[batch_trajetory_gen.py](babyai/batch_trajetory_gen.py)**: Primary script for generating BabyAI trajectories across all difficulty levels and token bins
- **[babyai_qa_generator.py](babyai/babyai_qa_generator.py)**: Implements `BabyAIQAGenerator` class for multi-type QA generation
- **[convert_to_dataset.py](babyai/convert_to_dataset.py)**: Converts raw trajectories to AMA-Bench dataset format

### TextWorld

- **[batch_generate_trajectories.py](textworld/batch_generate_trajectories.py)**: Primary script for generating TextWorld trajectories with async/concurrent generation
- **[textworld_label_generator.py](textworld/textworld_label_generator.py)**: Implements `TextWorldQAGenerator` and `TextWorldStateTracker` for QA generation
- **[textworld_facts_analyzer.py](textworld/textworld_facts_analyzer.py)**: Tracks game state using TextWorld's facts system

## Utility Scripts

- **analyze_trajectories.py**: Analyze token distribution, success rates, and bin statistics
- **convert_jsonl_to_json.py**: Convert JSONL format to individual JSON files for dataset
- **generate_qa_batch.py**: Batch process existing trajectories to add/regenerate QA pairs
- **add_qa_answer_ids.py**: Post-process QA pairs to add `relevant_turn_indices` field

## Key Differences: BabyAI vs TextWorld

| Aspect | BabyAI | TextWorld |
|--------|--------|-----------|
| **Environment** | Grid-based 2D navigation | Text-based interactive fiction |
| **Observation** | Grid → Natural language | Native text |
| **State Tracking** | Vision-based (grid cells) | Facts-based (entity relationships) |
| **Game Types** | Single env with varying difficulty | Multiple types (cooking, coin_collector, treasure_hunter) |
| **Concurrency** | Sequential (single-threaded) | Async/parallel (20-50 concurrent) |
| **Verbosity Levels** | N/A (fixed natural language) | Low/Medium/High based on difficulty |
| **Token Bins** | 8K, 16K | 4K, 8K, 16K, 32K, 64K, 128K |
