# 🧠 AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications

[![Website](https://img.shields.io/badge/Website-ama--bench.github.io-blue)](https://ama-bench.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-2602.22769-b31b1b.svg)](https://arxiv.org/abs/2602.22769)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/AMA-bench/AMA-bench)
[![Leaderboard](https://img.shields.io/badge/🤗%20Hugging%20Face-Leaderboard-blue)](https://huggingface.co/spaces/AMA-bench/AMA-bench-Leaderboard)

AMA-Bench is a benchmark for evaluating agent memory on long-horizon agent trajectories.

![AMA-Bench memory formulation](assets/memform.png)

## 📚 Category

- [🎬 Media](#media)
- [🗞️ News](#news)
- [🧭 Overview](#overview)
- [🧠 Memory Architecture (Two-Stage)](#memory-architecture-two-stage)
- [⚙️ Installation](#installation)
- [📦 Dataset](#dataset)
- [🚀 Quick Start](#quick-start)
- [📊 Evaluation](#evaluation)
- [🧩 Scripts](#scripts)
- [🏆 Submit to Leaderboard](#submit-to-leaderboard)
- [🏗️ Project Structure](#project-structure)
- [🧪 Synthetic Data Generation](#synthetic-data-generation)
- [🛠️ Add a New Method](#add-a-new-method)
- [📖 Citation](#citation)
- [📮 Contact](#contact)
- [📄 License](#license)

## 🎬 Media

- AMA-Bench was featured by dair.ai: https://x.com/dair_ai/status/2027776582262395054
- Special thanks to Vincent De Genova for the visual notes and homepage feature: https://vdegenova.github.io/visual-notes/visuals/ama-bench/

## 🗞️ News

- [ ] **[coming soon]** 🤖 AMA-Agent code release
- ✅ **[2026-03]** 🚀 [AMA-Hub](https://github.com/AMA-Bench/AMA-Hub) repo launched
- ✅ **[2026-03]** 📄 [AMA-Bench paper](https://arxiv.org/abs/2602.22769) is accepted by ICLR 2026 Memory Agent workshop and released on arXiv!
- ✅  **[2026-03]** 📦 [Dataset](https://huggingface.co/datasets/AMA-bench/AMA-bench) released on Hugging Face
- ✅  **[2026-03]** 🏆 [Leaderboard](https://huggingface.co/spaces/AMA-bench/AMA-bench-Leaderboard) launched on Hugging Face Spaces

---

## 🧭 Overview

AMA-Bench evaluates whether a method can:

- Build memory from long agent trajectories
- Retrieve relevant evidence for a given question
- Answer open-ended and MCQ questions robustly

All methods share a unified two-stage interface defined in [src/method/base_method.py](src/method/base_method.py):

- `memory_construction(traj_text, task="")` — build a memory object from trajectory text
- `memory_retrieve(memory, question)` — retrieve relevant context for a question

## 🧠 Memory Architecture (Two-Stage)

![Two-stage memory architecture](assets/mem_two_stage.png)

Each memory method is implemented in two stages:

| Stage | Function | Input | Output |
|---|---|---|---|
| Construction | `memory_construction` | trajectory text + task | memory object |
| Retrieval | `memory_retrieve` | memory object + question | context string |

---

## ⚙️ Installation

**Requirements:** Python 3.9–3.11 · Linux · CUDA GPU (recommended)

```bash
git clone https://github.com/AMA-Bench/AMA-Hub.git
cd AMA-Hub
python -m venv ama_venv
source ama_venv/bin/activate
pip install -r requirements.txt
```

---

## 📦 Dataset

Official dataset: [huggingface.co/datasets/AMA-bench/AMA-bench](https://huggingface.co/datasets/AMA-bench/AMA-bench)

```bash
huggingface-cli download AMA-bench/AMA-bench --repo-type dataset --local-dir ./dataset
```

The default evaluation path is `dataset/test/`. The dataset contains only a test split — there is no training split or `qa_set` directory.

```text
dataset/
└── test/
    └── open_end_qa_set.jsonl
```

---

## 🚀 Quick Start

### ▶️ Run end-to-end evaluation

```bash
bash scripts/run.sh
```

### 🤖 Run AMA-Agent

```bash
bash scripts/ama_agent.sh
```

### 💻 Run via CLI

```bash
python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B.yaml \
  --subset openend \
  --method longcontext \
  --test-dir dataset/test \
  --judge-config configs/llm_judge.yaml
```

---

## 📊 Evaluation

`src/run.py` runs answer generation followed by LLM-as-judge scoring, saving outputs to `results/`.

To evaluate an existing answer file separately:

```bash
bash scripts/evaluate.sh \
  --answers-file results/openend/answers_xxx.jsonl \
  --test-file dataset/test/open_end_qa_set.jsonl \
  --judge-config configs/llm_judge.yaml
```

### Judge Reliability

We performed a comprehensive consistency check for the LLM-as-judge setup, evaluating multiple judge models and analyzing their agreement patterns.

**Performance against human annotations**

| Metric | Value |
|---|---:|
| Accuracy | 92.67% |
| Precision | 96.45% |
| Recall | 92.68% |
| F1 Score | 94.53% |

**Overall Judge Accuracy Scores**

| Judge Model | Accuracy | Notes |
|---|---:|---|
| Qwen3-32B | 0.4880 | ⚠️ Highest accuracy, but shows leniency bias |
| Claude-4.6 | 0.3986 | Reference judge, balanced |
| GPT-5.4 | 0.3906 | Moderate strictness |
| GPT-5.2 | 0.3550 | More strict |
| DeepSeek-v3.2 | 0.3309 | Most strict judge |

**Model Agreement Rates**

*Highest Agreement (> 90%):*
- GPT-5.2 vs. GPT-5.4: **94.58%** ✅ (Highest agreement - same family)
- DeepSeek-v3.2 vs. GPT-5.2: **90.20%** ✅
- Claude-4.6 vs. GPT-5.2: **90.16%** ✅

*Qwen3-32B Agreement with Others:*
- vs. GPT-5.4: **92.80%**
- vs. GPT-5.2: **91.80%**
- vs. Claude-4.6: **88.80%**
- vs. DeepSeek-v3.2: **84.70%**

**Leniency Bias Analysis**

Qwen3-32B shows systematic leniency bias, particularly on:
- **SOFTWARE domain** (swebench): 100 lenient cases vs. 15 strict (vs. Claude-4.6)
- **Game domain**: 66 lenient cases vs. 17 strict
- **TEXT2SQL domain** (spider2): 62 lenient cases vs. 24 strict

*Example 1: Accepting Core Concepts Without Complete Details*

**Task**: babaisai (Game domain)
**Question**: Based on game mechanics, what hidden movement mechanic can be inferred when pushing blocks?

**Reference Answer**: When pushing a text block, that block and adjacent blocks forming a phrase move as a group **diagonally** (one step in push direction and one step to the left).

**Predicted Answer**: Pushing a block causes adjacent blocks in the same rule line to shift, suggesting rule blocks are connected and move together.

- **Qwen3-32B**: ✅ CORRECT (accepts answer capturing the core mechanic despite missing diagonal movement detail)
- **Claude-4.6**: ❌ INCORRECT (requires the specific diagonal movement detail)

*Example 2: Accepting Partial Explanations*

**Task**: crafter (Game domain)
**Question**: What inventory change occurred and why is it critical for future crafting?

**Reference Answer**: Agent gained **wood**. Critical because wood is needed to craft a **workbench**, and without a workbench, the agent cannot craft any other tools, halting technological progression.

**Predicted Answer**: Agent gained **wood**. Critical because wood is required to craft tools (e.g., Wood Pickaxe), which are prerequisites for mining stone and other resources.

- **Qwen3-32B**: ✅ CORRECT (accepts the wood→tools explanation despite skipping the intermediate workbench step)
- **Claude-4.6**: ❌ INCORRECT (requires explicit mention of the workbench as the critical intermediate step)

**Recommendations**: While Qwen3-32B achieves the highest accuracy, its leniency bias means it may accept incomplete answers. For critical evaluations, consider using Claude-4.6 as the reference judge or employing ensemble voting across multiple judges.

---

## 🧩 Scripts

| Script | Description |
|---|---|
| `scripts/launch_vllm_32B.sh` | Launch vLLM server from a YAML config |
| `scripts/run.sh` | End-to-end generation + evaluation pipeline |
| `scripts/evaluate.sh` | Evaluate an existing answers JSONL with LLM-as-judge |
| `scripts/run_cross_validation.sh` | Run cross-validation with multiple judges |
| `scripts/cross_validate.py` | Compare evaluation results from two judges |

**`launch_vllm_32B.sh`** — reads model and server settings from YAML, starts a vLLM OpenAI-compatible API, waits for the health endpoint, and logs to `vllm_server.log`.

**`run.sh`** — launches the vLLM server if needed, runs answer generation with the configured method, then evaluates with LLM-as-judge.

**`evaluate.sh`** — standalone judge evaluation for pre-generated answer files, useful for re-evaluating with a different judge model.

**`run_cross_validation.sh`** — evaluates answers with both GPT-5.2 and Qwen3-32B judges, then compares results to measure inter-rater reliability.

**`cross_validate.py`** — compares two evaluation result files and generates agreement statistics, including Cohen's Kappa and detailed disagreement analysis.

---

## 🏆 Submit to Leaderboard

Visit the [AMA-Bench Leaderboard](https://huggingface.co/spaces/AMA-bench/AMA-bench-Leaderboard) and upload a JSONL file where each line is:

```jsonl
{"episode_id": 0, "answer_list": ["(A)", "(B)(C)"], "reasoning_trace": "..."}
```

Required fields: `episode_id`, `answer_list`. `reasoning_trace` is optional.

---

## 📮 Contact

- yuz285@ucsd.edu

---

## 🏗️ Project Structure

```text
AMA-Bench/
├── src/
│   ├── method/             # Memory method implementations (BM25, embedding, AMA-Agent, …)
│   ├── synthetic_data_gen/ # Trajectory synthesis tools (BabyAI, TextWorld)
│   ├── run.py              # Main evaluation entry point
│   ├── evaluate.py         # LLM-as-judge evaluation
│   └── model_client.py     # Unified client for different LLM providers
├── configs/                # Model / judge / method configs
│   ├── llm_judge.yaml      # VLLM judge config (Qwen3-32B)
│   └── llm_judge_api.yaml  # API judge config (GPT-5.2, etc.)
├── scripts/                # Shell scripts for launching and evaluation
│   ├── run.sh
│   ├── evaluate.sh
│   ├── run_cross_validation.sh  # Cross-validation pipeline
│   └── cross_validate.py        # Cross-validation comparison tool
├── docs/                   # Documentation
│   └── CROSS_VALIDATION.md      # Cross-validation guide
├── dataset/                # Local dataset (test split, downloaded separately)
├── results/                # Prediction outputs and evaluation metrics
├── assets/                 # Figures
└── utils/                  # Shared utilities
```

### 🧪 Synthetic Data Generation

`src/synthetic_data_gen/` provides tools for generating trajectories used in the benchmark:

- **BabyAI** — grid-world navigation tasks with natural language instructions
- **TextWorld** — text-based interactive fiction environments

Both include trajectory generation, automatic QA pair creation via state tracking, and token-based length binning. See [src/synthetic_data_gen/README.md](src/synthetic_data_gen/README.md) for details.

---

## 🛠️ Add a New Method

Create a subclass of `BaseMethod` and implement both stages:

```python
from typing import override
from dataclasses import dataclass

from .base import *

@dataclass
class MyConfig(BaseConfig):
    # Define your method configuration here
    ...

@dataclass
class MyMemory(BaseMemory):
    # Define your memory object structure here
    ...

class MyMethod(BaseMethod):

    def __init__(
        self,
        config_path: os.PathLike = None,
        client: ModelClient = None,
        embedding_engine: EmbeddingEngine = None,
    ):
        super().__init__(config_path, client, embedding_engine)
        self.config = self._parse_config()
        ...

    @override
    def _parse_config(self) -> MyConfig:
        # Load your configuration from the config file in config_path
        config_dict = self._load_config(self.config_path)
        ...
    
    @override
    def memory_construction(self, traj_text: str, task: str = "") -> MyMemory:
        # Build and return a memory object from the trajectory
        ...

    @override
    def memory_retrieve(self, memory, question: str) -> str:
        # Return relevant context string for the question
        ...
```

Add your method implementation in `method/__init__.py` to make it importable:

```python
from .my_method import MyMethod
```

Then register it in [src/method_register.py](src/method_register.py) by modifying `_METHOD_REGISTRY` to make it available via `--method`.

```python
from method import *

# Registry of available methods
_METHOD_REGISTRY: Dict[str, Type[BaseMethod]] = {
    "bm25": BM25Method,
    "embedding": EmbeddingMethod,
    "longcontext": LongContextMethod,
    "ama_agent": AMAAgentMethod,

    # Add your own method
    "mymethod": MyMethod,
}
```

## Contributing

We welcome contributions! Please fork the repo, create a new branch for your feature or bug fix, and submit a pull request. Make sure to follow the existing code style and include tests for new functionality.

```bash
pre-commit install && pre-commit autoupdate
```

---

## 📖 Citation

```bibtex
@misc{zhao2026amabenchevaluatinglonghorizonmemory,
  title={AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications},
  author={Yujie Zhao and Boqin Yuan and Junbo Huang and Haocheng Yuan and Zhongming Yu and Haozhou Xu and Lanxiang Hu and Abhilash Shankarampeta and Zimeng Huang and Wentao Ni and Yuandong Tian and Jishen Zhao},
  year={2026},
  eprint={2602.22769},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2602.22769}
}
```

## 📄 License

MIT License. See [LICENSE](LICENSE).
