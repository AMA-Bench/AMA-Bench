# AMA-Bench: Agent Memory Architecture Benchmark

[![Website](https://img.shields.io/badge/Website-ama--bench.github.io-blue)](https://ama-bench.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-2602.22769-b31b1b.svg)](https://arxiv.org/abs/2602.22769)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/AMA-bench/AMA-bench)
[![Leaderboard](https://img.shields.io/badge/🤗%20Hugging%20Face-Leaderboard-blue)](https://huggingface.co/spaces/AMA-bench/AMA-bench-Leaderboard)

AMA-Bench is a benchmark for evaluating memory architectures on long-horizon agent trajectories.

## 🗞️ News

- [ ] **[coming soon]** 🤖 AMA-Agent code release
- ✅ **[2026-03]** 🚀 [AMA-Hub](https://github.com/AMA-Bench/AMA-Hub) repo launched
- ✅ **[2026-03]** 📄 [AMA-Bench paper](https://arxiv.org/abs/2602.22769) released on arXiv
- ✅  **[2026-03]** 📦 [Dataset](https://huggingface.co/datasets/AMA-bench/AMA-bench) released on Hugging Face
- ✅  **[2026-03]** 🏆 [Leaderboard](https://huggingface.co/spaces/AMA-bench/AMA-bench-Leaderboard) launched on Hugging Face Spaces

---

## Overview

AMA-Bench evaluates whether a method can:

- Build memory from long agent trajectories
- Retrieve relevant evidence for a given question
- Answer open-ended and MCQ questions robustly

All methods share a unified two-stage interface defined in [src/method/base_method.py](src/method/base_method.py):

- `memory_construction(traj_text, task="")` — build a memory object from trajectory text
- `memory_retrieve(memory, question)` — retrieve relevant context for a question

## Memory Architecture (Two-Stage)

![Two-stage memory architecture](assets/mem_two_stage.png)

Each memory method is implemented in two stages:

| Stage | Function | Input | Output |
|---|---|---|---|
| Construction | `memory_construction` | trajectory text + task | memory object |
| Retrieval | `memory_retrieve` | memory object + question | context string |

---

## Installation

**Requirements:** Python 3.9–3.11 · Linux · CUDA GPU (recommended)

```bash
git clone https://github.com/AMA-Bench/AMA-Hub.git
cd AMA-Hub
pip install -r requirements.txt
```

---

## Dataset

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

## Quick Start

### Run end-to-end evaluation

```bash
bash scripts/run.sh
```

### Run AMA-Agent

```bash
bash scripts/ama_agent.sh
```

### Run via CLI

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

## Evaluation

`src/run.py` runs answer generation followed by LLM-as-judge scoring, saving outputs to `results/`.

To evaluate an existing answer file separately:

```bash
bash scripts/evaluate.sh \
  --answers-file results/openend/answers_xxx.jsonl \
  --test-file dataset/test/open_end_qa_set.jsonl \
  --judge-config configs/llm_judge.yaml
```

---

## Scripts

| Script | Description |
|---|---|
| `scripts/launch_vllm_32B.sh` | Launch vLLM server from a YAML config |
| `scripts/run.sh` | End-to-end generation + evaluation pipeline |
| `scripts/evaluate.sh` | Evaluate an existing answers JSONL with LLM-as-judge |

**`launch_vllm_32B.sh`** — reads model and server settings from YAML, starts a vLLM OpenAI-compatible API, waits for the health endpoint, and logs to `vllm_server.log`.

**`run.sh`** — launches the vLLM server if needed, runs answer generation with the configured method, then evaluates with LLM-as-judge.

**`evaluate.sh`** — standalone judge evaluation for pre-generated answer files, useful for re-evaluating with a different judge model.

---

## Submit to Leaderboard

Visit the [AMA-Bench Leaderboard](https://huggingface.co/spaces/AMA-bench/AMA-bench-Leaderboard) and upload a JSONL file where each line is:

```jsonl
{"episode_id": 0, "answer_list": ["(A)", "(B)(C)"], "reasoning_trace": "..."}
```

Required fields: `episode_id`, `answer_list`. `reasoning_trace` is optional.

---

## Project Structure

```text
AMA-Bench/
├── src/
│   ├── method/             # Memory method implementations (BM25, embedding, AMA-Agent, …)
│   ├── synthetic_data_gen/ # Trajectory synthesis tools (BabyAI, TextWorld)
│   ├── run.py              # Main evaluation entry point
│   └── evaluate.py         # LLM-as-judge evaluation
├── configs/                # Model / judge / method configs
├── scripts/                # Shell scripts for launching and evaluation
├── dataset/                # Local dataset (test split, downloaded separately)
├── results/                # Prediction outputs and evaluation metrics
├── assets/                 # Figures
└── utils/                  # Shared utilities
```

### Synthetic Data Generation

`src/synthetic_data_gen/` provides tools for generating trajectories used in the benchmark:

- **BabyAI** — grid-world navigation tasks with natural language instructions
- **TextWorld** — text-based interactive fiction environments

Both include trajectory generation, automatic QA pair creation via state tracking, and token-based length binning. See [src/synthetic_data_gen/README.md](src/synthetic_data_gen/README.md) for details.

---

## Add a New Method

Create a subclass of `BaseMethod` and implement both stages:

```python
from src.method.base_method import BaseMethod

class MyMethod(BaseMethod):
    def memory_construction(self, traj_text: str, task: str = ""):
        # Build and return a memory object from the trajectory
        ...

    def memory_retrieve(self, memory, question: str) -> str:
        # Return relevant context string for the question
        ...
```

Then register it in [src/method_register.py](src/method_register.py) to make it available via `--method`.

---

## Citation

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

## License

MIT License. See [LICENSE](LICENSE).
