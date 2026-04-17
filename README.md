# Multimodal Sentiment Analysis using Deep Learning

A research-oriented project for sentiment classification using **text**, **audio**, and **video** modalities — featuring ResNet-18, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16, and fusion models including a **Hybrid Fusion Transformer** (ViT + AST).

The system performs feature extraction, fusion, training, and benchmarking across multiple architectures on GPU.

Extended experiments (ablations, robustness, telehealth scenarios) live under `[research_extensions/](research_extensions/README_research.md)`.

## Features

- Automatic dataset generation for text, audio, and video
- Pretrained embedding extraction (DistilBERT, Wav2Vec2, ViT-Base)
- Multiple deep learning models: ResNet-18, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16, and fusion variants (baseline, weighted, transformer, hybrid)
- Benchmark pipeline that trains, evaluates, and compares models
- GPU acceleration (CUDA supported)
- Comparison graphs for accuracy and F1-style metrics

## Project structure

```text
Multimodal-sentiment/
├── src/
│   ├── data_creation/          # Dataset generators, manifests
│   ├── feature_extraction/     # Pretrained embedding extractors
│   ├── models/
│   │   ├── cnn_variants/       # ResNet, EfficientNet, ConvNeXt
│   │   ├── transformer_variants/  # ViT-B/16
│   │   ├── fusion_variants/    # Baseline, weighted, transformer, hybrid fusion
│   │   ├── compare_all_models.py
│   │   ├── evaluate_model.py
│   │   ├── train_all_models.py
│   │   └── ...
│   ├── dataset.py
│   ├── utils.py
│   └── train.py
├── configs/                    # YAML configs (e.g. MOSI-style runs)
├── data/
│   ├── manifest_*.csv          # Splits (tracked)
│   ├── custom/                 # Sample data / manifests
│   └── features/               # Extracted embeddings (gitignored)
├── results/                    # Checkpoints, logs, plots (gitignored; created when you train)
├── research_extensions/        # Optional research scripts & configs (see README inside)
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository**
  ```bash
   git clone https://github.com/ssatviksingh/Multimodal-sentiment.git
   cd Multimodal-sentiment
  ```
2. **Create and activate a virtual environment**
  ```powershell
   python -m venv venv
   venv\Scripts\activate
  ```
   On macOS/Linux:
  ```bash
   python3 -m venv venv
   source venv/bin/activate
  ```
3. **Install dependencies**
  ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
  ```
4. **(Optional) CUDA** — install a PyTorch build that matches your GPU: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)

## How to run the project (demo)

Run everything from the **repository root** (the folder that contains `requirements.txt` and `src/`). Activate your virtual environment first.

### End-to-end benchmark (main demo)

This is the usual path: use existing data → extract embeddings → train all models → compare. After training, `**train_all_models` runs test-set evaluation only for HybridFusion** (see [Metrics](#metrics-where-the-numbers-live)). It then runs `compare_all_models`. To **retrain everything** even when checkpoints exist, use `--force-retrain`.


| Step                      | Command                                                                     | Notes                                                                                                                                                                                   |
| ------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Dataset                | `python src/data_creation/generate_large_dataset.py`                        | Creates samples and `data/manifest_*.csv`. Skip if your manifests and media are already in place.                                                                                       |
| 2. Embeddings             | `python src/feature_extraction/pretrained/extract_pretrained_embeddings.py` | Writes `data/features/{text,audio,video}/`. Needs GPU/CPU time and disk.                                                                                                                |
| 3. Train + eval + compare | `python -m src.models.train_all_models`                                     | Trains baselines + hybrid; skips training if `results/<name>_best.pt` exists unless you pass `--force-retrain`. Runs `**evaluate_model` once** (hybrid test set) after hybrid training. |
| 3b. Force full retrain    | `python -m src.models.train_all_models --force-retrain`                     | Ignores existing checkpoints so all models train on current features.                                                                                                                   |


**Hybrid test-set metrics** (`data/manifest_test.csv` + `data/features/`):

```bash
python -m src.models.evaluate_model
# Optional: python -m src.models.evaluate_model --checkpoint results/hybrid_fusion_best.pt --manifest data/manifest_test.csv
```

**Regenerate comparison charts** without retraining (if checkpoints already exist):

```bash
python -m src.models.compare_all_models
```

### Optional: smaller dataset

For a quicker smoke test of data generation (writes under `data/raw/` and manifests):

```bash
python src/data_creation/generate_dummy_dataset.py
```

You still need feature extraction and training afterward if you want the full pipeline.

### Optional: telehealth scenario demo (research extension)

Requires **pre-extracted** features under `data/features/`, a trained `**results/hybrid_fusion_best.pt`**, and the expanded manifest used by the script (`data/custom/manifest_train_expanded.csv`). Outputs go to `research_extensions/results/telehealth_demo/`.

```bash
python -m research_extensions.experiments.run_applied_scenario_telehealth
```

Equivalent entry point:

```bash
python -m research_extensions.scenarios.telehealth_pipeline_demo
```

See `[research_extensions/README_research.md](research_extensions/README_research.md)` for other experiments.

### Legacy single-script trainer

An alternate training entry point (different fusion setup, expects processed data under `--data_dir`):

```bash
python src/train.py --data_dir data/processed --epochs 3
```

Use this only if you are following that code path; the main demo above uses `train_all_models`.

## Dataset preparation

*(For the full command order, see [How to run the project (demo)](#how-to-run-the-project-demo).)*

Generate or expand data:

```bash
python src/data_creation/generate_large_dataset.py
```

This creates generated text, audio, and video samples, writes manifest files (`manifest_train.csv`, `manifest_val.csv`, `manifest_test.csv`), and stores paths under `data/`.

## Pretrained feature extraction

```bash
python src/feature_extraction/pretrained/extract_pretrained_embeddings.py
```

Embeddings are written under (ignored by git):

- `data/features/text/`
- `data/features/audio/`
- `data/features/video/`

## Training and evaluation

**Train the benchmark suite:**

```bash
python -m src.models.train_all_models
python -m src.models.train_all_models --force-retrain   # optional: ignore existing checkpoints
```

Trains CNNs, ViT, and hybrid fusion; saves checkpoints and logs under `results/` (local only; not committed).

**Test-set evaluation (HybridFusion only):**

```bash
python -m src.models.evaluate_model
```

CNN/ViT baselines do not use this script; use validation metrics in their `results/*_log.csv` files (see [Baseline models](#baseline-models-cnn--vit)).

## Metrics: where the numbers live


| What you need                                         | Where it is                                                                                         |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Hybrid — test accuracy / macro F1** (for reports)   | `results/evaluation_summary.txt` (from `evaluate_model`)                                            |
| **Hybrid — validation curves**                        | `results/hybrid_fusion_log.csv`, `hybrid_fusion_curve.png`                                          |
| **All models — best validation epoch (chart)**        | `python -m src.models.compare_all_models` → `results/comparison_chart.png` (reads best row per log) |
| **Fusion variants (weighted / transformer / hybrid)** | `python -m src.models.compare_results` → `results/accuracy_progression.csv` (only logs that exist)  |


Validation metrics come from training logs; test metrics for the hybrid come only from `evaluate_model`.

## Baseline models (CNN / ViT)

The ResNet, EfficientNet, ConvNeXt, and ViT training scripts under `src/models/cnn_variants/` and `transformer_variants/` currently train on **placeholder random image tensors** for quick pipeline checks. They do **not** consume `data/features/` multimodal embeddings. **Report numbers for the real multimodal task** should come from **HybridFusion** (`evaluate_model` + `hybrid_fusion_log.csv`). Comparing CNN/ViT validation logs to hybrid on the same table is **not** apples-to-apples until those baselines are reimplemented on the same embedding dataset.

## Comparison graphs

After training:

```bash
python -m src.models.compare_all_models
```

Produces `results/comparison_chart.png` from each model’s log (best validation epoch).

Fusion-only comparison:

```bash
python -m src.models.compare_results
```

## Results summary (example)

Illustrative numbers from an example run — your metrics will vary by data and seed.


| Model                | Accuracy | F1-Score |
| -------------------- | -------- | -------- |
| ResNet-18            | 68.9%    | 60.5%    |
| EfficientNet-B0      | 67.6%    | 66.2%    |
| ConvNeXt-Tiny        | 67.6%    | 66.2%    |
| ViT-B/16             | 67.6%    | 66.2%    |
| Hybrid-fusion (ours) | 70.4%    | 69.1%    |


## Key components


| Component                          | Description                                               |
| ---------------------------------- | --------------------------------------------------------- |
| `fusion_variants/`                 | Baseline, weighted, transformer, and hybrid fusion models |
| `train_all_models.py`              | Full training and evaluation pipeline                     |
| `compare_all_models.py`            | Model comparison and charts                               |
| `extract_pretrained_embeddings.py` | Modality-specific embeddings                              |
| `evaluate_model.py`                | Inference and metrics                                     |


## Research extensions

See `[research_extensions/README_research.md](research_extensions/README_research.md)` for telehealth-focused experiments, ablations, and analysis scripts. Outputs from those scripts go under `research_extensions/results/` (gitignored).

## Git (optional): SSH remote

If you use an SSH host alias (e.g. `github.com-ssatviksingh`):

```bash
git remote set-url origin git@github.com-ssatviksingh:ssatviksingh/Multimodal-sentiment.git
git push origin main
```

## Verification checklist

1. Use the same `data/features/` and manifests you trained on; if you regenerated features, run `python -m src.models.train_all_models --force-retrain` or delete stale `results/*_best.pt` first.
2. After training, confirm `results/evaluation_summary.txt` matches the hybrid test run you report.
3. `python -m src.models.compare_all_models` should list each model with best validation metrics (not repeated hybrid test prints).

