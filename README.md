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
3. **Install dependencies**
  ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
  ```
4. **(Optional) CUDA** — install a PyTorch build that matches your GPU: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)

## How to run the project (demo)

Run everything from the **repository root** (the folder that contains `requirements.txt` and `src/`). Activate your virtual environment first.

### End-to-end benchmark (main demo)

This is the usual path: build or use data → extract embeddings → train all models → compare. `train_all_models` also **evaluates** each model and **runs the comparison step** at the end, so you do not need to invoke `compare_all_models` separately unless you only want to regenerate charts.


| Step                      | Command                                                                     | Notes                                                                                                                                                                            |
| ------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Dataset                | `python src/data_creation/generate_large_dataset.py`                        | Creates samples and `data/manifest_*.csv`. Skip if your manifests and media are already in place.                                                                                |
| 2. Embeddings             | `python src/feature_extraction/pretrained/extract_pretrained_embeddings.py` | Writes `data/features/{text,audio,video}/`. Needs GPU/CPU time and disk.                                                                                                         |
| 3. Train + eval + compare | `python -m src.models.train_all_models`                                     | Trains ResNet-18, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16, and hybrid fusion; skips any model that already has `results/<name>_best.pt`; saves logs and plots under `results/`. |


**Evaluate only the hybrid fusion checkpoint** (after step 2 and a trained `results/hybrid_fusion_best.pt`):

```bash
python -m src.models.evaluate_model
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

This creates synthetic text, audio, and video samples, writes manifest files (`manifest_train.csv`, `manifest_val.csv`, `manifest_test.csv`), and stores paths under `data/`.

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
```

Trains CNNs, ViT, and fusion models; saves checkpoints and logs under `results/` (local only; not committed).

**Evaluate a single model:**

```bash
python -m src.models.evaluate_model
```

## Comparison graphs

After training:

```bash
python -m src.models.compare_all_models
```

Produces comparison outputs under `results/` (for example accuracy curves and comparison charts).

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