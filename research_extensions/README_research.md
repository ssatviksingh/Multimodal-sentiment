# Research Extensions for Telehealth Emotional Wellbeing

This folder contains **research-grade extensions** built on top of the core multimodal sentiment project, targeting the application domain of **remote mental health / emotional wellbeing monitoring in telehealth video sessions**.

## Structure

- `configs/` – YAML configs for ablation and evaluation experiments (no changes to core configs).
- `experiments/` – Scripts that **call** existing dataset/model code to run journal-level studies (ablation, robustness, temporal, real-time, applied scenario).
- `analysis/` – Plotting and table utilities to generate publication-ready figures from experiment outputs.
- `scenarios/` – Application-layer logic for telehealth, including wellbeing state mapping, end-to-end pipeline demo, and timeline visualization.
- `utils/` – Shared helpers for timing, metrics, and I/O used across experiments and analysis.

All files under `research_extensions/` are **non-invasive**: they import and reuse the existing dataset, feature extraction, and model implementations without modifying them. All new outputs (CSVs, PNGs, logs) are written under `research_extensions/results/`.

