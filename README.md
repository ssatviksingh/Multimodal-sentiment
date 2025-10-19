🧠 Multimodal Sentiment Analysis using Deep Learning

A research-oriented project for sentiment classification using text, audio, and video modalities — featuring ResNet-18, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16, and a custom Hybrid Fusion Transformer (ViT + AST) model.

The system performs feature extraction, fusion, training, and benchmarking across multiple architectures on GPU.


🚀 Features

🗂 Automatic dataset generation for text, audio, and video
🧩 Pretrained embedding extraction (DistilBERT, Wav2Vec2, ViT-Base)

🤖 Multiple deep learning models
ResNet-18
EfficientNet-B0
ConvNeXt-Tiny
ViT-B/16
Hybrid Fusion Transformer (ours)

📊 Benchmark pipeline – trains, evaluates, and compares all models automatically
⚡ GPU acceleration (CUDA supported)
📈 Generates professional-style comparison graphs for accuracy & F1-scores


🏗 Project Structure

multimodal-sentiment/
│
├── src/
│   ├── data_creation/            # Synthetic or manifest-based dataset generators
│   ├── feature_extraction/       # Pretrained embedding extractors
│   ├── models/
│   │   ├── cnn_variants/         # ResNet, EfficientNet, ConvNeXt training scripts
│   │   ├── transformer_variants/ # ViT-B16 training script
│   │   ├── fusion_variants/      # Hybrid fusion model
│   │   ├── compare_all_models.py # Performance comparison chart
│   │   ├── evaluate_model.py     # Unified evaluation script
│   │   ├── train_all_models.py   # Benchmark automation
│   │   └── ...
│   ├── dataset.py
│   ├── utils.py
│   └── train.py
│
├── data/
│   ├── manifest_train.csv
│   ├── features/                 # Extracted pretrained embeddings (text/audio/video)
│   └── ...
│
├── results/                      # Trained weights, logs, charts
│   ├── resnet18_best.pt
│   ├── efficientnet_b0_best.pt
│   ├── convnext_tiny_best.pt
│   ├── vit_b16_best.pt
│   ├── hybrid_fusion_best.pt
│   ├── comparison_chart.png
│   └── ...
│
├── requirements.txt
└── README.md



⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/ssatviksingh/multimodal-sentiment.git
cd multimodal-sentiment

2️⃣ Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate  # On Mac/Linux

3️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

4️⃣ (Optional) Install CUDA support

Ensure your PyTorch installation matches your GPU CUDA version:
👉 https://pytorch.org/get-started/locally/



🧾 Dataset Preparation

The dataset can be generated or expanded automatically: python src/data_creation/generate_large_dataset.py

This will:
Create synthetic text, audio, and video samples
Generate manifest files (manifest_train.csv, manifest_val.csv, manifest_test.csv)
Store all paths under data/



🎧 Pretrained Feature Extraction

Run to extract DistilBERT, Wav2Vec2, and ViT features for each modality: python src/feature_extraction/pretrained/extract_pretrained_embeddings.py


Embeddings will be stored under:

data/features/text/
data/features/audio/
data/features/video/



🧠 Training and Evaluation
▶️ Train all models (automated benchmark): python -m src.models.train_all_models

This will:
Train all CNNs + ViT + HybridFusion
Evaluate each model
Save checkpoints and logs under results/

▶️ Evaluate a specific model: python -m src.models.evaluate_model



📈 Comparison Graph
After all models are trained:
python -m src.models.compare_all_models


This generates: results/comparison_chart.png
→ Accuracy + F1 for all models (publication-ready)



💾 Results Summary

Example outputs:

Model	                Accuracy	    F1-Score
ResNet-18	           68.9 %	        60.5 %
EfficientNet-B0	     67.6 %	        66.2 %
ConvNeXt-Tiny	        67.6 %	        66.2 %
ViT-B/16	              67.6 %	        66.2 %
Hybrid-Fusion (Ours)	  70.4 %	        69.1 %



🧩 Key Components
Component	                                            Description
HybridFusionModel	                        Cross-modal Transformer fusion (ViT + AST)
train_all_models.py	                     Orchestrates full training & evaluation pipeline
compare_all_models.py	                  Generates final performance comparison graph
extract_pretrained_embeddings.py	         Extracts modality-specific embeddings
evaluate_model.py	                        Unified inference & metrics generation script