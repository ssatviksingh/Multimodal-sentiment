
# Multimodal Sentiment Analysis (Text, Audio, Visual)

Minimal starter repo. Steps:
1. Prepare manifest CSV (filename,audio_path,video_path,text,label) and run:
   python src/extract_features.py --manifest data/manifest_train.csv --output_dir data/processed/train
   python src/extract_features.py --manifest data/manifest_val.csv --output_dir data/processed/val
   python src/extract_features.py --manifest data/manifest_test.csv --output_dir data/processed/test

2. Train:
   python src/train.py --data_dir data/processed --out_dir checkpoints --batch_size 8 --epochs 3

3. Evaluate:
   python src/evaluate.py --data_dir data/processed --ckpt checkpoints/best_model.pt

4. Visualize results.csv:
   python src/visualize.py --results_csv results/experiments.csv

This scaffold uses RoBERTa for text and small CNNs for audio/vision features. Swap in Wav2Vec2 and ViT as you scale.

