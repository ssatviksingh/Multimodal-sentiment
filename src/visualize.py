
# visualize.py - simple visualization utilities for results CSV
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--out", default="results/plot_growth.png")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)  # expects columns: model, accuracy, f1, std
    df = df.sort_values('accuracy')
    plt.figure(figsize=(8,5))
    plt.bar(df['model'], df['accuracy'], yerr=df.get('std', None))
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Baseline -> Modern model growth')
    plt.tight_layout()
    plt.savefig(args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()

