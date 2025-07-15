#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random
from datetime import datetime

def extract_random_sample(dataset_path, output_dir="../data/predict"):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(dataset_path)
    random_idx = random.randint(0, len(df) - 1)
    sample_row = df.iloc[random_idx]
    
    label = sample_row.iloc[0]
    pixels = sample_row.iloc[1:].values
    image_array = pixels.reshape(28, 28)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"sample_{timestamp}_label{label}"
    
    image_path = os.path.join(output_dir, f"{base_name}.png")
    plt.figure(figsize=(3, 3))
    plt.imshow(image_array, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(image_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    sample_df = pd.DataFrame([sample_row.values], columns=df.columns)
    sample_df.to_csv(csv_path, index=False)
    
    raw_csv_path = os.path.join(output_dir, f"{base_name}_raw.csv")
    with open(raw_csv_path, 'w') as f:
        f.write(f"{label}")
        for pixel in pixels:
            f.write(f",{pixel}")
        f.write("\n")
    
    print(f"Image: {image_path}")
    print(f"CSV: {csv_path}")
    print(f"Raw CSV: {raw_csv_path}")
    print(f"Label: {label}")
    
    return image_path, csv_path, label

def main():
    parser = argparse.ArgumentParser(description='Extract random MNIST sample')
    parser.add_argument('dataset', help='Path to CSV dataset file')
    parser.add_argument('-o', '--output', default='../data/predict', help='Output directory')
    parser.add_argument('-n', '--samples', type=int, default=1, help='Number of samples')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset {args.dataset} not found")
        return
    
    for i in range(args.samples):
        print(f"Extracting sample {i+1}/{args.samples}...")
        extract_random_sample(args.dataset, args.output)
        print()

if __name__ == "__main__":
    main()