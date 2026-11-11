"""
Script to download the gold-standard Alzheimer's 4-Class MRI dataset 
and organize it into stratified Train, Validation, and Test directories.
"""

import os
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

def prepare_dataset():
    base_dir = Path("data/processed")
    samples_dir = Path("data/samples")
    
    print("[1/4] Loading Falah/Alzheimer_MRI dataset from Hugging Face...")
    hf_dataset = load_dataset("Falah/Alzheimer_MRI")
    
    class_names = hf_dataset["train"].features["label"].names
    print(f"Detected Classes ({len(class_names)}): {class_names}")
    
    train_data = hf_dataset["train"]
    test_data = hf_dataset["test"]
    
    print(f"Raw Train samples: {len(train_data)}")
    print(f"Raw Test samples:  {len(test_data)}")
    
    # Stratified split train into Train (80%) and Validation (20%)
    train_indices = list(range(len(train_data)))
    train_labels = [train_data[i]["label"] for i in range(len(train_data))]
    
    tr_idx, val_idx = train_test_split(
        train_indices,
        test_size=0.20,
        random_state=42,
        stratify=train_labels
    )
    
    splits = {
        "train": [(train_data[i]["image"], train_data[i]["label"], f"tr_{i:05d}") for i in tr_idx],
        "val": [(train_data[i]["image"], train_data[i]["label"], f"val_{i:05d}") for i in val_idx],
        "test": [(test_data[i]["image"], test_data[i]["label"], f"ts_{i:05d}") for i in range(len(test_data))]
    }
    
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_saved_per_class = {c: 0 for c in class_names}
    
    print("\n[2/4] Saving images into structured directory splits...")
    stats = {}
    for split_name, items in splits.items():
        print(f"Processing '{split_name}' split ({len(items)} images)...")
        split_counts = Counter()
        for img, label_idx, file_id in tqdm(items, desc=f"Writing {split_name}"):
            cname = class_names[label_idx]
            split_counts[cname] += 1
            
            target_dir = base_dir / split_name / cname
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure RGB image format
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            file_path = target_dir / f"{file_id}.jpg"
            img.save(file_path, quality=95)
            
            # Save 2 representative samples per class in data/samples
            if sample_saved_per_class[cname] < 2:
                sample_file = samples_dir / f"{cname}_sample_{sample_saved_per_class[cname] + 1}.jpg"
                img.save(sample_file, quality=95)
                sample_saved_per_class[cname] += 1
                
        stats[split_name] = split_counts
        
    print("\n[3/4] Dataset Preparation Summary:")
    print("-" * 65)
    header = f"{'Class Name':<25} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}"
    print(header)
    print("-" * 65)
    for cname in class_names:
        tr_c = stats["train"][cname]
        val_c = stats["val"][cname]
        ts_c = stats["test"][cname]
        tot = tr_c + val_c + ts_c
        print(f"{cname:<25} | {tr_c:<8} | {val_c:<8} | {ts_c:<8} | {tot:<8}")
    print("-" * 65)
    
    total_tr = sum(stats["train"].values())
    total_val = sum(stats["val"].values())
    total_ts = sum(stats["test"].values())
    print(f"{'TOTAL':<25} | {total_tr:<8} | {total_val:<8} | {total_ts:<8} | {total_tr+total_val+total_ts:<8}")
    print("-" * 65)
    print(f"\n[4/4] Data successfully saved to '{base_dir.resolve()}'.")
    print(f"Representative demo samples saved to '{samples_dir.resolve()}'.")

if __name__ == "__main__":
    prepare_dataset()

