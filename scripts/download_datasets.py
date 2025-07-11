#!/usr/bin/env python3
"""
Download a subset of the voice-is-cool/voxtube dataset from HuggingFace.
Downloads up to 1000 samples and saves them to the datasets folder.
"""

import os
import sys
from datasets import load_dataset

def download_voxtube_dataset(max_samples=10000, output_dir="datasets"):
    """
    Download a subset of the voice-is-cool/voxtube dataset.
    
    Args:
        max_samples (int): Maximum number of samples to download
        output_dir (str): Directory to save the dataset
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading voice-is-cool/voxtube dataset (max {max_samples} samples)...")
    print(f"Saving to: {output_dir}")
    
    try:
        # Load the dataset with streaming to limit samples
        dataset = load_dataset(
            "voice-is-cool/voxtube",
            streaming=True
        )
        
        # Take first max_samples from the train split and collect them
        train_dataset = dataset["train"].take(max_samples)
        samples = list(train_dataset)
        
        # Convert to Dataset format for saving
        from datasets import Dataset
        final_dataset = Dataset.from_list(samples)
        
        # Save the dataset
        save_path = os.path.join(output_dir, "voxtube_subset")
        final_dataset.save_to_disk(save_path)
        
        print(f"Successfully downloaded and saved {max_samples} samples to {save_path}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Default to datasets folder in workspace
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
    
    print("Starting dataset download...")
    download_voxtube_dataset(max_samples=10000, output_dir=datasets_dir)
    print("Dataset download complete!")