#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from datasets import Dataset

def create_dataset(args):
    """Create a dataset from a CSV file and save it in the format required for training."""
    print(f"Creating dataset from {args.input_file}...")
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file {args.input_file} not found.")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load data
    data = pd.read_csv(args.input_file)
    
    # Check required columns
    required_columns = ["text"]
    trait_columns = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    
    for col in required_columns:
        if col not in data.columns:
            print(f"ERROR: Required column '{col}' not found in input file.")
            return False
    
    # Check if we have at least one trait column
    trait_found = False
    for col in trait_columns:
        if col in data.columns:
            trait_found = True
            break
    
    if not trait_found:
        print(f"ERROR: At least one of the trait columns {trait_columns} must be present.")
        return False
    
    # Create dataset
    dataset = Dataset.from_pandas(data)
    
    # Save to disk
    dataset.save_to_disk(args.output_path)
    
    print(f"Dataset saved to {args.output_path}")
    return True

def create_sample_dataset(args):
    """Create a sample dataset for demonstration purposes."""
    print(f"Creating sample dataset at {args.output_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Create sample data
    data = {
        "text": [
            "I love exploring new ideas and concepts.",
            "I am very organized and detail-oriented.",
            "I enjoy being the center of attention at parties.",
            "I always try to be kind and considerate to others.",
            "I often worry about things that might go wrong.",
            "I prefer routine and familiar experiences.",
            "I sometimes procrastinate and leave tasks until the last minute.",
            "I prefer quiet activities over social gatherings.",
            "I can be critical and argumentative at times.",
            "I remain calm even in stressful situations."
        ],
        "openness": [0.9, 0.5, 0.6, 0.5, 0.4, 0.1, 0.5, 0.4, 0.6, 0.5],
        "conscientiousness": [0.6, 0.9, 0.5, 0.7, 0.5, 0.7, 0.2, 0.6, 0.4, 0.6],
        "extraversion": [0.7, 0.5, 0.9, 0.6, 0.3, 0.2, 0.5, 0.1, 0.5, 0.4],
        "agreeableness": [0.6, 0.6, 0.5, 0.9, 0.5, 0.6, 0.6, 0.7, 0.2, 0.6],
        "neuroticism": [0.4, 0.3, 0.4, 0.3, 0.9, 0.5, 0.6, 0.5, 0.7, 0.1]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Split into train/val/test sets
    train_df = df.iloc[:6]
    val_df = df.iloc[6:8]
    test_df = df.iloc[8:]
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Save to disk
    base_path = args.output_path
    train_dataset.save_to_disk(os.path.join(base_path, "train"))
    val_dataset.save_to_disk(os.path.join(base_path, "val"))
    test_dataset.save_to_disk(os.path.join(base_path, "test"))
    
    # Also save as CSV for reference
    train_df.to_csv(os.path.join(base_path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(base_path, "val.csv"), index=False)
    test_df.to_csv(os.path.join(base_path, "test.csv"), index=False)
    
    print(f"Sample datasets saved to {base_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create datasets for BIG5Chat training")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create dataset from file
    create_parser = subparsers.add_parser("create", help="Create dataset from a CSV file")
    create_parser.add_argument("--input-file", required=True, help="Input CSV file")
    create_parser.add_argument("--output-path", required=True, help="Output path for dataset")
    
    # Create sample dataset
    sample_parser = subparsers.add_parser("sample", help="Create a sample dataset")
    sample_parser.add_argument("--output-path", default="data/sample_dataset", help="Output path for sample dataset")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the requested command
    if args.command == "create":
        success = create_dataset(args)
    elif args.command == "sample":
        success = create_sample_dataset(args)
    else:
        parser.print_help()
        return
    
    if success:
        print(f"Command '{args.command}' completed successfully!")
    else:
        print(f"Command '{args.command}' failed!")
        exit(1)

if __name__ == "__main__":
    main() 