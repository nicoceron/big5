#!/bin/bash
set -e

# Create directories
mkdir -p data
mkdir -p logs
mkdir -p modelfiles

echo "=== BIG5Chat Training Workflow ==="
echo "This script will train the BIG5Chat personality models."

# Step 0: Check if Ollama is running and set up models
echo -e "\n0. Setting up Ollama personality models..."
python scripts/utils/setup_ollama_models.py --base-model gemma3

# Step 1: Create directories and prepare data
echo -e "\n1. Creating sample dataset..."
python scripts/data_preparation/create_datasets.py sample --output-path data/sample_dataset

# Step 2: Load the SODA dataset for training
echo -e "\n2. Loading SODA dataset..."
python scripts/data_preparation/load_soda.py --split train --output-dir data/soda --max-examples 1000 --require-topic --require-persona --skip-existing

# Skip classifier training if it takes too long
echo -e "\n3. Skipping classifier training (would take too long on CPU)..."
# python scripts/training/train.py train-classifier \
#    --train-dataset data/sample_dataset/train \
#    --val-dataset data/sample_dataset/val \
#    --test-dataset data/sample_dataset/test

# Step 3: Generate personality profiles for each trait
echo -e "\n4. Generating personality profiles for each Big Five trait..."

# Loop through all traits
for trait in o c e a n; do
  trait_name=""
  case $trait in
    o) trait_name="Openness" ;;
    c) trait_name="Conscientiousness" ;;
    e) trait_name="Extraversion" ;;
    a) trait_name="Agreeableness" ;;
    n) trait_name="Neuroticism" ;;
  esac
  
  echo -e "\n4.$trait. Generating profiles for $trait_name..."
  python scripts/training/train.py generate-profiles \
    --input-file data/soda/soda_train.csv \
    --output-file data/soda/profiles_${trait}.jsonl \
    --trait $trait \
    --alpha 0.5 \
    --chunk 1/1
done

# Step 4: Test the DExpert model with each trait
echo -e "\n5. Testing DExpert with each personality trait..."
for trait in o c e a n; do
  trait_name=""
  case $trait in
    o) trait_name="Openness" ;;
    c) trait_name="Conscientiousness" ;;
    e) trait_name="Extraversion" ;;
    a) trait_name="Agreeableness" ;;
    n) trait_name="Neuroticism" ;;
  esac
  
  echo -e "\n5.$trait. Testing personality adaptation for $trait_name..."
  python scripts/evaluation/run_personality_test.py --trait $trait --input-file data/soda/soda_train.csv \
    --output-file logs/personality_test_${trait}.json
done

echo -e "\nWorkflow complete! All personality traits have been tested."
echo -e "\nResults are stored in:"
echo -e "  - Profiles: data/soda/profiles_*.jsonl"
echo -e "  - Test results: logs/personality_test_*.json" 