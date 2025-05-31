#!/bin/bash
set -e

# Create directories
mkdir -p data
mkdir -p logs
mkdir -p modelfiles

echo "=== BIG5Chat Training Workflow ==="
echo "This script will train the BIG5Chat personality models with large datasets."

# Step 0: Check if Ollama is running and set up models
echo -e "\n0. Setting up Ollama personality models..."
python scripts/utils/setup_ollama_models.py --base-model gemma3

# Step 1: Create directories and prepare data
echo -e "\n1. Creating sample dataset..."
python scripts/data_preparation/create_datasets.py sample --output-path data/sample_dataset

# Step 2: Load large SODA dataset for training
echo -e "\n2. Loading large SODA dataset (50K examples)..."
python scripts/data_preparation/load_soda.py --split train --output-dir data/soda --max-examples 50000 --skip-existing

# Step 2b: Also load validation dataset
echo -e "\n2b. Loading validation dataset..."
python scripts/data_preparation/load_soda.py --split validation --output-dir data/soda --max-examples 10000 --skip-existing

# Step 2c: Load additional personality datasets
echo -e "\n2c. Loading additional datasets..."
python scripts/data_preparation/load_additional_datasets.py --output-dir data/additional

# Step 3: Train the RoBERTa classifier with large dataset
echo -e "\n3. Training RoBERTa classifier with large dataset..."
python scripts/training/train.py train-classifier \
   --train-dataset data/sample_dataset/train \
   --val-dataset data/sample_dataset/val \
   --test-dataset data/sample_dataset/test \
   --epochs 10 \
   --batch-size 32

# Step 4: Generate personality profiles for each trait with large dataset
echo -e "\n4. Generating personality profiles for each Big Five trait with large dataset..."

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
  
  echo -e "\n4.$trait. Generating profiles for $trait_name with large dataset..."
  python scripts/training/train.py generate-profiles \
    --input-file data/soda/soda_train.csv \
    --output-file data/soda/profiles_${trait}.jsonl \
    --trait $trait \
    --alpha 0.5 \
    --chunk 1/1 \
    --max-examples 10000
done

# Step 5: Fine-tune personality-specific LLM models
echo -e "\n5. Fine-tuning personality-specific models..."
for trait in o c e a n; do
  trait_name=""
  case $trait in
    o) trait_name="Openness" ;;
    c) trait_name="Conscientiousness" ;;
    e) trait_name="Extraversion" ;;
    a) trait_name="Agreeableness" ;;
    n) trait_name="Neuroticism" ;;
  esac
  
  echo -e "\n5.$trait. Fine-tuning model for $trait_name..."
  python scripts/training/finetune_personality_models.py \
    --trait $trait \
    --train-data data/soda/profiles_${trait}.jsonl \
    --base-model microsoft/DialoGPT-medium \
    --epochs 5 \
    --output-dir model_backup/personality_${trait}
done

# Step 6: Comprehensive evaluation
echo -e "\n6. Running comprehensive evaluation..."
python scripts/evaluation/comprehensive_evaluation.py \
  --test-datasets data/soda/soda_validation.csv \
  --models "model_backup/personality_*" \
  --output-dir eval_results/large_scale_eval

# Step 7: Test the DExpert model with each trait on larger test set
echo -e "\n7. Testing DExpert with each personality trait on large test set..."
for trait in o c e a n; do
  trait_name=""
  case $trait in
    o) trait_name="Openness" ;;
    c) trait_name="Conscientiousness" ;;
    e) trait_name="Extraversion" ;;
    a) trait_name="Agreeableness" ;;
    n) trait_name="Neuroticism" ;;
  esac
  
  echo -e "\n7.$trait. Testing personality adaptation for $trait_name on large dataset..."
  python scripts/evaluation/run_personality_test.py --trait $trait \
    --input-file data/soda/soda_validation.csv \
    --output-file logs/personality_test_${trait}_large.json \
    --num-examples 1000
done

echo -e "\nWorkflow complete! All personality models have been trained and tested on large datasets."
echo -e "\nResults are stored in:"
echo -e "  - Training profiles: data/soda/profiles_*.jsonl"
echo -e "  - Fine-tuned models: model_backup/personality_*"
echo -e "  - Evaluation results: eval_results/large_scale_eval"
echo -e "  - Test results: logs/personality_test_*_large.json" 