# BIG5Chat

BIG5Chat is a framework designed to generate personality-steered conversational agents based on the Big Five personality traits. The project aims to facilitate psychological studies by allowing AI agents to interact with each other while exhibiting different personality characteristics.

## Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running with Gemma models
- Required Python packages (install via `pip install -r requirements.txt`)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/big5.git
   cd big5
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is installed and running:

   ```bash
   # Check if Ollama is installed
   ollama --version

   # Start the Ollama service if not already running
   ollama serve
   ```

4. Verify available models:

   ```bash
   ollama list
   ```

   You should see models like `gemma3-openness`, `gemma3-conscientiousness`, etc.

## Using BIG5Chat

### Testing Personality Traits

Use the `scripts/evaluation/run_personality_test.py` script to test different personality traits:

```bash
# Test with openness trait
python scripts/evaluation/run_personality_test.py --trait o

# Test with conscientiousness trait
python scripts/evaluation/run_personality_test.py --trait c

# Test with extraversion trait
python scripts/evaluation/run_personality_test.py --trait e

# Test with agreeableness trait
python scripts/evaluation/run_personality_test.py --trait a

# Test with neuroticism trait
python scripts/evaluation/run_personality_test.py --trait n

# You can also provide a custom prompt
python scripts/evaluation/run_personality_test.py --trait o --prompt "How would you approach solving a difficult problem?"
```

### Training the RoBERTa Classifier

To train the personality classifier:

```bash
# Create sample dataset
python scripts/data_preparation/create_datasets.py sample

# Train the classifier
python scripts/training/train.py train-classifier \
  --train-dataset data/sample_dataset/train \
  --val-dataset data/sample_dataset/val \
  --test-dataset data/sample_dataset/test
```

### Running the Full Pipeline

To run the complete pipeline:

```bash
./run_all.sh
```

This will:

1. Create sample datasets
2. Train the classifier (if not disabled)
3. Test the DExpert model using local Ollama models

## Project Structure

```
big5/
├── data/                      # Data files and datasets
│   ├── sample_dataset/        # Sample training data
│   ├── sft_alpaca/            # SFT data in alpaca format
│   ├── soda/                  # SODA dialogue dataset
│   └── synthetic_psychgenerator/ # Synthetic personality data
├── llm_personality/           # Core personality framework
│   ├── classifier/            # RoBERTa-based classifier
│   ├── dexpert/               # DExpert model for steering outputs
│   └── profile_creation/      # Tools for creating profiles
├── modelfiles/                # Ollama model definitions
├── scripts/                   # Organized scripts
│   ├── data_preparation/      # Data preparation scripts
│   ├── evaluation/            # Evaluation scripts
│   ├── training/              # Training scripts
│   └── utils/                 # Utility scripts
├── external/                  # External dependencies
├── eval_results/              # Evaluation result files
├── logs/                      # Log files
├── requirements.txt           # Python dependencies
└── run_all.sh                 # Main script to run the pipeline
```

## Personality Traits

The Big Five personality traits used in this project are:

- **O**penness: Tendency to be open to new experiences, ideas, and creativity
- **C**onscientiousness: Tendency to be organized, disciplined, and achievement-oriented
- **E**xtraversion: Tendency to be outgoing, energetic, and social
- **A**greeableness: Tendency to be cooperative, compassionate, and trusting
- **N**euroticism: Tendency to experience negative emotions and stress

## Local Models

This project uses Ollama to run local Gemma3 models tuned for different personality traits:

- `gemma3-openness`: Tuned for responses exhibiting high openness
- `gemma3-conscientiousness`: Tuned for responses exhibiting high conscientiousness
- `gemma3-extraversion`: Tuned for responses exhibiting high extraversion
- `gemma3-agreeableness`: Tuned for responses exhibiting high agreeableness
- `gemma3-neuroticism`: Tuned for responses exhibiting high neuroticism
- `gemma3:12b`: Base model for comparison

## Troubleshooting

- If you encounter errors about missing models, make sure Ollama is running (`ollama serve`) and you have the required models installed.
- If you have issues with multiprocessing, try setting environment variable `PYTHONPATH=.` before running scripts.
- For MacOS users with Apple Silicon, the code is optimized to use the MPS backend instead of CUDA.

## License

[MIT License](LICENSE)
