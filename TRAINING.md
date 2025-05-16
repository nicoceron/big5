# Training BIG5Chat Models

This document explains how to train the different components of the BIG5Chat framework.

## Setup

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have the necessary datasets and resources.

## Training the RoBERTa Classifier

The classifier is used to predict personality traits based on text inputs.

```bash
python train.py train-classifier \
    --train-dataset data/train_data \
    --val-dataset data/val_data \
    --test-dataset data/test_data  # Optional
```

You need to prepare your datasets in the Hugging Face datasets format. To convert your data:

```python
from datasets import Dataset
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

# Convert to datasets format
dataset = Dataset.from_pandas(data)

# Save to disk
dataset.save_to_disk("data/train_data")
```

## Generating Personality Profiles

Generate personality-influenced profiles using Llama3:

```bash
python train.py generate-profiles \
    --input-file data/sample_input.csv \
    --output-file data/profiles.jsonl \
    --trait o \  # Choose from: o, c, e, a, n
    --alpha 0.5 \
    --chunk 1/1
```

- `trait`: Represents one of the Big Five personality traits:

  - o: Openness
  - c: Conscientiousness
  - e: Extraversion
  - a: Agreeableness
  - n: Neuroticism

- `alpha`: Controls the strength of personality influence (0.0-1.0)
- `chunk`: For parallel processing (format: "x/y")

## Running DExpert Model

The DExpert model adapts language model outputs based on personality traits:

```bash
python train.py run-dexpert
```

## Note on Models

- For the classifier, the code uses RoBERTa-large from Hugging Face.
- For profile generation and DExpert, the code uses Llama3-70B-Instruct.

Accessing these models, especially Llama3, may require API keys or special access permissions. Make sure you have the necessary credentials configured.

## Workflow

1. Start by training the classifier to recognize personality traits in text.
2. Generate personality profiles using the profile generation tool.
3. Use the DExpert model to steer text generation based on personality traits.

## Data Structure

The expected format for training data will vary by component, but generally:

- Classifier: Text inputs and personality trait scores
- Profile Generation: Topics and initial personas
- DExpert: Trained models and personality specifications
