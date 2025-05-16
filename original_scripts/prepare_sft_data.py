import pandas as pd
import json
import os
import argparse

# Define paths to the input CSV files
MESSAGES_CSV_PATH = "external/psychgenerator/processed_data/generated_posts.csv"
VARIABLES_CSV_PATH = "external/psychgenerator/data/generated_variables.csv"
OUTPUT_DIR = "data/sft_alpaca"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sft_data.jsonl")

# Define trait mapping and binarization thresholds
TRAIT_MAP = {
    "variable1": "Openness",
    "variable2": "Conscientiousness",
    "variable3": "Extraversion",
    "variable4": "Agreeableness",
    "variable5": "Neuroticism",
}
HIGH_THRESHOLD = 0.6
LOW_THRESHOLD = 0.4
NUM_INPUT_WORDS = 5

def prepare_sft_data(variables_csv_path, output_jsonl_path):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_jsonl_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        messages_df = pd.read_csv(MESSAGES_CSV_PATH)
        variables_df = pd.read_csv(variables_csv_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find input CSV files: {e}")
        return

    # Merge dataframes
    merged_df = pd.merge(messages_df, variables_df, on="user_id")

    sft_examples = []

    for _, row in merged_df.iterrows():
        message_text = str(row["message"])
        words = message_text.split()
        
        if len(words) <= NUM_INPUT_WORDS:
            print(f"Skipping short message_id {row['message_id']}: {message_text}")
            continue

        input_text = " ".join(words[:NUM_INPUT_WORDS])
        output_text = " ".join(words[NUM_INPUT_WORDS:])

        for var_col, trait_name in TRAIT_MAP.items():
            score = row[var_col]
            trait_level = None
            if score >= HIGH_THRESHOLD:
                trait_level = "high"
            elif score <= LOW_THRESHOLD:
                trait_level = "low"
            
            if trait_level:
                instruction = f"Help me complete the sentence with certain Big Five Personality: {trait_name} - {trait_level}."
                sft_examples.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text,
                })

    # Save to JSONL file
    with open(output_jsonl_path, 'w') as f:
        for example in sft_examples:
            f.write(json.dumps(example) + '\n')

    print(f"Successfully prepared {len(sft_examples)} SFT examples in Alpaca format.")
    print(f"Output saved to: {output_jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SFT data in Alpaca format from generated posts and personality variables.")
    parser.add_argument("--variables_csv", type=str, 
                        default="external/psychgenerator/data/generated_variables.csv", 
                        help="Path to the input variables CSV file with personality scores.")
    parser.add_argument("--output_file", type=str, 
                        default="data/sft_alpaca/sft_data.jsonl", 
                        help="Path to save the output SFT JSONL file.")
    # Potentially add --messages_csv if needed later

    args = parser.parse_args()
    prepare_sft_data(args.variables_csv, args.output_file) 