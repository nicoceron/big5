import pandas as pd
import os
import re
import argparse

# GENERATED_POSTS_CSV = "external/psychgenerator/processed_data/generated_posts.csv"
# OUTPUT_VARIABLES_CSV = "external/psychgenerator/data/generated_variables.csv"

# Define trait mapping (variableN to actual trait name if needed, but for now, use variableN)
# Based on prepare_sft_data.py's TRAIT_MAP:
# variable1: Openness, variable2: Conscientiousness, variable3: Extraversion, 
# variable4: Agreeableness, variable5: Neuroticism
ALL_VARIABLES = ["variable1", "variable2", "variable3", "variable4", "variable5"]
VARIABLE_TO_TRAIT_NAME = {
    "variable1": "Openness",
    "variable2": "Conscientiousness",
    "variable3": "Extraversion",
    "variable4": "Agreeableness",
    "variable5": "Neuroticism",
}

# Define scores for levels
LOW_SCORE = 0.1
MEDIUM_SCORE = 0.5
HIGH_SCORE = 0.9
NEUTRAL_SCORE = 0.5 # For non-target variables

def create_variables_for_generated_posts(posts_csv_path, output_csv_path, target_profile_args):
    try:
        posts_df = pd.read_csv(posts_csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {posts_csv_path}. Please generate posts first.")
        return

    unique_user_ids = posts_df['user_id'].unique()
    variables_data = []

    # Convert target_profile_args to a more usable dict: {'variable1': 'high', 'variable5': 'low'}
    profile_targets = {}
    if target_profile_args:
        for i in range(1, 6): # variable1 to variable5
            var_key = f"variable{i}"
            level = getattr(target_profile_args, f"target_{var_key}", None)
            if level:
                profile_targets[var_key] = level.lower()

    for user_id in unique_user_ids:
        user_scores = {"user_id": user_id}
        
        if profile_targets: # If a specific profile is being targeted
            print(f"Generating scores for user {user_id} based on target profile: {profile_targets}")
            for var_name in ALL_VARIABLES:
                target_level = profile_targets.get(var_name)
                if target_level == "low":
                    user_scores[var_name] = LOW_SCORE
                elif target_level == "high":
                    user_scores[var_name] = HIGH_SCORE
                elif target_level == "medium": # Allow medium if explicitly set
                    user_scores[var_name] = MEDIUM_SCORE
                else: # If not specified in profile, or specified as something else, use neutral
                    user_scores[var_name] = NEUTRAL_SCORE
        else: # Original behavior: infer from user_id
            match = re.match(r"gen_(variable[1-5])_(low|medium|high)_\\d+", user_id)
            if not match:
                print(f"Warning: Could not parse user_id: {user_id} for score inference. Skipping.")
                continue
            
            source_target_variable, source_level = match.groups()
            print(f"Generating scores for user {user_id} based on user_id inference (target: {source_target_variable}, level: {source_level})")

            for var_name in ALL_VARIABLES:
                if var_name == source_target_variable:
                    if source_level == "low":
                        user_scores[var_name] = LOW_SCORE
                    elif source_level == "medium":
                        user_scores[var_name] = MEDIUM_SCORE
                    elif source_level == "high":
                        user_scores[var_name] = HIGH_SCORE
                else:
                    user_scores[var_name] = NEUTRAL_SCORE
        
        variables_data.append(user_scores)

    if not variables_data:
        print("No user data processed or generated. Exiting.")
        return

    variables_df = pd.DataFrame(variables_data)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    variables_df.to_csv(output_csv_path, index=False)
    print(f"Successfully created {output_csv_path} with {len(variables_df)} users.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a variables.csv file with personality scores for generated posts.")
    parser.add_argument("--posts_csv", type=str, default="external/psychgenerator/processed_data/generated_posts.csv", help="Path to the input generated_posts.csv file.")
    parser.add_argument("--output_csv", type=str, default="external/psychgenerator/data/generated_variables.csv", help="Path to save the output variables CSV file.")
    
    # Arguments for specific profile targeting
    for i in range(1, 6): # variable1 to variable5
        trait_name = VARIABLE_TO_TRAIT_NAME[f"variable{i}"]
        parser.add_argument(f"--target_variable{i}", type=str, choices=['low', 'medium', 'high'], default=None, 
                            help=f"Target level for {trait_name} (variable{i}). If any target_variable is set, all users will get this profile.")

    args = parser.parse_args()

    # Check if any target_variableX is set to activate profile mode
    is_profile_targeting_active = any(getattr(args, f"target_variable{i}") for i in range(1, 6))

    if is_profile_targeting_active:
        print("Profile targeting activated. All users in the output CSV will conform to the specified profile.")
    else:
        print("No specific profile targeted. Scores will be inferred from user_ids in posts_csv (original behavior).")

    create_variables_for_generated_posts(args.posts_csv, args.output_csv, args if is_profile_targeting_active else None) 