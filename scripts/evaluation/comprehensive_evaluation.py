#!/usr/bin/env python3
import argparse
import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import ollama
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PersonalityEvaluator:
    def __init__(self, args):
        self.args = args
        self.results = {}
        
    def load_test_data(self):
        """Load test datasets."""
        logger.info("Loading test datasets...")
        
        test_datasets = []
        
        # Load main test dataset
        if os.path.exists(self.args.test_datasets):
            df = pd.read_csv(self.args.test_datasets)
            test_datasets.append(("main", df))
        
        # Load additional test files if they exist
        additional_files = [
            "data/additional/additional_personality_data.csv",
            "data/sample_dataset/test/dataset.arrow"
        ]
        
        for file_path in additional_files:
            if os.path.exists(file_path):
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    test_datasets.append((os.path.basename(file_path), df))
        
        logger.info(f"Loaded {len(test_datasets)} test datasets")
        return test_datasets
    
    def evaluate_ollama_models(self, test_data):
        """Evaluate Ollama-based personality models."""
        logger.info("Evaluating Ollama personality models...")
        
        traits = ['o', 'c', 'e', 'a', 'n']
        trait_names = {
            'o': 'Openness',
            'c': 'Conscientiousness', 
            'e': 'Extraversion',
            'a': 'Agreeableness',
            'n': 'Neuroticism'
        }
        
        results = {}
        
        for trait in traits:
            logger.info(f"Evaluating {trait_names[trait]}...")
            
            # Test high and low variants
            for level in ['high', 'low']:
                model_name = f"gemma3-{trait_names[trait].lower()}" + ("" if level == "high" else "-low")
                
                try:
                    # Test the model with sample prompts
                    test_prompts = [
                        "What are your thoughts on trying new experiences?",
                        "How do you approach planning and organization?",
                        "What's your preference for social situations?",
                        "How do you handle conflicts with others?",
                        "How do you typically respond to stress?"
                    ]
                    
                    responses = []
                    for prompt in test_prompts:
                        try:
                            response = ollama.chat(
                                model=model_name,
                                messages=[{"role": "user", "content": prompt}]
                            )
                            responses.append(response['message']['content'])
                        except Exception as e:
                            logger.warning(f"Error with model {model_name}: {e}")
                            responses.append("No response")
                    
                    # Analyze responses for personality indicators
                    personality_score = self.analyze_personality_indicators(responses, trait, level)
                    
                    results[f"{trait}_{level}"] = {
                        "model": model_name,
                        "responses": responses,
                        "personality_score": personality_score,
                        "trait": trait,
                        "level": level
                    }
                    
                except Exception as e:
                    logger.error(f"Could not evaluate model {model_name}: {e}")
        
        return results
    
    def analyze_personality_indicators(self, responses, trait, level):
        """Analyze responses for personality trait indicators."""
        # Simple keyword-based analysis
        trait_keywords = {
            'o': {
                'high': ['creative', 'innovative', 'curious', 'explore', 'new', 'imagination', 'abstract'],
                'low': ['traditional', 'conventional', 'practical', 'routine', 'familiar', 'stable']
            },
            'c': {
                'high': ['organized', 'planned', 'systematic', 'disciplined', 'goal', 'schedule', 'careful'],
                'low': ['flexible', 'spontaneous', 'adaptable', 'relaxed', 'flow', 'improvise']
            },
            'e': {
                'high': ['social', 'outgoing', 'energetic', 'people', 'party', 'excitement', 'attention'],
                'low': ['quiet', 'reserved', 'thoughtful', 'alone', 'introspective', 'private']
            },
            'a': {
                'high': ['cooperative', 'kind', 'trusting', 'harmony', 'considerate', 'empathy'],
                'low': ['direct', 'critical', 'assertive', 'logical', 'honest', 'skeptical']
            },
            'n': {
                'high': ['worry', 'stress', 'anxious', 'sensitive', 'emotional', 'overwhelmed'],
                'low': ['calm', 'stable', 'resilient', 'composed', 'relaxed', 'confident']
            }
        }
        
        keywords = trait_keywords.get(trait, {}).get(level, [])
        
        total_score = 0
        total_words = 0
        
        for response in responses:
            words = response.lower().split()
            total_words += len(words)
            
            for keyword in keywords:
                total_score += response.lower().count(keyword)
        
        # Normalize score
        personality_score = total_score / max(total_words, 1) * 100
        return personality_score
    
    def evaluate_fine_tuned_models(self, test_data):
        """Evaluate fine-tuned models if they exist."""
        logger.info("Evaluating fine-tuned models...")
        
        model_dirs = []
        if os.path.exists("model_backup"):
            for item in os.listdir("model_backup"):
                item_path = os.path.join("model_backup", item)
                if os.path.isdir(item_path) and "personality" in item:
                    model_dirs.append(item_path)
        
        results = {}
        
        for model_dir in model_dirs:
            try:
                logger.info(f"Evaluating model: {model_dir}")
                
                # Load the model
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModelForCausalLM.from_pretrained(model_dir)
                
                # Test on sample data
                test_prompts = [
                    "What motivates you in life?",
                    "How do you handle challenges?",
                    "What's important to you in relationships?"
                ]
                
                responses = []
                for prompt in test_prompts:
                    try:
                        inputs = tokenizer.encode(prompt, return_tensors="pt")
                        with torch.no_grad():
                            outputs = model.generate(
                                inputs, 
                                max_length=inputs.shape[1] + 50,
                                num_return_sequences=1,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id
                            )
                        
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        response = response[len(prompt):].strip()
                        responses.append(response)
                    except Exception as e:
                        logger.warning(f"Error generating response: {e}")
                        responses.append("No response")
                
                results[os.path.basename(model_dir)] = {
                    "model_path": model_dir,
                    "responses": responses,
                    "prompts": test_prompts
                }
                
            except Exception as e:
                logger.error(f"Could not evaluate model {model_dir}: {e}")
        
        return results
    
    def personality_coherence_test(self, responses, expected_trait, expected_level):
        """Test if responses are coherent with expected personality trait."""
        # Simple coherence scoring based on response consistency
        coherence_score = 0
        
        # Check for trait-appropriate language patterns
        trait_patterns = {
            'o_high': ['new', 'creative', 'explore', 'imagine'],
            'o_low': ['traditional', 'practical', 'routine'],
            'c_high': ['organized', 'plan', 'systematic'],
            'c_low': ['flexible', 'spontaneous'],
            'e_high': ['social', 'people', 'outgoing'],
            'e_low': ['quiet', 'thoughtful', 'alone'],
            'a_high': ['kind', 'cooperative', 'caring'],
            'a_low': ['direct', 'honest', 'logical'],
            'n_high': ['worry', 'stress', 'emotional'],
            'n_low': ['calm', 'stable', 'relaxed']
        }
        
        pattern_key = f"{expected_trait}_{expected_level}"
        expected_patterns = trait_patterns.get(pattern_key, [])
        
        for response in responses:
            response_lower = response.lower()
            for pattern in expected_patterns:
                if pattern in response_lower:
                    coherence_score += 1
        
        # Normalize by number of responses and patterns
        coherence_score = coherence_score / (len(responses) * len(expected_patterns)) if expected_patterns else 0
        return coherence_score
    
    def generate_evaluation_report(self):
        """Generate a comprehensive evaluation report."""
        logger.info("Generating evaluation report...")
        
        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Load test data
        test_datasets = self.load_test_data()
        
        # Discover and evaluate different model types
        model_dirs = self.discover_models()
        fine_tuned_results = {}
        
        if model_dirs:
            logger.info(f"Evaluating {len(model_dirs)} fine-tuned models...")
            fine_tuned_results = self.evaluate_discovered_models(model_dirs, test_datasets)
        else:
            logger.warning("No fine-tuned models found to evaluate")
        
        # Try to evaluate Ollama models (if available)
        ollama_results = {}
        try:
            ollama_results = self.evaluate_ollama_models(test_datasets)
        except Exception as e:
            logger.warning(f"Could not evaluate Ollama models: {e}")
        
        # Create comprehensive report
        report = {
            "evaluation_summary": {
                "ollama_models_tested": len(ollama_results),
                "fine_tuned_models_tested": len(fine_tuned_results),
                "test_datasets": len(test_datasets)
            },
            "ollama_results": ollama_results,
            "fine_tuned_results": fine_tuned_results,
            "performance_metrics": self.calculate_performance_metrics(ollama_results, fine_tuned_results)
        }
        
        # Save detailed results
        with open(os.path.join(self.args.output_dir, "evaluation_report.json"), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate visualizations
        self.create_visualizations(report)
        
        # Generate summary text report
        self.create_text_summary(report)
        
        logger.info(f"Evaluation report saved to {self.args.output_dir}")
        return report
    
    def calculate_performance_metrics(self, ollama_results, fine_tuned_results):
        """Calculate performance metrics across all models."""
        metrics = {}
        
        # Ollama model metrics
        if ollama_results:
            trait_scores = {}
            for result_key, result in ollama_results.items():
                trait = result['trait']
                level = result['level']
                score = result['personality_score']
                
                if trait not in trait_scores:
                    trait_scores[trait] = {'high': [], 'low': []}
                trait_scores[trait][level].append(score)
            
            # Calculate average scores per trait
            trait_averages = {}
            for trait, levels in trait_scores.items():
                trait_averages[trait] = {
                    'high_avg': np.mean(levels['high']) if levels['high'] else 0,
                    'low_avg': np.mean(levels['low']) if levels['low'] else 0
                }
            
            metrics['ollama_trait_scores'] = trait_averages
        
        # Fine-tuned model metrics
        if fine_tuned_results:
            metrics['fine_tuned_models'] = list(fine_tuned_results.keys())
        
        return metrics
    
    def create_visualizations(self, report):
        """Create visualization plots for the evaluation results."""
        logger.info("Creating visualization plots...")
        
        plt.style.use('default')
        
        # Plot 1: Personality scores by trait
        if 'ollama_results' in report and report['ollama_results']:
            fig, ax = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Personality Model Evaluation Results', fontsize=16)
            
            traits = ['o', 'c', 'e', 'a', 'n']
            trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
            
            for i, (trait, trait_name) in enumerate(zip(traits, trait_names)):
                row = i // 3
                col = i % 3
                
                high_scores = []
                low_scores = []
                
                for result_key, result in report['ollama_results'].items():
                    if result['trait'] == trait:
                        if result['level'] == 'high':
                            high_scores.append(result['personality_score'])
                        else:
                            low_scores.append(result['personality_score'])
                
                if high_scores or low_scores:
                    ax[row, col].bar(['High', 'Low'], 
                                   [np.mean(high_scores) if high_scores else 0,
                                    np.mean(low_scores) if low_scores else 0])
                    ax[row, col].set_title(trait_name)
                    ax[row, col].set_ylabel('Personality Score')
                    ax[row, col].set_ylim(0, max(max(high_scores + low_scores, default=[1]), 1))
            
            # Remove empty subplot
            fig.delaxes(ax[1, 2])
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.args.output_dir, 'personality_scores.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Visualizations saved")
    
    def create_text_summary(self, report):
        """Create a text summary of the evaluation results."""
        summary_text = f"""
# Personality Model Evaluation Report

## Summary
- Ollama models tested: {report['evaluation_summary']['ollama_models_tested']}
- Fine-tuned models tested: {report['evaluation_summary']['fine_tuned_models_tested']}
- Test datasets used: {report['evaluation_summary']['test_datasets']}

## Ollama Model Results
"""
        
        if report['ollama_results']:
            for result_key, result in report['ollama_results'].items():
                summary_text += f"\n### {result['trait'].upper()} - {result['level'].title()}\n"
                summary_text += f"- Model: {result['model']}\n"
                summary_text += f"- Personality Score: {result['personality_score']:.2f}\n"
                summary_text += f"- Sample Response: {result['responses'][0][:100]}...\n"
        
        if report['fine_tuned_results']:
            summary_text += "\n## Fine-tuned Model Results\n"
            for model_name, result in report['fine_tuned_results'].items():
                summary_text += f"\n### {model_name}\n"
                summary_text += f"- Model Path: {result['model_path']}\n"
                summary_text += f"- Sample Response: {result['responses'][0][:100] if result['responses'] else 'No response'}...\n"
        
        # Save summary
        with open(os.path.join(self.args.output_dir, 'evaluation_summary.md'), 'w') as f:
            f.write(summary_text)

    def discover_models(self):
        """Discover model directories based on the pattern."""
        logger.info(f"Discovering models with pattern: {self.args.models}")
        
        model_dirs = glob.glob(self.args.models)
        model_dirs = [d for d in model_dirs if os.path.isdir(d)]
        
        logger.info(f"Found {len(model_dirs)} model directories: {model_dirs}")
        return model_dirs

    def evaluate_discovered_models(self, model_dirs, test_datasets):
        """Evaluate the discovered fine-tuned models."""
        results = {}
        
        for model_dir in model_dirs:
            try:
                logger.info(f"Evaluating model: {model_dir}")
                
                # Extract trait from model directory name
                trait = None
                if "personality_o" in model_dir:
                    trait = "openness"
                elif "personality_c" in model_dir:
                    trait = "conscientiousness"
                elif "personality_e" in model_dir:
                    trait = "extraversion"
                elif "personality_a" in model_dir:
                    trait = "agreeableness"
                elif "personality_n" in model_dir:
                    trait = "neuroticism"
                
                # Load the model
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModelForCausalLM.from_pretrained(model_dir)
                
                # Test on sample data
                test_prompts = [
                    "What motivates you in life?",
                    "How do you handle challenges?",
                    "What's important to you in relationships?",
                    "Describe your ideal work environment.",
                    "How do you make important decisions?"
                ]
                
                responses = []
                for prompt in test_prompts:
                    try:
                        inputs = tokenizer.encode(prompt, return_tensors="pt")
                        with torch.no_grad():
                            outputs = model.generate(
                                inputs, 
                                max_length=inputs.shape[1] + 50,
                                num_return_sequences=1,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id
                            )
                        
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        response = response[len(prompt):].strip()
                        responses.append(response)
                    except Exception as e:
                        logger.warning(f"Error generating response for {prompt}: {e}")
                        responses.append("No response")
                
                # Calculate personality score
                personality_score = self.calculate_model_personality_score(responses, trait)
                
                results[os.path.basename(model_dir)] = {
                    "model_path": model_dir,
                    "trait": trait,
                    "responses": responses,
                    "prompts": test_prompts,
                    "personality_score": personality_score
                }
                
            except Exception as e:
                logger.error(f"Could not evaluate model {model_dir}: {e}")
                results[os.path.basename(model_dir)] = {
                    "model_path": model_dir,
                    "error": str(e),
                    "responses": [],
                    "personality_score": 0
                }
        
        return results
    
    def calculate_model_personality_score(self, responses, trait):
        """Calculate a personality score for the model based on trait-specific keywords."""
        if not trait or not responses:
            return 0
        
        trait_keywords = {
            'openness': ['creative', 'innovative', 'curious', 'explore', 'new', 'imagination', 'abstract', 'artistic'],
            'conscientiousness': ['organized', 'planned', 'systematic', 'disciplined', 'goal', 'schedule', 'careful', 'responsible'],
            'extraversion': ['social', 'outgoing', 'energetic', 'people', 'party', 'excitement', 'attention', 'talkative'],
            'agreeableness': ['cooperative', 'kind', 'trusting', 'harmony', 'considerate', 'empathy', 'helpful', 'compassionate'],
            'neuroticism': ['worry', 'stress', 'anxious', 'sensitive', 'emotional', 'overwhelmed', 'nervous', 'unstable']
        }
        
        keywords = trait_keywords.get(trait, [])
        
        total_score = 0
        total_words = 0
        
        for response in responses:
            if response and response != "No response":
                words = response.lower().split()
                total_words += len(words)
                
                for keyword in keywords:
                    total_score += response.lower().count(keyword)
        
        # Normalize score
        personality_score = (total_score / max(total_words, 1)) * 100
        return personality_score

def main():
    parser = argparse.ArgumentParser(description="Comprehensive personality model evaluation")
    parser.add_argument("--test-datasets", required=True,
                        help="Path to test dataset CSV file")
    parser.add_argument("--models", default="model_backup/personality_*",
                        help="Pattern for model directories to evaluate")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Create evaluator and run evaluation
    evaluator = PersonalityEvaluator(args)
    report = evaluator.generate_evaluation_report()
    
    logger.info("Comprehensive evaluation completed!")
    return 0

if __name__ == "__main__":
    exit(main()) 