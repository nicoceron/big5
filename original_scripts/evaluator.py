import torch
from transformers import AutoTokenizer
import json

class PersonalityEvaluator:
    def __init__(self, classifier_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(classifier_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    
    def evaluate_dataset(self, dataset_path):
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        results = {}
        
        for trait in dataset.keys():
            trait_accuracy = {"high": 0, "low": 0}
            
            for level in ["high", "low"]:
                correct = 0
                total = len(dataset[trait][level])
                
                for entry in dataset[trait][level]:
                    response = entry["response"]
                    inputs = self.tokenizer(response, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    scores = outputs.squeeze().cpu().numpy()
                    
                    trait_indices = {
                        "openness": 0,
                        "conscientiousness": 1,
                        "extraversion": 2,
                        "agreeableness": 3,
                        "neuroticism": 4
                    }
                    
                    trait_score = scores[trait_indices[trait]]
                    predicted_level = "high" if trait_score > 0.5 else "low"
                    
                    if predicted_level == level:
                        correct += 1
                
                trait_accuracy[level] = (correct / total) * 100
            
            results[trait] = trait_accuracy
            
        avg_accuracy = 0
        count = 0
        
        for trait, accuracies in results.items():
            for level, accuracy in accuracies.items():
                avg_accuracy += accuracy
                count += 1
                
        results["average"] = avg_accuracy / count
        
        return results
