#!/usr/bin/env python3
"""
Systematic testing script to evaluate both embedding models using test phrases.
Generates comprehensive statistics and performance metrics.
"""

import json
import time
import argparse
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class ModelEvaluator:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.models = {
            'mini': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'index_name': 'all-mini-lm-data',
                'model': None
            },
            'gemma': {
                'model_name': 'google/embeddinggemma-300m',
                'index_name': 'embedding-gemma-data', 
                'model': None
            }
        }
        self.results = {}
    
    def load_models(self):
        """Load both embedding models."""
        print("Loading models...")
        for model_key, config in self.models.items():
            print(f"  Loading {model_key}...")
            start_time = time.time()
            config['model'] = SentenceTransformer(config['model_name'])
            load_time = time.time() - start_time
            print(f"    Loaded in {load_time:.2f}s")
    
    def evaluate_query(self, model_key: str, query: str, target_id: str, target_type: str) -> Dict[str, Any]:
        """Evaluate a single query against a model."""
        config = self.models[model_key]
        index = self.pc.Index(config['index_name'])
        
        # Generate embedding and search
        start_time = time.time()
        query_embedding = config['model'].encode(query, convert_to_tensor=False).tolist()
        results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )
        search_time = time.time() - start_time
        
        # Analyze results
        found_target = False
        target_rank = None
        target_score = None
        
        for i, match in enumerate(results.matches, 1):
            if match.id == target_id:
                found_target = True
                target_rank = i
                target_score = match.score
                break
        
        # Check if we found the right type of result
        found_correct_type = False
        for match in results.matches[:3]:  # Check top 3
            if match.id.startswith(f"{target_type}_"):
                found_correct_type = True
                break
        
        return {
            'query': query,
            'target_id': target_id,
            'target_type': target_type,
            'found_target': found_target,
            'target_rank': target_rank,
            'target_score': target_score,
            'search_time': search_time,
            'found_correct_type': found_correct_type,
            'top_score': results.matches[0].score if results.matches else 0,
            'top_result_id': results.matches[0].id if results.matches else None
        }
    
    def run_systematic_tests(self, test_phrases_path: str, max_tests_per_category: int = 10):
        """Run systematic tests on both models."""
        # Load test phrases
        with open(test_phrases_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"Running systematic tests...")
        print(f"Max tests per category: {max_tests_per_category}")
        
        for model_key in self.models.keys():
            print(f"\n=== Testing {model_key.upper()} Model ===")
            self.results[model_key] = {}
            
            for category, items in test_data.items():
                print(f"\nTesting {category}...")
                self.results[model_key][category] = []
                
                # Test a subset of items
                test_items = items[:max_tests_per_category]
                
                for item in test_items:
                    target_id = item['target_id']
                    target_type = item['target_type']
                    
                    # Test each variation
                    for variation in item['variations'][:3]:  # Test first 3 variations
                        result = self.evaluate_query(model_key, variation, target_id, target_type)
                        self.results[model_key][category].append(result)
                        
                        # Print progress
                        status = "PASS" if result['found_target'] else "FAIL"
                        print(f"  {status} '{variation}' -> {result['target_rank'] or 'N/A'} (Score: {result['target_score'] or 'N/A'})")
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive statistics from test results."""
        stats = {}
        
        for model_key, model_results in self.results.items():
            stats[model_key] = {}
            
            for category, results in model_results.items():
                if not results:
                    continue
                
                # Calculate metrics
                total_tests = len(results)
                found_target = sum(1 for r in results if r['found_target'])
                found_correct_type = sum(1 for r in results if r['found_correct_type'])
                avg_search_time = sum(r['search_time'] for r in results) / total_tests
                avg_target_score = sum(r['target_score'] for r in results if r['target_score']) / max(1, found_target)
                avg_top_score = sum(r['top_score'] for r in results) / total_tests
                
                # Rank analysis
                ranks = [r['target_rank'] for r in results if r['target_rank']]
                avg_rank = sum(ranks) / len(ranks) if ranks else None
                rank_1 = sum(1 for r in results if r['target_rank'] == 1)
                rank_3 = sum(1 for r in results if r['target_rank'] and r['target_rank'] <= 3)
                
                stats[model_key][category] = {
                    'total_tests': total_tests,
                    'found_target': found_target,
                    'found_correct_type': found_correct_type,
                    'accuracy': found_target / total_tests,
                    'correct_type_rate': found_correct_type / total_tests,
                    'avg_search_time': avg_search_time,
                    'avg_target_score': avg_target_score,
                    'avg_top_score': avg_top_score,
                    'avg_rank': avg_rank,
                    'rank_1_count': rank_1,
                    'rank_3_count': rank_3,
                    'rank_1_rate': rank_1 / total_tests,
                    'rank_3_rate': rank_3 / total_tests
                }
        
        return stats
    
    def print_summary(self, stats: Dict[str, Any]):
        """Print a comprehensive summary of results."""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON RESULTS")
        print("="*80)
        
        for model_key, model_stats in stats.items():
            print(f"\n{model_key.upper()} MODEL PERFORMANCE")
            print("-" * 50)
            
            overall_accuracy = 0
            overall_correct_type = 0
            total_tests = 0
            
            for category, category_stats in model_stats.items():
                print(f"\n{category.upper()}:")
                print(f"  Accuracy: {category_stats['accuracy']:.1%} ({category_stats['found_target']}/{category_stats['total_tests']})")
                print(f"  Correct Type Rate: {category_stats['correct_type_rate']:.1%}")
                print(f"  Avg Search Time: {category_stats['avg_search_time']:.3f}s")
                print(f"  Avg Target Score: {category_stats['avg_target_score']:.3f}")
                print(f"  Rank 1 Rate: {category_stats['rank_1_rate']:.1%}")
                print(f"  Rank 3 Rate: {category_stats['rank_3_rate']:.1%}")
                
                overall_accuracy += category_stats['found_target']
                overall_correct_type += category_stats['found_correct_type']
                total_tests += category_stats['total_tests']
            
            if total_tests > 0:
                print(f"\nOVERALL {model_key.upper()}:")
                print(f"  Overall Accuracy: {overall_accuracy/total_tests:.1%}")
                print(f"  Overall Correct Type Rate: {overall_correct_type/total_tests:.1%}")
        
        # Direct comparison
        print(f"\nHEAD-TO-HEAD COMPARISON")
        print("-" * 50)
        
        if 'mini' in stats and 'gemma' in stats:
            mini_overall = sum(stats['mini'][cat]['found_target'] for cat in stats['mini']) / sum(stats['mini'][cat]['total_tests'] for cat in stats['mini'])
            gemma_overall = sum(stats['gemma'][cat]['found_target'] for cat in stats['gemma']) / sum(stats['gemma'][cat]['total_tests'] for cat in stats['gemma'])
            
            print(f"Overall Accuracy:")
            print(f"  all-MiniLM-L6-v2: {mini_overall:.1%}")
            print(f"  EmbeddingGemma:   {gemma_overall:.1%}")
            print(f"  Winner: {'all-MiniLM-L6-v2' if mini_overall > gemma_overall else 'EmbeddingGemma'}")
    
    def save_results(self, stats: Dict[str, Any], output_path: str):
        """Save results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'statistics': stats,
                'raw_results': self.results
            }, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Run systematic embedding model evaluation')
    parser.add_argument('--test-phrases', type=str, default='test_phrases.json', help='Path to test phrases file')
    parser.add_argument('--max-tests', type=int, default=10, help='Max tests per category')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output file path')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator()
    evaluator.load_models()
    evaluator.run_systematic_tests(args.test_phrases, args.max_tests)
    
    stats = evaluator.generate_statistics()
    evaluator.print_summary(stats)
    evaluator.save_results(stats, args.output)

if __name__ == "__main__":
    main()
