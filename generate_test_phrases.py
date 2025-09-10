#!/usr/bin/env python3
"""
Generate test phrases and variations from BI model data for semantic search evaluation.
"""

import json
import random
from typing import List, Dict, Any, Set
import argparse

class TestPhraseGenerator:
    def __init__(self, bi_model_path: str, dim_values_path: str):
        """Initialize with data files."""
        self.bi_model_path = bi_model_path
        self.dim_values_path = dim_values_path
        self.bi_data = None
        self.dim_data = None
        
    def load_data(self):
        """Load the JSON data files."""
        with open(self.bi_model_path, 'r', encoding='utf-8') as f:
            self.bi_data = json.load(f)
        
        with open(self.dim_values_path, 'r', encoding='utf-8') as f:
            self.dim_data = json.load(f)
    
    def extract_metrics(self) -> List[Dict[str, Any]]:
        """Extract all metrics with their variations."""
        metrics = []
        for metric in self.bi_data.get('metrics', []):
            metric_info = {
                'name': metric['name'],
                'descriptions': metric.get('descriptions', []),
                'activity_types': metric.get('activity_types', []),
                'type': 'metric'
            }
            metrics.append(metric_info)
        return metrics
    
    def extract_formulas(self) -> List[Dict[str, Any]]:
        """Extract all formulas with their variations."""
        formulas = []
        for formula in self.bi_data.get('formulas', []):
            formula_info = {
                'name': formula['name'],
                'descriptions': formula.get('descriptions', []),
                'metrics': formula.get('metrics', []),
                'dax_names': formula.get('dax_names', []),
                'type': 'formula'
            }
            formulas.append(formula_info)
        return formulas
    
    def extract_dimensions(self) -> List[Dict[str, Any]]:
        """Extract key dimensions."""
        dimensions = []
        for dim in self.bi_data.get('dimensions', []):
            dim_info = {
                'name': dim['name'],
                'descriptions': dim.get('descriptions', []),
                'activity_types': dim.get('activity_types', []),
                'type': 'dimension'
            }
            dimensions.append(dim_info)
        return dimensions
    
    def extract_dimension_values(self) -> List[Dict[str, Any]]:
        """Extract interesting dimension values."""
        dim_values = []
        for dim_name, values in self.dim_data.get('dim_values', {}).items():
            if isinstance(values, list) and values:
                # Take a sample of values for each dimension
                sample_values = random.sample(values, min(5, len(values)))
                for value in sample_values:
                    if value and len(str(value).strip()) > 0:
                        dim_values.append({
                            'dimension': dim_name,
                            'value': str(value).strip(),
                            'type': 'dimension_value'
                        })
        return dim_values
    
    def generate_metric_variations(self, metric: Dict[str, Any]) -> List[str]:
        """Generate natural language variations for a metric."""
        name = metric['name']
        descriptions = metric.get('descriptions', [])
        variations = []
        
        # Base variations
        variations.extend([
            name,
            name.lower(),
            name.replace(' ', ''),
            name.replace(' ', '_'),
        ])
        
        # Description-based variations
        for desc in descriptions:
            variations.extend([
                desc,
                desc.lower(),
                f"what is {desc.lower()}",
                f"show me {desc.lower()}",
                f"find {desc.lower()}",
            ])
        
        # Business context variations
        business_terms = ['metric', 'kpi', 'measure', 'statistic', 'number', 'count', 'rate', 'percentage']
        for term in business_terms:
            variations.extend([
                f"{term} {name.lower()}",
                f"{name.lower()} {term}",
                f"{term} for {name.lower()}",
            ])
        
        # Question variations
        variations.extend([
            f"how many {name.lower()}",
            f"what is the {name.lower()}",
            f"total {name.lower()}",
            f"sum of {name.lower()}",
            f"count of {name.lower()}",
        ])
        
        return list(set(variations))  # Remove duplicates
    
    def generate_formula_variations(self, formula: Dict[str, Any]) -> List[str]:
        """Generate natural language variations for a formula."""
        name = formula['name']
        descriptions = formula.get('descriptions', [])
        variations = []
        
        # Base variations
        variations.extend([
            name,
            name.lower(),
            name.replace(' ', ''),
            name.replace(' ', '_'),
        ])
        
        # Description-based variations
        for desc in descriptions:
            variations.extend([
                desc,
                desc.lower(),
                f"calculate {desc.lower()}",
                f"formula for {desc.lower()}",
                f"how to calculate {desc.lower()}",
            ])
        
        # Formula context variations
        formula_terms = ['formula', 'calculation', 'computation', 'rate', 'ratio', 'percentage']
        for term in formula_terms:
            variations.extend([
                f"{term} {name.lower()}",
                f"{name.lower()} {term}",
                f"{term} for {name.lower()}",
            ])
        
        return list(set(variations))
    
    def generate_dimension_variations(self, dimension: Dict[str, Any]) -> List[str]:
        """Generate natural language variations for a dimension."""
        name = dimension['name']
        descriptions = dimension.get('descriptions', [])
        variations = []
        
        # Base variations
        variations.extend([
            name,
            name.lower(),
            name.replace(' ', ''),
            name.replace(' ', '_'),
        ])
        
        # Description-based variations
        for desc in descriptions:
            variations.extend([
                desc,
                desc.lower(),
                f"filter by {desc.lower()}",
                f"group by {desc.lower()}",
                f"breakdown by {desc.lower()}",
            ])
        
        # Dimension context variations
        dim_terms = ['dimension', 'attribute', 'field', 'category', 'segment']
        for term in dim_terms:
            variations.extend([
                f"{term} {name.lower()}",
                f"{name.lower()} {term}",
                f"{term} for {name.lower()}",
            ])
        
        return list(set(variations))
    
    def generate_dimension_value_variations(self, dim_value: Dict[str, Any]) -> List[str]:
        """Generate natural language variations for dimension values."""
        dimension = dim_value['dimension']
        value = dim_value['value']
        variations = []
        
        # Base variations
        variations.extend([
            value,
            value.lower(),
            f"{dimension} {value}",
            f"{value} in {dimension}",
        ])
        
        # Filter variations
        variations.extend([
            f"where {dimension} is {value}",
            f"filter {dimension} = {value}",
            f"{dimension} equals {value}",
            f"show {dimension} {value}",
        ])
        
        return list(set(variations))
    
    def generate_all_test_phrases(self, max_variations_per_item: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Generate all test phrases with variations."""
        self.load_data()
        
        test_phrases = {
            'metrics': [],
            'formulas': [],
            'dimensions': [],
            'dimension_values': []
        }
        
        # Generate metric variations
        print("Generating metric variations...")
        for metric in self.extract_metrics():
            variations = self.generate_metric_variations(metric)
            # Limit variations to avoid too many
            selected_variations = random.sample(variations, min(max_variations_per_item, len(variations)))
            
            test_phrases['metrics'].append({
                'target': metric['name'],
                'target_type': 'metric',
                'target_id': f"metric_{metric['name'].replace(' ', '_')}",
                'variations': selected_variations,
                'metadata': metric
            })
        
        # Generate formula variations
        print("Generating formula variations...")
        for formula in self.extract_formulas():
            variations = self.generate_formula_variations(formula)
            selected_variations = random.sample(variations, min(max_variations_per_item, len(variations)))
            
            test_phrases['formulas'].append({
                'target': formula['name'],
                'target_type': 'formula',
                'target_id': f"formula_{formula['name'].replace(' ', '_')}",
                'variations': selected_variations,
                'metadata': formula
            })
        
        # Generate dimension variations
        print("Generating dimension variations...")
        for dimension in self.extract_dimensions()[:10]:  # Limit to top 10 dimensions
            variations = self.generate_dimension_variations(dimension)
            selected_variations = random.sample(variations, min(max_variations_per_item, len(variations)))
            
            test_phrases['dimensions'].append({
                'target': dimension['name'],
                'target_type': 'dimension',
                'target_id': f"dimension_{dimension['name'].replace(' ', '_')}",
                'variations': selected_variations,
                'metadata': dimension
            })
        
        # Generate dimension value variations
        print("Generating dimension value variations...")
        for dim_value in self.extract_dimension_values()[:20]:  # Limit to 20 dimension values
            variations = self.generate_dimension_value_variations(dim_value)
            selected_variations = random.sample(variations, min(max_variations_per_item, len(variations)))
            
            test_phrases['dimension_values'].append({
                'target': f"{dim_value['dimension']}: {dim_value['value']}",
                'target_type': 'dimension_value',
                'target_id': f"dim_value_{dim_value['dimension']}_{dim_value['value']}".replace(' ', '_'),
                'variations': selected_variations,
                'metadata': dim_value
            })
        
        return test_phrases
    
    def save_test_phrases(self, test_phrases: Dict[str, List[Dict[str, Any]]], output_path: str):
        """Save test phrases to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_phrases, f, indent=2, ensure_ascii=False)
        
        # Print summary
        total_phrases = sum(len(category) for category in test_phrases.values())
        total_variations = sum(
            len(item['variations']) 
            for category in test_phrases.values() 
            for item in category
        )
        
        print(f"\n📊 Test Phrases Summary:")
        print(f"Total categories: {len(test_phrases)}")
        print(f"Total test items: {total_phrases}")
        print(f"Total variations: {total_variations}")
        print(f"Saved to: {output_path}")
        
        for category, items in test_phrases.items():
            print(f"  {category}: {len(items)} items")

def main():
    parser = argparse.ArgumentParser(description='Generate test phrases from BI model data')
    parser.add_argument('--bi-model', type=str, default='bi_model.json', help='Path to bi_model.json')
    parser.add_argument('--dim-values', type=str, default='dim_values.json', help='Path to dim_values.json')
    parser.add_argument('--output', type=str, default='test_phrases.json', help='Output file path')
    parser.add_argument('--max-variations', type=int, default=10, help='Max variations per item')
    
    args = parser.parse_args()
    
    generator = TestPhraseGenerator(args.bi_model, args.dim_values)
    test_phrases = generator.generate_all_test_phrases(args.max_variations)
    generator.save_test_phrases(test_phrases, args.output)

if __name__ == "__main__":
    main()
