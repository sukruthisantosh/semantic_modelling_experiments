#!/usr/bin/env python3
"""
Enhanced semantic search evaluation script with model presets.
"""

import os
import time
import argparse
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model and index presets for easy switching
MODEL_PRESETS = {
    'mini': {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'index_name': 'all-mini-lm-data',
        'description': 'Current model (384 dim)'
    },
    'gemma': {
        'model_name': 'google/embeddinggemma-300m', 
        'index_name': 'embedding-gemma-data',
        'description': 'New EmbeddingGemma model (768 dim)'
    }
}

def list_presets():
    """List available model presets."""
    print("Available model presets:")
    print("-" * 50)
    for key, preset in MODEL_PRESETS.items():
        print(f"{key:8} | {preset['model_name']:40} | {preset['index_name']:20} | {preset['description']}")
    print()

def main():
    parser = argparse.ArgumentParser(description='Enhanced semantic search evaluation with presets')
    parser.add_argument('--model', type=str, help='Model name or preset (mini/gemma/openai)')
    parser.add_argument('--index', type=str, help='Pinecone index name (auto-detected if using preset)')
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    parser.add_argument('--list-presets', action='store_true', help='List available model presets')
    parser.add_argument('--compare', action='store_true', help='Compare results across all models')
    
    args = parser.parse_args()
    
    # Handle list presets
    if args.list_presets:
        list_presets()
        return
    
    # Validate query is provided for search operations
    if not args.query:
        print("Error: --query is required for search operations.")
        print("Use --list-presets to see available model presets.")
        return
    
    # Handle model presets (skip if comparing all models)
    if not args.compare:
        if args.model in MODEL_PRESETS:
            preset = MODEL_PRESETS[args.model]
            model_name = preset['model_name']
            index_name = preset['index_name'] if not args.index else args.index
            print(f"Using preset '{args.model}': {preset['description']}")
        else:
            if not args.model:
                print("Error: --model is required. Use --list-presets to see available options.")
                return
            model_name = args.model
            index_name = args.index
            if not index_name:
                print("Error: --index is required when using custom model name.")
                return
    
    # Handle comparison mode
    if args.compare:
        compare_models(args.query, args.top_k)
        return
    
    # Single model search
    search_single_model(model_name, index_name, args.query, args.top_k)

def search_single_model(model_name, index_name, query, top_k):
    """Search using a single model."""
    print(f"Testing: {query}")
    print(f"Model: {model_name}")
    print(f"Index: {index_name}")
    print("=" * 50)
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    
    # Connect to Pinecone
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(index_name)
    print("Connected!")
    
    # Search
    print(f"Searching for: '{query}'")
    start_time = time.time()
    
    # Generate embedding
    query_embedding = model.encode(query, convert_to_tensor=False).tolist()
    
    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    search_time = time.time() - start_time
    
    # Show results
    print(f"\nFound {len(results.matches)} results in {search_time:.3f}s")
    print("-" * 50)
    
    for i, match in enumerate(results.matches, 1):
        print(f"{i}. {match.id} (Score: {match.score:.4f})")
        if match.metadata and 'name' in match.metadata:
            print(f"   Name: {match.metadata['name']}")
        if match.metadata and 'type' in match.metadata:
            print(f"   Type: {match.metadata['type']}")
        print()

def compare_models(query, top_k):
    """Compare results across all available models."""
    print(f"Comparing models for query: '{query}'")
    print("=" * 80)
    
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    for preset_key, preset in MODEL_PRESETS.items():
        print(f"\n{preset_key.upper()} - {preset['description']}")
        print("-" * 60)
        
        try:
            # Load model
            start_time = time.time()
            model = SentenceTransformer(preset['model_name'])
            load_time = time.time() - start_time
            
            # Connect to index
            index = pc.Index(preset['index_name'])
            
            # Search
            start_time = time.time()
            query_embedding = model.encode(query, convert_to_tensor=False).tolist()
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            search_time = time.time() - start_time
            
            # Show top 3 results
            print(f"Load: {load_time:.2f}s | Search: {search_time:.3f}s")
            for i, match in enumerate(results.matches[:3], 1):
                name = match.metadata.get('name', 'N/A') if match.metadata else 'N/A'
                print(f"  {i}. {match.id} (Score: {match.score:.4f}) - {name}")
            
            if len(results.matches) > 3:
                print(f"  ... and {len(results.matches) - 3} more results")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()