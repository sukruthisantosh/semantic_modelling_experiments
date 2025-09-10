#!/usr/bin/env python3
"""
Simple semantic search evaluation script.
"""

import os
import time
import argparse
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='Simple semantic search evaluation')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--index', type=str, required=True, help='Pinecone index name')
    parser.add_argument('--query', type=str, required=True, help='Search query')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    
    args = parser.parse_args()
    
    print(f"Testing: {args.query}")
    print(f"Model: {args.model}")
    print(f"Index: {args.index}")
    print("=" * 50)
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    model = SentenceTransformer(args.model)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    
    # Connect to Pinecone
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(args.index)
    print("Connected!")
    
    # Search
    print(f"Searching for: '{args.query}'")
    start_time = time.time()
    
    # Generate embedding
    query_embedding = model.encode(args.query, convert_to_tensor=False).tolist()
    
    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=args.top_k,
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

if __name__ == "__main__":
    main()