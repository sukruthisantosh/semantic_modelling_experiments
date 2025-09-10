#!/usr/bin/env python3
"""
Script to search your Pinecone database for semantic queries.
"""

import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PineconeSearcher:
    def __init__(self):
        """Initialize the Pinecone searcher."""
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index("semantic-modeling-data")
        
        # Initialize OpenAI
        openai.api_key = self.openai_api_key
    
    def create_embedding(self, text: str):
        """Create embedding for search query."""
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            raise
    
    def search(self, query: str, top_k: int = 10):
        """Search for similar vectors."""
        try:
            # Create embedding for the query
            query_embedding = self.create_embedding(query)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return results
        except Exception as e:
            print(f"Error searching: {e}")
            raise
    
    def print_results(self, results):
        """Print search results in a readable format."""
        print(f"\nFound {len(results.matches)} results:\n")
        
        for i, match in enumerate(results.matches, 1):
            print(f"{i}. ID: {match.id}")
            print(f"   Score: {match.score:.4f}")
            print(f"   Type: {match.metadata.get('type', 'Unknown')}")
            
            if 'name' in match.metadata:
                print(f"   Name: {match.metadata['name']}")
            
            if 'descriptions' in match.metadata:
                descriptions = match.metadata['descriptions']
                if isinstance(descriptions, list):
                    print(f"   Description: {', '.join(descriptions)}")
                else:
                    print(f"   Description: {descriptions}")
            
            if 'dimension_name' in match.metadata:
                print(f"   Dimension: {match.metadata['dimension_name']}")
            
            if 'value' in match.metadata:
                print(f"   Value: {match.metadata['value']}")
            
            print("-" * 50)

def main():
    """Interactive search interface."""
    try:
        searcher = PineconeSearcher()
        
        print("Pinecone Semantic Search")
        print("=" * 40)
        print("Type your search query and press Enter.")
        print("Type 'quit' to exit.\n")
        
        while True:
            query = input("Search: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print(f"\nSearching for: '{query}'...")
            results = searcher.search(query, top_k=5)
            searcher.print_results(results)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
