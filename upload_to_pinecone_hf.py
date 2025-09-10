#!/usr/bin/env python3
"""
Script to upload JSON files to Pinecone using Hugging Face embedding models.
This script processes bi_model.json and dim_values.json files and creates
vector embeddings using Hugging Face models for semantic search and retrieval.

To use different models, change the MODEL_NAME variable at the top.
"""

import json
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - Change this to use different models
# =============================================================================
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Current model
MODEL_NAME = "google/embeddinggemma-300m"  # New model (when available)

# Index configuration
INDEX_NAME = "embedding-gemma-data"  # Change this for different models
DIMENSION = 768  # EmbeddingGemma dimension (change for other models)
METRIC = "cosine"

class HuggingFacePineconeUploader:
    def __init__(self, model_name: str, index_name: str, dimension: int):
        """Initialize the uploader with Hugging Face model."""
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Configuration
        self.model_name = model_name
        self.index_name = index_name
        self.dimension = dimension
        self.metric = METRIC
        
        # Initialize model
        self.model = None
        
    def load_model(self):
        """Load the Hugging Face embedding model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts using Hugging Face model."""
        if not self.model:
            self.load_model()
        
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            logger.info(f"Successfully created {len(embeddings)} embeddings")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def delete_index(self):
        """Delete the Pinecone index if it exists."""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name in existing_indexes:
                logger.info(f"Deleting existing index: {self.index_name}")
                self.pc.delete_index(self.index_name)
                logger.info(f"Index {self.index_name} deleted successfully")
            else:
                logger.info(f"Index {self.index_name} does not exist")
                
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            raise
    
    def create_index(self):
        """Create a Pinecone index if it doesn't exist."""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                logger.info(f"Index {self.index_name} created successfully")
            else:
                logger.info(f"Index {self.index_name} already exists")
                
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def process_bi_model_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process bi_model.json data and create vector records."""
        vectors = []
        
        # Process facts
        for fact in data.get('facts', []):
            text_content = f"Fact: {fact['name']}. Description: {', '.join(fact['descriptions'])}. Activity Type: {fact['activity_type']}"
            vectors.append({
                'id': f"fact_{fact['name'].replace(' ', '_')}",
                'text': text_content,
                'type': 'fact',
                'metadata': {
                    'name': fact['name'],
                    'activity_type': fact['activity_type'],
                    'descriptions': fact['descriptions']
                }
            })
        
        # Process dimensions
        for dimension in data.get('dimensions', []):
            text_content = f"Dimension: {dimension['name']}. Description: {', '.join(dimension['descriptions'])}. Activity Types: {', '.join(dimension['activity_types'])}"
            vectors.append({
                'id': f"dimension_{dimension['name'].replace(' ', '_')}",
                'text': text_content,
                'type': 'dimension',
                'metadata': {
                    'name': dimension['name'],
                    'activity_types': dimension['activity_types'],
                    'descriptions': dimension['descriptions']
                }
            })
        
        # Process metrics
        for metric in data.get('metrics', []):
            # Create multiple text variations for better semantic matching
            text_variations = [
                metric['name'],  # Just the name
                f"{metric['name']} - {', '.join(metric['descriptions'])}",  # Name with description
                f"Metric: {metric['name']}. Description: {', '.join(metric['descriptions'])}. Activity Types: {', '.join(metric['activity_types'])}"  # Full format
            ]
            
            for i, text_content in enumerate(text_variations):
                vectors.append({
                    'id': f"metric_{metric['name'].replace(' ', '_')}_{i}" if i > 0 else f"metric_{metric['name'].replace(' ', '_')}",
                    'text': text_content,
                    'type': 'metric',
                    'metadata': {
                        'name': metric['name'],
                        'activity_types': metric['activity_types'],
                        'descriptions': metric['descriptions'],
                        'text_variation': i
                    }
                })
        
        # Process formulas
        for formula in data.get('formulas', []):
            # Create multiple text variations for better semantic matching
            text_variations = [
                formula['name'],  # Just the name
                f"{formula['name']} - {', '.join(formula['descriptions'])}",  # Name with description
                f"Formula: {formula['name']}. Description: {', '.join(formula['descriptions'])}. Metrics: {', '.join(formula.get('metrics', []))}"  # Full format
            ]
            
            if 'dax_expression' in formula:
                text_variations[2] += f". DAX Expression: {formula['dax_expression']}"
            
            for i, text_content in enumerate(text_variations):
                vectors.append({
                    'id': f"formula_{formula['name'].replace(' ', '_')}_{i}" if i > 0 else f"formula_{formula['name'].replace(' ', '_')}",
                    'text': text_content,
                    'type': 'formula',
                    'metadata': {
                        'name': formula['name'],
                        'metrics': formula.get('metrics', []),
                        'descriptions': formula['descriptions'],
                        'dax_expression': formula.get('dax_expression', ''),
                        'dax_names': formula.get('dax_names', []),
                        'text_variation': i
                    }
                })
        
        # Process analysis areas
        for area in data.get('analysis_areas', []):
            text_content = f"Analysis Area: {area['name']}. Description: {', '.join(area['descriptions'])}. Facts: {', '.join(area.get('facts', []))}"
            vectors.append({
                'id': f"analysis_area_{area['name'].replace(' ', '_')}",
                'text': text_content,
                'type': 'analysis_area',
                'metadata': {
                    'name': area['name'],
                    'facts': area.get('facts', []),
                    'descriptions': area['descriptions']
                }
            })
        
        return vectors
    
    def process_dim_values_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process dim_values.json data and create vector records."""
        vectors = []
        
        for dimension_name, values in data.get('dim_values', {}).items():
            if isinstance(values, list) and values:
                # Create a vector for the dimension itself with limited metadata
                text_content = f"Dimension Values for {dimension_name}: {', '.join(str(v) for v in values[:10])}"  # Limit to first 10 values
                if len(values) > 10:
                    text_content += f" and {len(values) - 10} more values"
                
                vectors.append({
                    'id': f"dim_values_{dimension_name.replace(' ', '_')}",
                    'text': text_content,
                    'type': 'dimension_values',
                    'metadata': {
                        'dimension_name': dimension_name,
                        'value_count': len(values),
                        'sample_values': values[:5]  # Only store first 5 values in metadata
                    }
                })
                
                # Create individual vectors for each value (for more granular search)
                for i, value in enumerate(values):  # No limit - include all values
                    if str(value).strip():  # Skip empty values
                        text_content = f"Dimension: {dimension_name}. Value: {value}"
                        vectors.append({
                            'id': f"dim_value_{dimension_name.replace(' ', '_')}_{i}",
                            'text': text_content,
                            'type': 'dimension_value',
                            'metadata': {
                                'dimension_name': dimension_name,
                                'value': str(value)[:100],  # Limit value length
                                'value_index': i
                            }
                        })
        
        return vectors
    
    def check_metadata_size(self, metadata: Dict[str, Any]) -> bool:
        """Check if metadata size is within Pinecone limits."""
        metadata_json = json.dumps(metadata)
        return len(metadata_json.encode('utf-8')) <= 40000  # Leave some buffer under 40KB limit
    
    def upload_vectors(self, vectors: List[Dict[str, Any]]):
        """Upload vectors to Pinecone index."""
        try:
            index = self.pc.Index(self.index_name)
            
            # Create embeddings for all texts
            texts = [vector['text'] for vector in vectors]
            logger.info(f"Creating embeddings for {len(texts)} texts...")
            embeddings = self.create_embeddings(texts)
            
            # Check embedding dimensions
            if embeddings:
                actual_dim = len(embeddings[0])
                logger.info(f"Actual embedding dimension: {actual_dim}, Expected: {self.dimension}")
                if actual_dim != self.dimension:
                    logger.warning(f"Dimension mismatch! Expected {self.dimension}, got {actual_dim}")
            
            # Prepare vectors for upload, filtering out oversized metadata
            pinecone_vectors = []
            skipped_count = 0
            
            for i, vector in enumerate(vectors):
                if self.check_metadata_size(vector['metadata']):
                    pinecone_vectors.append({
                        'id': vector['id'],
                        'values': embeddings[i],
                        'metadata': vector['metadata']
                    })
                else:
                    logger.warning(f"Skipping vector {vector['id']} due to oversized metadata")
                    skipped_count += 1
            
            if skipped_count > 0:
                logger.info(f"Skipped {skipped_count} vectors due to metadata size limits")
            
            logger.info(f"Prepared {len(pinecone_vectors)} vectors for upload")
            
            # Upload in batches
            batch_size = 100
            for i in range(0, len(pinecone_vectors), batch_size):
                batch = pinecone_vectors[i:i + batch_size]
                logger.info(f"Uploading batch {i//batch_size + 1}/{(len(pinecone_vectors) + batch_size - 1)//batch_size}")
                try:
                    index.upsert(vectors=batch)
                    logger.info(f"Successfully uploaded batch {i//batch_size + 1}")
                except Exception as batch_error:
                    logger.error(f"Error uploading batch {i//batch_size + 1}: {batch_error}")
                    raise
            
            logger.info(f"Successfully uploaded {len(pinecone_vectors)} vectors to Pinecone")
            
        except Exception as e:
            logger.error(f"Error uploading vectors: {e}")
            raise
    
    def upload_files(self, bi_model_path: str, dim_values_path: str):
        """Main method to upload both JSON files to Pinecone."""
        try:
            # Delete and recreate index to ensure correct dimensions
            self.delete_index()
            self.create_index()
            
            # Load and process bi_model.json
            logger.info("Processing bi_model.json...")
            with open(bi_model_path, 'r', encoding='utf-8') as f:
                bi_model_data = json.load(f)
            
            bi_vectors = self.process_bi_model_data(bi_model_data)
            logger.info(f"Created {len(bi_vectors)} vectors from bi_model.json")
            
            # Load and process dim_values.json
            logger.info("Processing dim_values.json...")
            with open(dim_values_path, 'r', encoding='utf-8') as f:
                dim_values_data = json.load(f)
            
            dim_vectors = self.process_dim_values_data(dim_values_data)
            logger.info(f"Created {len(dim_vectors)} vectors from dim_values.json")
            
            # Combine all vectors
            all_vectors = bi_vectors + dim_vectors
            logger.info(f"Total vectors to upload: {len(all_vectors)}")
            
            # Upload to Pinecone
            self.upload_vectors(all_vectors)
            
            logger.info("Upload completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in upload process: {e}")
            raise

def main():
    """Main function to run the upload process."""
    try:
        # File paths
        bi_model_path = "/Users/suki/Documents/semantic_modelling_experiments/bi_model.json"
        dim_values_path = "/Users/suki/Documents/semantic_modelling_experiments/dim_values.json"
        
        # Check if files exist
        if not os.path.exists(bi_model_path):
            raise FileNotFoundError(f"File not found: {bi_model_path}")
        if not os.path.exists(dim_values_path):
            raise FileNotFoundError(f"File not found: {dim_values_path}")
        
        # Initialize uploader and upload files
        uploader = HuggingFacePineconeUploader(MODEL_NAME, INDEX_NAME, DIMENSION)
        uploader.upload_files(bi_model_path, dim_values_path)
        
    except Exception as e:
        logger.error(f"Failed to upload files: {e}")
        raise

if __name__ == "__main__":
    main()
