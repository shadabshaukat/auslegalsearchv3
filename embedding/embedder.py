"""
Embedding interface for auslegalsearchv3.
- Loads embedding model (default: sentence-transformers/all-MiniLM-L6-v2 or user-specified)
- For legal use, set model_name to a legal domain embedding model, e.g. BGE-Legal or openai-ada-002.
- Can select model via environment variable AUSLEGALSEARCH_EMBED_MODEL for deployment flexibility.
- Provides embed(texts) method for batch embeddings
- Extensible for custom, HuggingFace, or local Ollama/Llama4-compatible models
"""

from typing import List
import numpy as np
import os

try:
    from sentence_transformers import SentenceTransformer
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
except ImportError:
    SentenceTransformer = None
    DEFAULT_MODEL = None

# For best legal performance, consider one of:
#   "BAAI/bge-base-en-v1.5"   (general, but very strong)
#   "maastrichtlawtech/bge-legal-en-v1.5"   (if available on HuggingFace: legal-specific)
#   "thenlper/gte-large"      (generic but sometimes preferred)
#   Any OpenAI LLM embedding: "text-embedding-ada-002" (requires OpenAI API)
#   Or further specialized/finetuned models as field evolves

class Embedder:
    def __init__(self, model_name: str = None):
        if model_name is None:
            # Check env first (for deployment override)
            model_name = os.environ.get("AUSLEGALSEARCH_EMBED_MODEL") or DEFAULT_MODEL
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed.")
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts; returns ndarray [batch, dim].
        """
        return np.array(self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True))

    # TODO: add OpenAI embedding support, or local Ollama/REST/Llama/Instructor methods

"""
Usage examples:
# Default (MiniLM)
embedder = Embedder()
# Use legal-specific model (set env or pass string)
embedder = Embedder("BAAI/bge-base-en-v1.5")  # General SOTA
embedder = Embedder("maastrichtlawtech/bge-legal-en-v1.5")  # In-domain legal, if available
vec = embedder.embed(["example legal text"])
"""
