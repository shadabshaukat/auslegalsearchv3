"""
Embedding interface for auslegalsearchv3.
- Loads embedding model (default: nomic-ai/nomic-embed-text-v1.5, or user-specified model)
- For legal use, highly recommended: nomic-ai/nomic-embed-text-v1.5 (768d vectors, 2048 tokens context)
- Can select model via environment variable AUSLEGALSEARCH_EMBED_MODEL for deployment flexibility.
- Provides embed(texts) method for batch embeddings
- Extensible for custom, HuggingFace, or local Ollama/Llama4-compatible models
"""

from typing import List
import numpy as np
import os

try:
    from sentence_transformers import SentenceTransformer
    DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
except ImportError:
    SentenceTransformer = None
    DEFAULT_MODEL = None

class Embedder:
    def __init__(self, model_name: str = None):
        """
        Initialize embedder. If no model_name is given, will use:
        - Environment variable AUSLEGALSEARCH_EMBED_MODEL, OR
        - Default 'nomic-ai/nomic-embed-text-v1.5'.
        Passes trust_remote_code=True if needed for models with custom code (e.g., nomic-ai/nomic-embed-text-v1.5)
        """
        if model_name is None:
            model_name = os.environ.get("AUSLEGALSEARCH_EMBED_MODEL") or DEFAULT_MODEL
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed.")
        # Detect if trust_remote_code required for custom HuggingFace repository (nomic/bge/other)
        trust_remote = True if ("nomic-ai" in model_name or "trust_remote_code" in os.environ.get("AUSLEGALSEARCH_EMBEDDER_FLAGS", "")) else False
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, trust_remote_code=trust_remote)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts; returns ndarray [batch, dim].
        """
        return np.array(self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True))

    # Optionally add OpenAI embedding, Ollama/REST/Llama/Instructor models as needed in the future

"""
Usage:
# Default (Nomic, sets trust_remote_code)
embedder = Embedder()
# Use MiniLM or any other, disables trust_remote, unless environment var override
embedder = Embedder("all-MiniLM-L6-v2")
embedder = Embedder("BAAI/bge-base-en-v1.5")
vecs = embedder.embed(["example legal text 1", "example legal text 2"])
print(vecs.shape)  # (batch_size, 768)
"""
