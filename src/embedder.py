from __future__ import annotations
import json
from typing import List, Protocol
import numpy as np

import openai
import requests
from .config import Settings

class Embedder(Protocol):
    dim: int
    def embed(self, texts: List[str], batch_size: int = 128) -> np.ndarray: ...

class LocalSentenceTransformerEmbedder:
    """
    Snowflake/Jina 모두 sentence-transformers로 로딩.
    일부 모델은 remote code 필요할 수 있어 trust_remote_code=True 사용.
    """
    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        # device: "cuda" or "cpu" (None이면 자동)
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
        )
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine 검색용
        )
        return np.asarray(vecs, dtype=np.float32)

# B안
class OpenAIEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.api_url = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/embeddings"

    def embed(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {Settings.OPENAI_API_KEY}"  # GMS API 키를 사용
        }
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            data = {
                "model": self.model_name,
                "input": batch
            }
            
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
            
            if response.status_code == 200:
                response_data = response.json()
                embeddings.extend([embedding['embedding'] for embedding in response_data['data']])
            else:
                print(f"Error: {response.status_code}, {response.text}")
        
        return np.array(embeddings)