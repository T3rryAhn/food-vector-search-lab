from __future__ import annotations
from typing import List, Protocol
import numpy as np

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

# B안은 다음 단계에서 실제 구현 예정
class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        raise NotImplementedError("B안(OpenAIEmbedder)은 다음 단계에서 추가합니다.")
