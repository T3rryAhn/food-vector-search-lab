from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]

    data_path: Path = project_root / "data" / "food_data.csv"

    outputs_dir: Path = project_root / "outputs"
    indices_dir: Path = outputs_dir / "indices"
    reports_dir: Path = outputs_dir / "reports"

    # 모델 후보 2개
    model_snowflake: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
    model_jina: str = "jinaai/jina-embeddings-v3"  # 필요시 repo id만 바꾸면 됨

    # B안(OpenAI)에서 사용
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()

settings = Settings()
settings.outputs_dir.mkdir(parents=True, exist_ok=True)
settings.indices_dir.mkdir(parents=True, exist_ok=True)
settings.reports_dir.mkdir(parents=True, exist_ok=True)
