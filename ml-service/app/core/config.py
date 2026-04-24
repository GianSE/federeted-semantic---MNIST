import os
from pathlib import Path


def get_path(env_key: str, default: str) -> Path:
    value = os.getenv(env_key, default)
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


DATA_ROOT = get_path("DATA_ROOT", "/ml-data")
DATASETS_DIR = get_path("DATASETS_DIR", str(DATA_ROOT / "datasets"))
RUNS_DIR = get_path("RUNS_DIR", str(DATA_ROOT / "runs"))
LOGS_DIR = get_path("LOGS_DIR", str(DATA_ROOT / "logs"))
# Central experiment output root mounted by docker-compose.
RESULTADOS_ROOT = get_path("RESULTADOS_ROOT", "/resultados")
