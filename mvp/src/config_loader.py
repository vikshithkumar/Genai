from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import json

from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config"


class Category(BaseModel):
    id: str
    name: str
    description: str
    aliases: List[str]


class Taxonomy(BaseModel):
    categories: List[Category]
    unknown_category_id: str

    def id_to_name(self) -> Dict[str, str]:
        return {c.id: c.name for c in self.categories}

    def alias_to_id(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for c in self.categories:
            for alias in c.aliases:
                mapping[alias.lower()] = c.id
        return mapping


class Settings(BaseModel):
    low_confidence_threshold: float = 0.6
    random_seed: int = 42
    k_neighbors: int = 5


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_taxonomy() -> Taxonomy:
    path = CONFIG_DIR / "taxonomy.json"
    data = load_json(path)
    return Taxonomy.model_validate(data)


def load_settings() -> Settings:
    path = CONFIG_DIR / "settings.json"
    if not path.exists():
        return Settings()
    data = load_json(path)
    return Settings.model_validate(data)


def project_paths() -> Dict[str, Path]:
    """Convenience accessor for important directories."""

    return {
        "base": BASE_DIR,
        "data": BASE_DIR / "data",
        "models": BASE_DIR / "models",
        "config": CONFIG_DIR,
        "ui": BASE_DIR / "ui",
        "evaluation": BASE_DIR / "evaluation",
    }
