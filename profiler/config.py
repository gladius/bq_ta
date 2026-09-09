"""Configuration: column mapping from YAML, secrets from .env. No other module reads either."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MAPPING = os.path.join(HERE, "mapping.yaml")
DEFAULT_ENV = os.path.join(HERE, ".env")

# Volume work goes to the cheaper model; judgment work to the stronger one.
MODEL_BULK = "gpt-4.1-mini"
MODEL_JUDGE = "gpt-4.1"


@dataclass
class Mapping:
    """Where the fields we need live in this particular export."""

    request_id: str = "request_id"
    timestamp: str = "startTime"
    model: str = "model"
    request: str = "request_payload"
    response: str = "response_payload"
    call_type: Optional[str] = "call_type"
    app_id_paths: List[str] = field(default_factory=list)
    json_is_string: bool = True
    max_rows_per_app: int = 1000
    chat_call_types: List[str] = field(default_factory=list)

    @property
    def source_columns(self) -> List[str]:
        """Columns that must exist in the CSV, including any that carry an app id path."""
        needed = {self.request_id, self.timestamp, self.model, self.request, self.response}
        if self.call_type:
            needed.add(self.call_type)
        for path in self.app_id_paths:
            needed.add(path.split(".")[0])
        return sorted(c for c in needed if c)


def load_mapping(path: Optional[str] = None) -> Mapping:
    with open(path or DEFAULT_MAPPING, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    columns = raw.get("columns") or {}
    options = raw.get("options") or {}
    return Mapping(
        request_id=columns.get("request_id", "request_id"),
        timestamp=columns.get("timestamp", "startTime"),
        model=columns.get("model", "model"),
        request=columns.get("request", "request_payload"),
        response=columns.get("response", "response_payload"),
        call_type=columns.get("call_type"),
        app_id_paths=list(raw.get("app_id_paths") or []),
        json_is_string=bool(options.get("json_is_string", True)),
        max_rows_per_app=int(options.get("max_rows_per_app", 1000)),
        chat_call_types=list(options.get("chat_call_types") or []),
    )


def load_api_key(env_path: Optional[str] = None) -> Optional[str]:
    """Read LLM_KEY from .env or the environment. Never logged, never written to output."""
    for name in ("LLM_KEY", "OPENAI_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    path = env_path or DEFAULT_ENV
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() in ("LLM_KEY", "OPENAI_API_KEY"):
                return value.strip().strip('"').strip("'")
    return None


@dataclass
class Settings:
    """Everything a run needs, resolved once and passed down explicitly."""

    mapping: Mapping
    api_key: Optional[str]
    use_llm: bool = True
    model_bulk: str = MODEL_BULK
    model_judge: str = MODEL_JUDGE
    out_dir: str = os.path.join(HERE, "out")
    audit_dir: str = os.path.join(HERE, "audit")
    cache_dir: str = os.path.join(HERE, ".cache_llm")
    # Memory across runs: which callsites we have already profiled. Deleting this file
    # loses history only - it never changes what a run measures.
    registry_path: str = os.path.join(HERE, "registry.sqlite")
    # grouping parameters, all measured in the spike (sandbox/spike/DECISIONS.md,
    # local only - the findings are summarised in STRESS_FINDINGS.md)
    tau: float = 0.6                 # D26: 0.6 is the usable ceiling
    min_doc_freq: int = 2            # D27: strips per-call noise only
    min_node_fraction: float = 0.02  # relative floor, for small exports
    min_node_floor: int = 3
    # 8 workers saturated a 30k tokens/min limit on the judge model; 4 keeps the
    # pipeline fast without spending the run on backoff
    llm_workers: int = 4
    ambiguous_low: float = 0.45      # LLM adjudication band
    ambiguous_high: float = 0.75

    @property
    def llm_enabled(self) -> bool:
        return bool(self.use_llm and self.api_key)


def load_settings(mapping_path: Optional[str] = None, use_llm: bool = True,
                  **overrides: Any) -> Settings:
    settings = Settings(mapping=load_mapping(mapping_path), api_key=load_api_key(),
                        use_llm=use_llm)
    for key, value in overrides.items():
        if value is not None and hasattr(settings, key):
            setattr(settings, key, value)
    for directory in (settings.out_dir, settings.audit_dir, settings.cache_dir):
        os.makedirs(directory, exist_ok=True)
    return settings
