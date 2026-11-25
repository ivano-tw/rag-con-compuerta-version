"""Cliente de la API de Claude con salidas estructuradas y caché en disco.

La caché (data/processed/llm_cache.jsonl) evita repetir llamadas (y costos) al
re-ejecutar los experimentos: la clave es el hash de (modelo, sistema, usuario, esquema).
"""
from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

import anthropic
from pydantic import BaseModel
from tqdm import tqdm

from . import config

T = TypeVar("T", bound=BaseModel)
_CACHE_PATH = config.PROCESSED / "llm_cache.jsonl"
_lock = threading.Lock()
_cache: dict[str, dict] | None = None
_client: anthropic.Anthropic | None = None

# Modelos con fallback del lado del servidor ante rechazos (stop_reason == "refusal").
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2"}


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=5, timeout=120.0)
    return _client


def _load_cache() -> dict[str, dict]:
    global _cache
    if _cache is None:
        _cache = {}
        if _CACHE_PATH.exists():
            for line in _CACHE_PATH.read_text().splitlines():
                rec = json.loads(line)
                _cache[rec["key"]] = rec
    return _cache


def usage_summary() -> dict:
    tot: dict[str, dict[str, int]] = {}
    for rec in _load_cache().values():
        m = tot.setdefault(rec["model"], {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        m["calls"] += 1
        m["input_tokens"] += rec.get("input_tokens", 0)
        m["output_tokens"] += rec.get("output_tokens", 0)
    return tot


def structured(
    model: str,
    system: str,
    user: str,
    schema: type[T],
    effort: str = "low",
    max_tokens: int = 16000,
) -> T | None:
    """Una llamada con salida validada contra `schema`. Devuelve None si el modelo
    rechaza la solicitud."""
    key = hashlib.sha256(
        json.dumps([model, system, user, schema.__name__, json.dumps(schema.model_json_schema()), effort]).encode()
    ).hexdigest()
    cache = _load_cache()
    if key in cache:
        data = cache[key]["output"]
        return None if data is None else schema.model_validate(data)

    kwargs = {}
    if model in _FALLBACK_MODELS:
        kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    resp = _get_client().beta.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=schema,
        output_config={"effort": effort},
        **kwargs,
    )
    parsed = None if resp.stop_reason == "refusal" else resp.parsed_output
    rec = {
        "key": key,
        "model": model,
        "output": None if parsed is None else parsed.model_dump(),
        "stop_reason": resp.stop_reason,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    with _lock:
        cache[key] = rec
        with _CACHE_PATH.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return parsed


def pmap(fn: Callable, items: Iterable, desc: str = "") -> list:
    items = list(items)
    with ThreadPoolExecutor(max_workers=config.LLM_CONCURRENCY) as ex:
        return list(tqdm(ex.map(fn, items), total=len(items), desc=desc))
