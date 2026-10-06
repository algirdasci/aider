"""
Hostinger Router model metadata caching and lookup.

This module keeps a local cached copy of the Hostinger Router model list
(downloaded from the public ``https://router.hostinger.com/v1/models``) and
exposes a helper class that returns metadata for a given model in a format
compatible with litellm's ``get_model_info``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import requests

HOSTINGER_ROUTER_API_BASE = "https://router.hostinger.com/v1"
CREDITS_PER_USD = 100  # router pricing is in credits per token; 1 credit = $0.01


def _credits_to_usd(val: str | None) -> float | None:
    """Convert a credits-per-token price string to USD per token."""
    if val in (None, "", "0"):
        return 0.0 if val == "0" else None
    try:
        return float(val) / CREDITS_PER_USD
    except Exception:  # noqa: BLE001
        return None


class HostingerRouterModelManager:
    MODELS_URL = HOSTINGER_ROUTER_API_BASE + "/models"
    CACHE_TTL = 60 * 60 * 24  # 24 h

    def __init__(self) -> None:
        self.cache_dir = Path.home() / ".aider" / "caches"
        self.cache_file = self.cache_dir / "hostinger_router_models.json"
        self.content: Dict | None = None
        self.verify_ssl: bool = True
        self._cache_loaded = False

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def set_verify_ssl(self, verify_ssl: bool) -> None:
        """Enable/disable SSL verification for API requests."""
        self.verify_ssl = verify_ssl

    def get_model_info(self, model: str) -> Dict:
        """
        Return metadata for *model* or an empty ``dict`` when unknown.

        ``model`` should use the aider naming convention, e.g.
        ``hostinger_router/claude-sonnet-5``.
        """
        self._ensure_content()
        if not self.content or "data" not in self.content:
            return {}

        route = self._strip_prefix(model)
        record = next((item for item in self.content["data"] if item.get("id") == route), None)
        if not record:
            return {}

        context_len = record.get("context_length") or record.get("max_input_tokens") or None
        max_output = record.get("max_output_tokens") or None

        pricing = record.get("pricing", {})
        cache_read = _credits_to_usd(pricing.get("input_cache_read"))
        cache_write = _credits_to_usd(pricing.get("input_cache_write"))
        input_cost = _credits_to_usd(pricing.get("prompt"))

        info = {
            "max_input_tokens": context_len,
            "max_tokens": max_output or context_len,
            "max_output_tokens": max_output,
            "input_cost_per_token": input_cost,
            "output_cost_per_token": _credits_to_usd(pricing.get("completion")),
            "litellm_provider": "hostinger_router",
        }

        # The proxy defaults cache prices to the input price when caching is
        # not offered; only report cache pricing when it differs.
        if cache_read is not None and input_cost is not None and cache_read != input_cost:
            info["cache_read_input_token_cost"] = cache_read
            if cache_write is not None and cache_write != input_cost:
                info["cache_creation_input_token_cost"] = cache_write
            info["supports_prompt_caching"] = True

        architecture = record.get("architecture") or {}
        input_modalities = architecture.get("input_modalities") or []
        if "image" in input_modalities:
            info["supports_vision"] = True

        params = record.get("supported_parameters") or []
        if "tools" in params:
            info["supports_function_calling"] = True
        if "tool_choice" in params:
            info["supports_tool_choice"] = True
        if "reasoning" in params or "reasoning_effort" in params:
            info["supports_reasoning"] = True
        if "structured_outputs" in params or "response_format" in params:
            info["supports_response_schema"] = True

        endpoints = record.get("supported_endpoints") or []
        if "/v1/embeddings" in endpoints and "/v1/chat/completions" not in endpoints:
            info["mode"] = "embedding"

        return info

    def list_chat_models(self) -> list:
        """Return all chat-capable model ids with the hostinger_router/ prefix."""
        self._ensure_content()
        if not self.content or "data" not in self.content:
            return []

        out = []
        for item in self.content["data"]:
            endpoints = item.get("supported_endpoints")
            if endpoints and "/v1/chat/completions" not in endpoints:
                continue
            model_id = item.get("id")
            if model_id:
                out.append("hostinger_router/" + model_id)
        return out

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _strip_prefix(self, model: str) -> str:
        prefix = "hostinger_router/"
        return model[len(prefix) :] if model.startswith(prefix) else model

    def _ensure_content(self) -> None:
        self._load_cache()
        if not self.content:
            self._update_cache()

    def _load_cache(self) -> None:
        if self._cache_loaded:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.cache_file.exists():
                cache_age = time.time() - self.cache_file.stat().st_mtime
                if cache_age < self.CACHE_TTL:
                    try:
                        self.content = json.loads(self.cache_file.read_text())
                    except json.JSONDecodeError:
                        self.content = None
        except OSError:
            # Cache directory might be unwritable; ignore.
            pass

        self._cache_loaded = True

    def _update_cache(self) -> None:
        base = os.environ.get("HOSTINGER_ROUTER_API_BASE") or HOSTINGER_ROUTER_API_BASE
        url = base.rstrip("/") + "/models"
        headers = {}
        api_key = os.environ.get("HOSTINGER_ROUTER_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=10,
                verify=self.verify_ssl,
            )
            if response.status_code == 200:
                self.content = response.json()
                try:
                    self.cache_file.write_text(json.dumps(self.content, indent=2))
                except OSError:
                    pass  # Non-fatal if we can't write the cache
        except Exception as ex:  # noqa: BLE001
            print(f"Failed to fetch Hostinger Router model list: {ex}")
            try:
                self.cache_file.write_text("{}")
            except OSError:
                pass
