from pathlib import Path
from unittest.mock import patch

from aider.models import HOSTINGER_ROUTER_API_BASE, Model, model_info_manager

ROUTER_MODELS_PAYLOAD = {
    "data": [
        {
            "id": "claude-sonnet-5",
            "context_length": 1000000,
            "max_input_tokens": 1000000,
            "max_output_tokens": 128000,
            "pricing": {
                "prompt": "0.00024",
                "completion": "0.0012",
                "input_cache_read": "0.000024",
                "input_cache_write": "0.0003",
                "input_cache_write_1h": "0.00048",
            },
            "supported_parameters": [
                "include_reasoning",
                "max_completion_tokens",
                "max_tokens",
                "reasoning",
                "reasoning_effort",
                "response_format",
                "structured_outputs",
                "tool_choice",
                "tools",
            ],
            "supported_endpoints": ["/v1/chat/completions", "/v1/messages"],
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
        },
        {
            "id": "text-embedding-3-small",
            "context_length": 8191,
            "max_input_tokens": 8191,
            "max_output_tokens": None,
            "pricing": {
                "prompt": "0.0000024",
                "completion": "0",
                "input_cache_read": "0.0000024",
                "input_cache_write": "0.0000024",
                "input_cache_write_1h": "0.0000024",
            },
            "supported_parameters": ["max_tokens"],
            "supported_endpoints": ["/v1/embeddings"],
            "architecture": {"input_modalities": ["text"], "output_modalities": None},
        },
    ]
}


class DummyResponse:
    def __init__(self, json_data, status_code=200):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data


def _patched_manager(monkeypatch, tmp_path, payload=ROUTER_MODELS_PAYLOAD):
    monkeypatch.setenv("HOSTINGER_ROUTER_API_KEY", "hr-test-key")
    monkeypatch.setattr("requests.get", lambda *a, **k: DummyResponse(payload))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    from aider.hostinger_router import HostingerRouterModelManager

    return HostingerRouterModelManager()


def test_hostinger_router_manager_get_model_info(monkeypatch, tmp_path):
    manager = _patched_manager(monkeypatch, tmp_path)
    info = manager.get_model_info("hostinger_router/claude-sonnet-5")

    assert info["max_input_tokens"] == 1000000
    assert info["max_output_tokens"] == 128000
    assert abs(info["input_cost_per_token"] - 0.0000024) < 1e-12
    assert abs(info["output_cost_per_token"] - 0.000012) < 1e-12
    assert abs(info["cache_read_input_token_cost"] - 0.00000024) < 1e-12
    assert info["litellm_provider"] == "hostinger_router"
    assert info["supports_vision"] is True
    assert info["supports_function_calling"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_prompt_caching"] is True
    assert info.get("mode") != "embedding"


def test_hostinger_router_manager_embedding_model(monkeypatch, tmp_path):
    manager = _patched_manager(monkeypatch, tmp_path)
    info = manager.get_model_info("hostinger_router/text-embedding-3-small")

    # cache prices equal the input price -> caching treated as unset
    assert info["mode"] == "embedding"
    assert "cache_read_input_token_cost" not in info
    assert "supports_prompt_caching" not in info


def test_hostinger_router_manager_list_chat_models(monkeypatch, tmp_path):
    manager = _patched_manager(monkeypatch, tmp_path)
    models = manager.list_chat_models()

    assert "hostinger_router/claude-sonnet-5" in models
    assert "hostinger_router/text-embedding-3-small" not in models


def test_hostinger_router_manager_fetches_without_key(monkeypatch, tmp_path):
    # /v1/models is public; the catalog loads with no key and no auth header
    monkeypatch.delenv("HOSTINGER_ROUTER_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    captured = {}

    def spy_get(url, **kwargs):
        captured.update(kwargs)
        return DummyResponse(ROUTER_MODELS_PAYLOAD)

    monkeypatch.setattr("requests.get", spy_get)
    from aider.hostinger_router import HostingerRouterModelManager

    manager = HostingerRouterModelManager()
    info = manager.get_model_info("hostinger_router/claude-sonnet-5")
    assert info["litellm_provider"] == "hostinger_router"
    assert captured["headers"] == {}


def test_hostinger_router_manager_unknown_model(monkeypatch, tmp_path):
    manager = _patched_manager(monkeypatch, tmp_path)
    assert manager.get_model_info("hostinger_router/no-such-model") == {}


def test_model_info_manager_uses_hostinger_router_manager(monkeypatch, tmp_path):
    from aider.models import model_info_manager

    _patched_manager(monkeypatch, tmp_path)
    monkeypatch.setattr(model_info_manager, "local_model_metadata", {})
    monkeypatch.setattr(
        model_info_manager, "hostinger_router_manager", _patched_manager(monkeypatch, tmp_path)
    )

    info = model_info_manager.get_model_info("hostinger_router/claude-sonnet-5")
    assert info["litellm_provider"] == "hostinger_router"
    assert info["max_input_tokens"] == 1000000


def test_list_models_includes_live_catalog(monkeypatch, tmp_path):
    from aider import models
    from aider.models import model_info_manager

    manager = _patched_manager(monkeypatch, tmp_path)
    monkeypatch.setattr(model_info_manager, "hostinger_router_manager", manager)
    monkeypatch.setattr(models.model_info_manager, "hostinger_router_manager", manager)

    matches = models.fuzzy_match_models("hostinger_router/")
    assert "hostinger_router/claude-sonnet-5" in matches


def test_hostinger_router_shim_rewrites_model_and_injects_credentials(monkeypatch):
    monkeypatch.setenv("HOSTINGER_ROUTER_API_KEY", "hr-test-key")
    monkeypatch.delenv("HOSTINGER_ROUTER_API_BASE", raising=False)

    model = Model("hostinger_router/claude-sonnet-5")
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)

        class FakeResponse:
            pass

        return FakeResponse()

    with patch("aider.models.litellm.completion", side_effect=fake_completion):
        model.send_completion([{"role": "user", "content": "hi"}], None, stream=False)

    assert captured["model"] == "openai/claude-sonnet-5"
    assert captured["api_base"] == HOSTINGER_ROUTER_API_BASE
    assert captured["api_key"] == "hr-test-key"


def test_hostinger_router_shim_respects_user_overrides(monkeypatch):
    monkeypatch.setenv("HOSTINGER_ROUTER_API_KEY", "hr-env-key")
    monkeypatch.setenv("HOSTINGER_ROUTER_API_BASE", "https://router.eu.example/v1")

    model = Model("hostinger_router/claude-sonnet-5")
    model.extra_params = dict(model.extra_params or {})
    model.extra_params["api_base"] = "https://router.custom.example/v1"
    model.extra_params["api_key"] = "hr-explicit-key"
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)

        class FakeResponse:
            pass

        return FakeResponse()

    with patch("aider.models.litellm.completion", side_effect=fake_completion):
        model.send_completion([{"role": "user", "content": "hi"}], None, stream=False)

    # explicit extra_params win over env and defaults
    assert captured["api_base"] == "https://router.custom.example/v1"
    assert captured["api_key"] == "hr-explicit-key"


def test_hostinger_router_shim_uses_api_base_env(monkeypatch):
    monkeypatch.setenv("HOSTINGER_ROUTER_API_KEY", "hr-test-key")
    monkeypatch.setenv("HOSTINGER_ROUTER_API_BASE", "https://router.eu.example/v1")

    model = Model("hostinger_router/claude-sonnet-5")
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)

        class FakeResponse:
            pass

        return FakeResponse()

    with patch("aider.models.litellm.completion", side_effect=fake_completion):
        model.send_completion([{"role": "user", "content": "hi"}], None, stream=False)

    assert captured["api_base"] == "https://router.eu.example/v1"


def test_hostinger_router_fast_validate_environment(monkeypatch):
    monkeypatch.delenv("HOSTINGER_ROUTER_API_KEY", raising=False)
    model = Model("hostinger_router/claude-sonnet-5")
    res = model.fast_validate_environment()
    # fast path only fires when the key IS present
    assert res is None

    monkeypatch.setenv("HOSTINGER_ROUTER_API_KEY", "hr-test-key")
    res = model.fast_validate_environment()
    assert res == dict(keys_in_environment=["HOSTINGER_ROUTER_API_KEY"], missing_keys=[])


def test_hostinger_router_validate_environment_warns_without_key(monkeypatch):
    monkeypatch.delenv("HOSTINGER_ROUTER_API_KEY", raising=False)
    model = Model("hostinger_router/claude-sonnet-5")
    res = model.validate_environment()
    assert "HOSTINGER_ROUTER_API_KEY" in res["missing_keys"]


def test_hostinger_router_settings_resolution():
    model = Model("hostinger_router/claude-sonnet-5")
    assert model.edit_format == "diff"
    assert model.use_repo_map is True
    assert model.cache_control is True
    assert model.editor_edit_format == "editor-diff"
    assert model.extra_params.get("max_tokens") == 128000
    assert "reasoning_effort" in (model.accepts_settings or [])

    weak = Model("hostinger_router/claude-haiku-4.5")
    assert weak.extra_params.get("max_tokens") == 64000


def test_hostinger_router_model_metadata(monkeypatch, tmp_path):
    manager = _patched_manager(monkeypatch, tmp_path)
    monkeypatch.setattr(model_info_manager, "hostinger_router_manager", manager)
    model = Model("hostinger_router/claude-sonnet-5")
    info = model.info
    assert info.get("litellm_provider") == "hostinger_router"
    assert info.get("max_input_tokens") == 1000000
    assert info.get("supports_vision") is True
    assert info.get("supports_prompt_caching") is True


def test_hostinger_router_model_name_preserved_for_metadata(monkeypatch):
    # the shim rewrites only the litellm call kwargs; self.name stays branded
    monkeypatch.setenv("HOSTINGER_ROUTER_API_KEY", "hr-test-key")
    model = Model("hostinger_router/claude-sonnet-5")
    assert model.name == "hostinger_router/claude-sonnet-5"
    assert model.is_hostinger_router()
