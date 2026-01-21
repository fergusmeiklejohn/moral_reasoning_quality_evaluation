import os
from pathlib import Path

import pytest
import yaml

from src.config.loader import ConfigLoader


def write_config(tmp_path: Path) -> ConfigLoader:
    models_yaml = {
        "openai": {
            "api_key": "${OPENAI_API_KEY}",
            "endpoint": "https://api.openai.com",
            "models": {
                "gpt-5.1": {
                    "name": "gpt-5.1",
                    "supports_seed": True,
                    "default_max_tokens": 100,
                }
            },
        },
        "mock": {
            "models": {
                "test-model": {"name": "test-model", "supports_seed": True}
            }
        },
    }

    experiments_yaml = {
        "pilot": {
            "experiment_id": "pilot_123",
            "experiment_type": "pilot",
            "models": ["gpt-5.1"],
            "dilemma_ids": ["dilemma-1"],
            "temperatures": [0.0],
            "num_runs": 1,
        }
    }

    (tmp_path / "models.yaml").write_text(yaml.safe_dump(models_yaml))
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiments_yaml))
    return ConfigLoader(config_dir=tmp_path)


def test_env_substitution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    loader = write_config(tmp_path)
    cfg = loader.load_models_config()
    assert cfg["openai"]["api_key"] == "fake-key"


def test_get_model_config_includes_provider_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    loader = write_config(tmp_path)
    model_cfg = loader.get_model_config("openai", "gpt-5.1")
    assert model_cfg["api_key"] == "env-key"
    # Provider-level endpoint should propagate
    assert model_cfg["endpoint"] == "https://api.openai.com"
    assert model_cfg["name"] == "gpt-5.1"


def test_missing_experiment_raises(tmp_path: Path):
    loader = write_config(tmp_path)
    with pytest.raises(KeyError):
        loader.get_experiment_config("nonexistent")


def test_missing_model_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    loader = write_config(tmp_path)
    with pytest.raises(KeyError):
        loader.get_model_config("openai", "does-not-exist")


def test_loads_env_file_for_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """ConfigLoader should hydrate environment from a colocated .env file."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from_env_file\n")
    models_yaml = {
        "openai": {
            "api_key": "${OPENAI_API_KEY}",
            "models": {"gpt-5.1": {"name": "gpt-5.1", "supports_seed": True}},
        }
    }
    (config_dir / "models.yaml").write_text(yaml.safe_dump(models_yaml))

    loader = ConfigLoader(config_dir=config_dir)
    cfg = loader.load_models_config()
    assert cfg["openai"]["api_key"] == "from_env_file"
