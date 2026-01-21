"""
Configuration loader for experiments and models.

Handles loading and parsing YAML configuration files with environment variable substitution.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dotenv import load_dotenv

from ..data.schemas import ExperimentConfig


class ConfigLoader:
    """Loads configuration from YAML files."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration loader.

        Args:
            config_dir: Path to config directory. If None, uses default location.
        """
        if config_dir is None:
            # Default location
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"

        self.config_dir = Path(config_dir)
        self.models_config: Optional[Dict[str, Any]] = None
        self.experiment_configs: Optional[Dict[str, Any]] = None
        # Load .env once so API keys and other settings are available
        load_dotenv(self.config_dir.parent / ".env", override=False)

    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in config.

        Replaces ${VAR_NAME} with the value of environment variable VAR_NAME.

        Args:
            config: Configuration dict or value

        Returns:
            Configuration with substituted values
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Find ${VAR_NAME} patterns
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, config)
            for var_name in matches:
                env_value = os.getenv(var_name, '')
                if not env_value:
                    print(f"Warning: Environment variable '{var_name}' not set")
                config = config.replace(f'${{{var_name}}}', env_value)
            return config
        else:
            return config

    def load_models_config(self) -> Dict[str, Any]:
        """
        Load model configuration.

        Returns:
            Dictionary with model configurations
        """
        if self.models_config is not None:
            return self.models_config

        config_file = self.config_dir / "models.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Models config not found: {config_file}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Substitute environment variables
        config = self._substitute_env_vars(config)
        self.models_config = config
        return config

    def load_experiment_configs(self) -> Dict[str, Any]:
        """
        Load all experiment configurations.

        Returns:
            Dictionary with experiment configurations
        """
        if self.experiment_configs is not None:
            return self.experiment_configs

        config_file = self.config_dir / "experiment.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Experiment config not found: {config_file}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.experiment_configs = config
        return config

    def get_experiment_config(self, experiment_name: str) -> ExperimentConfig:
        """
        Get configuration for a specific experiment.

        Args:
            experiment_name: Name of the experiment (e.g., "pilot", "phase1_consistency")

        Returns:
            ExperimentConfig object

        Raises:
            KeyError: If experiment not found
        """
        configs = self.load_experiment_configs()

        if experiment_name not in configs:
            available = list(configs.keys())
            raise KeyError(
                f"Experiment '{experiment_name}' not found. "
                f"Available: {available}"
            )

        exp_config = configs[experiment_name]

        # Create ExperimentConfig object
        # Generate experiment ID if not present
        if "experiment_id" not in exp_config:
            exp_config["experiment_id"] = f"{experiment_name}_{self._get_timestamp()}"

        return ExperimentConfig(**exp_config)

    def get_experiment_config_raw(self, experiment_name: str) -> Dict[str, Any]:
        """
        Get raw configuration dictionary for a specific experiment.

        Used for V2 experiments and other cases where we need the raw dict
        rather than a validated ExperimentConfig object.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Raw configuration dictionary

        Raises:
            KeyError: If experiment not found
        """
        configs = self.load_experiment_configs()

        if experiment_name not in configs:
            available = list(configs.keys())
            raise KeyError(
                f"Experiment '{experiment_name}' not found. "
                f"Available: {available}"
            )

        exp_config = configs[experiment_name].copy()

        # Generate experiment ID if not present
        if "experiment_id" not in exp_config:
            exp_config["experiment_id"] = f"{experiment_name}_{self._get_timestamp()}"

        return exp_config

    def get_model_config(self, provider: str, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model_name: Model identifier

        Returns:
            Dictionary with model configuration

        Raises:
            KeyError: If provider or model not found
        """
        models_config = self.load_models_config()

        if provider not in models_config:
            available = list(models_config.keys())
            raise KeyError(
                f"Provider '{provider}' not found. "
                f"Available: {available}"
            )

        provider_config = models_config[provider]

        # Get API key if available
        api_key = provider_config.get("api_key", None)

        # Get model-specific config
        if "models" in provider_config:
            if model_name not in provider_config["models"]:
                available = list(provider_config["models"].keys())
                raise KeyError(
                    f"Model '{model_name}' not found for provider '{provider}'. "
                    f"Available: {available}"
                )
            model_config = provider_config["models"][model_name].copy()
            model_config["api_key"] = api_key
            # Add provider-level config (like endpoint)
            for key in ["endpoint", "base_url"]:
                if key in provider_config:
                    model_config[key] = provider_config[key]
            return model_config
        else:
            # Return provider config with API key
            return {"api_key": api_key, **provider_config}

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def __repr__(self) -> str:
        return f"ConfigLoader(config_dir={self.config_dir})"
