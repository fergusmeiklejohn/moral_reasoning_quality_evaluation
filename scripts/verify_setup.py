#!/usr/bin/env python3
"""
Verify that the research framework is set up correctly.

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_imports():
    """Verify all required imports work."""
    print("Checking Python imports...")
    required_packages = [
        ("yaml", "pyyaml"),
        ("pydantic", "pydantic"),
        ("tqdm", "tqdm"),
        ("tenacity", "tenacity"),
        ("rich", "rich"),
        ("requests", "requests"),
    ]

    optional_packages = [
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("google.generativeai", "google-generativeai"),
        ("sentence_transformers", "sentence-transformers"),
    ]

    all_good = True

    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - REQUIRED")
            all_good = False

    print("\nChecking optional packages...")
    for module_name, package_name in optional_packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ⚠ {package_name} - optional (needed for specific features)")

    return all_good


def verify_structure():
    """Verify project structure."""
    print("\nChecking project structure...")

    project_root = Path(__file__).parent.parent

    required_paths = [
        "src",
        "src/models",
        "src/dilemmas",
        "src/experiments",
        "src/data",
        "src/analysis",
        "config",
        "config/models.yaml",
        "config/experiment.yaml",
        "data/dilemmas",
        "data/dilemmas/dilemmas.json",
        "scripts",
    ]

    all_good = True

    for path_str in required_paths:
        path = project_root / path_str
        if path.exists():
            print(f"  ✓ {path_str}")
        else:
            print(f"  ✗ {path_str} - MISSING")
            all_good = False

    return all_good


def verify_config():
    """Verify configuration files."""
    print("\nChecking configuration...")

    try:
        from src.config.loader import ConfigLoader

        config_loader = ConfigLoader()

        # Try loading models config
        models_config = config_loader.load_models_config()
        print(f"  ✓ models.yaml loaded ({len(models_config)} providers)")

        # Try loading experiment configs
        exp_configs = config_loader.load_experiment_configs()
        print(f"  ✓ experiment.yaml loaded ({len(exp_configs)} experiments)")

        # Check for API keys
        print("\nChecking API keys...")
        import os

        api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        }

        any_key_set = False
        for key, value in api_keys.items():
            if value and value != "":
                print(f"  ✓ {key} is set")
                any_key_set = True
            else:
                print(f"  ⚠ {key} not set (optional)")

        if not any_key_set:
            print("\n  ⚠ No API keys set. You can only use mock or local models.")
            print("  Create .env file from .env.example and add your API keys.")

        return True

    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        return False


def verify_dilemmas():
    """Verify dilemmas load correctly."""
    print("\nChecking dilemmas...")

    try:
        from src.dilemmas.loader import DilemmaLoader

        loader = DilemmaLoader()
        dilemmas = loader.get_all_dilemmas()

        print(f"  ✓ Loaded {len(dilemmas)} dilemmas")

        # Check categories
        from collections import Counter
        from src.data.schemas import DilemmaCategory

        categories = Counter(d.category for d in dilemmas)
        for category, count in categories.items():
            print(f"    - {category.value}: {count} dilemmas")

        # Check perturbation coverage
        with_perturbations = sum(
            1 for d in dilemmas
            if d.perturbation_variants
        )
        print(f"  ℹ {with_perturbations} dilemmas have perturbation variants")

        return True

    except Exception as e:
        print(f"  ✗ Dilemma loading error: {e}")
        return False


def verify_models():
    """Verify model providers can be initialized."""
    print("\nChecking model providers...")

    try:
        all_good = True
        from src.models import create_provider
        from src.config.loader import ConfigLoader
        from src.data.schemas import Choice

        # Test mock provider
        mock_provider = create_provider("mock", "test-model")
        print(f"  ✓ Mock provider works")

        # Test if mock provider can generate
        response = mock_provider.generate("Test prompt", temperature=0.0)
        if response.parsed_choice == Choice.ERROR:
            print(f"  ✗ Mock provider failed to generate a valid response")
            all_good = False
        else:
            print(f"  ✓ Mock provider can generate responses")

        # Inspect configured local providers
        config_loader = ConfigLoader()
        models_config = config_loader.load_models_config()
        local_providers = []

        for provider_name in ("vllm", "ollama"):
            provider_config = models_config.get(provider_name, {})
            model_count = len(provider_config.get("models", {}) or {})
            if model_count:
                local_providers.append((provider_name, model_count))

        if local_providers:
            for provider_name, model_count in local_providers:
                print(f"  ✓ {provider_name} configured with {model_count} local model(s)")
        else:
            print("  ⚠ No local (vLLM/Ollama) providers configured")

        # Run lightweight connectivity checks for the primary Ollama models
        primary_ollama_models = [
            ("gpt-oss", "gpt-oss:latest", "Ollama/GPT-OSS"),
            ("qwen3", "qwen3:latest", "Ollama/Qwen3"),
        ]
        connectivity_failed = False
        ollama_config = models_config.get("ollama")
        ollama_models = (ollama_config or {}).get("models", {}) or {}
        if not ollama_config:
            print("  ⚠ Ollama provider not configured; skipping connectivity check")
        elif not ollama_models:
            print("  ⚠ Ollama configured without models; skipping connectivity check")
        else:
            endpoint = ollama_config.get("endpoint")
            for alias, default_tag, label in primary_ollama_models:
                if alias not in ollama_models:
                    print(f"  ⚠ Ollama configured but {alias} model missing; update config/models.yaml if you plan to use it")
                    continue

                model_tag = ollama_models[alias].get("name", default_tag)
                print(f"  ⏳ Checking {label} connectivity ({model_tag})...")
                try:
                    provider = create_provider("ollama", model_tag, endpoint=endpoint)
                    if provider.validate_connection():
                        print(f"  ✓ {label} reachable")
                    else:
                        print(f"  ⚠ {label} not reachable; is Ollama running and is the model pulled?")
                        connectivity_failed = True
                except Exception as e:
                    print(f"  ✗ {label} connectivity check failed: {e}")
                    connectivity_failed = True

        if connectivity_failed:
            all_good = False

        return all_good

    except Exception as e:
        print(f"  ✗ Model provider error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("="*80)
    print("Moral Decision Consistency Research Framework")
    print("Setup Verification")
    print("="*80 + "\n")

    checks = [
        ("Python imports", verify_imports),
        ("Project structure", verify_structure),
        ("Configuration", verify_config),
        ("Dilemmas", verify_dilemmas),
        ("Model providers", verify_models),
    ]

    results = []

    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed with error: {e}")
            results.append((name, False))
        print()

    # Summary
    print("="*80)
    print("Verification Summary")
    print("="*80 + "\n")

    all_passed = True
    for name, result in results:
        if result:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            all_passed = False

    print()

    if all_passed:
        print("🎉 All checks passed! You're ready to run experiments.")
        print("\nNext steps:")
        print("  1. Review DECISIONS.md for areas needing your input")
        print("  2. Set up API keys in .env (if using cloud models)")
        print("  3. Run a test: uv run python scripts/run_experiment.py --phase pilot")
    else:
        print("⚠ Some checks failed. Please review the errors above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: uv sync")
        print("  - Create .env file: cp .env.example .env")
        print("  - Check that you're in the project root directory")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
