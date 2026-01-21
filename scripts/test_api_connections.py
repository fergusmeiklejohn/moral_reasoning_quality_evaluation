#!/usr/bin/env python3
"""
API connection tests for Ethics Bowl providers.

Tests each configured provider to verify:
1. Connection works
2. Response format is correct
3. Conversation/multi-turn works
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import create_provider, get_provider_from_model_name
from src.config.loader import ConfigLoader


# Test models for each provider
TEST_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-5.2",
    "google": "gemini-3-pro-preview",
    "grok": "grok-4-1-fast-reasoning",
}

# Simple test prompt
SIMPLE_PROMPT = "Respond with exactly: 'API connection successful'"

# Conversation test messages
CONVERSATION_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Keep responses brief."},
    {"role": "user", "content": "What is 2 + 2? Reply with just the number."},
]


def test_provider(provider_name: str, model_name: str, config_loader: ConfigLoader) -> dict:
    """
    Test a single provider.

    Returns dict with test results.
    """
    results = {
        "provider": provider_name,
        "model": model_name,
        "connection": False,
        "simple_generation": False,
        "conversation": False,
        "response_format": False,
        "errors": [],
    }

    print(f"\n{'='*60}")
    print(f"Testing: {provider_name} ({model_name})")
    print(f"{'='*60}")

    # Get API key from config
    try:
        models_config = config_loader.load_models_config()
        provider_config = models_config.get(provider_name, {})
        api_key = provider_config.get("api_key")

        if not api_key or api_key.startswith("your_"):
            results["errors"].append(f"API key not configured for {provider_name}")
            print(f"  ❌ API key not configured")
            return results

        # Get additional config (like base_url for grok)
        extra_config = {}
        if "base_url" in provider_config:
            extra_config["base_url"] = provider_config["base_url"]

    except Exception as e:
        results["errors"].append(f"Config error: {e}")
        print(f"  ❌ Config error: {e}")
        return results

    # Create provider
    try:
        provider = create_provider(
            provider_name=provider_name,
            model_name=model_name,
            api_key=api_key,
            **extra_config
        )
        results["connection"] = True
        print(f"  ✓ Provider created successfully")
    except Exception as e:
        results["errors"].append(f"Provider creation failed: {e}")
        print(f"  ❌ Provider creation failed: {e}")
        return results

    # Test 1: Simple generation
    print(f"\n  Test 1: Simple generation...")
    try:
        response = provider.generate(
            prompt=SIMPLE_PROMPT,
            temperature=0.0,
            max_tokens=50,
        )

        if response.raw_text and len(response.raw_text) > 0:
            results["simple_generation"] = True
            print(f"  ✓ Simple generation works")
            print(f"    Response: {response.raw_text[:100]}...")

            # Check response format
            if hasattr(response, 'tokens_used') or hasattr(response, 'response_time_seconds'):
                results["response_format"] = True
                print(f"    Tokens: {response.tokens_used}, Time: {response.response_time_seconds:.2f}s")
        else:
            results["errors"].append("Empty response from simple generation")
            print(f"  ❌ Empty response")

    except Exception as e:
        results["errors"].append(f"Simple generation failed: {e}")
        print(f"  ❌ Simple generation failed: {e}")

    # Test 2: Conversation (multi-turn)
    print(f"\n  Test 2: Conversation (multi-turn)...")
    try:
        response = provider.generate_conversation(
            messages=CONVERSATION_MESSAGES,
            temperature=0.0,
            max_tokens=50,
        )

        if response.raw_text and len(response.raw_text) > 0:
            results["conversation"] = True
            print(f"  ✓ Conversation works")
            print(f"    Response: {response.raw_text[:100]}...")

            # Check if response contains "4"
            if "4" in response.raw_text:
                print(f"    ✓ Correct answer detected")
            else:
                print(f"    ⚠ Expected '4' in response")
        else:
            results["errors"].append("Empty response from conversation")
            print(f"  ❌ Empty response")

    except Exception as e:
        results["errors"].append(f"Conversation failed: {e}")
        print(f"  ❌ Conversation failed: {e}")

    # Test 3: Ethics Bowl style prompt
    print(f"\n  Test 3: Ethics Bowl style prompt...")
    ethics_prompt = """You are participating in an Ethics Bowl. Briefly analyze this scenario:

A self-driving car must choose between hitting one pedestrian or swerving into a wall, injuring the passenger.

In 2-3 sentences, identify the core ethical tension."""

    try:
        response = provider.generate_conversation(
            messages=[
                {"role": "system", "content": "You are a thoughtful ethics discussant."},
                {"role": "user", "content": ethics_prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        if response.raw_text and len(response.raw_text) > 50:
            print(f"  ✓ Ethics prompt works")
            print(f"    Response preview: {response.raw_text[:150]}...")
        else:
            print(f"  ⚠ Response seems short: {response.raw_text}")

    except Exception as e:
        results["errors"].append(f"Ethics prompt failed: {e}")
        print(f"  ❌ Ethics prompt failed: {e}")

    return results


def main():
    """Run all API tests."""
    print("=" * 60)
    print("Ethics Bowl API Connection Tests")
    print("=" * 60)

    config_loader = ConfigLoader()
    all_results = []

    for provider_name, model_name in TEST_MODELS.items():
        results = test_provider(provider_name, model_name, config_loader)
        all_results.append(results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Provider':<15} {'Model':<25} {'Status':<10}")
    print("-" * 60)

    all_passed = True
    for r in all_results:
        passed = r["connection"] and r["simple_generation"] and r["conversation"]
        status = "✓ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        print(f"{r['provider']:<15} {r['model']:<25} {status:<10}")

        if r["errors"]:
            for err in r["errors"]:
                print(f"  └─ {err}")

    print("\n" + "-" * 60)

    if all_passed:
        print("All API tests passed! Ready for Ethics Bowl tournaments.")
        return 0
    else:
        print("Some tests failed. Check API keys and configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
