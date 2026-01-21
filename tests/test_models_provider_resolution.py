from src.models import get_provider_from_model_name


def test_get_provider_from_model_name_handles_mock_aliases():
    assert get_provider_from_model_name("mock") == "mock"
    assert get_provider_from_model_name("mock-model") == "mock"
