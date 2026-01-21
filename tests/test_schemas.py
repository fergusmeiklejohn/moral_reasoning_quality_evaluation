import pytest

from src.data.schemas import Choice, ExperimentConfig, ModelResponse


def test_choice_parser_variants():
    assert ModelResponse.parse_choice("CHOICE A") == Choice.A
    assert ModelResponse.parse_choice("a") == Choice.A
    assert ModelResponse.parse_choice("CHOICE B") == Choice.B
    assert ModelResponse.parse_choice("refuse") == Choice.REFUSE
    assert ModelResponse.parse_choice("nonsense") == Choice.ERROR


def test_experiment_config_defaults():
    cfg = ExperimentConfig(
        experiment_id="test",
        experiment_type="pilot",
        models=["mock"],
        dilemma_ids=["d1"],
    )
    assert cfg.top_p == 1.0
    assert cfg.num_runs == 30
    assert cfg.test_reversed_order
    assert cfg.randomize_dilemma_order
    # Perturbation defaults include none/relevant/irrelevant
    assert len(cfg.perturbation_types) == 3
