from collections import defaultdict

import pytest

from src.analysis.metrics import MetricsCalculator
from src.data.schemas import (
    Choice,
    ExperimentRun,
    PerturbationType,
    SyntheticErrorTransform,
)


def build_runs(experiment_run_factory, choices):
    runs = []
    for idx, choice in enumerate(choices, start=1):
        resp = experiment_run_factory().response.model_copy()
        resp.raw_text = choice.value
        resp.parsed_choice = choice
        runs.append(
            experiment_run_factory(
                run_id=f"run-{idx}",
                run_number=idx,
                response=resp,
            )
        )
    return runs


def test_basic_metrics(experiment_run_factory, mock_similarity_model):
    calc = MetricsCalculator()
    runs = build_runs(experiment_run_factory, [Choice.A, Choice.A, Choice.B])
    assert calc.calculate_choice_consistency_rate(runs) == pytest.approx(2 / 3)
    assert calc.calculate_refusal_rate(runs) == 0
    flip = calc.calculate_flip_pattern(runs)
    assert flip["num_flips"] == 1


def test_perturbation_sensitivity(mock_similarity_model, experiment_run_factory):
    calc = MetricsCalculator()
    baseline = build_runs(experiment_run_factory, [Choice.A, Choice.A])
    perturbed = build_runs(experiment_run_factory, [Choice.B, Choice.B])
    result = calc.calculate_perturbation_sensitivity(
        baseline, perturbed, PerturbationType.RELEVANT
    )
    assert result["decision_changed"] is True
    assert result["appropriate_sensitivity"] is True

    # Irrelevant perturbation should prefer stability
    result_irrel = calc.calculate_perturbation_sensitivity(
        baseline, baseline, PerturbationType.IRRELEVANT
    )
    assert result_irrel["appropriate_sensitivity"] is True


def test_cross_model_agreement(experiment_run_factory):
    calc = MetricsCalculator()
    runs_by_model = defaultdict(list)
    for model, choice in (("m1", Choice.A), ("m2", Choice.A), ("m3", Choice.B)):
        resp = experiment_run_factory().response.model_copy()
        resp.parsed_choice = choice
        runs_by_model[model].append(
            ExperimentRun(
                **experiment_run_factory(
                    model_name=model,
                    response=resp,
                ).model_dump()
            )
    )
    agreement = calc.calculate_cross_model_agreement(runs_by_model)
    assert agreement["agreement_rate"] == pytest.approx(2 / 3)
    assert agreement["monoculture_risk"] == "low"


def test_mss_quadrant():
    calc = MetricsCalculator()
    assert calc.calculate_mss_quadrant(0.95, 0.8) == "principled"
    assert calc.calculate_mss_quadrant(0.95, 0.5) == "brittle_monoculture"
    assert calc.calculate_mss_quadrant(0.5, 0.8) == "principled_underdetermined"
    assert calc.calculate_mss_quadrant(0.5, 0.5) == "chaotic"


def test_type_c_metrics(type_c_record_factory, experiment_run_factory):
    calc = MetricsCalculator()
    record = type_c_record_factory()
    run = experiment_run_factory(
        perturbation_type=PerturbationType.SYNTHETIC_ERROR,
        type_c_record=record,
    )
    metrics = calc.calculate_type_c_metrics([run])
    assert metrics["total_runs"] == 1
    assert metrics["localization_accuracy"] == 1
    assert metrics["repair_success_rate"] == 1
    assert 0 <= metrics["minimality_score"] <= 1
    assert 0 <= metrics["counterfactual_coherence"] <= 1
    assert 0 <= metrics["explanation_alignment"] <= 1
