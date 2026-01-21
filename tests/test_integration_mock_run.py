import json
from pathlib import Path

import pytest
import yaml

from scripts import analyze_results
from src.config.loader import ConfigLoader
from src.dilemmas.loader import DilemmaLoader
from src.experiments.phase1_consistency import Phase1Runner
from src.data.schemas import PerturbationType
from src.data.storage import ExperimentStorage


@pytest.mark.integration
def test_mock_phase1_run_and_analysis(
    tmp_path: Path,
    sample_dilemmas_file: Path,
    storage: ExperimentStorage,
    experiment_config_factory,
    mock_similarity_model,
    monkeypatch: pytest.MonkeyPatch,
):
    # Prepare isolated config directory with mock model
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    models_yaml = {
        "mock": {"models": {"mock-model": {"name": "mock-model", "supports_seed": True}}}
    }
    experiments_yaml = {
        "pilot": {
            "experiment_id": "mock_integration",
            "experiment_type": "pilot",
            "models": ["mock-model"],
            "dilemma_ids": ["dilemma-1"],
            "temperatures": [0.0],
            "num_runs": 2,
            "test_reversed_order": False,
            "randomize_dilemma_order": False,
            "rate_limit_per_minute": 1000,
        }
    }
    (config_dir / "models.yaml").write_text(yaml.safe_dump(models_yaml))
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(experiments_yaml))

    config_loader = ConfigLoader(config_dir=config_dir)
    config = config_loader.get_experiment_config("pilot")
    dilemma_loader = DilemmaLoader(dilemmas_file=sample_dilemmas_file)

    # Avoid sleeping in the runner
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    runner = Phase1Runner(
        config=config,
        config_loader=config_loader,
        dilemma_loader=dilemma_loader,
        storage=storage,
    )
    experiment_id = runner.run_experiment()

    runs = storage.load_runs(experiment_id)
    # 1 model * 1 dilemma * 1 temp * 2 runs
    assert len(runs) == 2
    summary = storage.get_experiment_summary(experiment_id)
    assert summary["total_runs"] == 2

    # Point analyze_experiment to the same storage location
    def _patched_storage():
        return storage

    monkeypatch.setattr(analyze_results, "ExperimentStorage", _patched_storage)

    # Capture analysis output to ensure it completes without errors
    analyze_results.analyze_experiment(experiment_id)


@pytest.mark.integration
def test_phase1_runner_handles_perturbations(
    sample_dilemmas_file: Path,
    storage: ExperimentStorage,
    experiment_config_factory,
    monkeypatch: pytest.MonkeyPatch,
    mock_similarity_model,
    capsys,
):
    config = experiment_config_factory(
        experiment_id="perturbation_smoke",
        models=["mock"],
        test_perturbations=True,
        perturbation_types=[PerturbationType.NONE, PerturbationType.RELEVANT],
        num_runs=1,
        temperatures=[0.0],
        test_reversed_order=False,
        randomize_dilemma_order=False,
        rate_limit_per_minute=1000,
    )
    dilemma_loader = DilemmaLoader(dilemmas_file=sample_dilemmas_file)

    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    runner = Phase1Runner(
        config=config,
        config_loader=ConfigLoader(),
        dilemma_loader=dilemma_loader,
        storage=storage,
    )
    experiment_id = runner.run_experiment()

    runs = storage.load_runs(experiment_id)
    assert len(runs) == 2  # 1 model * 1 dilemma * 1 temp * 2 perturbations
    assert {run.perturbation_type for run in runs} == {
        PerturbationType.NONE,
        PerturbationType.RELEVANT,
    }

    def _patched_storage():
        return storage

    monkeypatch.setattr(analyze_results, "ExperimentStorage", _patched_storage)

    analyze_results.analyze_experiment(experiment_id)
    captured = capsys.readouterr().out
    assert "Perturbation Analysis" in captured
