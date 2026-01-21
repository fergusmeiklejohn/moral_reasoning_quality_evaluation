import json
from pathlib import Path

from src.data.storage import ExperimentStorage


def test_save_and_load_run(storage, experiment_run_factory):
    run = experiment_run_factory()
    storage.save_run(run.experiment_id, run)

    loaded = storage.load_runs(run.experiment_id)
    assert len(loaded) == 1
    assert loaded[0].response.parsed_choice == run.response.parsed_choice


def test_save_config_and_summary(storage, experiment_config_factory, experiment_run_factory):
    cfg = experiment_config_factory()
    storage.save_experiment_config(cfg.experiment_id, cfg)

    # Write a couple of runs to exercise summary stats
    storage.save_runs_batch(
        cfg.experiment_id,
        [
            experiment_run_factory(response=experiment_run_factory().response),
            experiment_run_factory(run_id="run-2"),
        ],
    )

    summary = storage.get_experiment_summary(cfg.experiment_id)
    assert summary["total_runs"] == 2
    assert summary["num_models"] == 1
    assert summary["num_dilemmas"] == 1


def test_create_backup(storage, experiment_run_factory):
    run = experiment_run_factory()
    storage.save_run(run.experiment_id, run)
    backup_dir = storage.create_backup(run.experiment_id, "test_backup")
    assert backup_dir.exists()
    assert (backup_dir / "all_runs.jsonl").exists()
    # Backup config may not exist if not saved; ensure no exception
    assert backup_dir.is_dir()
