import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pytest

# Ensure src/ is importable without relying on external PYTHONPATH tweaks
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schemas import (  # pylint: disable=wrong-import-position
    Choice,
    DilemmaCategory,
    ExperimentConfig,
    ExperimentRun,
    ModelResponse,
    PerturbationType,
    ReasoningGraph,
    ReasoningStep,
    SyntheticErrorInjection,
    SyntheticErrorTransform,
    TypeCRecord,
    TypeCRepairMetadata,
)
from src.data.storage import ExperimentStorage  # pylint: disable=wrong-import-position


@pytest.fixture
def project_root() -> Path:
    """Path to the repository root for relative fixture files."""
    return PROJECT_ROOT


@pytest.fixture
def temp_results_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Isolated results directory so tests never touch data/results/."""
    return tmp_path_factory.mktemp("results")


@pytest.fixture
def storage(temp_results_dir: Path) -> ExperimentStorage:
    """Storage object that writes into a temporary location."""
    return ExperimentStorage(results_dir=temp_results_dir)


@pytest.fixture
def sample_dilemmas_file(tmp_path: Path) -> Path:
    """Write a minimal dilemmas.json to a temp file for loader tests."""
    dilemmas = {
        "dilemmas": [
            {
                "id": "dilemma-1",
                "category": "philosophical",
                "title": "Trolley switch",
                "description": "A runaway trolley is headed toward five workers.",
                "choice_a": "Pull the lever to divert the trolley.",
                "choice_b": "Do nothing.",
                "context": "Classic trolley problem",
                "perturbation_variants": {
                    "relevant": {
                        "perturbation_type": "relevant",
                        "description": "Now, the trolley will kill a doctor needed for future cures.",
                        "choice_a": "Pull the lever to divert toward the doctor.",
                        "choice_b": "Do nothing.",
                        "perturbed_elements": ["victim_role"],
                        "expected_change": "choice_b",
                    },
                    "irrelevant": {
                        "perturbation_type": "irrelevant",
                        "description": "The trolley is painted blue instead of red.",
                        "choice_a": "Pull the lever to divert the trolley.",
                        "choice_b": "Do nothing.",
                        "perturbed_elements": ["color"],
                        "expected_change": "no_change",
                    },
                },
            }
        ]
    }
    path = tmp_path / "dilemmas.json"
    path.write_text(json.dumps(dilemmas, indent=2))
    return path


@pytest.fixture
def experiment_config_factory() -> Callable[..., ExperimentConfig]:
    """Factory for lightweight ExperimentConfig objects."""

    def _factory(**overrides: Any) -> ExperimentConfig:
        base_config: Dict[str, Any] = {
            "experiment_id": "test_experiment",
            "experiment_type": "pilot",
            "models": ["mock-model"],
            "dilemma_ids": ["dilemma-1"],
            "temperatures": [0.0],
            "top_p": 1.0,
            "num_runs": 1,
            "test_perturbations": False,
            "perturbation_types": [PerturbationType.NONE],
            "test_reversed_order": False,
            "randomize_dilemma_order": False,
            "fixed_seed": None,
            "rate_limit_per_minute": 1000,  # kept high to avoid sleeps during tests
            "retry_attempts": 1,
            "backup_every_n_queries": 10,
        }
        base_config.update(overrides)
        return ExperimentConfig(**base_config)

    return _factory


@pytest.fixture
def experiment_run_factory() -> Callable[..., ExperimentRun]:
    """Factory for ExperimentRun instances with a simple ModelResponse."""

    def _factory(**overrides: Any) -> ExperimentRun:
        response: Optional[ModelResponse] = overrides.pop("response", None)
        if response is None:
            response = ModelResponse(
                raw_text="CHOICE A\nsample reasoning",
                parsed_choice=Choice.A,
                reasoning="sample reasoning",
                response_time_seconds=0.01,
            )

        base_run: Dict[str, Any] = {
            "experiment_id": "test_experiment",
            "run_id": "run-1",
            "model_name": "mock-model",
            "model_version": "1.0",
            "provider": "mock",
            "dilemma_id": "dilemma-1",
            "dilemma_category": DilemmaCategory.PHILOSOPHICAL,
            "perturbation_type": PerturbationType.NONE,
            "position_order": "original",
            "temperature": 0.0,
            "top_p": 1.0,
            "random_seed": 42,
            "run_number": 1,
            "response": response,
            "error": None,
            "notes": None,
            "type_c_record": None,
        }
        base_run.update(overrides)
        return ExperimentRun(**base_run)

    return _factory


@pytest.fixture
def type_c_record_factory() -> Callable[..., TypeCRecord]:
    """Factory for minimal Type C records used by metrics tests."""

    def _factory(**overrides: Any) -> TypeCRecord:
        graph = overrides.pop(
            "initial_graph",
            ReasoningGraph(
                dilemma_id="dilemma-1",
                model_name="mock-model",
                steps=[
                    ReasoningStep(step_number=1, claim="Assess harm", depends_on=[]),
                    ReasoningStep(step_number=2, claim="Minimize casualties", depends_on=[1]),
                ],
                final_choice=Choice.A,
            ),
        )

        injection = overrides.pop(
            "injected_error",
            SyntheticErrorInjection(
                step_number=2,
                transform=SyntheticErrorTransform.SIGN_FLIP,
                original_claim="Minimize casualties",
                perturbed_claim="Maximize casualties",
                depends_on=[1],
            ),
        )

        repair_metadata = overrides.pop(
            "repair_metadata",
            TypeCRepairMetadata(
                identified_step=2,
                error_explanation="Swapped sign",
                repaired_steps=[ReasoningStep(step_number=2, claim="Restore minimize")],
                downstream_steps_touched=[2],
                final_choice=Choice.A,
                raw_response_json={"identified_step": 2},
            ),
        )

        initial_response = overrides.pop(
            "initial_response",
            ModelResponse(
                raw_text="CHOICE A\nstructured reasoning",
                parsed_choice=Choice.A,
                reasoning="structured reasoning",
                response_time_seconds=0.01,
            ),
        )

        repair_response = overrides.pop(
            "repair_response",
            ModelResponse(
                raw_text="CHOICE A\nrepair reasoning",
                parsed_choice=Choice.A,
                reasoning="repair reasoning",
                response_time_seconds=0.01,
            ),
        )

        base_record: Dict[str, Any] = {
            "initial_graph": graph,
            "injected_error": injection,
            "corrupted_steps": graph.steps,
            "initial_response": initial_response,
            "repair_response": repair_response,
            "repair_metadata": repair_metadata,
        }
        base_record.update(overrides)
        return TypeCRecord(**base_record)

    return _factory


@pytest.fixture
def mock_similarity_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch MetricsCalculator to avoid downloading embedding models during tests."""
    from src.analysis.metrics import MetricsCalculator

    class _DummyModel:
        def encode(self, texts):
            return [[float(len(text))] for text in texts]

    def _stub(self: MetricsCalculator):
        return _DummyModel()

    monkeypatch.setattr(MetricsCalculator, "_get_similarity_model", _stub, raising=True)
