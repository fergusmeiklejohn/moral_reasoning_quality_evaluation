import pytest

from src.dilemmas.loader import DilemmaLoader
from src.data.schemas import PerturbationType


def test_prompt_generation_order(sample_dilemmas_file):
    loader = DilemmaLoader(dilemmas_file=sample_dilemmas_file)
    dilemma = loader.get_dilemma("dilemma-1")

    prompt = dilemma.get_prompt(reversed_order=False)
    assert "CHOICE A: Pull the lever" in prompt
    assert "CHOICE B: Do nothing." in prompt

    prompt_reversed = dilemma.get_prompt(reversed_order=True)
    assert "CHOICE A: Do nothing." in prompt_reversed
    assert "CHOICE B: Pull the lever" in prompt_reversed


def test_perturbation_lookup(sample_dilemmas_file):
    loader = DilemmaLoader(dilemmas_file=sample_dilemmas_file)
    dilemma, variant = loader.get_dilemma_with_perturbation(
        "dilemma-1", PerturbationType.RELEVANT
    )
    assert dilemma.id == "dilemma-1"
    assert variant is not None
    assert "doctor" in variant.description


def test_missing_perturbation_raises(sample_dilemmas_file):
    loader = DilemmaLoader(dilemmas_file=sample_dilemmas_file)
    with pytest.raises(KeyError):
        loader.get_dilemma_with_perturbation("missing", PerturbationType.NONE)
