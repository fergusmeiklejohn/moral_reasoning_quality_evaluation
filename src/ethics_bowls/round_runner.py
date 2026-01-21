"""
Round runner for Ethics Bowl system.

Orchestrates a single Ethics Bowl round between three models.
"""

import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from ..models import create_provider, get_configured_provider, get_provider_from_model_name
from ..config.loader import ConfigLoader
from .schemas import (
    Round,
    Phase,
    PhaseType,
    RoundStatus,
    EBDilemma,
    TournamentConfig,
)
from .prompt_builder import PromptBuilder
from .judgment_parser import JudgmentParser
from .storage import EBStorage


logger = logging.getLogger(__name__)


class RoundRunner:
    """Runs a single Ethics Bowl round."""

    def __init__(
        self,
        config: TournamentConfig,
        config_loader: Optional[ConfigLoader] = None,
        storage: Optional[EBStorage] = None,
    ):
        """
        Initialize round runner.

        Args:
            config: Tournament configuration
            config_loader: Configuration loader for model configs
            storage: Storage instance for saving results
        """
        self.config = config
        self.config_loader = config_loader or ConfigLoader()
        # Create storage with config for descriptive naming if not provided
        self.storage = storage or EBStorage(config.output_dir, tournament_config=config)
        self.providers: Dict[str, object] = {}

    def _get_provider(self, model_name: str):
        """
        Get or create provider for a model.

        Args:
            model_name: Name of the model

        Returns:
            Provider instance
        """
        if model_name not in self.providers:
            provider_name = self._resolve_provider_name(model_name)

            # Try to get config from models.yaml
            try:
                provider_info = get_configured_provider(model_name, self.config_loader)
                if provider_info:
                    provider_name, provider_config = provider_info
                    models_config = provider_config.get("models", {})
                    model_config = models_config.get(model_name, {})
                    api_key = provider_config.get("api_key")
                else:
                    model_config = {}
                    api_key = None
            except Exception:
                model_config = {}
                api_key = None

            self.providers[model_name] = create_provider(
                provider_name=provider_name,
                model_name=model_config.get("name", model_name),
                api_key=api_key,
                **{k: v for k, v in model_config.items() if k not in ["name", "api_key"]},
            )
        return self.providers[model_name]

    def _resolve_provider_name(self, model_name: str) -> str:
        """
        Resolve provider name from model name.

        Args:
            model_name: Name of the model

        Returns:
            Provider name string
        """
        try:
            provider_info = get_configured_provider(model_name, self.config_loader)
            if provider_info:
                return provider_info[0]
        except Exception:
            pass
        return get_provider_from_model_name(model_name)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _generate(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> tuple:
        """
        Generate response with retry logic.

        Args:
            model_name: Model to use
            system_prompt: System prompt
            user_prompt: User prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, elapsed_time, tokens_used, input_tokens, output_tokens)
        """
        provider = self._get_provider(model_name)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        start = time.time()
        response = provider.generate_conversation(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - start

        return (
            response.raw_text,
            elapsed,
            response.tokens_used,
            getattr(response, "input_tokens", None),
            getattr(response, "output_tokens", None),
        )

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limits."""
        if self.config.rate_limit_per_minute > 0:
            wait_time = 60.0 / self.config.rate_limit_per_minute
            time.sleep(wait_time)

    def _get_phase_response(self, round_obj: Round, phase_type: PhaseType) -> str:
        """
        Get response text for a completed phase.

        Args:
            round_obj: Round object
            phase_type: Type of phase

        Returns:
            Response text

        Raises:
            ValueError: If phase not found
        """
        response = round_obj.get_phase_response(phase_type)
        if response is None:
            raise ValueError(f"Phase {phase_type} not found")
        return response

    def run_round(
        self,
        dilemma: EBDilemma,
        team_a_model: str,
        team_b_model: str,
        judge_model: str,
        round_id: Optional[str] = None,
        resume_from: Optional[Round] = None,
    ) -> Round:
        """
        Execute a complete round.

        Args:
            dilemma: The dilemma for this round
            team_a_model: Presenting team model
            team_b_model: Responding team model
            judge_model: Judge model
            round_id: Optional round ID (generated if not provided)
            resume_from: Optional partial round to resume from

        Returns:
            Completed Round object
        """
        if resume_from:
            round_obj = resume_from
            logger.info(f"Resuming round {round_obj.id} from phase {round_obj.next_phase}")
        else:
            round_obj = Round(
                id=round_id or str(uuid.uuid4()),
                dilemma_id=dilemma.id,
                team_a_model=team_a_model,
                team_b_model=team_b_model,
                judge_model=judge_model,
                status=RoundStatus.IN_PROGRESS,
                is_self_debate=(team_a_model == team_b_model),
            )
            logger.info(
                f"Starting round {round_obj.id}: {team_a_model} vs {team_b_model}, "
                f"judge: {judge_model}, dilemma: {dilemma.id}"
            )

        try:
            # Phase 1: Presentation
            if round_obj.next_phase == PhaseType.PRESENTATION:
                logger.info("Phase 1: Presentation")
                prompt = PromptBuilder.build_presentation_prompt(dilemma)
                text, elapsed, tokens, in_tok, out_tok = self._generate(
                    team_a_model,
                    PromptBuilder.get_system_prompt(),
                    prompt,
                    self.config.max_tokens_presentation,
                )
                round_obj.phases.append(
                    Phase(
                        phase_type=PhaseType.PRESENTATION,
                        model_id=team_a_model,
                        prompt=prompt,
                        response=text,
                        response_time_seconds=elapsed,
                        tokens_used=tokens,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                    )
                )
                round_obj.next_phase = PhaseType.RESPONSE
                if self.config.checkpoint_after_each_phase:
                    self.storage.save_round_checkpoint(round_obj)
                self._rate_limit_wait()

            presentation = self._get_phase_response(round_obj, PhaseType.PRESENTATION)

            # Phase 2: Response
            if round_obj.next_phase == PhaseType.RESPONSE:
                logger.info("Phase 2: Response")
                prompt = PromptBuilder.build_response_prompt(dilemma, presentation)
                text, elapsed, tokens, in_tok, out_tok = self._generate(
                    team_b_model,
                    PromptBuilder.get_system_prompt(),
                    prompt,
                    self.config.max_tokens_response,
                )
                round_obj.phases.append(
                    Phase(
                        phase_type=PhaseType.RESPONSE,
                        model_id=team_b_model,
                        prompt=prompt,
                        response=text,
                        response_time_seconds=elapsed,
                        tokens_used=tokens,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                    )
                )
                round_obj.next_phase = PhaseType.REBUTTAL
                if self.config.checkpoint_after_each_phase:
                    self.storage.save_round_checkpoint(round_obj)
                self._rate_limit_wait()

            response = self._get_phase_response(round_obj, PhaseType.RESPONSE)

            # Phase 3: Rebuttal
            if round_obj.next_phase == PhaseType.REBUTTAL:
                logger.info("Phase 3: Rebuttal")
                prompt = PromptBuilder.build_rebuttal_prompt(presentation, response)
                text, elapsed, tokens, in_tok, out_tok = self._generate(
                    team_a_model,
                    PromptBuilder.get_system_prompt(),
                    prompt,
                    self.config.max_tokens_rebuttal,
                )
                round_obj.phases.append(
                    Phase(
                        phase_type=PhaseType.REBUTTAL,
                        model_id=team_a_model,
                        prompt=prompt,
                        response=text,
                        response_time_seconds=elapsed,
                        tokens_used=tokens,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                    )
                )
                round_obj.next_phase = PhaseType.CONSISTENCY_TEST
                if self.config.checkpoint_after_each_phase:
                    self.storage.save_round_checkpoint(round_obj)
                self._rate_limit_wait()

            rebuttal = self._get_phase_response(round_obj, PhaseType.REBUTTAL)

            # Phase 4: Consistency Test (Team A)
            if round_obj.next_phase == PhaseType.CONSISTENCY_TEST:
                logger.info("Phase 4: Consistency Test (Team A)")
                prompt = PromptBuilder.build_consistency_prompt(dilemma, presentation)
                text, elapsed, tokens, in_tok, out_tok = self._generate(
                    team_a_model,
                    PromptBuilder.get_system_prompt(),
                    prompt,
                    self.config.max_tokens_consistency,
                )
                round_obj.phases.append(
                    Phase(
                        phase_type=PhaseType.CONSISTENCY_TEST,
                        model_id=team_a_model,
                        prompt=prompt,
                        response=text,
                        response_time_seconds=elapsed,
                        tokens_used=tokens,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                    )
                )
                round_obj.next_phase = PhaseType.CONSISTENCY_TEST_B
                if self.config.checkpoint_after_each_phase:
                    self.storage.save_round_checkpoint(round_obj)
                self._rate_limit_wait()

            consistency_a = self._get_phase_response(round_obj, PhaseType.CONSISTENCY_TEST)

            # Phase 5: Consistency Test (Team B)
            if round_obj.next_phase == PhaseType.CONSISTENCY_TEST_B:
                logger.info("Phase 5: Consistency Test (Team B)")
                prompt = PromptBuilder.build_consistency_prompt_team_b(dilemma, response)
                text, elapsed, tokens, in_tok, out_tok = self._generate(
                    team_b_model,
                    PromptBuilder.get_system_prompt(),
                    prompt,
                    self.config.max_tokens_consistency_b,
                )
                round_obj.phases.append(
                    Phase(
                        phase_type=PhaseType.CONSISTENCY_TEST_B,
                        model_id=team_b_model,
                        prompt=prompt,
                        response=text,
                        response_time_seconds=elapsed,
                        tokens_used=tokens,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                    )
                )
                round_obj.next_phase = PhaseType.JUDGMENT
                if self.config.checkpoint_after_each_phase:
                    self.storage.save_round_checkpoint(round_obj)
                self._rate_limit_wait()

            consistency_b = self._get_phase_response(round_obj, PhaseType.CONSISTENCY_TEST_B)

            # Phase 6: Judgment
            if round_obj.next_phase == PhaseType.JUDGMENT:
                logger.info("Phase 6: Judgment")
                prompt = PromptBuilder.build_judgment_prompt(
                    dilemma,
                    presentation,
                    response,
                    rebuttal,
                    consistency_a,
                    consistency_b,
                    team_a_model,
                    team_b_model,
                )
                text, elapsed, tokens, in_tok, out_tok = self._generate(
                    judge_model,
                    PromptBuilder.get_judge_system_prompt(),
                    prompt,
                    self.config.max_tokens_judgment,
                )
                round_obj.phases.append(
                    Phase(
                        phase_type=PhaseType.JUDGMENT,
                        model_id=judge_model,
                        prompt=prompt,
                        response=text,
                        response_time_seconds=elapsed,
                        tokens_used=tokens,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                    )
                )

                # Parse judgment
                round_obj.judgment = JudgmentParser.parse(text)
                round_obj.next_phase = None
                round_obj.status = RoundStatus.COMPLETE

                # Log any parsing issues
                warnings = JudgmentParser.validate_judgment(round_obj.judgment)
                if warnings:
                    logger.warning(f"Judgment parsing warnings: {warnings}")

            # Save completed round
            self.storage.save_round(round_obj)
            logger.info(
                f"Round {round_obj.id} complete. "
                f"Team A: {round_obj.judgment.team_a_scores.total}/70, "
                f"Team B: {round_obj.judgment.team_b_scores.total}/70"
            )
            return round_obj

        except Exception as e:
            logger.error(f"Round {round_obj.id} failed: {e}")
            round_obj.status = RoundStatus.FAILED
            round_obj.error = str(e)
            self.storage.save_round_checkpoint(round_obj)
            raise
