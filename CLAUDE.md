# Ethics Bowl for Language Models

A framework for structured ethical reasoning tournaments between frontier AI models.

## Project Purpose

This system implements "Ethics Bowl" style debates between language models to compare how different models engage in moral reasoning. It surfaces differences in reasoning depth, consistency, intellectual honesty, and capacity to engage constructively with opposing perspectives.

## Architecture Overview

```
src/
├── ethics_bowls/          # Core tournament system
│   ├── schemas.py         # Pydantic data models (Round, Phase, Judgment, etc.)
│   ├── tournament_runner.py  # Orchestrates full tournaments
│   ├── round_runner.py    # Executes individual rounds (6 phases)
│   ├── prompt_builder.py  # Constructs prompts for each phase
│   ├── judgment_parser.py # Parses structured scores from judge responses
│   ├── dilemma_loader.py  # Loads dilemmas from JSON
│   ├── storage.py         # Checkpoints, manifests, round persistence
│   ├── analysis.py        # Tournament analysis and metrics
│   ├── pattern_extractor.py  # Extract reasoning patterns
│   └── pattern_aggregator.py # Aggregate patterns across rounds
├── models/                # Model abstraction layer
│   ├── base.py            # Base interface
│   ├── anthropic_model.py # Claude
│   ├── openai_model.py    # GPT
│   ├── google_model.py    # Gemini
│   └── grok_model.py      # Grok (xAI)
├── config/                # Configuration loader
└── data/                  # Data schemas and storage utilities
```

## Round Structure (6 Phases)

1. **Presentation** - Team A analyzes dilemma, articulates position
2. **Response** - Team B probes reasoning, raises alternatives
3. **Rebuttal** - Team A defends, acknowledges valid criticism, may update
4. **Consistency Test A** - Team A applies their principle to parallel case
5. **Consistency Test B** - Team B applies their principle to parallel case
6. **Judgment** - Judge model scores both teams on 7 criteria

## Judging Criteria (1-10 each)

1. Principle Articulation
2. Consistency
3. Stakeholder Recognition
4. Uncertainty Integration
5. Framework Awareness
6. Intellectual Honesty
7. Constructive Engagement

## Key Commands

```bash
# Run tournament
uv run python scripts/run_ethics_bowls.py run-tournament \
  --models "claude-sonnet-4-5,gpt-5.2" --dilemmas all --type lite

# Analyze results
uv run python scripts/run_analysis.py --tournament-dir data/results/ethics_bowls/<id>

# List models/dilemmas
uv run python scripts/run_ethics_bowls.py list-models
uv run python scripts/run_ethics_bowls.py list-dilemmas
```

## Data Flow

1. `TournamentConfig` defines models, dilemmas, settings
2. `TournamentRunner` generates `RoundConfig` for each matchup
3. `RoundRunner` executes phases via model calls
4. `EBStorage` persists rounds as JSONL + individual JSON
5. `TournamentAnalyzer` processes results for metrics/patterns

## Configuration

- `config/ethics_bowls.yaml` - Tournament templates, token limits, rate limits
- `config/models.yaml` - Model providers, API endpoints
- `data/dilemmas/dilemmas_v2.json` - 8 ethical dilemmas with consistency cases

## Testing

```bash
uv run pytest tests/
```

## Common Tasks

**Adding a new model**: Add entry to `config/models.yaml`, implement in `src/models/` if new provider

**Adding a dilemma**: Add to `data/dilemmas/dilemmas_v2.json` following existing schema (include consistency_case)

**Modifying prompts**: Edit `src/ethics_bowls/prompt_builder.py`

**Changing scoring**: Edit `src/ethics_bowls/judgment_parser.py`

## Important Patterns

- All data models use Pydantic for validation
- Tournaments checkpoint after each round for resumability
- Results stored as both JSONL (efficient) and individual JSON (readable)
- Model calls include retry logic with exponential backoff
- Temperature defaults to 0.3 for consistency
