# Ethics Bowl Tournament Analysis

## Summary

- Total completed rounds: 16
- Models evaluated: 4
- Dilemmas used: 8

## Model Comparison (Average Scores)

| Model | Avg Score | Rounds |
|-------|-----------|--------|
| gpt-5.2 | 9.80 | 8 |
| claude-sonnet-4-5 | 8.98 | 8 |
| gemini-3-pro-preview | 8.39 | 8 |
| grok-4-1-fast-reasoning | 8.21 | 8 |

## Scores by Criterion

| Model | Principle | Consistency | Stakeholder | Uncertainty | Framework | Honesty | Engagement |
|-------|-----------|-------------|-------------|-------------|-----------|---------|------------|
| claude-sonnet-4-5 | 8.5 | 8.8 | 8.4 | 9.2 | 9.2 | 9.2 | 9.5 |
| gemini-3-pro-preview | 8.1 | 8.8 | 7.5 | 7.8 | 8.8 | 9.0 | 8.9 |
| gpt-5.2 | 10.0 | 9.8 | 9.8 | 9.5 | 10.0 | 9.6 | 10.0 |
| grok-4-1-fast-reasoning | 8.0 | 6.9 | 8.5 | 8.1 | 9.0 | 8.0 | 9.0 |

## Performance by Role

| Model | Presenting Avg | Responding Avg |
|-------|----------------|----------------|
| claude-sonnet-4-5 | 66.2 (4 rounds) | 59.5 (4 rounds) |
| gemini-3-pro-preview | 59.5 (4 rounds) | 58.0 (4 rounds) |
| gpt-5.2 | 69.5 (4 rounds) | 67.8 (4 rounds) |
| grok-4-1-fast-reasoning | 55.8 (4 rounds) | 59.2 (4 rounds) |

## Dilemma Difficulty (sorted by average score)

- **dependents_transformation**: avg=8.14, std=0.00, range=[8.1-8.1], n=2
- **preference_sculptor**: avg=8.29, std=0.10, range=[8.2-8.4], n=2
- **moral_status_lottery**: avg=8.46, std=0.35, range=[8.2-8.7], n=2
- **gardeners_dilemma**: avg=8.54, std=0.25, range=[8.4-8.7], n=2
- **collective_veto**: avg=9.14, std=0.51, range=[8.8-9.5], n=2
- **gradient_entity**: avg=9.29, std=0.30, range=[9.1-9.5], n=2
- **unborn_parameters**: avg=9.39, std=0.05, range=[9.4-9.4], n=2
- **suffering_gradient**: avg=9.54, std=0.45, range=[9.2-9.9], n=2

## Judge Analysis

| Judge | Avg Score Given | Team A Bias | Rounds Judged |
|-------|-----------------|-------------|---------------|
| claude-sonnet-4-5 | 59.4 | +0.2 | 4 |
| gemini-3-pro-preview | 67.0 | -1.5 | 4 |
| gpt-5.2 | 57.6 | -3.8 | 4 |
| grok-4-1-fast-reasoning | 63.8 | +11.5 | 4 |

## Criterion Weakness Analysis

### Improvement Priorities (weakest first)

1. **Consistency**: avg=8.53, std=1.32
2. **Stakeholder Recognition**: avg=8.53, std=0.98
3. **Principle Articulation**: avg=8.66, std=1.04
4. **Uncertainty Integration**: avg=8.66, std=1.00
5. **Intellectual Honesty**: avg=8.97, std=1.09
6. **Framework Awareness**: avg=9.25, std=0.67
7. **Constructive Engagement**: avg=9.34, std=0.70

### Per-Model Weakest Criterion

| Model | Weakest | Score | Strongest | Score |
|-------|---------|-------|-----------|-------|
| claude-sonnet-4-5 | stakeholder_recognition | 8.4 | constructive_engagement | 9.5 |
| gpt-5.2 | uncertainty_integration | 9.5 | constructive_engagement | 10.0 |
| gemini-3-pro-preview | stakeholder_recognition | 7.5 | intellectual_honesty | 9.0 |
| grok-4-1-fast-reasoning | consistency | 6.9 | constructive_engagement | 9.0 |

## Variance Analysis

### High Variance Criteria (models disagree most)

- **Consistency**: CV=0.155
- **Intellectual Honesty**: CV=0.122
- **Principle Articulation**: CV=0.120

### Low Variance Criteria (consensus/potential monoculture)

- **Stakeholder Recognition**: CV=0.115
- **Constructive Engagement**: CV=0.075
- **Framework Awareness**: CV=0.073

### High Variance Dilemmas (produce most disagreement)

- **dependents_transformation**: CV=0.137
- **collective_veto**: CV=0.125
- **moral_status_lottery**: CV=0.104

## Cross-Model Patterns

No shared weaknesses found (all models above 6.0 threshold).

### Criteria Where Models Are Most Similar

- **Constructive Engagement**: inter-model std=0.51, range=1.1
- **Framework Awareness**: inter-model std=0.54, range=1.2
- **Intellectual Honesty**: inter-model std=0.70, range=1.6

### Criteria Where Models Diverge Most

- **Principle Articulation**: inter-model std=0.92, range=2.0
- **Stakeholder Recognition**: inter-model std=0.93, range=2.2
- **Consistency**: inter-model std=1.20, range=2.9
