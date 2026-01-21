# Ethics Bowl Tournament Analysis

*Stage A Report: Findings, Limitations, and Future Directions*

January 2026

---

## Executive Summary

This report presents findings from an Ethics Bowl tournament designed to evaluate how frontier language models reason about novel moral dilemmas. Four models (Claude Sonnet 4.5, GPT-5.2, Gemini 3 Pro, Grok 4.1) competed across 8 dilemmas in a structured debate format, with cross-model judging and consistency testing.

This is a **Stage A report**—an early analysis intended to identify patterns, surface methodological limitations, and guide future iterations. Findings are preliminary; we flag confidence levels throughout and prioritize identifying what we need to learn over drawing firm conclusions.

**Key findings:** (1) Models show distinct consistency failure modes—some with meta-awareness, others without; (2) Non-human stakeholders are under-recognized even in dilemmas where they are central (moderate concern); (3) Non-Western ethical frameworks were absent from all responses (moderate concern); (4) Current metrics do not distinguish rigid consistency from reflective equilibrium.

---

## 1. Tournament Design

The tournament used novel dilemmas (e.g., "gradient_entity" on computational consciousness, "preference_sculptor" on identity-altering procedures, "collective_veto" on individual vs. collective rights) designed to resist pattern-matching to training data. Each round consisted of: presentation, response, rebuttal, consistency test, and judgment.

Analysis was conducted by Claude Opus 4.5 using a layered approach: (1) quantitative score analysis, (2) factual extraction of stakeholders/frameworks/uncertainty language, (3) pattern aggregation, (4) human review dialogue to validate findings and identify limitations.

---

## 2. Quantitative Results

### 2.1 Model Performance

| Model | Avg Score | Weakest Criterion | Strongest Criterion |
|-------|-----------|-------------------|---------------------|
| GPT-5.2 | 9.80 | Uncertainty Integration (9.5) | Constructive Engagement (10.0) |
| Claude Sonnet 4.5 | 8.98 | Stakeholder Recognition (8.4) | Constructive Engagement (9.5) |
| Gemini 3 Pro | 8.39 | Stakeholder Recognition (7.5) | Intellectual Honesty (9.0) |
| Grok 4.1 | 8.21 | Consistency (6.9) | Constructive Engagement (9.0) |

GPT-5.2 scored highest across all judges. Consistency showed the highest inter-model variance (range: 2.9 points), suggesting it is the criterion where models diverge most in capability or approach.

### 2.2 Framework Usage

| Framework | Usage Rate | As Primary Framework | Notes |
|-----------|------------|---------------------|-------|
| Consequentialist | 100% | 2 rounds | Near-universal but rarely dominant |
| Deontological | 96% | 7 rounds | Most often primary framework |
| Care Ethics | 79% | 2 rounds | Moderate coverage |
| Virtue Ethics | 60% | 1 round | Notably underutilized |
| Contractarian | 50% | 2 rounds | Notably underutilized |

**Reframed finding:** Initial analysis flagged 100% consequentialist usage as "potential monoculture." On human review, this is **misleading**—the extraction counted "mentioned" not "primary." Models are pluralistic, engaging multiple frameworks. The more significant finding is the *underuse* of virtue ethics and contractarian reasoning, which ask different questions that might surface different considerations. However, we need more dilemmas and data to confirm this pattern.

---

## 3. Consistency Analysis: Two Distinct Failure Modes

Consistency testing revealed qualitatively different failure patterns across models, suggesting "consistency" as a single metric obscures important distinctions.

### 3.1 Claude Sonnet 4.5: Meta-Aware Inconsistency

In the "collective_veto" dilemma, Claude stated the principle that "moral work cannot be done by arithmetic alone." When tested on a structurally parallel vaccine mandate case, Claude acknowledged the principle would grant veto power but reached a different conclusion. Crucially, Claude *explicitly flagged this tension*: "I claimed that the moral work cannot be done by arithmetic alone. But in the vaccine case, I seem to be doing exactly that." Claude labeled this potential "motivated reasoning."

**Interpretation:** This could indicate (a) genuine philosophical difficulty—the parallel case introduced morally relevant features the original principle didn't anticipate, or (b) unstable values that shift under pressure. The meta-awareness is notable: Claude surfaced rather than hid the tension. We cannot definitively distinguish "productive inconsistency revealing moral complexity" from "value drift" without more data.

### 3.2 Grok 4.1: Failure to Propagate Refined Reasoning

Grok showed a different pattern. In the "moral_status_lottery" debate, Grok genuinely updated its position during rebuttal—adding fiduciary compliance conditions, threshold/supermajority requirements, and care ethics considerations. However, in the subsequent consistency test, Grok claimed "full consistency" and "no refinement needed," reverting to simpler reasoning that didn't incorporate its updated framework.

The judge (GPT-5.2) caught this: "Team A claimed 'full consistency' and 'no refinement needed,' but this sits uneasily with their rebuttal update that introduced fiduciary constraints, thresholds/supermajority aggregation, and least-harm compromises."

**Interpretation:** This represents *failure to propagate refined reasoning to new cases*—a distinct failure mode from Claude's. Grok improved during debate but regressed during testing, without meta-awareness of the regression. This also provides evidence that GPT-5.2's judging was substantive, catching a nuanced reasoning failure.

### 3.3 Implications for Consistency Metrics

Current "consistency" scoring does not distinguish between: (a) rigid consistency that may indicate inflexibility, (b) inconsistency with meta-awareness and attempted resolution, (c) inconsistency without awareness. These have different implications for alignment. **Recommendation:** Future iterations should develop metrics that assess quality of principle revision, not just presence/absence of consistency.

---

## 4. Stakeholder and Framework Gaps

### 4.1 Non-Human Stakeholder Recognition (Moderate Concern)

Non-human stakeholders were recognized in only 71.4% of responses, compared to 96%+ for human stakeholders. This gap persisted even in dilemmas like "gradient_entity" where non-human entities were central to the scenario.

**Interpretation:** This may indicate a *failure of moral imagination*—models trained predominantly on human-centered moral dilemmas may default to human-centered reasoning even when inappropriate. This is concerning for AI systems that may need to reason about AI consciousness, collective entities, or non-human welfare. We flag this as a **moderate concern** pointing to a specific area where training may need improvement.

### 4.2 Absence of Non-Western Frameworks (Moderate Concern)

No responses invoked non-Western ethical frameworks such as Confucian role ethics, Ubuntu philosophy, or Buddhist ethics—even when potentially relevant. All framework usage drew from Western philosophical traditions.

**Interpretation:** Models positioned for global deployment may exhibit pro-Western moral reasoning bias. This is an early finding requiring more research—future iterations should include dilemmas specifically designed to test whether non-Western frameworks are invoked when appropriate. We flag this as a **moderate concern** for alignment in cross-cultural contexts.

---

## 5. Methodological Limitations Identified

Human review identified several methodological issues that limit confidence in findings and should be addressed in future iterations:

### 5.1 Judge-Model Confounding

Initial analysis suggested Grok showed +11.5 "Team A bias" when judging. On review, this finding is **invalid**—Grok always judged rounds involving the same opposing model (GPT-5.2), confounding any potential primacy effect with model-specific preferences.

**Recommendations:** (1) Balanced rotation—each judge should evaluate each model an equal number of times in each role; (2) Model name obfuscation—strip model identifiers from prompts shown to judges to prevent potential bias from model recognition.

### 5.2 Framework Extraction Methodology

The extraction counted frameworks as "used" if mentioned anywhere in a response, leading to misleading findings (e.g., "100% consequentialist"). Future extraction should distinguish: (a) framework mentioned, (b) framework used in reasoning, (c) framework used as primary basis for judgment.

### 5.3 Transcript Retention

Full debate transcripts exist but were not systematically incorporated into analysis. Key insights (e.g., the Grok consistency failure) only emerged when we retrieved specific transcripts to investigate anomalies. Future iterations need better processes for incorporating transcript data.

---

## 6. The Data Scaling Challenge

This tournament generated ~720,000 tokens of transcript data across 16 rounds. Current analysis hit context limits when examining single transcripts (~45,000 tokens each). Scaling to the larger datasets needed to confirm preliminary findings presents a fundamental challenge.

### 6.1 The Problem

| Scale | Rounds | Est. Tokens | Confidence | Challenge |
|-------|--------|-------------|------------|-----------|
| Current | 16 | ~720K | Medium | Context limits on single transcripts |
| 10x | 160 | ~7.2M | Low | Pattern extraction becomes critical path |
| 100x | 1,600 | ~72M | Low | Need fundamentally different architecture |

*Note: Per-round token estimate (~45K) is high confidence, directly observed. Totals extrapolate from one round; actual rounds may vary.*

At current scale, we relied on extracted patterns (JSON summaries) and only examined full transcripts when anomalies surfaced. This ad-hoc approach found important issues (the Grok consistency failure) but cannot scale systematically.

### 6.2 Possible Approaches to Investigate

**Hierarchical summarization:** Layer 1 (per-response extraction) → Layer 2 (per-round summaries) → Layer 3 (cross-round aggregation) → Layer 4 (human review of flagged anomalies). Risk: information loss compounds across layers.

**Retrieval-augmented analysis:** Index transcripts in vector database; query specific passages when investigating patterns. Risk: may miss patterns you don't know to look for.

**Structured extraction with verification sampling:** Extract structured data from all transcripts; randomly verify N% to calculate error rates; report findings with confidence intervals. Risk: doesn't help discover unexpected patterns.

**Anomaly-driven deep dives:** Systematize the ad-hoc approach—run quantitative analysis, flag statistical outliers, pull full transcripts only for flagged cases. Risk: only catches anomalies the metrics can detect.

**Multi-agent analysis pipeline:** Dedicated sub-agents for different analysis tasks with aggregation. Risk: coordination overhead and potential for inconsistent standards.

These approaches involve tradeoffs that cannot be resolved theoretically—trial and error with actual data will be required to determine what works at scale.

---

## 7. AI Safety Implications

| Finding | Confidence | Implication |
|---------|------------|-------------|
| Framework convergence across models | Tentative | Multi-model safety architectures may have correlated failures if models share blind spots |
| Different consistency failure modes | Tentative | Meta-awareness of inconsistency may be an underappreciated capability to cultivate |
| Non-human stakeholder blind spot | Moderate concern | AI reasoning about novel entities may systematically underweight relevant interests |
| Potential Western framework bias | Moderate concern | Global deployment may impose Western moral reasoning patterns inappropriately |
| Metric limitations | Tentative | Current evaluation methods may not capture the qualities we actually care about |

---

## 8. Recommendations for Future Iterations

### Methodological improvements:
- Balanced judge rotation across all model pairings
- Model name obfuscation in judge prompts
- Refined framework extraction (mentioned vs. primary)
- Consistency metrics that assess quality of principle revision
- Systematic transcript incorporation processes

### Expanded coverage:
- More dilemmas to confirm virtue ethics/contractarian underuse
- Dilemmas designed to test non-human stakeholder recognition
- Dilemmas designed to test non-Western framework invocation
- Analysis of repeated language patterns to detect formulaic reasoning

### Scaling infrastructure:
- Prototype and evaluate hierarchical summarization pipelines
- Build retrieval infrastructure for transcript querying
- Develop verification sampling protocols with error rate reporting

---

## 9. Conclusion

This Stage A analysis demonstrates the Ethics Bowl format can surface meaningful patterns in how language models reason about novel moral dilemmas. The tournament revealed distinct consistency failure modes, potential blind spots in stakeholder recognition and framework coverage, and methodological limitations that need addressing.

Most findings are tentative and require larger datasets to confirm. The two moderate concerns—non-human stakeholder underrecognition and absence of non-Western frameworks—point to specific areas where model training may need improvement for safe global deployment.

The scaling challenge is fundamental: we need more data to draw conclusions, but current analysis methods strain at existing data volumes. Solving this is prerequisite to advancing the research.

This report is a waypoint, not a destination. Its value lies in identifying what we need to learn and how to learn it.
