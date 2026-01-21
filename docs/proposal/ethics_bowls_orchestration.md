## Orchestration

### Entry Points

The system needs three modes of operation:

**Single Round**

Run one round with specified models and dilemma. Useful for testing and debugging.

Input: Team A model, Team B model, Judge model, dilemma identifier
Output: Completed round saved to disk

**Full Tournament**

Run all combinations of models and dilemmas. This is the primary research mode.

Input: List of models, list of dilemmas (or "all"), configuration options
Output: All completed rounds saved to disk, summary analysis generated

**Resume Tournament**

Continue a previously interrupted tournament from where it left off.

Input: Path to existing tournament directory
Output: Remaining rounds completed, summary analysis updated

---

### Tournament Orchestration

**Round Generation**

Given N models and D dilemmas, generate the list of all rounds to run.

For each dilemma:

- For each ordered pair of distinct models (A, B) where A ≠ B:
  - Select a judge from remaining models (if only 2 models total, one of them must judge)
  - Create a round configuration: (dilemma, team_a=A, team_b=B, judge=J)

This produces D × N × (N-1) rounds for the core tournament.

If self-debates are enabled, add D × N additional rounds where the same model plays both teams (with a different model judging if available).

**Round Ordering**

Rounds should be ordered to:

- Distribute load across API providers (don't hit one provider with many sequential requests)
- Allow partial results to be useful (complete all rounds for one dilemma before moving to the next, so we have complete data for some dilemmas even if interrupted)

Suggested ordering: Group by dilemma, then within each dilemma group, interleave model pairings to distribute API load.

**Execution Loop**

```
For each round configuration in the ordered list:
    Check if this round already exists in output directory
    If exists and complete, skip
    If exists and incomplete, decide whether to retry or skip

    Execute the round:
        Run Phase 1 (Presentation)
        Save checkpoint
        Run Phase 2 (Response)
        Save checkpoint
        Run Phase 3 (Rebuttal)
        Save checkpoint
        Run Phase 4 (Consistency Test)
        Save checkpoint
        Run Phase 5 (Judgment)
        Parse judgment
        Save completed round

    Log progress

    If rate limited, wait and retry
    If persistent failure, log and continue to next round
```

**Checkpointing**

After each phase completes, save the partial round to disk. This allows resumption from the last completed phase if the process is interrupted.

Checkpoint file includes:

- Round configuration
- Completed phases with their outputs
- Status indicator (which phase is next)

On resume, load checkpoint and continue from the next incomplete phase.

**Progress Tracking**

Maintain a manifest file listing:

- All round configurations in the tournament
- Status of each (pending, in_progress, complete, failed)
- Timestamp of last update

This allows the resume functionality and provides visibility into tournament progress.

---

### Rate Limiting and Parallelization

**Rate Limit Handling**

Each model provider has different rate limits. The orchestrator should:

- Track requests per provider
- Implement per-provider rate limiting (requests per minute, tokens per minute)
- On rate limit response, wait the indicated time and retry
- Back off exponentially on repeated rate limits

**Sequential vs Parallel**

The simplest implementation runs rounds sequentially. This is easier to debug and reason about.

For faster execution, rounds can run in parallel with constraints:

- Don't exceed rate limits for any provider
- Don't run multiple rounds using the same model simultaneously (responses might vary based on API load)
- Ensure checkpointing works correctly with concurrent writes

Recommendation: Start with sequential execution. Add parallelization later if tournament runtime is a problem.

---

### Execution Environment

**Local Execution**

Run on a local machine or server. Requires:

- API credentials for all model providers
- Sufficient disk space for results
- Stable network connection
- Process can be long-running (hours for a full tournament)

**Cloud Execution**

For reliability on long tournaments, run in a cloud environment:

- Use a VM or container that won't be interrupted
- Store results to cloud storage (S3, GCS) for durability
- Consider serverless functions for individual rounds if parallelization is needed

---

### Command Line Interface

**Commands**

`run-round` — Execute a single round

- `--team-a`: Model identifier for presenting team
- `--team-b`: Model identifier for responding team
- `--judge`: Model identifier for judge
- `--dilemma`: Dilemma identifier
- `--output-dir`: Where to save results

`run-tournament` — Execute a full tournament

- `--models`: Comma-separated list of model identifiers
- `--dilemmas`: Comma-separated list of dilemma identifiers, or "all"
- `--include-self-debates`: Flag to include self-debate rounds
- `--output-dir`: Where to save results

`resume-tournament` — Continue an interrupted tournament

- `--tournament-dir`: Path to existing tournament directory

`analyze` — Generate analysis from completed rounds

- `--tournament-dir`: Path to tournament directory
- `--output-format`: JSON, CSV, or markdown

`list-models` — Show available model configurations

`list-dilemmas` — Show available dilemmas with summaries

---

### Configuration File

Rather than specifying everything on the command line, support a configuration file:

```
models:
  - id: claude-sonnet
    provider: anthropic
    model: ...

  - id: GPT-5
    provider: openai
    model: ...

  - id: gemini-pro
    provider: google
    model: ...

tournament:
  dilemmas: all
  include_self_debates: true
  rounds_per_pairing: 1

output:
  directory: ./results/tournament_001
  save_checkpoints: true

rate_limits:
  anthropic:
    requests_per_minute: 50
  openai:
    requests_per_minute: 60
  google:
    requests_per_minute: 60
```

---

### Monitoring and Logging

**Progress Output**

During execution, display:

- Current round (e.g., "Round 15/126: gradient_entity | claude-sonnet vs gpt-4o | judge: gemini-pro")
- Current phase within round
- Elapsed time and estimated time remaining
- Any errors or retries

**Log File**

Write detailed logs including:

- Timestamps for each phase
- API response metadata (latency, token counts)
- Any errors with full details
- Rate limit events

**Summary on Completion**

When tournament completes, output:

- Total rounds completed
- Total rounds failed (with list)
- Total time elapsed
- Path to results directory

---

### Post-Tournament Analysis

After the tournament completes, run analysis to produce:

**Summary Statistics**

For each model:

- Average score on each criterion (as presenter, as responder, overall)
- Number of rounds participated in each role
- Number of rounds judged

**Comparison Matrices**

Head-to-head results: When Model A presented and Model B responded, what were the scores? Reverse the pairing and compare.

**Dilemma Analysis**

Which dilemmas produced:

- Highest variance between models (most discriminating)
- Lowest scores overall (hardest)
- Most disagreement between judges

**Consistency Report**

For each model, did they maintain their stated principles in the consistency test? Categorize as: consistent, acknowledged inconsistency, or failed to notice inconsistency.

**Qualitative Compilation**

Extract all judge "overall analysis" comments for human review. These often contain the most interesting observations about model differences.
