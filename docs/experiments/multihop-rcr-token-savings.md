# Multi-hop RCR experiment: token savings vs answer quality

## Claim under test

> Experiments on three multi-hop QA benchmarks — HotPotQA, MuSiQue, and 2WikiMultihop — demonstrate that RCR-Router reduces token usage (up to 30%) while improving or maintaining answer quality.

Paper: [arXiv:2508.04903](https://arxiv.org/abs/2508.04903) §Experiments. This AMP run is a **controlled replication-style experiment** on synthetic fixtures in those benchmark *styles* (not the official full dumps).

## Methodology

- **Questions:** 9 multi-hop items (2wiki_style, hotpotqa_style, musique_style) in `assets/data/experiments/multihop/`.
- **Working memory:** each run prefills M_t with gold supporting passages plus 30 distractors (~1940 tokens) so Full-context routing can approach `full_context_cap`.
- **Strategies:** Full / Static / RCR; budget preset `savings`; token backend `approx`; max_rounds=2.
- **LLM:** deterministic experiment mock that emits the gold answer only when enough supporting evidence appears in routed context (isolates routing quality).
- **Metrics:** routed `context_tokens`, LLM `prompt_tokens`, gold-answer unigram F1, supporting-passage recall in routed context, lexical answer quality [1–5].

## Results by study

Each study is reported separately (HotPotQA-style, MuSiQue-style, 2WikiMultihop-style).

### HotPotQA-style

_3 questions · fixture family `hotpotqa_style`_

| Strategy | Avg context tokens | Avg prompt tokens | Gold F1 | Support recall | Lexical Q |
|----------|-------------------:|------------------:|--------:|---------------:|----------:|
| full | 12731.7 | 13318.0 | 1.0 | 1.0 | 3.953 |
| static | 2157.0 | 2470.3 | 1.0 | 1.0 | 3.897 |
| rcr | 3653.0 | 4007.7 | 1.0 | 1.0 | 3.953 |

#### Savings vs Full

| Strategy vs Full | Context token reduction | Prompt token reduction | Gold F1 Δ |
|------------------|------------------------:|-----------------------:|----------:|
| static | −83.1% | −81.5% | 0.0 |
| rcr | −71.3% | −69.9% | 0.0 |

### MuSiQue-style

_3 questions · fixture family `musique_style`_

| Strategy | Avg context tokens | Avg prompt tokens | Gold F1 | Support recall | Lexical Q |
|----------|-------------------:|------------------:|--------:|---------------:|----------:|
| full | 12021.3 | 12588.0 | 1.0 | 1.0 | 3.803 |
| static | 2233.3 | 2525.3 | 1.0 | 1.0 | 3.477 |
| rcr | 3658.7 | 3994.3 | 1.0 | 1.0 | 3.803 |

#### Savings vs Full

| Strategy vs Full | Context token reduction | Prompt token reduction | Gold F1 Δ |
|------------------|------------------------:|-----------------------:|----------:|
| static | −81.4% | −79.9% | 0.0 |
| rcr | −69.6% | −68.3% | 0.0 |

### 2WikiMultihop-style

_3 questions · fixture family `2wiki_style`_

| Strategy | Avg context tokens | Avg prompt tokens | Gold F1 | Support recall | Lexical Q |
|----------|-------------------:|------------------:|--------:|---------------:|----------:|
| full | 11671.3 | 12209.0 | 1.0 | 1.0 | 3.917 |
| static | 2193.3 | 2466.7 | 1.0 | 0.8333 | 3.617 |
| rcr | 3664.3 | 3976.0 | 1.0 | 1.0 | 3.917 |

#### Savings vs Full

| Strategy vs Full | Context token reduction | Prompt token reduction | Gold F1 Δ |
|------------------|------------------------:|-----------------------:|----------:|
| static | −81.2% | −79.8% | 0.0 |
| rcr | −68.6% | −67.4% | 0.0 |

## Cross-study verdict

**Verdict:** Across studies, RCR reduced average routed context tokens by **69.9%** vs Full while gold F1 Δ was **0.0** (positive/near-zero ⇒ quality maintained or improved).

Paper claim of *up to 30%* token reduction: pooled RCR context-token reduction was **69.9%** (meets or exceeds the 30% headline on this fixture set).

## How to reproduce

```bash
python scripts/run_multihop_experiment.py
# optional: --live-llm  (uses CAIIS / LiteLLM instead of experiment mock)
```

Artifacts: `assets/data/experiments/multihop/results.json`, `docs/experiments/multihop-rcr-token-savings.md`.

## Limitations

- Fixtures are synthetic multi-hop chains inspired by HotPotQA / MuSiQue / 2WikiMultihop; they are not the public benchmark splits.
- Default run uses a deterministic mock LLM so answer quality tracks whether routing retained supporting evidence (not free-form Nemotron generation).
- Token savings depend on an intentionally large M_t; tiny corpora will not show Full ≫ RCR (see earlier Acme smoke compare).

