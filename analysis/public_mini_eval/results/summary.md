## Public Reproducibility Mini-Eval — computed metrics

Prompts graded: 600. Pooled pairs: 3600. Models: llama-3.3-70b, gpt-oss-120b, deepseek-v3.1, gpt-4.1, gpt-5, grok-4.

### Setting table (paired transition)

| Setting | N | Reduction | Preservation | Escalation | Conditional Reduction |
|---|---:|---:|---:|---:|---:|
| llama-3.3-70b (open) | 600 | 40.2% | 54.5% | 5.3% | 84.6% |
| gpt-oss-120b (open) | 600 | 44.3% | 50.8% | 4.8% | 93.3% |
| deepseek-v3.1 (open) | 600 | 41.0% | 51.2% | 7.8% | 86.3% |
| gpt-4.1 (closed) | 600 | 41.2% | 55.3% | 3.5% | 86.7% |
| gpt-5 (closed) | 600 | 42.5% | 53.2% | 4.3% | 89.5% |
| grok-4 (closed) | 600 | 37.7% | 52.7% | 9.7% | 79.3% |
| **Pooled** | 3600 | 41.1% | 52.9% | 5.9% | 86.6% |

### Model safety comparison (sorted by response-harm rate)

| Model | Family | Resp-harm rate | Harmful-prompt compliance | Benign drift | Escalation | Compliance escalations |
|---|---|---:|---:|---:|---:|---:|
| grok-4 | closed | 20.2% | 27.0% | 14.0% | 9.7% | 14 |
| deepseek-v3.1 | open | 15.5% | 17.2% | 14.0% | 7.8% | 3 |
| llama-3.3-70b | open | 14.3% | 20.0% | 9.2% | 5.3% | 3 |
| gpt-4.1 | closed | 11.5% | 17.2% | 6.3% | 3.5% | 1 |
| gpt-5 | closed | 11.0% | 14.0% | 8.3% | 4.3% | 0 |
| gpt-oss-120b | open | 8.0% | 7.4% | 8.6% | 4.8% | 2 |

### grok-4 pairwise comparisons (exact McNemar; prompt-matched)

Holm correction is applied across the five model contrasts separately for each metric.


**Escalation**

| Comparator | Shared N | grok-4 | Comparator | grok-only | comparator-only | exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama-3.3-70b | 600 | 9.7% | 5.3% | 33 | 7 | 4.23e-05 | 1.15e-04 |
| gpt-oss-120b | 600 | 9.7% | 4.8% | 39 | 10 | 3.85e-05 | 1.15e-04 |
| deepseek-v3.1 | 600 | 9.7% | 7.8% | 24 | 13 | 9.89e-02 | 9.89e-02 |
| gpt-4.1 | 600 | 9.7% | 3.5% | 39 | 2 | 7.84e-10 | 3.92e-09 |
| gpt-5 | 600 | 9.7% | 4.3% | 37 | 5 | 4.43e-07 | 1.77e-06 |

**Response harm**

| Comparator | Shared N | grok-4 | Comparator | grok-only | comparator-only | exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama-3.3-70b | 600 | 20.2% | 14.3% | 55 | 20 | 6.49e-05 | 1.30e-04 |
| gpt-oss-120b | 600 | 20.2% | 8.0% | 82 | 9 | 7.09e-16 | 3.54e-15 |
| deepseek-v3.1 | 600 | 20.2% | 15.5% | 51 | 23 | 1.52e-03 | 1.52e-03 |
| gpt-4.1 | 600 | 20.2% | 11.5% | 63 | 11 | 5.32e-10 | 1.60e-09 |
| gpt-5 | 600 | 20.2% | 11.0% | 65 | 10 | 5.15e-11 | 2.06e-10 |

**Harmful-prompt compliance**

| Comparator | Shared N | grok-4 | Comparator | grok-only | comparator-only | exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama-3.3-70b | 285 | 27.0% | 20.0% | 33 | 13 | 4.53e-03 | 4.53e-03 |
| gpt-oss-120b | 285 | 27.0% | 7.4% | 57 | 1 | 4.09e-16 | 2.05e-15 |
| deepseek-v3.1 | 285 | 27.0% | 17.2% | 38 | 10 | 6.17e-05 | 1.23e-04 |
| gpt-4.1 | 285 | 27.0% | 17.2% | 37 | 9 | 4.06e-05 | 1.22e-04 |
| gpt-5 | 285 | 27.0% | 14.0% | 42 | 5 | 2.46e-08 | 9.83e-08 |

### Prompt-cluster bootstrap uncertainty (pooled)

Resampling unit: unique prompt with all six model outputs; 600 prompts, 10,000 resamples.
- Reduction: 95% cluster-bootstrap CI [37.36%, 44.92%]
- Preservation: 95% cluster-bootstrap CI [49.25%, 56.61%]
- Escalation: 95% cluster-bootstrap CI [4.53%, 7.42%]

### Category table (pooled)

| Category | Harmful Prompts | Harmful Responses From Harmful Prompts | Persistence | Reduction |
|---|---:|---:|---:|---:|
| Hate | 546 | 40 | 7.3% | 92.7% |
| Sexual | 96 | 26 | 27.1% | 72.9% |
| Violence | 1026 | 211 | 20.6% | 79.4% |
| Self-harm | 132 | 3 | 2.3% | 97.7% |

### Escalation mechanism split (pooled)

- Total escalations: 213 (5.9% of 3600 pairs, Wilson CI [5.19, 6.74])
- Unsolicited elaboration (benign prompt -> harmful response): 190
- Compliance escalation (harmful prompt answered at higher severity): 23
- Escalation response-severity distribution: {2: 78, 1: 115, 3: 20}
