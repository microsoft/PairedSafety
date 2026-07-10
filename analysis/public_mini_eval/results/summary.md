## Public Reproducibility Mini-Eval — computed metrics

Prompts graded: 600. Pooled pairs: 3579. Models: llama-3.3-70b, gpt-oss-120b, deepseek-v3.1, gpt-4.1, gpt-5, grok-4.

### Setting table (paired transition)

| Setting | N | Reduction | Preservation | Escalation | Conditional Reduction |
|---|---:|---:|---:|---:|---:|
| llama-3.3-70b (open) | 594 | 40.2% | 54.4% | 5.4% | 84.8% |
| gpt-oss-120b (open) | 600 | 44.3% | 50.8% | 4.8% | 93.3% |
| deepseek-v3.1 (open) | 600 | 41.0% | 51.2% | 7.8% | 86.3% |
| gpt-4.1 (closed) | 592 | 41.2% | 55.2% | 3.5% | 86.8% |
| gpt-5 (closed) | 593 | 42.5% | 53.5% | 4.0% | 89.7% |
| grok-4 (closed) | 600 | 37.7% | 52.7% | 9.7% | 79.3% |
| **Pooled** | 3579 | 41.2% | 52.9% | 5.9% | 86.7% |

### Model safety comparison (sorted by response-harm rate)

| Model | Family | Resp-harm rate | Harmful-prompt compliance | Benign drift | Escalation | Compliance escalations |
|---|---|---:|---:|---:|---:|---:|
| grok-4 | closed | 20.2% | 27.0% | 14.0% | 9.7% | 14 |
| deepseek-v3.1 | open | 15.5% | 17.2% | 14.0% | 7.8% | 3 |
| llama-3.3-70b | open | 14.3% | 19.9% | 9.3% | 5.4% | 3 |
| gpt-4.1 | closed | 11.3% | 16.7% | 6.4% | 3.5% | 1 |
| gpt-5 | closed | 10.6% | 13.9% | 7.7% | 4.0% | 0 |
| gpt-oss-120b | open | 8.0% | 7.4% | 8.6% | 4.8% | 2 |

### grok-4 pairwise comparisons (exact McNemar; prompt-matched)

Holm correction is applied across the five model contrasts separately for each metric.


**Escalation**

| Comparator | Shared N | grok-4 | Comparator | grok-only | comparator-only | exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama-3.3-70b | 594 | 9.8% | 5.4% | 33 | 7 | 4.23e-05 | 1.15e-04 |
| gpt-oss-120b | 600 | 9.7% | 4.8% | 39 | 10 | 3.85e-05 | 1.15e-04 |
| deepseek-v3.1 | 600 | 9.7% | 7.8% | 24 | 13 | 9.89e-02 | 9.89e-02 |
| gpt-4.1 | 592 | 9.8% | 3.5% | 39 | 2 | 7.84e-10 | 3.92e-09 |
| gpt-5 | 593 | 9.4% | 4.0% | 37 | 5 | 4.43e-07 | 1.77e-06 |

**Response harm**

| Comparator | Shared N | grok-4 | Comparator | grok-only | comparator-only | exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama-3.3-70b | 594 | 20.2% | 14.3% | 55 | 20 | 6.49e-05 | 1.30e-04 |
| gpt-oss-120b | 600 | 20.2% | 8.0% | 82 | 9 | 7.09e-16 | 3.54e-15 |
| deepseek-v3.1 | 600 | 20.2% | 15.5% | 51 | 23 | 1.52e-03 | 1.52e-03 |
| gpt-4.1 | 592 | 20.1% | 11.3% | 63 | 11 | 5.32e-10 | 1.60e-09 |
| gpt-5 | 593 | 19.7% | 10.6% | 64 | 10 | 8.96e-11 | 3.58e-10 |

**Harmful-prompt compliance**

| Comparator | Shared N | grok-4 | Comparator | grok-only | comparator-only | exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama-3.3-70b | 282 | 27.0% | 19.9% | 33 | 13 | 4.53e-03 | 4.53e-03 |
| gpt-oss-120b | 285 | 27.0% | 7.4% | 57 | 1 | 4.09e-16 | 2.05e-15 |
| deepseek-v3.1 | 285 | 27.0% | 17.2% | 38 | 10 | 6.17e-05 | 1.23e-04 |
| gpt-4.1 | 281 | 26.7% | 16.7% | 37 | 9 | 4.06e-05 | 1.22e-04 |
| gpt-5 | 281 | 26.7% | 13.9% | 41 | 5 | 4.41e-08 | 1.76e-07 |

### Prompt-cluster bootstrap uncertainty (pooled)

Resampling unit: unique prompt with all available model outputs; 600 prompts, 10,000 resamples.
- Reduction: 95% cluster-bootstrap CI [37.37%, 44.94%]
- Preservation: 95% cluster-bootstrap CI [49.23%, 56.60%]
- Escalation: 95% cluster-bootstrap CI [4.52%, 7.40%]

### Category table (pooled)

| Category | Harmful Prompts | Harmful Responses From Harmful Prompts | Persistence | Reduction |
|---|---:|---:|---:|---:|
| Hate | 542 | 40 | 7.4% | 92.6% |
| Sexual | 96 | 26 | 27.1% | 72.9% |
| Violence | 1019 | 207 | 20.3% | 79.7% |
| Self-harm | 131 | 3 | 2.3% | 97.7% |

### Escalation mechanism split (pooled)

- Total escalations: 211 (5.9% of 3579 pairs, Wilson CI [5.17, 6.72])
- Unsolicited elaboration (benign prompt -> harmful response): 188
- Compliance escalation (harmful prompt answered at higher severity): 23
- Escalation response-severity distribution: {2: 78, 1: 113, 3: 20}
