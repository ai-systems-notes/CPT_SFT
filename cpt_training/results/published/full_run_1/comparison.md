# Transformers vs Unsloth CPT comparison

Mode: `full`

| Framework | token/s | peak allocated MiB | nvidia-smi peak MiB | train sec | held-out loss (before → after) |
| --- | ---: | ---: | ---: | ---: | ---: |
| transformers_pytorch | 4754.61 | 8474.05 | 9450.00 | 321.18 | 2.540318 → 2.145524 |
| unsloth | 4761.86 | 7672.57 | 8060.00 | 320.55 | 2.540201 → 2.145465 |

- Unsloth throughput ratio: 1.0015x
- Unsloth peak allocated difference: -801.48 MiB
- Formal benchmark: no

A single result pair is provisional. Use three repeated runs and an aggregate median for the final benchmark.
