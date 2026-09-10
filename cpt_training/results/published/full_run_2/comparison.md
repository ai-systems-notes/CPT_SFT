# Transformers vs Unsloth CPT comparison

Mode: `full`

| Framework | token/s | peak allocated MiB | nvidia-smi peak MiB | train sec | held-out loss (before → after) |
| --- | ---: | ---: | ---: | ---: | ---: |
| transformers_pytorch | 4981.27 | 8474.05 | 9450.00 | 305.72 | 2.540318 → 2.145567 |
| unsloth | 4778.68 | 7672.57 | 8060.00 | 319.12 | 2.540201 → 2.145478 |

- Unsloth throughput ratio: 0.9593x
- Unsloth peak allocated difference: -801.48 MiB
- Formal benchmark: no

A single result pair is provisional. Use three repeated runs and an aggregate median for the final benchmark.
