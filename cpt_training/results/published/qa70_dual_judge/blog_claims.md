# Quantitative claims behind the CPT/SFT write-up

- QA items: 70
- Conditions: 12
- Graded answers: 840
- Bootstrap resamples: 20,000 (seed 20260815)

All deltas are paired within an item, so item difficulty and judge
leniency cancel. `*` marks a 95% CI that excludes zero.

## Judge agreement

- Judges: `terra` and `gemini`
- Paired scores: 840
- Pearson r: **0.8688**
- Significance calls agreeing: **24/25**

## Paired effects

### overall_70

| Comparison | Judge | Δ mean | 95% CI | W/L/T |
| :--- | :--- | ---: | :--- | :--- |
| `base` → `transformers_cpt` | terra | +0.800* | [+0.257, +1.357] | 29/16/25 |
| `base` → `transformers_cpt` | gemini | +0.886* | [+0.400, +1.414] | 24/15/31 |
| `base` → `unsloth_cpt` | terra | +0.786* | [+0.229, +1.371] | 28/15/27 |
| `base` → `unsloth_cpt` | gemini | +0.886* | [+0.414, +1.414] | 25/13/32 |
| `transformers_cpt` → `transformers_cpt_sft_v3` | terra | +0.043 | [-0.514, +0.614] | 22/23/25 |
| `transformers_cpt` → `transformers_cpt_sft_v3` | gemini | +0.214 | [-0.300, +0.729] | 26/17/27 |
| `unsloth_cpt` → `unsloth_cpt_sft_v3` | terra | +0.114 | [-0.414, +0.671] | 25/23/22 |
| `unsloth_cpt` → `unsloth_cpt_sft_v3` | gemini | +0.229 | [-0.257, +0.714] | 23/15/32 |
| `base` → `base_sft_v3` | terra | +0.829* | [+0.171, +1.529] | 24/19/27 |
| `base` → `base_sft_v3` | gemini | +1.057* | [+0.471, +1.671] | 30/17/23 |

### sft_seen_30

| Comparison | Judge | Δ mean | 95% CI | W/L/T |
| :--- | :--- | ---: | :--- | :--- |
| `base` → `transformers_cpt` | terra | +1.200* | [+0.267, +2.267] | 13/7/10 |
| `base` → `transformers_cpt` | gemini | +1.233* | [+0.400, +2.167] | 12/7/11 |
| `base` → `unsloth_cpt` | terra | +1.300* | [+0.300, +2.433] | 12/6/12 |
| `base` → `unsloth_cpt` | gemini | +1.367* | [+0.567, +2.300] | 12/4/14 |
| `transformers_cpt` → `transformers_cpt_sft_v3` | terra | +0.367 | [-0.633, +1.433] | 8/8/14 |
| `transformers_cpt` → `transformers_cpt_sft_v3` | gemini | +0.133 | [-0.700, +1.000] | 9/8/13 |
| `unsloth_cpt` → `unsloth_cpt_sft_v3` | terra | +0.433 | [-0.500, +1.433] | 9/7/14 |
| `unsloth_cpt` → `unsloth_cpt_sft_v3` | gemini | +0.100 | [-0.633, +0.867] | 6/6/18 |
| `base` → `base_sft_v3` | terra | +1.767* | [+0.667, +2.933] | 16/9/5 |
| `base` → `base_sft_v3` | gemini | +1.867* | [+0.900, +2.900] | 16/6/8 |

### heldout_40

| Comparison | Judge | Δ mean | 95% CI | W/L/T |
| :--- | :--- | ---: | :--- | :--- |
| `base` → `transformers_cpt` | terra | +0.500 | [-0.100, +1.125] | 16/9/15 |
| `base` → `transformers_cpt` | gemini | +0.625* | [+0.025, +1.250] | 12/8/20 |
| `base` → `unsloth_cpt` | terra | +0.400 | [-0.200, +1.025] | 16/9/15 |
| `base` → `unsloth_cpt` | gemini | +0.525 | [-0.050, +1.150] | 13/9/18 |
| `transformers_cpt` → `transformers_cpt_sft_v3` | terra | -0.200 | [-0.825, +0.425] | 14/15/11 |
| `transformers_cpt` → `transformers_cpt_sft_v3` | gemini | +0.275 | [-0.350, +0.900] | 17/9/14 |
| `unsloth_cpt` → `unsloth_cpt_sft_v3` | terra | -0.125 | [-0.750, +0.500] | 16/16/8 |
| `unsloth_cpt` → `unsloth_cpt_sft_v3` | gemini | +0.325 | [-0.325, +0.975] | 17/9/14 |
| `base` → `base_sft_v3` | terra | +0.125 | [-0.625, +0.925] | 8/10/22 |
| `base` → `base_sft_v3` | gemini | +0.450 | [-0.175, +1.175] | 14/11/15 |

### heldout_claude_20

| Comparison | Judge | Δ mean | 95% CI | W/L/T |
| :--- | :--- | ---: | :--- | :--- |
| `base` → `transformers_cpt` | terra | +1.000* | [+0.200, +1.900] | 10/4/6 |
| `base` → `transformers_cpt` | gemini | +1.050* | [+0.100, +2.050] | 9/5/6 |
| `base` → `unsloth_cpt` | terra | +0.800 | [-0.050, +1.750] | 10/4/6 |
| `base` → `unsloth_cpt` | gemini | +0.800 | [-0.100, +1.800] | 9/6/5 |
| `transformers_cpt` → `transformers_cpt_sft_v3` | terra | -0.450 | [-1.150, +0.250] | 6/9/5 |
| `transformers_cpt` → `transformers_cpt_sft_v3` | gemini | -0.150 | [-0.750, +0.500] | 6/7/7 |
| `unsloth_cpt` → `unsloth_cpt_sft_v3` | terra | -0.400 | [-1.150, +0.350] | 7/10/3 |
| `unsloth_cpt` → `unsloth_cpt_sft_v3` | gemini | -0.050 | [-0.700, +0.600] | 7/7/6 |
| `base` → `base_sft_v3` | terra | +0.050 | [-0.500, +0.700] | 3/5/12 |
| `base` → `base_sft_v3` | gemini | +0.050 | [-0.500, +0.600] | 7/6/7 |

### heldout_codex_20

| Comparison | Judge | Δ mean | 95% CI | W/L/T |
| :--- | :--- | ---: | :--- | :--- |
| `base` → `transformers_cpt` | terra | +0.000 | [-0.800, +0.800] | 6/5/9 |
| `base` → `transformers_cpt` | gemini | +0.200 | [-0.350, +0.950] | 3/3/14 |
| `base` → `unsloth_cpt` | terra | +0.000 | [-0.800, +0.800] | 6/5/9 |
| `base` → `unsloth_cpt` | gemini | +0.250 | [-0.300, +1.000] | 4/3/13 |
| `transformers_cpt` → `transformers_cpt_sft_v3` | terra | +0.050 | [-1.000, +1.100] | 8/6/6 |
| `transformers_cpt` → `transformers_cpt_sft_v3` | gemini | +0.700 | [-0.350, +1.750] | 11/2/7 |
| `unsloth_cpt` → `unsloth_cpt_sft_v3` | terra | +0.150 | [-0.850, +1.150] | 9/6/5 |
| `unsloth_cpt` → `unsloth_cpt_sft_v3` | gemini | +0.700 | [-0.400, +1.750] | 10/2/8 |
| `base` → `base_sft_v3` | terra | +0.200 | [-1.150, +1.650] | 5/5/10 |
| `base` → `base_sft_v3` | gemini | +0.850 | [-0.200, +2.150] | 7/5/8 |

## CPT corpus composition (train split)

| Category | Documents | Tokens | Share | Asked about by the 70 questions |
| :--- | ---: | ---: | ---: | :--- |
| `claude_code` | 160 | 964,056 | 63.24% | yes |
| `openai_platform_api` | 101 | 293,167 | 19.23% | **no** |
| `codex_cli` | 111 | 267,107 | 17.52% | yes |

## Corpus exposure vs recall (held-out)

- r(log min occurrences, CPT score) = **-0.1391**
- r(log total occurrences, CPT score) = **0.0647**

| Min occurrences | Items | base | CPT | Δ |
| :--- | ---: | ---: | ---: | ---: |
| 0 | 4 | 1.25 | 3.5 | +2.250 |
| 1-4 | 18 | 3.333 | 3.778 | +0.444 |
| 20+ | 11 | 2.545 | 3 | +0.455 |
| 5-19 | 7 | 2.714 | 2.429 | -0.286 |

## Duplication

| Metric | claude_code | openai_codex |
| :--- | ---: | ---: |
| Near-duplicate document pairs | 0 | 0 |
| Unique 8-gram % | 97.74 | 97.93 |
| Unique substantive line % | 85.54 | 88.51 |
| Raw lines repeated 5+ times % | 24.69 | 35.5 |

## Question source provenance

- In CPT train split: 64 questions
- In CPT test split (never trained on): **4**
- Not in the corpus: 2

Held-out questions whose source was never trained on: `heldout_claude_014`, `heldout_claude_018`, `heldout_codex_002`, `heldout_codex_014`
