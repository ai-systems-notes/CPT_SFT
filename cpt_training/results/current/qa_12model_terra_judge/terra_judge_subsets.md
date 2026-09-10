# Terra Judge subset analysis

## overall

Items: 70; reference quality: `{'good': 70}`

| Rank | Model | Average / 10 | Good-reference avg | Wins (ties included) | Unique wins |
|---:|---|---:|---:|---:|---:|
| 1 | `unsloth_cpt_sft_v3` | 3.400 | 3.400 | 21 | 0 |
| 2 | `transformers_cpt_sft_v3` | 3.343 | 3.343 | 21 | 0 |
| 3 | `base_sft_v3` | 3.329 | 3.329 | 23 | 9 |
| 4 | `transformers_cpt` | 3.300 | 3.300 | 19 | 0 |
| 5 | `unsloth_cpt` | 3.286 | 3.286 | 19 | 0 |
| 6 | `unsloth_cpt_sft_v2` | 3.157 | 3.157 | 24 | 1 |
| 7 | `unsloth_cpt_sft_v1` | 3.071 | 3.071 | 22 | 1 |
| 8 | `transformers_cpt_sft_v2` | 3.057 | 3.057 | 22 | 0 |
| 9 | `transformers_cpt_sft_v1` | 3.029 | 3.029 | 20 | 0 |
| 10 | `base` | 2.500 | 2.500 | 16 | 1 |
| 11 | `base_sft_v2` | 2.443 | 2.443 | 16 | 3 |
| 12 | `base_sft_v1` | 2.043 | 2.043 | 13 | 1 |

### SFT average-score delta vs source

| Version | Base family | Transformers CPT family | Unsloth CPT family |
|---|---:|---:|---:|
| v1 | -0.457 | -0.271 | -0.214 |
| v2 | -0.057 | -0.243 | -0.129 |
| v3 | +0.829 | +0.043 | +0.114 |

## sft_seen

Items: 30; reference quality: `{'good': 30}`

| Rank | Model | Average / 10 | Good-reference avg | Wins (ties included) | Unique wins |
|---:|---|---:|---:|---:|---:|
| 1 | `base_sft_v3` | 3.867 | 3.867 | 12 | 5 |
| 2 | `unsloth_cpt_sft_v3` | 3.833 | 3.833 | 10 | 0 |
| 3 | `transformers_cpt_sft_v3` | 3.667 | 3.667 | 10 | 0 |
| 4 | `transformers_cpt_sft_v2` | 3.467 | 3.467 | 10 | 0 |
| 5 | `unsloth_cpt_sft_v2` | 3.467 | 3.467 | 9 | 0 |
| 6 | `unsloth_cpt` | 3.400 | 3.400 | 8 | 0 |
| 7 | `transformers_cpt` | 3.300 | 3.300 | 7 | 0 |
| 8 | `unsloth_cpt_sft_v1` | 2.933 | 2.933 | 9 | 0 |
| 9 | `transformers_cpt_sft_v1` | 2.900 | 2.900 | 9 | 0 |
| 10 | `base_sft_v2` | 2.300 | 2.300 | 7 | 1 |
| 11 | `base` | 2.100 | 2.100 | 5 | 1 |
| 12 | `base_sft_v1` | 1.967 | 1.967 | 7 | 0 |

### SFT average-score delta vs source

| Version | Base family | Transformers CPT family | Unsloth CPT family |
|---|---:|---:|---:|
| v1 | -0.133 | -0.400 | -0.467 |
| v2 | +0.200 | +0.167 | +0.067 |
| v3 | +1.767 | +0.367 | +0.433 |

## heldout

Items: 40; reference quality: `{'good': 40}`

| Rank | Model | Average / 10 | Good-reference avg | Wins (ties included) | Unique wins |
|---:|---|---:|---:|---:|---:|
| 1 | `transformers_cpt` | 3.300 | 3.300 | 12 | 0 |
| 2 | `unsloth_cpt` | 3.200 | 3.200 | 11 | 0 |
| 3 | `unsloth_cpt_sft_v1` | 3.175 | 3.175 | 13 | 1 |
| 4 | `transformers_cpt_sft_v1` | 3.125 | 3.125 | 11 | 0 |
| 5 | `transformers_cpt_sft_v3` | 3.100 | 3.100 | 11 | 0 |
| 6 | `unsloth_cpt_sft_v3` | 3.075 | 3.075 | 11 | 0 |
| 7 | `unsloth_cpt_sft_v2` | 2.925 | 2.925 | 15 | 1 |
| 8 | `base_sft_v3` | 2.925 | 2.925 | 11 | 4 |
| 9 | `base` | 2.800 | 2.800 | 11 | 0 |
| 10 | `transformers_cpt_sft_v2` | 2.750 | 2.750 | 12 | 0 |
| 11 | `base_sft_v2` | 2.550 | 2.550 | 9 | 2 |
| 12 | `base_sft_v1` | 2.100 | 2.100 | 6 | 1 |

### SFT average-score delta vs source

| Version | Base family | Transformers CPT family | Unsloth CPT family |
|---|---:|---:|---:|
| v1 | -0.700 | -0.175 | -0.025 |
| v2 | -0.250 | -0.550 | -0.275 |
| v3 | +0.125 | -0.200 | -0.125 |

## heldout_claude_code

Items: 20; reference quality: `{'good': 20}`

| Rank | Model | Average / 10 | Good-reference avg | Wins (ties included) | Unique wins |
|---:|---|---:|---:|---:|---:|
| 1 | `transformers_cpt` | 4.200 | 4.200 | 7 | 0 |
| 2 | `unsloth_cpt` | 4.000 | 4.000 | 6 | 0 |
| 3 | `transformers_cpt_sft_v3` | 3.750 | 3.750 | 5 | 0 |
| 4 | `unsloth_cpt_sft_v3` | 3.600 | 3.600 | 5 | 0 |
| 5 | `transformers_cpt_sft_v1` | 3.500 | 3.500 | 6 | 0 |
| 6 | `unsloth_cpt_sft_v1` | 3.350 | 3.350 | 5 | 0 |
| 7 | `unsloth_cpt_sft_v2` | 3.300 | 3.300 | 8 | 1 |
| 8 | `base_sft_v3` | 3.250 | 3.250 | 3 | 0 |
| 9 | `base` | 3.200 | 3.200 | 4 | 0 |
| 10 | `base_sft_v2` | 3.100 | 3.100 | 3 | 0 |
| 11 | `transformers_cpt_sft_v2` | 3.100 | 3.100 | 7 | 0 |
| 12 | `base_sft_v1` | 2.250 | 2.250 | 1 | 0 |

### SFT average-score delta vs source

| Version | Base family | Transformers CPT family | Unsloth CPT family |
|---|---:|---:|---:|
| v1 | -0.950 | -0.700 | -0.650 |
| v2 | -0.100 | -1.100 | -0.700 |
| v3 | +0.050 | -0.450 | -0.400 |

## heldout_openai_codex

Items: 20; reference quality: `{'good': 20}`

| Rank | Model | Average / 10 | Good-reference avg | Wins (ties included) | Unique wins |
|---:|---|---:|---:|---:|---:|
| 1 | `unsloth_cpt_sft_v1` | 3.000 | 3.000 | 8 | 1 |
| 2 | `transformers_cpt_sft_v1` | 2.750 | 2.750 | 5 | 0 |
| 3 | `base_sft_v3` | 2.600 | 2.600 | 8 | 4 |
| 4 | `unsloth_cpt_sft_v2` | 2.550 | 2.550 | 7 | 0 |
| 5 | `unsloth_cpt_sft_v3` | 2.550 | 2.550 | 6 | 0 |
| 6 | `transformers_cpt_sft_v3` | 2.450 | 2.450 | 6 | 0 |
| 7 | `base` | 2.400 | 2.400 | 7 | 0 |
| 8 | `transformers_cpt` | 2.400 | 2.400 | 5 | 0 |
| 9 | `unsloth_cpt` | 2.400 | 2.400 | 5 | 0 |
| 10 | `transformers_cpt_sft_v2` | 2.400 | 2.400 | 5 | 0 |
| 11 | `base_sft_v2` | 2.000 | 2.000 | 6 | 2 |
| 12 | `base_sft_v1` | 1.950 | 1.950 | 5 | 1 |

### SFT average-score delta vs source

| Version | Base family | Transformers CPT family | Unsloth CPT family |
|---|---:|---:|---:|
| v1 | -0.450 | +0.350 | +0.600 |
| v2 | -0.400 | +0.000 | +0.150 |
| v3 | +0.200 | +0.050 | +0.150 |
