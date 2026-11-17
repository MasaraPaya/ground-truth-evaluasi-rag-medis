# Comparison: graphrag vs baseline

Paired questions: 100

| Metric | baseline | graphrag | Δ (system − baseline) | n paired | Wilcoxon p |
|---|---|---|---|---|---|
| context_precision | 0.582 | 0.810 | +0.228 | 83 | 0.0000 |
| context_recall | 0.450 | 0.692 | +0.242 | 100 | 0.0000 |
| faithfulness | 0.774 | 0.800 | +0.026 | 98 | 0.2588 |
| answer_relevancy | 0.603 | 0.669 | +0.066 | 100 | 0.1311 |
| kg_context_recall | 0.550 | 0.863 | +0.312 | 92 | 0.0000 |

## Hypothesis H1

H1: context recall of graphrag is at least 0.2 higher than baseline.
Observed Δ context recall = +0.242 → H1 supported.

## By category (cross-domain = multi-hop questions): baseline → graphrag (Δ)

| Group | context_precision | context_recall | faithfulness | answer_relevancy | kg_context_recall |
|---|---|---|---|---|---|
| disease-drug (n=25) | 0.30 → 0.81 (+0.51) | 0.14 → 0.64 (+0.50) | 0.70 → 0.72 (+0.02) | 0.34 → 0.59 (+0.25) | 0.33 → 0.94 (+0.61) |
| disease-phenotype (n=20) | 0.61 → 0.78 (+0.17) | 0.38 → 0.64 (+0.26) | 0.73 → 0.73 (-0.00) | 0.75 → 0.60 (-0.16) | 0.42 → 0.85 (+0.43) |
| drug-side-effect (n=15) | 0.80 → 0.82 (+0.02) | 0.62 → 0.72 (+0.10) | 0.82 → 0.86 (+0.04) | 0.70 → 0.64 (-0.07) | 0.71 → 0.82 (+0.10) |
| disease-gene (n=15) | 0.78 → 0.92 (+0.14) | 0.73 → 0.82 (+0.09) | 0.89 → 0.90 (+0.00) | 0.78 → 0.90 (+0.12) | 0.79 → 0.82 (+0.03) |
| entity-definition (n=10) | 0.83 → 0.92 (+0.09) | 1.00 → 1.00 (+0.00) | 1.00 → 1.00 (+0.00) | 0.89 → 0.92 (+0.04) | 1.00 → 1.00 (+0.00) |
| cross-domain (n=15) | 0.44 → 0.67 (+0.23) | 0.24 → 0.48 (+0.24) | 0.65 → 0.74 (+0.09) | 0.37 → 0.52 (+0.15) | 0.27 → 0.72 (+0.44) |

## By source group: baseline → graphrag (Δ)

| Group | context_precision | context_recall | faithfulness | answer_relevancy | kg_context_recall |
|---|---|---|---|---|---|
| PrimeKG (n=55) | 0.61 → 0.96 (+0.35) | 0.67 → 0.98 (+0.31) | 0.96 → 0.98 (+0.01) | 0.77 → 0.98 (+0.21) | 0.68 → 0.98 (+0.30) |
| MedQA + PrimeKG (n=20) | 0.50 → 0.80 (+0.30) | 0.20 → 0.65 (+0.45) | 0.25 → 0.41 (+0.16) | 0.54 → 0.54 (-0.00) | 0.35 → 0.98 (+0.63) |
| Harrison + PrimeKG (n=17) | 0.52 → 0.54 (+0.02) | 0.12 → 0.13 (+0.01) | 0.78 → 0.69 (-0.08) | 0.20 → 0.15 (-0.05) | 0.38 → 0.34 (-0.04) |
| Harrison only (no KG context) (n=8) | 0.73 → 0.25 (-0.47) | 0.23 → 0.00 (-0.23) | 0.77 → 0.77 (-0.01) | 0.47 → 0.00 (-0.47) | - |

## By difficulty: baseline → graphrag (Δ)

| Group | context_precision | context_recall | faithfulness | answer_relevancy | kg_context_recall |
|---|---|---|---|---|---|
| easy (n=50) | 0.64 → 0.96 (+0.32) | 0.73 → 1.00 (+0.27) | 0.97 → 0.98 (+0.02) | 0.83 → 0.97 (+0.15) | 0.72 → 1.00 (+0.28) |
| hard (n=20) | 0.50 → 0.80 (+0.30) | 0.20 → 0.65 (+0.45) | 0.25 → 0.41 (+0.16) | 0.54 → 0.54 (-0.00) | 0.35 → 0.98 (+0.63) |
| medium (n=30) | 0.54 → 0.53 (-0.01) | 0.15 → 0.20 (+0.05) | 0.80 → 0.75 (-0.05) | 0.27 → 0.25 (-0.02) | 0.35 → 0.44 (+0.09) |
