# Comparison: graphrag vs baseline

Paired questions: 100

| Metric | baseline | graphrag | Δ (system − baseline) | n paired | Wilcoxon p |
|---|---|---|---|---|---|
| context_precision | 0.575 | 0.772 | +0.198 | 100 | 0.0000 |
| context_recall | 0.458 | 0.661 | +0.204 | 100 | 0.0000 |
| faithfulness | 0.776 | 0.810 | +0.034 | 98 | 0.1171 |
| answer_relevancy | 0.604 | 0.676 | +0.072 | 100 | 0.1116 |
| kg_context_recall | 0.548 | 0.863 | +0.314 | 92 | 0.0000 |

## Hypothesis H1

H1: context recall of graphrag is at least 0.2 higher than baseline.
Observed Δ context recall = +0.204 → H1 supported.

## By category (cross-domain = multi-hop questions): baseline → graphrag (Δ)

| Group | context_precision | context_recall | faithfulness | answer_relevancy | kg_context_recall |
|---|---|---|---|---|---|
| disease-drug (n=25) | 0.28 → 0.75 (+0.47) | 0.14 → 0.62 (+0.48) | 0.71 → 0.72 (+0.01) | 0.34 → 0.62 (+0.28) | 0.33 → 0.94 (+0.61) |
| disease-phenotype (n=20) | 0.61 → 0.80 (+0.19) | 0.40 → 0.56 (+0.16) | 0.74 → 0.75 (+0.00) | 0.76 → 0.60 (-0.16) | 0.42 → 0.85 (+0.43) |
| drug-side-effect (n=15) | 0.79 → 0.76 (-0.03) | 0.61 → 0.69 (+0.08) | 0.84 → 0.86 (+0.02) | 0.69 → 0.63 (-0.06) | 0.70 → 0.82 (+0.11) |
| disease-gene (n=15) | 0.74 → 0.87 (+0.13) | 0.73 → 0.82 (+0.09) | 0.93 → 0.94 (+0.02) | 0.79 → 0.91 (+0.12) | 0.79 → 0.82 (+0.03) |
| entity-definition (n=10) | 0.83 → 0.87 (+0.03) | 1.00 → 1.00 (+0.00) | 1.00 → 1.00 (+0.00) | 0.89 → 0.92 (+0.03) | 1.00 → 1.00 (+0.00) |
| cross-domain (n=15) | 0.45 → 0.61 (+0.16) | 0.28 → 0.46 (+0.18) | 0.59 → 0.75 (+0.16) | 0.37 → 0.51 (+0.14) | 0.27 → 0.72 (+0.44) |

## By source group: baseline → graphrag (Δ)

| Group | context_precision | context_recall | faithfulness | answer_relevancy | kg_context_recall |
|---|---|---|---|---|---|
| PrimeKG (n=55) | 0.61 → 0.95 (+0.34) | 0.67 → 0.98 (+0.31) | 0.97 → 0.99 (+0.03) | 0.77 → 0.98 (+0.21) | 0.67 → 0.98 (+0.31) |
| MedQA + PrimeKG (n=20) | 0.46 → 0.75 (+0.29) | 0.25 → 0.49 (+0.25) | 0.21 → 0.40 (+0.19) | 0.55 → 0.53 (-0.01) | 0.35 → 0.98 (+0.63) |
| Harrison + PrimeKG (n=17) | 0.54 → 0.46 (-0.08) | 0.12 → 0.13 (+0.01) | 0.82 → 0.68 (-0.14) | 0.20 → 0.15 (-0.05) | 0.38 → 0.34 (-0.04) |
| Harrison only (no KG context) (n=8) | 0.72 → 0.26 (-0.46) | 0.23 → 0.00 (-0.23) | 0.81 → 0.85 (+0.03) | 0.46 → 0.09 (-0.38) | - |

## By difficulty: baseline → graphrag (Δ)

| Group | context_precision | context_recall | faithfulness | answer_relevancy | kg_context_recall |
|---|---|---|---|---|---|
| easy (n=50) | 0.64 → 0.95 (+0.31) | 0.72 → 1.00 (+0.28) | 0.97 → 1.00 (+0.03) | 0.83 → 0.98 (+0.15) | 0.71 → 1.00 (+0.29) |
| hard (n=20) | 0.46 → 0.75 (+0.29) | 0.25 → 0.49 (+0.25) | 0.21 → 0.40 (+0.19) | 0.55 → 0.53 (-0.01) | 0.35 → 0.98 (+0.63) |
| medium (n=30) | 0.54 → 0.50 (-0.05) | 0.15 → 0.21 (+0.06) | 0.83 → 0.77 (-0.06) | 0.27 → 0.27 (+0.00) | 0.35 → 0.44 (+0.09) |
