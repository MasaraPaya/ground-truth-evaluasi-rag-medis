# Korelasi skor RAGAS dengan penilaian manual

Jawaban yang dinilai: 50 (25 soal × 2 sistem)
Skor manual = rata-rata 3 kriteria: Akurasi, Kelengkapan, Kejelasan

## Pasangan utama

| Manual | RAGAS | Pearson r (p) | Spearman ρ (p) | n | Kekuatan (r) |
|---|---|---|---|---|---|
| Rata-rata manual | Rata-rata RAGAS | 0.665 (0.0000) | 0.634 (0.0000) | 50 | sedang |
| Akurasi | faithfulness | 0.086 (0.5585) | 0.163 (0.2619) | 49 | lemah |
| Kelengkapan | context_recall | 0.703 (0.0000) | 0.707 (0.0000) | 50 | kuat |
| Kejelasan | answer_relevancy | 0.721 (0.0000) | 0.662 (0.0000) | 50 | kuat |

## Matriks Spearman ρ (semua kriteria × semua metrik)

| Manual \ RAGAS | faithfulness | answer_relevancy | context_precision | context_recall | Rata-rata RAGAS |
|---|---|---|---|---|---|
| Akurasi | 0.16 | 0.56* | 0.54* | 0.61* | 0.62* |
| Kelengkapan | 0.14 | 0.50* | 0.49* | 0.71* | 0.65* |
| Kejelasan | 0.15 | 0.66* | 0.46* | 0.50* | 0.58* |
| Rata-rata manual | 0.14 | 0.55* | 0.54* | 0.65* | 0.63* |

\* p < 0,05

## Skor manual per sistem (setelah label dibuka)

| Kriteria | Naive RAG | Simple GraphRAG | Δ | Wilcoxon p |
|---|---|---|---|---|
| Akurasi | 3.52 | 4.00 | +0.48 | 0.1578 |
| Kelengkapan | 3.28 | 3.76 | +0.48 | 0.1649 |
| Kejelasan | 4.60 | 4.64 | +0.04 | 0.7055 |
| Rata-rata manual | 3.80 | 4.13 | +0.33 | 0.1550 |

## Per sistem: Spearman ρ rata-rata manual vs rata-rata RAGAS

- naive_rag: ρ = 0.485 (p = 0.0140, n = 25)
- simple_graphrag: ρ = 0.759 (p = 0.0000, n = 25)

Interpretasi kekuatan (Schober dkk., 2018): |r| < 0,4 lemah · 0,4 ≤ |r| < 0,7 sedang · |r| ≥ 0,7 kuat. H2: rata-rata RAGAS vs rata-rata manual, Pearson r ≥ 0,6.
Metrik RAGAS yang NaN untuk suatu jawaban (judge gagal atau tidak terdefinisi) tidak ikut dihitung pada pasangan itu.
