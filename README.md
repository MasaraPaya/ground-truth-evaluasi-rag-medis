# Modul Evaluasi RAG Medis Berbasis Knowledge Graph: Ground-Truth Dataset dan RAGAS

Kode, *ground-truth dataset*, dan hasil evaluasi untuk artikel:

> Rauf Hafizh Asmenta, Yuda Munarko, Diah Risqiwati. *Dataset Acuan dan Modul Evaluasi Sistem Tanya Jawab Medis Berbasis Graf Pengetahuan*. JATISI (Jurnal Teknik Informatika dan Sistem Informasi), dalam proses pengajuan.

Isi repositori:

- *Ground-truth dataset* berisi 100 pasangan pertanyaan-jawaban (QA) penyakit fisiologis dengan 352 konteks rujukan. Sumbernya PrimeKG, MedQA (USMLE), dan Harrison's Principles of Internal Medicine edisi ke-20.
- *Evaluation harness* berbasis RAGAS 0.4.3 dengan metrik context precision, context recall, faithfulness, answer relevancy, dan KG context recall (tanpa LLM).
- Dua sistem uji: *naive RAG* (kemiripan kosinus top-10) dan *Simple GraphRAG* (traversal 1–3 hop pada PrimeKG). Keduanya memakai LLM dan *prompt* yang sama.
- Prosedur validasi manual beserta analisis korelasi skor RAGAS dengan penilaian manusia.

## Struktur

```
data/
  ground_truth_dataset.json        dataset final (100 QA): question, answer, contexts, source, category, difficulty
  arsip/ground_truth_dataset_sebelum_revisi_validator.json
                                   dataset sebelum perbaikan redaksi dari validator (dipakai run penilaian manual)
  dataset_prevalidation_log.csv    pemeriksaan setiap QA sebelum validasi ahli
  harrison_verification_log.csv    pemeriksaan QA Harrison's terhadap halaman buku yang dirujuk
  pair_blocklist.txt               edge/entitas PrimeKG yang dikeluarkan saat membangkitkan soal
  medqa_selected.csv               20 soal MedQA yang dipilih
  curation_template.csv            25 QA Harrison's (pertanyaan, jawaban, entitas kunci, rujukan bab/halaman)
  validasi_ahli/                   skor validasi ahli per QA dan log perbaikan redaksi
  manual/                          sampel 25 soal, kunci label sistem A/B, skor penilaian manual 50 jawaban
scripts/
  generate_qa_dataset.py           membangun dataset (subperintah medqa, generate, merge)
  mock_baseline.py                 sistem naive RAG
  simple_graphrag.py               sistem Simple GraphRAG
  evaluate_graphrag.py             evaluasi RAGAS + KG context recall; subperintah compare untuk uji Wilcoxon
  buat_lembar_rq3.py               sampel dan lembar penilaian manual (blind, Jawaban A/B)
  correlation_analysis.py          korelasi skor manual dengan skor RAGAS (Pearson, Spearman)
  google_form/                     Apps Script pembuat Google Form validasi dataset dan penilaian manual
results/
  baseline_scores_per_question.csv skor per soal naive RAG (run final), termasuk jawaban dan konteks sistem
  graphrag_scores_per_question.csv skor per soal Simple GraphRAG (run final)
  baseline_scores.json             ringkasan skor naive RAG (run final): rata-rata per kategori, kelompok sumber,
                                   tingkat kesulitan, model judge, waktu run
  graphrag_scores.json             ringkasan skor Simple GraphRAG (run final)
  comparison_report.md             perbandingan kedua sistem (run final)
  correlation_report.md            korelasi skor manual dan RAGAS
  manual_vs_ragas.csv              skor manual dan skor RAGAS per jawaban
  run_penilaian_manual/            run evaluasi yang jawabannya dinilai manual (lihat Catatan 1)
```

## Instalasi

Diuji dengan Python 3.11–3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # lalu isi OPENAI_API_KEY
```

Semua perintah dijalankan dari folder utama repositori.

## 1. Mereproduksi angka artikel (tanpa API key)

Skor per soal kedua sistem sudah disertakan, sehingga semua tabel dan uji statistik dapat dihitung ulang tanpa memanggil LLM:

```bash
python scripts/evaluate_graphrag.py compare                                   # Tabel 2 dan 3, H1
python scripts/evaluate_graphrag.py compare --dir results/run_penilaian_manual  # run sebelum perbaikan redaksi
python scripts/correlation_analysis.py                                        # Tabel 4, H2
```

| Bagian artikel | Sumber |
|---|---|
| 3.1 dan Tabel 1 (komposisi dataset, validasi ahli) | `data/ground_truth_dataset.json`, `data/validasi_ahli/` |
| 3.2, Tabel 2, Tabel 3, Gambar 2 | `results/comparison_report.md` |
| 3.2 (akurasi opsi MedQA), 3.5 (soal *easy* dan *multi-hop*) | `results/*_scores_per_question.csv` (kolom `response`, `difficulty`, `category`) |
| 3.3, Tabel 4, Gambar 3 | `results/correlation_report.md`, `results/manual_vs_ragas.csv` |
| 3.4 (faithfulness vs akurasi) | `results/manual_vs_ragas.csv` |
| 3.5 (variasi antar-*run*, +0,242 vs +0,204) | `results/run_penilaian_manual/comparison_report.md` dan `results/comparison_report.md` |

## 2. Menjalankan ulang seluruh pipeline

### Data yang tidak disertakan

Berkas berikut tidak disertakan karena ukurannya besar atau hak ciptanya milik pihak lain. Simpan di folder `data/`.

| Berkas | Sumber |
|---|---|
| `kg_subset.csv` | `kg.csv` PrimeKG (Chandak dkk., 2023; Harvard Dataverse, doi:10.7910/DVN/IXA7BM) yang disaring pada enam relasi: disease_phenotype_positive, indication, contraindication, off-label use, disease_protein, dan drug_effect. Semua edge keenam relasi diambil, lalu urutan barisnya diacak dengan seed 42 (lihat potongan kode di bawah) |
| `disease_features.csv` | `disease_features.tab` dari Dataverse PrimeKG yang sama (format tab tetap terbaca) |
| `medqa_train.jsonl` | MedQA (Jin dkk., 2021), set USMLE berbahasa Inggris, `train.jsonl` 5 opsi (10.178 soal) |
| Harrison's Principles of Internal Medicine ed. 20 | buku; hanya rujukan bab dan halaman yang dicantumkan di `curation_template.csv` |

```python
import pandas as pd
REL = ["disease_phenotype_positive", "indication", "contraindication",
       "off-label use", "disease_protein", "drug_effect"]
kg = pd.read_csv("kg.csv", low_memory=False)
sub = pd.concat([kg[kg["relation"] == r] for r in REL], ignore_index=True)
sub = sub.sample(frac=1, random_state=42).reset_index(drop=True)  # urutan baris diacak, seed 42
sub.to_csv("data/kg_subset.csv", index=False)
```

Urutan relasi dan pengacakan di atas mengikuti cara subset dibuat dalam penelitian ini. Urutan baris perlu sama
karena memengaruhi urutan kandidat saat membangkitkan soal dan fakta yang terpilih saat skornya sama.

### Urutan perintah

```bash
# a. Dataset (hasilnya draf awal; lihat Catatan 3)
python scripts/generate_qa_dataset.py medqa
python scripts/generate_qa_dataset.py generate --priority data/medqa_diseases.txt data/medqa_drugs.txt --priority-only
python scripts/generate_qa_dataset.py merge

# b. Cek pipeline evaluasi: jawaban dan konteks ground truth dipakai sebagai keluaran sistem, skor harus mendekati 1
python scripts/evaluate_graphrag.py --dummy --limit 5

# c. Kedua sistem (opsi --descriptions harus sama pada keduanya)
python scripts/mock_baseline.py   --descriptions data/disease_features.csv   # -> results/baseline_outputs.csv
python scripts/simple_graphrag.py --descriptions data/disease_features.csv   # -> results/graphrag_outputs.csv

# d. Evaluasi RAGAS dan perbandingan
python scripts/evaluate_graphrag.py --name baseline
python scripts/evaluate_graphrag.py --name graphrag --max-workers 8
python scripts/evaluate_graphrag.py compare

# e. Validasi manual
python scripts/buat_lembar_rq3.py        # sampel 25 soal + lembar penilaian + kunci A/B
python scripts/correlation_analysis.py   # setelah skor manual terisi
```

Run baru menimpa berkas di `results/`. LLM generator dan *judge* sama-sama `gpt-4o-mini` (dapat diganti lewat `LLM_MODEL` dan `JUDGE_MODEL` di `.env`). Embedding answer relevancy memakai `sentence-transformers/all-MiniLM-L6-v2`. Karena *judge* LLM tidak sepenuhnya deterministik, skor run ulang dapat sedikit berbeda (lihat Catatan 4).

## Catatan

1. **Dua run evaluasi.** Lembar penilaian manual dibuat dari run sebelum perbaikan redaksi dataset (`results/run_penilaian_manual/`, memakai `data/arsip/...`). Setelah validasi ahli, 15 QA diperbaiki redaksinya dan kedua sistem dijalankan ulang (run final di `results/`). Perbandingan sistem (Subbab 3.2) memakai run final. Korelasi dengan penilaian manual (Subbab 3.3–3.4) memakai run yang dinilai, karena hanya run itu yang memasangkan setiap skor manual dengan teks jawaban yang sama. Pada run tersebut, context precision gagal dihitung *judge* pada 5 dari 50 jawaban yang dinilai; rata-rata RAGAS kelima jawaban itu dihitung dari tiga metrik.
2. **Kriteria penilaian manual.** Formulir meminta empat kriteria: akurasi, kelengkapan, kejelasan, dan keamanan. Penilai bukan klinisi, sehingga keamanan tidak dipakai dalam analisis dan skor manual adalah rata-rata tiga kriteria pertama. Kolom `Keamanan` tetap disertakan sebagai data mentah; `correlation_analysis.py --include-keamanan` menghitung versi empat kriteria. Petunjuk dan rubrik di `buat_lembar_rq3.py` dan `google_form/buat_form_rq3.gs` dibiarkan persis seperti yang dilihat penilai.
3. **Dataset final.** `generate_qa_dataset.py` menghasilkan draf awal. Dataset final adalah draf tersebut setelah pemeriksaan pra-validasi (`dataset_prevalidation_log.csv`), pemeriksaan QA Harrison's terhadap halaman yang dirujuk (`harrison_verification_log.csv`), dan perbaikan redaksi dari validator (`validasi_ahli/log_revisi_pasca_validasi.csv`). Menjalankan ulang skrip tidak akan menghasilkan berkas yang identik.
4. **Judge.** Instruksi context precision dan context recall disesuaikan agar *judge* mengenali konteks berbentuk tripel `head --[relasi]--> tail` (lihat `PRECISION_INSTRUCTION` dan `RECALL_INSTRUCTION` di `evaluate_graphrag.py`; `--ragas-default-prompts` mengembalikan instruksi asli). Selisih context recall antar-sistem adalah +0,242 pada run yang dinilai dan +0,204 pada run final.
5. **KG context recall.** Proporsi tripel rujukan yang muncul pada konteks sistem. Pencocokan bersifat eksak setelah semua huruf diubah menjadi huruf kecil dan spasi dinormalisasi, tanpa memperhatikan arah relasi karena PrimeKG menyimpan edge dalam dua arah. Metrik ini tidak terdefinisi untuk 8 soal Harrison's tanpa konteks rujukan.
6. **Berkas ringkasan JSON.** `results/baseline_scores.json` dan `results/graphrag_scores.json` adalah keluaran asli run final (4 November 2026). Berkas ini ditulis oleh versi `evaluate_graphrag.py` sebelum penamaannya dirapikan: kunci `RQ1_threshold` dan `RQ1_meets_threshold` sama dengan `threshold` dan `meets_threshold` pada versi sekarang, sedangkan bagian `research_metrics` (ringkasan ulang context precision, context recall, dan faithfulness) tidak lagi ditulis.
7. **Ambang.** Ambang ≥ 0,7 (kelayakan skor, deskriptif), selisih 0,2 (H1), dan r ≥ 0,6 (H2) ditetapkan pada tahap perancangan penelitian, sebelum pengujian.

## Lisensi

Kode: [isi lisensi]. Data turunan PrimeKG dan MedQA mengikuti ketentuan lisensi sumber aslinya. Teks Harrison's tidak disertakan.
