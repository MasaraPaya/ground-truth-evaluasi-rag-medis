"""
correlation_analysis.py — korelasi skor RAGAS dengan skor penilaian manual.

Skor manual setiap jawaban adalah rata-rata tiga kriteria: Akurasi, Kelengkapan, Kejelasan (skala 1-5).
Formulir juga meminta skor Keamanan, tetapi kriteria ini tidak dipakai dalam analisis: menilai potensi bahaya
klinis memerlukan klinisi, sedangkan penilai bukan klinisi. Opsi --include-keamanan memasukkannya kembali
(rata-rata empat kriteria).

Keluaran
  results/correlation_report.md            Pearson + Spearman (r, p, n), skor manual per sistem
  results/manual_vs_ragas.csv              satu baris per jawaban: skor manual di samping skor RAGAS-nya
  (dengan --include-keamanan nama berkas diberi akhiran _4kriteria)

Jalankan dari folder utama repositori:
  python scripts/correlation_analysis.py
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = ["Akurasi", "Kelengkapan", "Kejelasan"]
SAFETY = "Keamanan"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
# criterion -> RAGAS metric that measures the closest idea (reported as the main pairs)
MAIN_PAIRS = [("Rata-rata manual", "Rata-rata RAGAS"), ("Akurasi", "faithfulness"),
              ("Kelengkapan", "context_recall"), ("Kejelasan", "answer_relevancy")]


def corr(x, y):
    d = pd.concat([x, y], axis=1).apply(pd.to_numeric, errors="coerce").dropna()
    n = len(d)
    if n < 5 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return None
    pr, pp = pearsonr(d.iloc[:, 0], d.iloc[:, 1])
    sr, sp = spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    return {"pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp, "n": n}


def strength(r):
    a = abs(r)  # bands of Schober et al. (2018): < 0.40 weak, 0.40-0.69 moderate, >= 0.70 strong
    return "lemah" if a < 0.4 else "sedang" if a < 0.7 else "kuat"


def read_scores(path, criteria):
    """Manual scores from the exported CSV (id_jawaban) or from the filled sheet (ID Jawaban)."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        df = pd.read_excel(path, sheet_name="Penilaian")
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"ID Jawaban": "id_jawaban"})
    missing = [c for c in ["id_jawaban"] + criteria if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing column(s) {', '.join(missing)}")
    df = df[["id_jawaban"] + criteria].copy()
    for c in criteria:  # the form exports e.g. "5 (Sangat baik)"; keep the number only
        df[c] = pd.to_numeric(df[c].astype(str).str.extract(r"^\s*(\d)")[0], errors="coerce")
    done = df[criteria].notna().all(axis=1)
    if not done.all():
        print(f"Warning: {(~done).sum()} answer(s) not fully scored; they are left out")
    return df[done].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=str(ROOT / "data" / "manual" / "hasil_penilaian_manual.csv"),
                    help="manual scores: exported .csv or the filled .xlsx sheet")
    ap.add_argument("--key", default=str(ROOT / "data" / "manual" / "kunci_label_sistem.csv"))
    ap.add_argument("--baseline", default=str(ROOT / "results" / "run_penilaian_manual" / "baseline_scores_per_question.csv"))
    ap.add_argument("--graphrag", default=str(ROOT / "results" / "run_penilaian_manual" / "graphrag_scores_per_question.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--include-keamanan", action="store_true",
                    help="also use Keamanan (manual mean over 4 criteria instead of 3)")
    args = ap.parse_args()

    criteria = CRITERIA + ([SAFETY] if args.include_keamanan else [])
    suffix = "_4kriteria" if args.include_keamanan else ""

    manual = read_scores(args.scores, criteria)
    manual["Rata-rata manual"] = manual[criteria].mean(axis=1)

    key = pd.read_csv(args.key, encoding="utf-8-sig")
    ragas = []
    for system, path in (("naive_rag", args.baseline), ("simple_graphrag", args.graphrag)):
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["sistem"] = system
        ragas.append(df[["qid", "sistem", "category", "group"] + METRICS])
    ragas = pd.concat(ragas).rename(columns={"qid": "qid_dataset"})
    ragas["Rata-rata RAGAS"] = ragas[METRICS].mean(axis=1, skipna=True)

    d = manual.merge(key, on="id_jawaban").merge(ragas, on=["qid_dataset", "sistem"], how="left")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_dir / f"manual_vs_ragas{suffix}.csv", index=False, encoding="utf-8-sig")

    L = ["# Korelasi skor RAGAS dengan penilaian manual", "",
         f"Jawaban yang dinilai: {len(d)} ({d['no_soal'].nunique()} soal × 2 sistem)",
         f"Skor manual = rata-rata {len(criteria)} kriteria: {', '.join(criteria)}", "",
         "## Pasangan utama", "",
         "| Manual | RAGAS | Pearson r (p) | Spearman ρ (p) | n | Kekuatan (r) |", "|---|---|---|---|---|---|"]
    for a, b in MAIN_PAIRS:
        c = corr(d[a], d[b])
        if c:
            L.append(f"| {a} | {b} | {c['pearson_r']:.3f} ({c['pearson_p']:.4f}) | {c['spearman_rho']:.3f} "
                     f"({c['spearman_p']:.4f}) | {c['n']} | {strength(c['pearson_r'])} |")
        else:
            L.append(f"| {a} | {b} | - | - | - | data kurang / tidak bervariasi |")

    L += ["", "## Matriks Spearman ρ (semua kriteria × semua metrik)", "",
          "| Manual \\ RAGAS | " + " | ".join(METRICS + ["Rata-rata RAGAS"]) + " |",
          "|---" * (len(METRICS) + 2) + "|"]
    for a in criteria + ["Rata-rata manual"]:
        cells = []
        for b in METRICS + ["Rata-rata RAGAS"]:
            c = corr(d[a], d[b])
            cells.append("-" if not c else f"{c['spearman_rho']:.2f}{'*' if c['spearman_p'] < 0.05 else ''}")
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    L += ["", "\\* p < 0,05", ""]

    L += ["## Skor manual per sistem (setelah label dibuka)", "",
          "| Kriteria | Naive RAG | Simple GraphRAG | Δ | Wilcoxon p |", "|---|---|---|---|---|"]
    piv = d.pivot_table(index="no_soal", columns="sistem", values=criteria + ["Rata-rata manual"])
    for a in criteria + ["Rata-rata manual"]:
        try:
            pair = piv[a][["naive_rag", "simple_graphrag"]].dropna()
        except KeyError:
            continue
        diff = pair["simple_graphrag"] - pair["naive_rag"]
        p = (f"{wilcoxon(pair['simple_graphrag'], pair['naive_rag']).pvalue:.4f}"
             if len(pair) >= 6 and (diff != 0).any() else "-")
        L.append(f"| {a} | {pair['naive_rag'].mean():.2f} | {pair['simple_graphrag'].mean():.2f} | "
                 f"{diff.mean():+.2f} | {p} |")

    L += ["", "## Per sistem: Spearman ρ rata-rata manual vs rata-rata RAGAS", ""]
    for system, g in d.groupby("sistem"):
        c = corr(g["Rata-rata manual"], g["Rata-rata RAGAS"])
        L.append(f"- {system}: " + ("-" if not c else f"ρ = {c['spearman_rho']:.3f} (p = {c['spearman_p']:.4f}, "
                                                        f"n = {c['n']})"))
    L += ["", "Interpretasi kekuatan (Schober dkk., 2018): |r| < 0,4 lemah · 0,4 ≤ |r| < 0,7 sedang · "
              "|r| ≥ 0,7 kuat. H2: rata-rata RAGAS vs rata-rata manual, Pearson r ≥ 0,6.",
          "Metrik RAGAS yang NaN untuk suatu jawaban (judge gagal atau tidak terdefinisi) tidak ikut dihitung "
          "pada pasangan itu."]

    report = out_dir / f"correlation_report{suffix}.md"
    report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nWrote {report}\nWrote {out_dir / f'manual_vs_ragas{suffix}.csv'}")


if __name__ == "__main__":
    main()
