"""
buat_lembar_rq3.py — sampel dan lembar penilaian manual (oleh manusia) atas jawaban sistem,
untuk menguji apakah skor RAGAS sesuai dengan penilaian manusia.

  * 25 soal dipilih secara stratified per kategori, sebanding dengan komposisi dataset (6/5/4/4/2/4),
    secara acak dengan seed tetap
  * setiap soal memuat jawaban kedua sistem (Naive RAG dan Simple GraphRAG) sebagai "Jawaban A" dan "Jawaban B";
    sistem yang menjadi A diacak per soal sehingga penilai tidak dapat mengenali sistemnya (blind)
  * 4 kriteria berskala 1-5: Akurasi, Kelengkapan, Kejelasan, Keamanan
    (Keamanan ikut dinilai, tetapi tidak dipakai dalam analisis; lihat correlation_analysis.py)

Keluaran (folder data/manual/):
  Lembar_Penilaian_Manual_RQ3.xlsx   -> lembar penilaian (isi hanya sel kuning)
  kunci_label_sistem.csv             -> sistem di balik A/B; JANGAN dibuka atau dibagikan sebelum penilaian selesai
  sampel_rq3.csv                     -> daftar soal sampel

Lembar dibuat dari run evaluasi di results/run_penilaian_manual/, yang memakai dataset sebelum perbaikan
redaksi pasca-validasi ahli (data/arsip/), sehingga jawaban yang dinilai adalah jawaban dari run tersebut.
Pada penelitian ini lembar diubah menjadi Google Form (scripts/google_form/buat_form_rq3.gs, dengan rubrik dan
petunjuk yang sama); jawaban penilai yang diekspor ada di data/manual/hasil_penilaian_manual.csv.
Petunjuk dan rubrik di bawah dibiarkan persis seperti yang dilihat penilai.

Jalankan dari folder utama repositori, setelah kedua sistem dievaluasi:
  python scripts/buat_lembar_rq3.py
Setelah skor terisi:
  python scripts/correlation_analysis.py
"""
import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
QUOTA = {"disease-drug": 6, "disease-phenotype": 5, "drug-side-effect": 4, "disease-gene": 4,
         "entity-definition": 2, "cross-domain": 4}
CAT = {"disease-drug": "Disease–Drug", "disease-phenotype": "Disease–Phenotype", "drug-side-effect": "Drug–Side Effect",
       "disease-gene": "Disease–Gene", "entity-definition": "Entity Definition", "cross-domain": "Lintas Domain"}
CRITERIA = ["Akurasi", "Kelengkapan", "Kejelasan", "Keamanan"]
RUBRIC = {
    "Akurasi": ["Sebagian besar isi salah secara medis atau bertentangan dengan jawaban acuan",
                "Banyak kesalahan; hanya sebagian kecil yang benar",
                "Benar sebagian; ada kesalahan yang cukup penting",
                "Hampir seluruhnya benar; hanya kesalahan kecil",
                "Seluruh isi benar secara medis dan sesuai jawaban acuan"],
    "Kelengkapan": ["Tidak menjawab pertanyaan / tidak memuat poin penting jawaban acuan",
                    "Hanya sedikit poin penting yang tercakup",
                    "Sekitar separuh poin penting tercakup",
                    "Sebagian besar poin penting tercakup",
                    "Semua poin penting jawaban acuan tercakup"],
    "Kejelasan": ["Sulit dipahami, berbelit, atau tidak nyambung dengan pertanyaan",
                  "Kurang jelas; banyak bagian yang membingungkan",
                  "Cukup jelas, tetapi ada bagian yang bertele-tele atau ambigu",
                  "Jelas dan runtut; sedikit perbaikan",
                  "Sangat jelas, ringkas, langsung menjawab pertanyaan"],
    "Keamanan": ["Berpotensi membahayakan (mis. menyarankan obat/tindakan yang keliru dan berisiko)",
                 "Ada informasi menyesatkan yang cukup berisiko",
                 "Ada informasi yang kurang tepat, risikonya rendah",
                 "Aman; ada sedikit hal yang perlu diberi catatan",
                 "Aman; tidak ada informasi yang menyesatkan atau berbahaya"],
}

F = "Arial"
def font(**k):
    return Font(name=F, size=k.pop("size", 10), **k)
YELLOW = PatternFill("solid", fgColor="FFF2CC")
HEAD = PatternFill("solid", fgColor="1F4E5A")
GREY = PatternFill("solid", fgColor="F2F2F2")
thin = Side(style="thin", color="BFBFBF")
BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
WRAP = Alignment(wrap_text=True, vertical="top")
CENTER = Alignment(horizontal="center", vertical="top", wrap_text=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(ROOT / "data" / "arsip" / "ground_truth_dataset_sebelum_revisi_validator.json"))
    ap.add_argument("--baseline", default=str(ROOT / "results" / "run_penilaian_manual" / "baseline_scores_per_question.csv"))
    ap.add_argument("--graphrag", default=str(ROOT / "results" / "run_penilaian_manual" / "graphrag_scores_per_question.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "manual"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true", help="overwrite an existing (possibly filled-in) sheet")
    args = ap.parse_args()
    sheet = Path(args.out_dir) / "Lembar_Penilaian_Manual_RQ3.xlsx"
    if sheet.exists() and not args.force:
        raise SystemExit(f"{sheet} already exists (it may already be filled in). Use --force to overwrite it.")

    ds = json.load(open(args.dataset, encoding="utf-8"))
    systems = {"naive_rag": pd.read_csv(args.baseline, encoding="utf-8-sig").set_index("user_input"),
               "simple_graphrag": pd.read_csv(args.graphrag, encoding="utf-8-sig").set_index("user_input")}
    rng = random.Random(args.seed)
    by_cat = {}
    for qid, rec in enumerate(ds, 1):
        by_cat.setdefault(rec["category"], []).append((qid, rec))
    sample = []
    for cat, n in QUOTA.items():
        sample += rng.sample(by_cat[cat], n)
    sample.sort(key=lambda x: x[0])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, key = [], []
    # which system is "A": balanced (about half each) and in random order
    first = ["naive_rag"] * (len(sample) // 2) + ["simple_graphrag"] * (len(sample) - len(sample) // 2)
    rng.shuffle(first)
    for no, (qid, rec) in enumerate(sample, 1):
        order = [first[no - 1], "simple_graphrag" if first[no - 1] == "naive_rag" else "naive_rag"]
        for label, system in zip("AB", order):
            aid = f"{no:02d}-{label}"
            rows.append({"no": no, "aid": aid, "cat": CAT[rec["category"]], "q": rec["question"],
                         "ref": rec["answer"], "resp": str(systems[system].loc[rec["question"], "response"])})
            key.append({"id_jawaban": aid, "no_soal": no, "qid_dataset": qid, "kategori": rec["category"],
                        "sistem": system})

    with open(out / "kunci_label_sistem.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    with open(out / "sampel_rq3.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["no_soal", "qid_dataset", "kategori", "sumber", "difficulty", "pertanyaan"])
        for no, (qid, rec) in enumerate(sample, 1):
            w.writerow([no, qid, rec["category"], rec["source"], rec["difficulty"], rec["question"]])

    wb = Workbook()
    # ------------------------------------------------------------------ Petunjuk
    P = wb.active
    P.title = "Petunjuk"
    P.column_dimensions["A"].width = 18
    for c in "BCDEF":
        P.column_dimensions[c].width = 30
    r = 1
    P.cell(r, 1, "Lembar Penilaian Manual Jawaban Sistem (RQ3)").font = font(size=14, bold=True); r += 1
    P.cell(r, 1, "Skripsi: Pengembangan Ground-Truth Dataset Penyakit Fisiologis dan Evaluasi Akurasi Penalaran "
                 "GraphRAG Menggunakan Metrik RAGAS").font = font(italic=True); r += 1
    P.cell(r, 1, "Penilai: ______________________   Tanggal: ____________").font = font(); r += 2
    P.cell(r, 1, "Cara mengisi").font = font(size=12, bold=True); r += 1
    steps = [
        f"1. Sheet 'Penilaian' berisi {len(sample)} pertanyaan. Setiap pertanyaan punya dua jawaban dari dua sistem "
        f"berbeda (Jawaban A dan Jawaban B), jadi ada {len(rows)} baris yang dinilai.",
        "2. Isi hanya sel kuning: empat skor (angka 1–5) dan Catatan (opsional, sangat membantu untuk skor 1–3).",
        "3. Nilai setiap jawaban sendiri-sendiri memakai rubrik di bawah, bukan dengan membandingkan A dan B.",
        "4. Kolom 'Jawaban acuan' adalah jawaban ground truth dataset, sebagai pegangan untuk Akurasi dan Kelengkapan. "
        "Pengetahuan medis penilai tetap dipakai: jawaban yang benar walau berbeda dari acuan tetap boleh diberi skor tinggi.",
        "5. Identitas sistem sengaja disembunyikan. Mohon tidak mencari tahu sistem mana yang menghasilkan jawaban "
        "sampai penilaian selesai.",
    ]
    for s in steps:
        P.cell(r, 1, s).font = font(); P.cell(r, 1).alignment = WRAP
        P.merge_cells(start_row=r, start_column=1, end_row=r, end_column=6); P.row_dimensions[r].height = 30; r += 1
    r += 1
    P.cell(r, 1, "Rubrik penilaian (1 = sangat kurang, 5 = sangat baik)").font = font(size=12, bold=True); r += 1
    hdr = ["Skor"] + CRITERIA
    for c, h in enumerate(hdr, 1):
        cell = P.cell(r, c, h); cell.font = font(bold=True, color="FFFFFF"); cell.fill = HEAD; cell.border = BOX
        cell.alignment = CENTER
    r += 1
    for s in range(5, 0, -1):
        P.cell(r, 1, s).font = font(bold=True); P.cell(r, 1).alignment = CENTER; P.cell(r, 1).border = BOX
        for c, crit in enumerate(CRITERIA, 2):
            cell = P.cell(r, c, RUBRIC[crit][s - 1]); cell.font = font(); cell.alignment = WRAP; cell.border = BOX
        P.row_dimensions[r].height = 45; r += 1
    r += 1
    P.cell(r, 1, "Contoh pengisian (fiktif, bukan bagian data)").font = font(size=12, bold=True); r += 1
    ex_h = ["ID", "Jawaban yang dinilai", "Akurasi", "Kelengkapan", "Kejelasan", "Keamanan"]
    for c, h in enumerate(ex_h, 1):
        cell = P.cell(r, c, h); cell.font = font(bold=True); cell.fill = GREY; cell.border = BOX
    r += 1
    ex = ["00-A", "Drugs indicated for hypothyroidism include levothyroxine.", 5, 3, 5, 5]
    for c, v in enumerate(ex, 1):
        cell = P.cell(r, c, v); cell.font = font(); cell.border = BOX; cell.alignment = WRAP if c == 2 else CENTER
    P.row_dimensions[r].height = 30; r += 1
    P.cell(r, 1, "Catatan contoh: benar, tetapi tidak menyebut liothyronine yang ada di jawaban acuan → "
                 "Kelengkapan 3.").font = font(italic=True)
    P.merge_cells(start_row=r, start_column=1, end_row=r, end_column=6)

    # ------------------------------------------------------------------ Penilaian
    S = wb.create_sheet("Penilaian")
    cols = [("No", 5), ("ID Jawaban", 9), ("Kategori", 14), ("Pertanyaan", 48), ("Jawaban acuan", 38),
            ("Jawaban yang dinilai", 55), ("Akurasi", 10), ("Kelengkapan", 13), ("Kejelasan", 11), ("Keamanan", 11),
            ("Rata-rata", 10), ("Catatan", 30)]
    for c, (h, wdt) in enumerate(cols, 1):
        cell = S.cell(1, c, h); cell.font = font(bold=True, color="FFFFFF"); cell.fill = HEAD
        cell.alignment = CENTER; cell.border = BOX
        S.column_dimensions[cell.column_letter].width = wdt
    dv = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True,
                        showErrorMessage=True, errorTitle="Skor tidak valid", error="Isi angka bulat 1 sampai 5.")
    S.add_data_validation(dv)
    for i, row in enumerate(rows, 2):
        vals = [row["no"], row["aid"], row["cat"], row["q"], row["ref"], row["resp"]]
        for c, v in enumerate(vals, 1):
            cell = S.cell(i, c, v); cell.font = font(); cell.border = BOX
            cell.alignment = CENTER if c <= 3 else WRAP
        for c in range(7, 11):
            cell = S.cell(i, c); cell.fill = YELLOW; cell.border = BOX; cell.alignment = CENTER; cell.font = font()
        dv.add(f"G{i}:J{i}")
        S.cell(i, 11, f'=IF(COUNT(G{i}:J{i})=4,AVERAGE(G{i}:J{i}),"")').font = font()
        S.cell(i, 11).alignment = CENTER; S.cell(i, 11).border = BOX; S.cell(i, 11).number_format = "0.00"
        S.cell(i, 12).fill = YELLOW; S.cell(i, 12).border = BOX; S.cell(i, 12).alignment = WRAP
        long = max(len(row["q"]) / 48, len(row["resp"]) / 55, len(row["ref"]) / 38)
        S.row_dimensions[i].height = min(400, max(45, 13 * (long + 1)))
        if row["aid"].endswith("B"):          # thicker separation between questions
            for c in range(1, 13):
                S.cell(i, c).border = Border(left=thin, right=thin, top=thin, bottom=Side(style="medium", color="1F4E5A"))
    S.freeze_panes = "G2"

    # ------------------------------------------------------------------ Rekap
    R = wb.create_sheet("Rekap")
    R.column_dimensions["A"].width = 30
    R.column_dimensions["B"].width = 14
    n = len(rows) + 1
    R["A1"] = "Rekap penilaian (otomatis)"; R["A1"].font = font(size=12, bold=True)
    R["A3"] = "Jawaban yang sudah dinilai lengkap"; R["B3"] = f"=COUNT(Penilaian!K2:K{n})"
    R["A4"] = "Total jawaban"; R["B4"] = len(rows)
    for j, (crit, col) in enumerate(zip(CRITERIA + ["Rata-rata"], "GHIJK"), 6):
        R.cell(j, 1, f"Rata-rata {crit}" if crit != "Rata-rata" else "Rata-rata keseluruhan")
        R.cell(j, 2, f'=IFERROR(AVERAGE(Penilaian!{col}2:{col}{n}),"")').number_format = "0.00"
    R["A12"] = "Catatan: rekap per sistem dihitung oleh correlation_analysis.py setelah penilaian selesai."
    R["A12"].font = font(italic=True)
    for row in R.iter_rows(min_row=3, max_row=11):
        for cell in row:
            if cell.font.name != F:
                cell.font = font()

    for ws in (P, S):                       # printable: landscape, all columns on one page width
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True
    S.print_title_rows = "1:1"

    path = out / "Lembar_Penilaian_Manual_RQ3.xlsx"
    wb.save(path)
    print(f"{len(sample)} questions, {len(rows)} answers")
    for cat, n_ in QUOTA.items():
        print(f"  {cat:<18} {n_}")
    print(f"Wrote {path}\nWrote {out / 'kunci_label_sistem.csv'} (keep closed until scoring is done)\n"
          f"Wrote {out / 'sampel_rq3.csv'}")


if __name__ == "__main__":
    main()
