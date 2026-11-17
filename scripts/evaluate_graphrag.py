"""
evaluate_graphrag.py — evaluasi RAGAS sebuah sistem QA terhadap ground-truth dataset.

Masukan
  data/ground_truth_dataset.json     question, answer, contexts, source, category, difficulty
  results/<name>_outputs.csv         question, response, retrieved_contexts (list JSON)  [, system]
                                     (ditulis oleh mock_baseline.py dan simple_graphrag.py)

Pemetaan kolom RAGAS
  question -> user_input        answer   -> reference        contexts -> reference_contexts
  response -> response          retrieved_contexts -> retrieved_contexts

Metrik (ragas 0.4.3, kelas legacy yang berjalan dengan ragas.evaluate)
  context_precision   LLMContextPrecisionWithReference
  context_recall      LLMContextRecall                   (H1: GraphRAG >= baseline + 0,2)
  faithfulness        Faithfulness
  answer_relevancy    ResponseRelevancy (memerlukan embedding)
  kg_context_recall   proporsi tripel KG rujukan yang ditemukan pada konteks sistem, tanpa LLM:
                      pencocokan eksak setelah diubah ke huruf kecil, arah edge diabaikan (lihat triple_recall);
                      NaN untuk soal Harrison's tanpa konteks rujukan

Cara pakai (dari folder utama repositori)
  python scripts/evaluate_graphrag.py --dummy --limit 5            # cek pipeline: ground truth sebagai keluaran
  python scripts/evaluate_graphrag.py --name baseline              # results/baseline_outputs.csv
  python scripts/evaluate_graphrag.py --name graphrag              # results/graphrag_outputs.csv
  python scripts/evaluate_graphrag.py --name graphrag --no-llm     # hanya kg_context_recall, tanpa API key
                                                                   # (menulis graphrag_nollm_*, tidak menimpa)
  python scripts/evaluate_graphrag.py compare                      # laporan baseline vs graphrag
  python scripts/evaluate_graphrag.py compare --dir results/run_penilaian_manual   # sama, untuk run sebelumnya

Keluaran
  results/<name>_scores_per_question.csv   satu baris per soal dengan semua metrik
  results/<name>_scores.json               rata-rata keseluruhan / per kategori / per kelompok sumber, cek >= 0,7
  results/comparison_report.md             (compare) selisih, cek H1, uji Wilcoxon berpasangan

Judge LLM adalah OpenAI GPT, default gpt-4o-mini (OPENAI_API_KEY); ganti dengan --model atau JUDGE_MODEL di .env.
gpt-3.5-turbo sempat dicoba sebagai judge lalu tidak dipakai: pada uji 5 soal, model ini memberi context precision
0 untuk konteks yang memuat tripel rujukan secara harfiah, dan context recall 1,0 untuk konteks yang tidak memuat
satu pun tripel rujukan.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, RESULTS_DIR = ROOT / "data", ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mock_baseline import load_dotenv, make_chat_llm  # noqa: E402  (shared .env / LLM setup)

METRIC_ORDER = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "kg_context_recall"]
RAGAS_NAME = {  # ragas column name -> our name
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
}
# Adapted judge instructions (ragas' few-shot examples are kept). Our contexts are atomic KG triples and many
# reference answers list several items, so with the default wording the judge rejects a triple that supports
# only part of the answer: in a traced run "breast cancer --[indication]--> Bevacizumab" got verdict 0 because
# "it does not provide information about Drostanolone propionate", and a one-sentence reference made recall 0/1.
# --ragas-default-prompts switches back to the original instructions.
PRECISION_INSTRUCTION = (
    "Given question, answer and context verify if the context was useful in arriving at the given answer. "
    "The context can be a single knowledge-graph fact written as 'head --[relation]--> tail'. "
    'Give verdict as "1" if the context supports at least one fact stated in the answer (it does not have to '
    'support the whole answer) and "0" if it supports none of them. Give the verdict with json output.')
RECALL_INSTRUCTION = (
    "Given a context, and an answer, split the answer into atomic statements, one statement per individual fact "
    "(an answer listing several items, e.g. 'X include A, B and C', gives one statement per item), and classify "
    "if each statement can be attributed to the given context or not. The context can contain knowledge-graph "
    "facts written as 'head --[relation]--> tail'. Use only 'Yes' (1) or 'No' (0) as a binary classification. "
    "Output json with reason.")
SCORE_THRESHOLD = 0.7  # descriptive "good enough" level per metric, set when the study was designed
H1_DELTA = 0.2         # H1: context recall of GraphRAG is at least this much higher than the baseline


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def norm_q(q):
    return " ".join(str(q).split()).lower()


def parse_list(value):
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
        return []
    s = str(value).strip()
    try:
        out = json.loads(s)
        return [str(v) for v in out] if isinstance(out, list) else [str(out)]
    except json.JSONDecodeError:
        return [p.strip() for p in s.split(" || ") if p.strip()]  # fallback: "a || b || c"


def source_group(rec):
    if not rec["contexts"]:
        return "Harrison only (no KG context)"
    if rec["source"].startswith("MedQA"):
        return "MedQA + PrimeKG"
    if "Harrison" in rec["source"]:
        return "Harrison + PrimeKG"
    return "PrimeKG"


def load_rows(dataset_path, outputs_path, dummy, limit):
    records = json.load(open(dataset_path, encoding="utf-8"))
    if limit:
        records = records[:limit]
    if dummy:
        # dummy mode: a "perfect" system that returns the ground truth. Checks the pipeline end to end;
        # scores should be close to 1 (the Harrison-only questions get retrieved = [] -> NaN for some metrics).
        outputs = {norm_q(r["question"]): {"response": r["answer"], "retrieved_contexts": r["contexts"]}
                   for r in records}
    else:
        if not Path(outputs_path).exists():
            sys.exit(f"{outputs_path} not found. Run mock_baseline.py (or the GraphRAG pipeline) first.")
        df = pd.read_csv(outputs_path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
        missing = {"question", "response", "retrieved_contexts"} - set(df.columns)
        if missing:
            sys.exit(f"{outputs_path} is missing column(s): {', '.join(sorted(missing))}")
        outputs = {norm_q(r["question"]): {"response": r["response"],
                                           "retrieved_contexts": parse_list(r["retrieved_contexts"])}
                   for _, r in df.iterrows()}

    rows, unanswered = [], []
    for i, rec in enumerate(records):
        out = outputs.get(norm_q(rec["question"]))
        if out is None:
            unanswered.append(i)
            continue
        rows.append({"qid": i + 1, "category": rec["category"], "source": rec["source"],
                     "difficulty": rec["difficulty"], "group": source_group(rec),
                     "user_input": rec["question"], "reference": rec["answer"],
                     "reference_contexts": rec["contexts"],
                     "response": out["response"] or "", "retrieved_contexts": out["retrieved_contexts"]})
    if unanswered:
        print(f"WARNING: {len(unanswered)} dataset question(s) have no system output and are skipped "
              f"(dataset rows {', '.join(str(i + 1) for i in unanswered[:10])}{' ...' if len(unanswered) > 10 else ''})")
    if not rows:
        sys.exit("No questions to evaluate (question text in the outputs CSV must match the dataset).")
    return rows


# ---------------------------------------------------------------------------
# ragas
# ---------------------------------------------------------------------------
def norm_triple(text):
    """'Head --[rel]--> Tail (MONDO:...)' -> ('head', 'rel', 'tail'); None if not a triple."""
    m = re.match(r"\s*(.+?)\s*--\[(.+?)\]-->\s*(.+?)\s*(\([A-Z]+[:_][^)]*\))?\s*$", str(text))
    if not m:
        return None
    return tuple(" ".join(x.lower().split()) for x in m.group(1, 2, 3))


def triple_recall(reference, retrieved):
    """Share of reference triples that appear among the retrieved contexts. Triples are compared after
    lower-casing and whitespace normalisation; direction is ignored, because PrimeKG stores every edge
    both ways. A retrieved context that is not a triple (free-text path or sentence) counts as a hit when it
    mentions both the head and the tail of the reference triple."""
    ref = [t for t in (norm_triple(c) for c in reference) if t]
    if not ref:
        return np.nan
    got = {t for t in (norm_triple(c) for c in retrieved) if t}
    got |= {(t[2], t[1], t[0]) for t in got}
    free = [" ".join(str(c).lower().split()) for c in retrieved if norm_triple(c) is None]
    hit = sum(1 for t in ref if t in got or any(t[0] in f and t[2] in f for f in free))
    return hit / len(ref)


def make_trace_handler(path):
    """LangChain callback that writes every judge prompt + raw reply to a JSONL file (--trace)."""
    from langchain_core.callbacks import BaseCallbackHandler

    class TraceHandler(BaseCallbackHandler):
        def __init__(self):
            self.prompts = {}
            self.f = open(path, "w", encoding="utf-8")

        def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
            self.prompts[run_id] = [str(m.content) for m in messages[0]]

        def on_llm_end(self, response, *, run_id, **kw):
            replies = [g.text for gens in response.generations for g in gens]
            prompt = self.prompts.pop(run_id, [])
            first = prompt[-1] if prompt else ""
            self.f.write(json.dumps({"task": first[:90], "prompt": prompt, "reply": replies},
                                    ensure_ascii=False) + "\n")
            self.f.flush()

        def on_llm_error(self, error, *, run_id, **kw):
            self.f.write(json.dumps({"prompt": self.prompts.pop(run_id, []), "error": str(error)},
                                    ensure_ascii=False) + "\n")
            self.f.flush()

    return TraceHandler()


def make_embeddings(model_name):
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=model_name,
                                                            encode_kwargs={"normalize_embeddings": True}))


def run_ragas(rows, llm, embeddings, metric_names, max_workers, timeout, batch_size, dummy=False,
              default_prompts=False):
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
                               ResponseRelevancy)
    from ragas.run_config import RunConfig

    factory = {"context_precision": LLMContextPrecisionWithReference, "context_recall": LLMContextRecall,
               "faithfulness": Faithfulness, "answer_relevancy": ResponseRelevancy}
    ctx_metrics = [factory[m]() for m in metric_names if m in ("context_precision", "context_recall", "faithfulness")]
    if not default_prompts:
        for met in ctx_metrics:
            if isinstance(met, LLMContextPrecisionWithReference):
                met.context_precision_prompt.instruction = PRECISION_INSTRUCTION
            elif isinstance(met, LLMContextRecall):
                met.context_recall_prompt.instruction = RECALL_INSTRUCTION
    ragas_llm = LangchainLLMWrapper(llm) if llm is not None else None

    results = {r["qid"]: {} for r in rows}
    run_config = RunConfig(timeout=timeout, max_retries=10, max_wait=60, max_workers=max_workers)

    def run_llm(subset, metrics):
        ds = EvaluationDataset([SingleTurnSample(user_input=r["user_input"], response=r["response"],
                                                 retrieved_contexts=r["retrieved_contexts"] or None,
                                                 reference=r["reference"]) for r in subset])
        res = evaluate(ds, metrics=metrics, llm=ragas_llm, embeddings=embeddings, run_config=run_config,
                       batch_size=batch_size, raise_exceptions=False, show_progress=True)
        df = res.to_pandas()
        for r, (_, s) in zip(subset, df.iterrows()):
            for col, ours in RAGAS_NAME.items():
                if col in df.columns:
                    results[r["qid"]][ours] = s[col]

    # context metrics need retrieved contexts (ragas errors on an empty list)
    judged = [r for r in rows if r["retrieved_contexts"]]
    empty = [r for r in rows if not r["retrieved_contexts"]]
    if ctx_metrics and judged:
        run_llm(judged, ctx_metrics)
    if ctx_metrics and empty:
        if dummy:  # only the Harrison-only questions (no reference contexts) -> undefined
            print(f"{len(empty)} question(s) without contexts -> context metrics NaN (dummy mode)")
        else:      # the system retrieved nothing: precision/recall 0, faithfulness undefined
            print(f"{len(empty)} question(s) with no retrieved contexts -> context_precision/recall = 0, "
                  f"faithfulness = NaN")
            for r in empty:
                for m in ("context_precision", "context_recall"):
                    if m in metric_names:
                        results[r["qid"]][m] = 0.0
    # answer relevancy only needs question + response, so every question is scored
    if "answer_relevancy" in metric_names:
        answered = [r for r in rows if r["response"].strip()]
        if answered:
            run_llm(answered, [ResponseRelevancy()])

    # exact-match recall of the reference KG triples (no LLM); undefined without reference contexts.
    # ragas' NonLLMContextRecall is not used: its fuzzy threshold (0.5) counts e.g.
    # "X --[contraindication]--> Y" as a hit for "X --[indication]--> Z" because the strings look alike.
    if "kg_context_recall" in metric_names:
        for r in rows:
            if r["reference_contexts"]:
                results[r["qid"]]["kg_context_recall"] = triple_recall(r["reference_contexts"],
                                                                       r["retrieved_contexts"])
    return results


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def mean_or_none(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return None if s.empty else round(float(s.mean()), 4)


def summarise(df, metrics, name, args):
    def block(sub):
        return {"n": int(len(sub)),
                **{m: mean_or_none(sub[m]) for m in metrics},
                **{f"{m}_n_scored": int(pd.to_numeric(sub[m], errors="coerce").notna().sum()) for m in metrics}}

    overall = block(df)
    summary = {
        "system": name,
        "run_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "dummy_mode": args.dummy,
        "judge_llm": "none (--no-llm)" if args.no_llm else "openai:" + args.model,
        "embeddings": args.embeddings,
        "judge_prompts": "ragas default" if args.ragas_default_prompts else "adapted for KG triples (see PRECISION_INSTRUCTION / RECALL_INSTRUCTION)",
        "n_questions": int(len(df)),
        "overall": overall,
        "by_category": {c: block(g) for c, g in df.groupby("category", sort=False)},
        "by_source_group": {c: block(g) for c, g in df.groupby("group", sort=False)},
        "by_difficulty": {c: block(g) for c, g in df.groupby("difficulty", sort=False)},
        "threshold": SCORE_THRESHOLD,
        "meets_threshold": {m: (overall[m] is not None and overall[m] >= SCORE_THRESHOLD)
                                for m in ("context_precision", "context_recall", "faithfulness") if m in metrics},
    }
    return summary


def print_summary(summary, metrics):
    print(f"\n=== {summary['system']}  ({summary['n_questions']} questions) ===")
    head = f"{'':<42}" + "".join(f"{m[:17]:>18}" for m in metrics)
    print(head)

    def line(label, b):
        cells = "".join(f"{('-' if b[m] is None else f'{b[m]:.3f}'):>18}" for m in metrics)
        print(f"{label[:41]:<42}{cells}")

    line("OVERALL", summary["overall"])
    for title in ("by_category", "by_source_group"):
        print(f"-- {title.replace('_', ' ')}")
        for k, b in summary[title].items():
            line(f"  {k} (n={b['n']})", b)
    print(f"Threshold (>= {SCORE_THRESHOLD}): " + ", ".join(f"{m}={'yes' if ok else 'no'}"
                                                         for m, ok in summary["meets_threshold"].items()))


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def compare(args):
    results_dir = Path(args.dir)
    base_csv = results_dir / f"{args.baseline}_scores_per_question.csv"
    sys_csv = results_dir / f"{args.system}_scores_per_question.csv"
    for p in (base_csv, sys_csv):
        if not p.exists():
            sys.exit(f"{p} not found. Evaluate both systems first.")
    b = pd.read_csv(base_csv, encoding="utf-8-sig")
    s = pd.read_csv(sys_csv, encoding="utf-8-sig")
    m = s.merge(b, on="user_input", suffixes=("_sys", "_base"))
    metrics = [c for c in METRIC_ORDER if f"{c}_sys" in m.columns and f"{c}_base" in m.columns]
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    lines = [f"# Comparison: {args.system} vs {args.baseline}", "",
             f"Paired questions: {len(m)}", "",
             "| Metric | " + args.baseline + " | " + args.system + " | Δ (system − baseline) | n paired | Wilcoxon p |",
             "|---|---|---|---|---|---|"]
    h1 = None
    for met in metrics:
        pair = m[[f"{met}_sys", f"{met}_base"]].apply(pd.to_numeric, errors="coerce").dropna()
        if pair.empty:
            continue
        ms, mb = pair[f"{met}_sys"].mean(), pair[f"{met}_base"].mean()
        p = "-"
        diff = pair[f"{met}_sys"] - pair[f"{met}_base"]
        if wilcoxon is not None and len(pair) >= 6 and (diff != 0).any():
            p = f"{wilcoxon(pair[f'{met}_sys'], pair[f'{met}_base'], zero_method='wilcox').pvalue:.4f}"
        lines.append(f"| {met} | {mb:.3f} | {ms:.3f} | {ms - mb:+.3f} | {len(pair)} | {p} |")
        if met == "context_recall":
            h1 = ms - mb
    lines += ["", "## Hypothesis H1", "",
              f"H1: context recall of {args.system} is at least {H1_DELTA} higher than {args.baseline}."]
    lines.append("Context recall not available." if h1 is None else
                 f"Observed Δ context recall = {h1:+.3f} → H1 {'supported' if h1 >= H1_DELTA else 'not supported'}.")

    def breakdown(title, col):
        col = f"{col}_sys" if f"{col}_sys" in m.columns else col
        if col not in m.columns:
            return []
        out = ["", f"## {title}: {args.baseline} → {args.system} (Δ)", "",
               "| Group | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
        for key, g in m.groupby(col, sort=False):
            cells = []
            for met in metrics:
                pair = g[[f"{met}_sys", f"{met}_base"]].apply(pd.to_numeric, errors="coerce").dropna()
                if pair.empty:
                    cells.append("-")
                    continue
                ms, mb = pair[f"{met}_sys"].mean(), pair[f"{met}_base"].mean()
                cells.append(f"{mb:.2f} → {ms:.2f} ({ms - mb:+.2f})")
            out.append(f"| {key} (n={len(g)}) | " + " | ".join(cells) + " |")
        return out

    lines += breakdown("By category (cross-domain = multi-hop questions)", "category")
    lines += breakdown("By source group", "group")
    lines += breakdown("By difficulty", "difficulty")
    if wilcoxon is None:
        lines += ["", "_scipy not installed: Wilcoxon test skipped (pip install scipy)._"]
    out = results_dir / "comparison_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")


# ---------------------------------------------------------------------------
def main():
    load_dotenv()
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        ap = argparse.ArgumentParser(prog="evaluate_graphrag.py compare")
        ap.add_argument("--baseline", default="baseline")
        ap.add_argument("--system", default="graphrag")
        ap.add_argument("--dir", default=str(RESULTS_DIR), help="folder with the two *_scores_per_question.csv")
        compare(ap.parse_args(sys.argv[2:]))
        return

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DATA_DIR / "ground_truth_dataset.json"))
    ap.add_argument("--name", default="graphrag", help="system name: reads results/<name>_outputs.csv")
    ap.add_argument("--outputs", help="override the outputs CSV path")
    ap.add_argument("--dummy", action="store_true",
                    help="use the ground truth as system output (pipeline check; name becomes 'dummy')")
    ap.add_argument("--no-llm", action="store_true",
                    help="only the non-LLM metric (kg_context_recall); no API key needed")
    ap.add_argument("--metrics", nargs="+", choices=METRIC_ORDER, default=METRIC_ORDER)
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
                    help="judge LLM (default gpt-4o-mini, or JUDGE_MODEL in .env)")
    ap.add_argument("--embeddings", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="HuggingFace model for answer_relevancy")
    ap.add_argument("--max-workers", type=int, default=2, help="parallel judge calls (keep low for rate limits)")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, help="only the first N dataset questions (for testing)")
    ap.add_argument("--ragas-default-prompts", action="store_true",
                    help="use ragas' original precision/recall instructions instead of the KG-triple adaptation")
    ap.add_argument("--trace", action="store_true",
                    help="write every judge prompt and raw reply to results/<name>_judge_trace.jsonl")
    args = ap.parse_args()

    name = ("dummy" if args.dummy else args.name) + ("_nollm" if args.no_llm else "")  # never overwrite full runs
    outputs = args.outputs or str(RESULTS_DIR / f"{args.name}_outputs.csv")
    metrics = ["kg_context_recall"] if args.no_llm else list(args.metrics)

    rows = load_rows(args.dataset, outputs, args.dummy, args.limit)
    print(f"Evaluating {len(rows)} question(s) for '{name}' with: {', '.join(metrics)}")

    llm = embeddings = None
    if any(m != "kg_context_recall" for m in metrics):
        llm = make_chat_llm(args.model)
        if args.trace:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            trace_path = RESULTS_DIR / f"{name}_judge_trace.jsonl"
            llm.callbacks = [make_trace_handler(trace_path)]
            print(f"Tracing judge calls to {trace_path}")
        if "answer_relevancy" in metrics:
            embeddings = make_embeddings(args.embeddings)
    scores = run_ragas(rows, llm, embeddings, metrics, args.max_workers, args.timeout, args.batch_size,
                       dummy=args.dummy, default_prompts=args.ragas_default_prompts)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = []
    for r in rows:
        row = {k: r[k] for k in ("qid", "category", "source", "difficulty", "group", "user_input")}
        row.update({m: scores[r["qid"]].get(m, np.nan) for m in metrics})
        row.update({"response": r["response"], "reference": r["reference"],
                    "retrieved_contexts": json.dumps(r["retrieved_contexts"], ensure_ascii=False),
                    "reference_contexts": json.dumps(r["reference_contexts"], ensure_ascii=False)})
        table.append(row)
    df = pd.DataFrame(table)
    per_q = RESULTS_DIR / f"{name}_scores_per_question.csv"
    df.to_csv(per_q, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    summary = summarise(df, metrics, name, args)
    out_json = RESULTS_DIR / f"{name}_scores.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(summary, metrics)
    failed = {m: int(df[m].isna().sum()) for m in metrics}
    if any(failed.values()):
        print("NaN per metric (undefined, or judge failure -> re-run / check rate limits): "
              + ", ".join(f"{m}={n}" for m, n in failed.items() if n))
    print(f"\nWrote {per_q}\nWrote {out_json}")


if __name__ == "__main__":
    main()
