"""
mock_baseline.py — sistem naive RAG di atas PrimeKG (sistem pembanding untuk Simple GraphRAG).

  retrieval : kemiripan kosinus sentence embedding terhadap edge di data/kg_subset.csv
              (tanpa traversal graph, tanpa query expansion, tanpa Cypher)
  generasi  : LLM menjawab dari top-k edge yang diambil (default), atau jawaban template tetap
              ("Based on the retrieved data: ...") dengan --generator template, tanpa API key

Jalankan dari folder utama repositori:

  python scripts/mock_baseline.py --descriptions data/disease_features.csv            # run yang dipakai di artikel
  python scripts/mock_baseline.py --descriptions data/disease_features.csv --limit 5  # uji cepat
  python scripts/mock_baseline.py --generator template                                # tanpa API key

Keluaran: results/baseline_outputs.csv dengan kolom
  question, response, retrieved_contexts (list JSON), retrieved_scores (list JSON), system
simple_graphrag.py menulis format yang sama (results/graphrag_outputs.csv), sehingga evaluate_graphrag.py
menilai kedua sistem dengan cara yang sama.

Indeks edge (embedding) dibangun sekali dan disimpan di data/baseline_index/. Run dapat dilanjutkan:
soal yang sudah ada di CSV keluaran dilewati (hapus CSV untuk mengulang dari awal).

Pengaturan LLM (sama dengan evaluate_graphrag.py): --model, default gpt-4o-mini (atau LLM_MODEL di .env).
  Memerlukan OPENAI_API_KEY.
API key dapat ditulis di berkas .env di folder utama repositori (KEY=value per baris, lihat .env.example).
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, RESULTS_DIR = ROOT / "data", ROOT / "results"

# same verbalisation as the ground-truth contexts: "head --[relation]--> tail"
# relation -> (head node type, display name); other relations keep their PrimeKG name and x -> y order
RELATION_DISPLAY = {
    "indication": ("disease", "indication"),
    "contraindication": ("disease", "contraindication"),
    "off-label use": ("disease", "off_label_use"),
    "disease_phenotype_positive": ("disease", "phenotype_present"),
    "disease_phenotype_negative": ("disease", "phenotype_absent"),
    "drug_effect": ("drug", "side_effect"),
    "disease_protein": ("disease", "associated_gene"),
}

SYSTEM_PROMPT = (
    "You are a biomedical question-answering assistant. Answer the question using only the knowledge-graph "
    "facts provided. Each fact has the form 'head --[relation]--> tail'. If the facts are not sufficient, say "
    "what can be concluded from them and do not add outside knowledge. "
    "Keep the answer concise (at most 4 sentences)."
)
# added only for multiple-choice (MedQA) questions; with it on every question, gpt-3.5-turbo (used in early
# tests) invented "Option B" for open questions
OPTIONS_PROMPT = " The question lists answer options: state the letter and text of the best option first, then a one-sentence justification."
HAS_OPTIONS = re.compile(r"\n\s*Options:\s*\n\s*A[.)]", re.I)


# ---------------------------------------------------------------------------
# environment / LLM
# ---------------------------------------------------------------------------
def load_dotenv(path=ROOT / ".env"):
    """Minimal .env reader (KEY=value), so no extra dependency is needed."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_chat_llm(model, temperature=0.0):
    """OpenAI GPT chat model (default gpt-4o-mini); needs OPENAI_API_KEY."""
    from langchain_openai import ChatOpenAI
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set: put  OPENAI_API_KEY=sk-...  in the .env file in the repository root")
    return ChatOpenAI(model=model, temperature=temperature, max_retries=6)


# ---------------------------------------------------------------------------
# edge index
# ---------------------------------------------------------------------------
def load_edges(kg_path, relations=None):
    """Unique edges of kg_subset.csv as text; PrimeKG's reversed duplicates are dropped."""
    seen, texts = set(), []
    cols = ["relation", "x_type", "x_name", "y_type", "y_name"]
    for chunk in pd.read_csv(kg_path, usecols=cols, dtype=str, chunksize=500_000):
        chunk = chunk.dropna()
        if relations:
            chunk = chunk[chunk["relation"].isin(relations)]
        for rel, xt, xn, yt, yn in chunk.itertuples(index=False):
            head_type, shown = RELATION_DISPLAY.get(rel, (None, rel))
            if head_type and yt == head_type and xt != head_type:
                xn, yn = yn, xn  # orient e.g. disease --[indication]--> drug
            key = (rel, *sorted((xn.strip().lower(), yn.strip().lower())))
            if key in seen:
                continue
            seen.add(key)
            texts.append(f"{xn.strip()} --[{shown}]--> {yn.strip()}")
    return texts


def load_descriptions(path, kg_path):
    """Optional: node descriptions from PrimeKG disease_features as '<KG name> --[definition]--> text'.
    Uses the same loader as the dataset generator (and simple_graphrag.py), so the text is identical."""
    from generate_qa_dataset import load_definitions
    out = []
    for kg_name, text in load_definitions(path, kg_path).values():
        line = f"{kg_name} --[definition]--> {text}"
        if line not in out:
            out.append(line)
    return out


class EdgeIndex:
    def __init__(self, kg_path, model_name, cache_dir, relations=None, descriptions=None, batch_size=256):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # the KG edges and the (optional) descriptions are cached separately, so adding --descriptions
        # later only embeds the descriptions and reuses the edge index
        self.texts, self.emb = self._cached(
            cache_dir, "edges", [str(kg_path), *self._stat(kg_path), model_name, sorted(relations or []), ""],
            lambda: load_edges(kg_path, relations))
        if descriptions:
            d_texts, d_emb = self._cached(
                cache_dir, "descriptions", [str(descriptions), *self._stat(descriptions), str(kg_path), model_name, 2],
                lambda: load_descriptions(descriptions, kg_path))
            self.texts = self.texts + d_texts
            self.emb = np.concatenate([np.asarray(self.emb), d_emb])

    @staticmethod
    def _stat(path):
        st = Path(path).stat()
        return [st.st_size, int(st.st_mtime)]

    def _cached(self, cache_dir, kind, sig, build):
        tag = hashlib.md5(json.dumps(sig).encode()).hexdigest()[:10]
        emb_path, txt_path = cache_dir / f"{kind}_{tag}.npy", cache_dir / f"{kind}_{tag}.txt"
        if emb_path.exists() and txt_path.exists():
            print(f"Loading cached index {emb_path.name} ...")
            return txt_path.read_text(encoding="utf-8").split("\n"), np.load(emb_path, mmap_mode="r")
        texts = build()
        print(f"{len(texts):,} unique {kind} to embed (one-time; this can take a while on CPU) ...")
        emb = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True,
                                normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
        np.save(emb_path, emb)
        txt_path.write_text("\n".join(texts), encoding="utf-8")
        print(f"Cached {kind} index in {cache_dir}")
        return texts, emb

    def search(self, question, top_k=5, chunk=200_000):
        q = self.model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        best_idx, best_sc = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        for start in range(0, len(self.texts), chunk):  # chunked to keep memory low
            sc = np.asarray(self.emb[start:start + chunk], dtype=np.float32) @ q
            k = min(top_k, len(sc))
            part = np.argpartition(-sc, k - 1)[:k]
            best_idx = np.concatenate([best_idx, part + start])
            best_sc = np.concatenate([best_sc, sc[part]])
        order = np.argsort(-best_sc)[:top_k]
        return [self.texts[i] for i in best_idx[order]], [float(s) for s in best_sc[order]]


# ---------------------------------------------------------------------------
# answering
# ---------------------------------------------------------------------------
def template_answer(contexts):
    """Fixed template answer (no LLM), in English like the dataset; for testing without an API key."""
    return "Based on the retrieved data: " + ". ".join(contexts[:2])


def llm_answer(llm, question, contexts):
    from langchain_core.messages import HumanMessage, SystemMessage
    facts = "\n".join(f"- {c}" for c in contexts)
    system = SYSTEM_PROMPT + (OPTIONS_PROMPT if HAS_OPTIONS.search(question) else "")
    msg = llm.invoke([SystemMessage(content=system),
                      HumanMessage(content=f"Facts:\n{facts}\n\nQuestion:\n{question}\n\nAnswer:")])
    return " ".join(str(msg.content).split())


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DATA_DIR / "ground_truth_dataset.json"))
    ap.add_argument("--kg", default=str(DATA_DIR / "kg_subset.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "baseline_outputs.csv"))
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--cache-dir", default=str(DATA_DIR / "baseline_index"))
    ap.add_argument("--relations", nargs="+", help="only index these PrimeKG relations (default: all in kg_subset)")
    ap.add_argument("--descriptions", help="also index disease definitions from this disease_features file")
    ap.add_argument("--top-k", type=int, default=10, help="same budget as simple_graphrag.py --max-contexts")
    ap.add_argument("--generator", choices=["llm", "template"], default="llm")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                    help="generator LLM; ideally the same model the GraphRAG pipeline uses")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between LLM calls (rate limits)")
    ap.add_argument("--limit", type=int, help="only the first N questions (for testing)")
    args = ap.parse_args()

    records = json.load(open(args.dataset, encoding="utf-8"))
    if args.limit:
        records = records[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8-sig") as f:
            done = {r["question"] for r in csv.DictReader(f)}
        print(f"{len(done)} question(s) already answered in {out_path.name}; they will be skipped")
    todo = [r for r in records if r["question"] not in done]
    if not todo:
        print("Nothing to do.")
        return

    llm = make_chat_llm(args.model) if args.generator == "llm" else None
    index = EdgeIndex(args.kg, args.embedding_model, args.cache_dir, args.relations, args.descriptions)

    write_header = not out_path.exists()
    with open(out_path, "a", encoding="utf-8-sig" if write_header else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "response", "retrieved_contexts", "retrieved_scores", "system"])
        if write_header:
            w.writeheader()
        for i, rec in enumerate(todo, 1):
            contexts, scores = index.search(rec["question"], args.top_k)
            if llm is None:
                response = template_answer(contexts)
            else:
                try:
                    response = llm_answer(llm, rec["question"], contexts)
                except Exception as e:  # keep going; the row can be re-run later
                    print(f"  ! LLM error on question {i}: {e}")
                    if type(e).__name__ in ("AuthenticationError", "PermissionDeniedError") or "insufficient_quota" in str(e):
                        sys.exit("\nStopped: problem with the OpenAI API key or credit (see the error above). "
                                 "Fix OPENAI_API_KEY in .env, then run the same command again; "
                                 "answered questions are kept.")
                    continue
                time.sleep(args.sleep)
            w.writerow({"question": rec["question"], "response": response,
                        "retrieved_contexts": json.dumps(contexts, ensure_ascii=False),
                        "retrieved_scores": json.dumps([round(s, 4) for s in scores]),
                        "system": f"mock_baseline (cosine top-{args.top_k}, {args.generator})"})
            f.flush()
            print(f"[{i}/{len(todo)}] {rec['category']:<18} {response[:90]}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
