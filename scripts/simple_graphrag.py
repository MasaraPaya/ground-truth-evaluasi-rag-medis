"""
simple_graphrag.py — Simple GraphRAG di atas PrimeKG (sistem yang dibandingkan dengan naive RAG).

  1. deteksi entitas : nama node PrimeKG (penyakit, obat, fenotipe, gen) yang muncul di pertanyaan
  2. traversal graph : 1-3 hop pada kg_subset.csv, dimuat ke memori sebagai graph biasa (tanpa vector store,
                       tanpa Neo4j)
       - jalur hingga 3 hop yang menghubungkan entitas yang terdeteksi (dan, pada soal pilihan ganda,
         menghubungkan setiap opsi jawaban dengan entitas pada batang soal)
       - tetangga bersama dua entitas bila pertanyaan menanyakan kesamaan ("in common" / "both")
       - tetangga 1 hop sepanjang relasi yang ditanyakan (obat -> indication, efek samping -> drug_effect,
         gen -> disease_protein, gejala/ciri -> disease_phenotype_positive)
  3. konteks         : fakta dengan skor tertinggi sebagai teks "head --[relation]--> tail" (format sama dengan
                       dataset dan naive RAG), paling banyak --max-contexts fakta
  4. LLM             : model dan prompt yang sama dengan mock_baseline.py, sehingga yang berbeda hanya retrieval

Jalankan dari folder utama repositori:

  python scripts/simple_graphrag.py --descriptions data/disease_features.csv            # run yang dipakai di artikel
  python scripts/simple_graphrag.py --descriptions data/disease_features.csv --limit 5  # uji cepat
  python scripts/simple_graphrag.py --generator none --limit 5   # hanya retrieval (tanpa biaya API), untuk cek konteks
(--descriptions menambahkan deskripsi node untuk soal "What is X?"; baseline harus memakai opsi yang sama)

Lalu evaluasi dengan cara yang sama seperti baseline:
  python scripts/evaluate_graphrag.py --name graphrag --max-workers 8
  python scripts/evaluate_graphrag.py compare

Kolom keluaran: question, response, retrieved_contexts (list JSON), retrieved_scores, entities, system.
Run dapat dilanjutkan: soal yang sudah ada di CSV keluaran dilewati (hapus CSV untuk mengulang dari awal).
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mock_baseline import (RELATION_DISPLAY, DATA_DIR, RESULTS_DIR, load_dotenv,  # noqa: E402
                           make_chat_llm, llm_answer, template_answer)
from generate_qa_dataset import GENERIC_PHENOTYPE, display, load_definitions  # noqa: E402

KG_COLUMNS = ["relation", "x_type", "x_name", "y_type", "y_name"]

# question wording -> relation(s) the question asks about
INTENT = [
    (re.compile(r"\b(side effects?|adverse|toxicit\w*|overdose)\b", re.I), "drug_effect"),
    (re.compile(r"\bcontraindicat\w*\b", re.I), "contraindication"),
    (re.compile(r"\b(drugs?|medications?|medicines?|treat\w*|therap\w*|indicated|antibiotics?|"
                r"pharmacotherapy|manag\w*|prevent\w*|prophyla\w*|used (for|to|in))\b", re.I), "indication"),
    (re.compile(r"\b(genes?|proteins?|genetic\w*|mutations?|inherit\w*)\b", re.I), "disease_protein"),
    (re.compile(r"\b(features?|symptoms?|signs?|characteri[sz]\w*|findings?|complications?|"
                r"manifestations?|diagnos\w*)\b", re.I), "disease_phenotype_positive"),
]
SHARED = re.compile(r"\b(both|shared?|in common|common to)\b", re.I)
DEFINITION = re.compile(r"^\s*what (is|are) (an? |the )?(?P<name>[^?]+)\?\s*$", re.I)
OPTIONS_SPLIT = re.compile(r"\n\s*Options:\s*\n", re.I)
OPTION_LINE = re.compile(r"^\s*([A-J])[.)]\s*(.+?)\s*$")

# frequent words that happen to be PrimeKG node names but carry no meaning in a question
STOP = {"disease", "syndrome", "disorder", "patient", "patients", "treatment", "drug", "drugs", "medication",
        "history", "examination", "diagnosis", "infection", "pain", "use", "used", "normal", "abnormal",
        "cell", "cells", "type", "acute", "chronic", "severe", "mild", "risk", "most", "first", "line",
        "test", "level", "levels", "rate", "time", "day", "days", "year", "years", "week", "weeks",
        "man", "woman", "boy", "girl", "child", "left", "right", "upper", "lower", "physical", "change",
        "changes", "increase", "decrease", "sign", "signs", "symptom", "symptoms", "feature", "features",
        "gene", "genes", "protein", "proteins", "effect", "effects", "cause", "causes", "factor", "lead",
        "all", "can", "may", "not", "one", "two", "positive", "negative", "unknown", "other", "others",
        "urea", "nitrogen", "oxygen", "diffuse", "water"}
NEGATION = {"no", "not", "denies", "denied", "without", "negative"}
HUB_DEGREE = 400          # nodes with more edges are not used as intermediate steps of a path
MAX_FRONTIER = 20000      # safety cap for the path search


class Graph:
    """PrimeKG facts as an in-memory graph. Nodes are (lower-case name, type); every edge keeps its text."""

    def __init__(self, kg_path, relations=None):
        self.text, self.rel, self.head, self.tail = [], [], [], []
        self.adj = defaultdict(list)                 # node -> [edge ids]
        self.name = {}                               # node -> name as written in PrimeKG
        seen = set()
        for chunk in pd.read_csv(kg_path, usecols=KG_COLUMNS, dtype=str, chunksize=500_000):
            chunk = chunk.dropna()
            if relations:
                chunk = chunk[chunk["relation"].isin(relations)]
            for rel, xt, xn, yt, yn in chunk.itertuples(index=False):
                xn, yn = xn.strip(), yn.strip()
                head_type, shown = RELATION_DISPLAY.get(rel, (None, rel))
                if head_type and yt == head_type and xt != head_type:      # same orientation as the baseline
                    xn, yn, xt, yt = yn, xn, yt, xt
                key = (rel, *sorted((xn.lower(), yn.lower())))
                if key in seen:
                    continue
                seen.add(key)
                h, t = (xn.lower(), xt), (yn.lower(), yt)
                self.name.setdefault(h, xn)
                self.name.setdefault(t, yn)
                eid = len(self.text)
                self.text.append(f"{xn} --[{shown}]--> {yn}")
                self.rel.append(rel)
                self.head.append(h)
                self.tail.append(t)
                self.adj[h].append(eid)
                self.adj[t].append(eid)
        # surface forms used for entity detection: the name, and the name without "(disease)" etc.
        self.surface = defaultdict(set)
        for node in self.adj:
            for form in {node[0], display(node[0]).lower()}:
                if len(form) >= 3 and form not in STOP and not form.isdigit():
                    self.surface[form].add(node)
        self.max_words = 8

    def degree(self, node):
        return len(self.adj[node])

    def other(self, eid, node):
        return self.tail[eid] if self.head[eid] == node else self.head[eid]

    # ------------------------------------------------------------------ entity detection
    def find_entities(self, text, max_words=None):
        """Longest non-overlapping matches of node names; gene symbols must match case exactly."""
        toks = [(m.group(0), m.start()) for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)]
        words = [t.lower() for t, _ in toks]
        found, i = [], 0
        while i < len(words):
            for n in range(min(max_words or self.max_words, len(words) - i), 0, -1):
                phrase = " ".join(words[i:i + n])
                nodes = self.surface.get(phrase)
                if not nodes:
                    continue
                original = " ".join(t for t, _ in toks[i:i + n])
                nodes = {nd for nd in nodes if nd[1] != "gene/protein" or self.name[nd] == original}
                if not nodes:
                    continue
                negated = any(w in NEGATION for w in words[max(0, i - 3):i])
                found.append({"phrase": phrase, "nodes": nodes, "pos": toks[i][1], "words": n, "negated": negated})
                i += n
                break
            else:
                i += 1
        return found

    # ------------------------------------------------------------------ traversal
    def shortest_path(self, a, b, max_hops=3):
        """Edge ids of a shortest path a -> b of at most max_hops edges (hubs are not used as midpoints)."""
        if a == b:
            return []
        prev = {a: None}
        frontier = [a]
        for _ in range(max_hops):
            nxt = []
            for node in frontier:
                if node != a and self.degree(node) > HUB_DEGREE:
                    continue
                for eid in self.adj[node]:
                    o = self.other(eid, node)
                    if o in prev:
                        continue
                    prev[o] = (node, eid)
                    if o == b:
                        path = []
                        while prev[o] is not None:
                            o, e = prev[o]
                            path.append(e)
                        return path[::-1]
                    nxt.append(o)
                    if len(nxt) > MAX_FRONTIER:
                        break
            frontier = nxt
        return None


def split_question(question):
    """(stem, {letter: option text}) for MedQA-style questions; (question, {}) otherwise."""
    parts = OPTIONS_SPLIT.split(question, maxsplit=1)
    if len(parts) == 1:
        return question, {}
    opts = {}
    for line in parts[1].splitlines():
        m = OPTION_LINE.match(line)
        if m:
            opts[m.group(1)] = m.group(2)
    return parts[0], opts


def ask_sentence(stem):
    sents = [s for s in re.split(r"(?<=[.?!])\s+", stem.strip()) if s]
    return sents[-1] if sents else stem


def intents(stem):
    ask = ask_sentence(stem)
    rels = [rel for rx, rel in INTENT if rx.search(ask)]
    return rels or [rel for rx, rel in INTENT if rx.search(stem)]


class SimpleGraphRAG:
    def __init__(self, graph, definitions=None, max_contexts=10, max_seeds=6):
        self.g = graph
        self.defs = definitions or {}
        self.k = max_contexts
        self.max_seeds = max_seeds

    def _usable(self, node):
        return not (node[1] == "effect/phenotype" and GENERIC_PHENOTYPE.search(node[0]))

    def entity_groups(self, stem, vignette=False):
        """Detected entities of the stem as groups (one group = one phrase, which may match several nodes,
        e.g. 'pneumonia' the disease and 'Pneumonia' the phenotype), best first: phrases in the question
        sentence, diseases/drugs and longer phrases first. In a clinical vignette negated findings
        ('no fever', 'denies chest pain') are dropped. Each multi-word phrase also contributes the shorter
        names inside it ('infective endocarditis' -> 'endocarditis'), ranked after all full phrases.
        Returns [(score, phrase, nodes, parent phrase or None)]."""
        ask_start = len(stem) - len(ask_sentence(stem))
        main, sub = {}, {}
        for ent in self.g.find_entities(stem):
            nodes = {n for n in ent["nodes"] if self._usable(n)}
            if (vignette and ent["negated"]) or not nodes:
                continue
            score = ent["words"] * 0.5 + (2 if ent["pos"] >= ask_start else 0) \
                + (1 if any(n[1] in ("disease", "drug") for n in nodes) else 0)
            if ent["phrase"] not in main or main[ent["phrase"]][0] < score:
                main[ent["phrase"]] = (score, nodes | main.get(ent["phrase"], (0, set()))[1], None)
            inner = ent["phrase"].replace("-", " ")
            n_inner = len(inner.split())
            if n_inner >= 2:
                for part in self.g.find_entities(inner, max_words=n_inner - 1):
                    pn = {n for n in part["nodes"] if self._usable(n)}
                    if pn and part["phrase"] not in main and part["phrase"] not in sub:
                        sub[part["phrase"]] = (score - 1.5, pn, ent["phrase"])
        order = lambda d: sorted(((v[0], ph, v[1], v[2]) for ph, v in d.items()), key=lambda x: (-x[0], x[1]))
        return order(main) + [g for g in order(sub) if g[1] not in main]

    def option_nodes(self, opts):
        out = {}
        for letter, text in opts.items():
            exact = self.g.surface.get(text.lower())
            nodes = set(exact) if exact else {n for e in self.g.find_entities(text) for n in e["nodes"]}
            out[letter] = {n for n in nodes if self._usable(n)}
        return out

    def retrieve(self, question):
        g = self.g
        stem, opts = split_question(question)
        rels = intents(stem)
        groups = self.entity_groups(stem, vignette=bool(opts))
        top = groups[:self.max_seeds]
        stem_nodes = {n for _, _, nodes, _ in groups for n in nodes}       # every entity of the stem
        group_of = {n: i for i, (_, _, nodes, _) in enumerate(top) for n in nodes}
        phrase_of = [ph for _, ph, _, _ in top]
        parent_of = [par for _, _, _, par in top]
        seeds = [n for _, _, nodes, _ in top for n in sorted(nodes)]
        options = self.option_nodes(opts)
        cand = {}

        def add(eid, score):
            if cand.get(eid, -1) < score:
                cand[eid] = score

        # 0. node description (attribute of the detected entity) for "What is X?"
        extra = []
        m = DEFINITION.match(stem)
        if m and self.defs:
            name = m.group("name").strip().lower()
            keys = [name, f"{name} (disease)"] + [n[0] for n in seeds if display(n[0]).lower() == name]
            hits = []
            for key in keys:                        # an ambiguous name may have two descriptions: keep both
                if key in self.defs and self.defs[key] not in hits:
                    hits.append(self.defs[key])
            for kg_name, text in hits[:2]:
                extra.append((f"{kg_name} --[definition]--> {text}", 10.0))

        # 1. multiple choice: facts linking each answer option to the entities of the stem
        #    (1 hop; if an option has no direct link, a path of up to 3 hops to the main entities)
        for letter, onodes in sorted(options.items()):
            links = [eid for o in onodes for eid in g.adj[o] if g.other(eid, o) in stem_nodes]
            support = len({g.other(eid, o) for o in onodes for eid in g.adj[o]} & stem_nodes)
            for eid in links:
                add(eid, 4.0 + 0.25 * min(support, 8))
            if not links:
                for o in onodes:
                    for s in seeds[:3]:
                        path = g.shortest_path(o, s, max_hops=3)
                        if path:
                            for eid in path:
                                add(eid, 3.0 + 1.0 / len(path))

        # 2. shared neighbours of the two main entities ("both", "shared", "in common"); the most specific
        #    shared neighbours (fewest connections overall) first, and both facts of a pair kept together
        if SHARED.search(ask_sentence(stem)) and len(top) >= 2:
            want = set(rels) or None

            def neigh(nodes):
                out = {}
                for a in nodes:
                    for e in g.adj[a]:
                        if (not want or g.rel[e] in want) and self._usable(g.other(e, a)):
                            out.setdefault(g.other(e, a), e)
                return out
            na, nb = neigh(top[0][2]), neigh(top[1][2])
            shared = sorted(set(na) & set(nb), key=lambda n: (g.degree(n), n))
            for i, node in enumerate(shared):
                add(na[node], 5.0 - 0.001 * i)
                add(nb[node], 5.0 - 0.001 * i)

        # 3. paths of up to 3 hops between different entities of the stem (multi-hop links);
        #    paths that use the relation the question asks about count more
        for a, b in combinations(seeds, 2):
            ga, gb = group_of[a], group_of[b]
            if ga == gb or parent_of[ga] == phrase_of[gb] or parent_of[gb] == phrase_of[ga]:
                continue
            path = g.shortest_path(a, b, max_hops=3)
            if path:
                on_topic = not rels or any(g.rel[e] in rels for e in path)
                for eid in path:
                    add(eid, (3.0 if on_topic else 1.5) + 1.0 / len(path))

        # 4. 1-hop neighbours along the relation the question asks about; neighbours that are also linked
        #    to another entity of the question rank higher
        for gi, (gscore, _, nodes, _) in enumerate(top):
            base = 2.0 + 0.5 * (gscore >= 2) - 0.1 * gi
            for seed in sorted(nodes):
                for eid in g.adj[seed]:
                    o = g.other(eid, seed)
                    if not self._usable(o):
                        continue
                    if rels and g.rel[eid] in rels:
                        bonus = 0.3 * len({group_of[x] for e2 in g.adj[o] for x in [g.other(e2, o)]
                                           if x in group_of and group_of[x] != gi}) if g.degree(o) <= 2000 else 0
                        add(eid, (base if g.head[eid] == seed else base - 1.0) + bonus)  # seed as subject first
                    elif not rels:
                        add(eid, 0.5 - 0.1 * gi)

        # 5. nothing found along the asked relation: fall back to all 1-hop facts of the entities
        if not cand:
            for gi, (_, _, nodes, _) in enumerate(top):
                for seed in sorted(nodes):
                    for eid in g.adj[seed]:
                        if self._usable(g.other(eid, seed)):
                            add(eid, 0.5 - 0.1 * gi)

        ranked = sorted(cand.items(), key=lambda kv: (-kv[1], g.text[kv[0]]))
        items = (extra + [(g.text[e], round(s, 3)) for e, s in ranked])[:self.k]
        entities = [g.name[n] for n in seeds] + [f"{l}:{'/'.join(sorted(g.name[n] for n in ns))}"
                                                 for l, ns in sorted(options.items()) if ns]
        return [t for t, _ in items], [s for _, s in items], entities, rels


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DATA_DIR / "ground_truth_dataset.json"))
    ap.add_argument("--kg", default=str(DATA_DIR / "kg_subset.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "graphrag_outputs.csv"))
    ap.add_argument("--descriptions", help="disease_features file: add the node description for 'What is X?' "
                                           "(use the same choice for the baseline to keep the comparison fair)")
    ap.add_argument("--max-contexts", type=int, default=10)
    ap.add_argument("--generator", choices=["llm", "template", "none"], default="llm",
                    help="none = retrieval only, writes the contexts without calling the LLM")
    ap.add_argument("--model", default=None, help="generator LLM (default: same as mock_baseline.py)")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--limit", type=int)
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

    llm = None
    if args.generator == "llm":
        import os
        llm = make_chat_llm(args.model or os.getenv("LLM_MODEL", "gpt-4o-mini"))

    print(f"Loading PrimeKG graph from {args.kg} ...")
    graph = Graph(args.kg)
    print(f"{len(graph.text):,} facts, {len(graph.adj):,} nodes")
    defs = load_definitions(args.descriptions, args.kg) if args.descriptions else None
    rag = SimpleGraphRAG(graph, defs, max_contexts=args.max_contexts)

    write_header = not out_path.exists()
    fields = ["question", "response", "retrieved_contexts", "retrieved_scores", "entities", "system"]
    with open(out_path, "a", encoding="utf-8-sig" if write_header else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for i, rec in enumerate(todo, 1):
            contexts, scores, entities, rels = rag.retrieve(rec["question"])
            if args.generator == "llm":
                try:
                    response = llm_answer(llm, rec["question"], contexts)
                except Exception as e:
                    print(f"  ! LLM error on question {i}: {e}")
                    if type(e).__name__ in ("AuthenticationError", "PermissionDeniedError") or "insufficient_quota" in str(e):
                        sys.exit("\nStopped: problem with the OpenAI API key or credit (see the error above). "
                                 "Fix OPENAI_API_KEY in .env, then run the same command again; "
                                 "answered questions are kept.")
                    continue
                time.sleep(args.sleep)
            elif args.generator == "template":
                response = template_answer(contexts)
            else:
                response = ""
            w.writerow({"question": rec["question"], "response": response,
                        "retrieved_contexts": json.dumps(contexts, ensure_ascii=False),
                        "retrieved_scores": json.dumps(scores), "entities": json.dumps(entities, ensure_ascii=False),
                        "system": f"simple_graphrag (entities -> 1-3 hop traversal, max {args.max_contexts} facts, "
                                  f"{args.generator})"})
            f.flush()
            print(f"[{i}/{len(todo)}] {rec['category']:<18} entities={entities[:3]} facts={len(contexts)}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
