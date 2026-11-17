"""
generate_qa_dataset.py — membangun ground-truth dataset QA (100 soal) dari empat sumber:
  * primekg                   pertanyaan dari tripel PrimeKG dengan satu template per relasi (45)
  * medqa                     soal pilihan ganda MedQA (USMLE) yang penyakit/obatnya ada di PrimeKG, diambil apa
                              adanya; contexts = tripel PrimeKG yang menghubungkan soal dan jawaban (20)
  * harrison                  QA konseptual yang ditulis dari bab Harrison's di curation_template.csv,
                              lalu digabungkan dengan perintah merge (25)
  * primekg_node_description  "What is X?" dari deskripsi penyakit di PrimeKG (10)
Kuota per kategori dan sumber ada di CATEGORIES di bawah.

Perintah (jalankan dari folder utama repositori):

  1) medqa     — mencocokkan nama PrimeKG di data/medqa_train.jsonl
       python scripts/generate_qa_dataset.py medqa
       -> data/medqa_diseases.txt, data/medqa_drugs.txt (daftar prioritas)
          data/medqa_candidates.csv (semua soal MedQA yang terhubung ke PrimeKG, untuk diperiksa)

  2) generate  — membangun dataset
       python scripts/generate_qa_dataset.py generate --priority data/medqa_diseases.txt data/medqa_drugs.txt --priority-only
       membaca data/kg_subset.csv, data/pair_blocklist.txt, data/medqa_train.jsonl,
               data/disease_features.csv (deskripsi node PrimeKG)
       -> data/ground_truth_dataset.json
          data/medqa_selected.csv     20 soal MedQA yang dipilih; ubah lalu jalankan generate lagi untuk mengganti pilihan
          data/curation_template.csv  25 slot Harrison's (+ slot yang tidak dapat diisi sumber otomatis)

  3) merge     — setelah QA Harrison's ditulis di curation_template.csv, gabungkan kembali (dapat diulang)
       python scripts/generate_qa_dataset.py merge
       key_entities dicocokkan ke PrimeKG (eksak, lalu fuzzy >= 0,9), tripel pendukung menjadi contexts,
       dan kg_supported = yes/partial/no diusulkan (PERIKSA MANUAL).

Dua berkas keluaran:
  data/ground_truth_dataset.json  dataset, hanya record yang lengkap, satu record per QA:
      {"question", "answer", "contexts": ["head --[relation]--> tail", ...], "source", "category", "difficulty"}
  data/ground_truth_work.json     berkas kerja untuk merge: semua slot + "id" dan "metadata"
      (source_type, key_entities, kg_supported, harrison_ref, medqa_id, status)
Untuk RAGAS 0.4, evaluasi memetakan question -> user_input, answer -> reference, contexts -> reference_contexts.

Dataset yang dipublikasikan adalah keluaran ini setelah pemeriksaan manual dan perbaikan redaksi dari validator
(lihat data/dataset_prevalidation_log.csv dan data/validasi_ahli/), sehingga menjalankan ulang skrip
menghasilkan draf pertama.
"""
import argparse
import csv
import difflib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ===========================================================================
# PrimeKG loading
# ===========================================================================
KG_COLUMNS = ["relation", "x_type", "x_name", "y_type", "y_name"]

# relation in PrimeKG -> (center node type, target node type, verbaliser for one triple)
RELATION_SPECS = {
    "indication": ("disease", "drug", lambda c, t: f"{c} --[indication]--> {t}"),
    "disease_phenotype_positive": ("disease", "effect/phenotype", lambda c, t: f"{c} --[phenotype_present]--> {t}"),
    "drug_effect": ("drug", "effect/phenotype", lambda c, t: f"{c} --[side_effect]--> {t}"),
    "disease_protein": ("disease", "gene/protein", lambda c, t: f"{c} --[associated_gene]--> {t}"),
}


def load_kg(path, relations=None):
    """Load only the needed columns/relations (kg_subset.csv can be large)."""
    relations = set(relations or RELATION_SPECS)
    chunks = []
    for chunk in pd.read_csv(path, usecols=KG_COLUMNS, dtype=str, chunksize=500_000, low_memory=False):
        chunks.append(chunk[chunk["relation"].isin(relations)])
    df = pd.concat(chunks, ignore_index=True).dropna()
    for col in ("x_name", "y_name"):
        df[col] = df[col].str.strip()
    return df.drop_duplicates()


def display(name):
    """Name as shown in questions/answers: drop trailing qualifiers like '(disease)'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip() or name


# HPO terms that are not clinical features (inheritance modes, onset, vague "Abnormality of ..." groupers)
GENERIC_PHENOTYPE = re.compile(
    r"inheritance|^sporadic$|^somatic mutation$|penetrance|expressivity|phenotypic variability|"
    r"genetic anticipation|\bonset$|^(progressive|chronic|acute|episodic|recurrent)$|abnormality of |^abnormal .*(morphology|physiology)$|^heterogeneous$", re.I)


def build_adjacency(df, blocklist=None, blocked_entities=None):
    """
    Return {relation: {center_name: set(target_names)}}, normalising direction.
    PrimeKG stores many edges in both directions, so each row is oriented by node type
    (this replaces the old get_pair() fix for reversed entities).
    """
    blocklist, blocked = blocklist or set(), blocked_entities or set()

    def keys(name):
        return {name.lower(), display(name).lower()}

    adj = {rel: defaultdict(set) for rel in RELATION_SPECS}
    for rel, (ctype, ttype, _) in RELATION_SPECS.items():
        sub = df[df["relation"] == rel]
        fwd = sub[(sub["x_type"] == ctype) & (sub["y_type"] == ttype)]
        rev = sub[(sub["x_type"] == ttype) & (sub["y_type"] == ctype)]
        pairs = list(zip(fwd["x_name"], fwd["y_name"])) + list(zip(rev["y_name"], rev["x_name"]))
        for c, t in pairs:
            if keys(c) & blocked or any((k, t.lower()) in blocklist for k in keys(c)):
                continue
            if ttype == "effect/phenotype" and GENERIC_PHENOTYPE.search(t):
                continue
            adj[rel][c].add(t)
    return adj


def verbalise(rel, center, target):
    return RELATION_SPECS[rel][2](center, target)


def read_lines(path):
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

# ===========================================================================
# Generate: KG portion + manual slots
# ===========================================================================
# Configuration: category quotas (total per category, of which `manual` are curated)
# ---------------------------------------------------------------------------
# quota = number of items per source; totals per category: 25/20/15/15/10/15
CATEGORIES = {
    "disease_drug":      {"code": "DDR", "slug": "disease-drug", "label": "Disease–Drug",
                          "quota": {"primekg": 12, "medqa": 6, "harrison": 7}},
    "disease_phenotype": {"code": "DPH", "slug": "disease-phenotype", "label": "Disease–Phenotype",
                          "quota": {"primekg": 8, "medqa": 6, "harrison": 6}},
    "drug_side_effect":  {"code": "DSE", "slug": "drug-side-effect", "label": "Drug–Side Effect",
                          "quota": {"primekg": 8, "medqa": 3, "harrison": 4}},
    "disease_gene":      {"code": "DGN", "slug": "disease-gene", "label": "Disease–Gene",
                          "quota": {"primekg": 12, "harrison": 3}},
    "entity_definition": {"code": "DEF", "slug": "entity-definition", "label": "Entity Definition",
                          "quota": {"primekg_node_description": 10}},
    "cross_domain":      {"code": "XDM", "slug": "cross-domain", "label": "Cross-Domain",
                          "quota": {"primekg": 5, "medqa": 5, "harrison": 5}},
}
SOURCES = ["primekg", "medqa", "primekg_node_description", "harrison"]
DIFFICULTY = {"primekg": "easy", "medqa": "hard", "primekg_node_description": "easy", "harrison": "medium"}
HARRISON_NOTE = ("Write from Harrison's (cite chapter in harrison_ref): a conceptual question "
                 "(mechanism, pathophysiology, why/how) and a 1-3 sentence answer; name key entities as in PrimeKG")

# Single-relation categories: relation, answer-size bounds, question templates, answer template
SINGLE_HOP = {
    "disease_drug": {
        "relation": "indication", "min": 2, "max": 8,
        "questions": ["Which drugs are indicated for the treatment of {e}?",
                      "What medications are used to treat {e}?"],
        "answer": "Drugs indicated for {e} include {items}.",
    },
    "disease_phenotype": {
        "relation": "disease_phenotype_positive", "min": 3, "max": 10,
        "questions": ["What clinical features are associated with {e}?",
                      "Which signs and symptoms are characteristic of {e}?"],
        "answer": "Clinical features associated with {e} include {items}.",
    },
    "drug_side_effect": {
        "relation": "drug_effect", "min": 3, "max": 10,
        "questions": ["What are the known adverse effects of {e}?",
                      "Which side effects are associated with {e}?"],
        "answer": "Known adverse effects of {e} include {items}.",
    },
    "disease_gene": {
        "relation": "disease_protein", "min": 2, "max": 8,
        "questions": ["Which genes are associated with {e}?",
                      "What genes or proteins are implicated in {e}?"],
        "answer": "Genes associated with {e} include {items}.",
    },
}

# Disease-name quality filter (keeps common clinical diseases, drops obscure genetic entries)
BAD_DISEASE_TERMS = ("obsolete", "autosomal", "x-linked", "susceptibility", "with or without",
                     "due to", "familial", "congenital", "deficiency of", "syndrome type")
BAD_DRUG = re.compile(r"\bacid$|^silver$|^water$|peroxide$|^(thiamine|pyridoxine|riboflavin|niacin)$", re.I)
CROSS_POOL_SIZE = 600   # candidate diseases considered for cross-domain pairing
CROSS_MAX_DRUG_DEGREE = 15        # ignore "do-everything" drugs (steroids, broad antibiotics) as links
CROSS_MAX_PHENOTYPE_DEGREE = 80  # ignore very common phenotypes as links


def capitalise(name):
    return name[0].upper() + name[1:]


def join_items(items):
    items = sorted(items, key=str.lower)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def disease_ok(name, whitelist):
    if whitelist:
        return name.lower() in whitelist
    n = name.lower()
    return (len(n) <= 50 and n.count("-") <= 1 and not re.search(r"\d", n)
            and not any(t in n for t in BAD_DISEASE_TERMS))


def ordered_candidates(names, priority, rng, usage):
    """Priority (MedQA) entities first, by MedQA rank; less-used entities first; the rest shuffled."""
    names = list(names)
    rng.shuffle(names)
    last = len(priority)
    return sorted(names, key=lambda n: (n.lower() not in priority, usage[n.lower()],
                                        priority.get(n.lower(), last)))


SOURCE_LABEL = {
    "primekg": "PrimeKG",
    "medqa": "MedQA (USMLE) + PrimeKG",
    "primekg_node_description": "PrimeKG node description",
    "harrison": "Harrison's Principles of Internal Medicine ed.20",
}
HARRISON_WITH_KG = "PrimeKG + Harrison's Principles of Internal Medicine ed.20"


DATASET_FIELDS = ["question", "answer", "contexts", "source", "category", "difficulty"]


def save_dataset(samples, work_path, out_path):
    """
    work file  = every record incl. empty slots, with id + metadata (used by merge; keep it for yourself)
    out file   = the dataset: completed records only, with the six dataset fields
    """
    with open(work_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    clean = [{k: s[k] for k in DATASET_FIELDS} for s in samples if s["question"] and s["answer"]]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    return len(clean)


def make_sample(sid, cat, source, question="", answer="", contexts=None, entities=None,
                difficulty=None, medqa_id=None):
    """Record with the six dataset fields; bookkeeping fields live under "metadata"."""
    filled = bool(question and answer)
    return {
        "id": sid,
        "question": question,
        "answer": answer,
        "contexts": contexts or [],
        "source": SOURCE_LABEL[source],
        "category": CATEGORIES[cat]["slug"],
        "difficulty": difficulty or DIFFICULTY[source],
        "metadata": {
            "source_type": source,
            "key_entities": entities or [],
            "kg_supported": ("yes" if contexts else "no") if filled and source != "harrison" else None,
            "harrison_ref": None,
            "medqa_id": medqa_id,
            "status": "generated" if filled else "pending_curation",
        },
    }


SELFREF_STOP = {"disease", "syndrome", "disorder", "abnormal", "abnormality", "acute", "chronic", "primary",
                "secondary", "congenital", "type", "deficiency", "infection", "tumor", "cancer", "neoplasm"}


def self_referential(disease, phenotype):
    """True if the phenotype is the disease itself or a subtype of it (shares a key word)."""
    d, p = display(disease).lower(), phenotype.lower()
    if d in p or p in d:
        return True
    words = lambda x: {w for w in re.findall(r"[a-z]{5,}", x) if w not in SELFREF_STOP}
    return bool(words(d) & words(p))


def gen_single_hop(cat, n, adj, filt, priority, rng, usage):
    spec = SINGLE_HOP[cat]
    rel = spec["relation"]
    graph = adj[rel]
    if rel == "disease_phenotype_positive":  # drop subtypes of the disease itself (e.g. leukemia -> AML)
        graph = {c: {t for t in ts if not self_referential(c, t)} for c, ts in graph.items()}
    pool = [c for c, t in graph.items() if spec["min"] <= len(t) <= spec["max"] and filt(cat, c)]
    out, chosen = [], []
    for e in ordered_candidates(pool, priority, rng, usage):
        if len(out) >= n:
            break
        targets = sorted(graph[e], key=str.lower)
        tset = set(targets)
        if any(len(tset & c) / len(tset | c) >= 0.6 for c in chosen):  # near-duplicate answer
            continue
        chosen.append(tset)
        q = rng.choice(spec["questions"]).format(e=display(e))
        a = spec["answer"].format(e=display(e), items=join_items(targets))
        ctx = [verbalise(rel, e, t) for t in targets]
        out.append((q, a, ctx, [e] + targets))
        usage[e.lower()] += 1
    return out, len(pool)


def gen_cross_domain(n, adj, filt, priority, rng, usage):
    """Two diseases linked through shared phenotypes or shared indicated drugs (2-hop)."""
    variants = [
        ("disease_phenotype_positive", "Which clinical features are shared by {a} and {b}?",
         "{A} and {b} share the following clinical features: {items}.", 3, 8),
        ("indication", "Which drugs are indicated for both {a} and {b}?",
         "Drugs indicated for both {a} and {b} include {items}.", 1, 3),
    ]
    found = []
    for rel, qt, at, lo, hi in variants:
        g = adj[rel]
        degree = Counter(t for targets in g.values() for t in targets)
        max_deg = CROSS_MAX_DRUG_DEGREE if rel == "indication" else CROSS_MAX_PHENOTYPE_DEGREE
        pool = [d for d in g if filt("cross_domain", d) and len(g[d]) >= lo]
        pool = ordered_candidates(pool, priority, rng, usage)[:CROSS_POOL_SIZE]
        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                shared = {t for t in g[a] & g[b] if degree[t] <= max_deg}
                # skip near-duplicate diseases (e.g., subtypes sharing almost everything)
                if lo <= len(shared) <= hi and len(shared) < 0.8 * min(len(g[a]), len(g[b])):
                    found.append((rel, qt, at, a, b, shared))
    rng.shuffle(found)
    found.sort(key=lambda f: -((f[3].lower() in priority) + (f[4].lower() in priority)))

    out, used, per_variant = [], set(), Counter()
    cap = n // 2 + n % 2  # balance phenotype vs drug variants
    for rel, qt, at, a, b, shared in found:
        if len(out) >= n:
            break
        if a in used or b in used or per_variant[rel] >= cap:
            continue
        items = sorted(shared, key=str.lower)
        ctx = [verbalise(rel, d, t) for t in items for d in (a, b)]
        out.append((qt.format(a=display(a), b=display(b)), at.format(A=capitalise(display(a)), a=display(a), b=display(b),
                                                     items=join_items(items)), ctx, [a, b] + items))
        used |= {a, b}
        per_variant[rel] += 1
        usage[a.lower()] += 1
        usage[b.lower()] += 1
    if len(out) < n:  # if one variant ran dry, let the other fill the gap
        for rel, qt, at, a, b, shared in found:
            if len(out) >= n:
                break
            if a in used or b in used:
                continue
            items = sorted(shared, key=str.lower)
            ctx = [verbalise(rel, d, t) for t in items for d in (a, b)]
            out.append((qt.format(a=display(a), b=display(b)), at.format(A=capitalise(display(a)), a=display(a), b=display(b),
                                                         items=join_items(items)), ctx, [a, b] + items))
            used |= {a, b}
    return out, len(found)


def cmd_generate(args):
    rng = random.Random(args.seed)
    priority = {}  # name -> rank (files from the medqa command are already sorted by score)
    for p in args.priority or []:
        for rank, x in enumerate(read_lines(p)):
            priority[x.lower()] = min(rank, priority.get(x.lower(), rank))
    whitelist = {x.lower() for x in read_lines(args.whitelist)}
    if args.blocklist is None and (DATA_DIR / "pair_blocklist.txt").exists():
        args.blocklist = str(DATA_DIR / "pair_blocklist.txt")
    lines = read_lines(args.blocklist)
    def_excluded = {ln.split(":", 1)[1].strip().lower() for ln in lines if ln.lower().startswith("def:")}
    lines = [ln for ln in lines if not ln.lower().startswith("def:")]
    blocklist = {tuple(p.strip().lower() for p in ln.split("||")) for ln in lines if "||" in ln}
    blocked_entities = {ln.strip().lower() for ln in lines if "||" not in ln}
    if lines:
        print(f"Blocklist: {len(blocklist)} edge(s), {len(blocked_entities)} entit(ies) from {args.blocklist}")
    if args.priority_only and not priority:
        raise SystemExit("--priority-only needs --priority <file>")

    print(f"Loading KG from {args.kg} ...")
    adj = build_adjacency(load_kg(args.kg), blocklist, blocked_entities)

    priority_has_drugs = any(d.lower() in priority for d in adj["drug_effect"])

    def filt(cat, name):
        if cat == "drug_side_effect" and BAD_DRUG.search(name):
            return False
        if cat == "drug_side_effect":  # drug-centred: no disease rules; MedQA drugs only if requested
            return not (args.priority_only and priority_has_drugs) or name.lower() in priority
        if args.priority_only:  # disease categories restricted to MedQA-matched diseases
            return name.lower() in priority and (not whitelist or name.lower() in whitelist)
        if not whitelist and name.lower() in priority:  # MedQA-matched diseases bypass the name heuristic
            return True
        return disease_ok(name, whitelist)

    usage = Counter()
    by_cat = {cat: [] for cat in CATEGORIES}   # (source, sample-without-id)
    pools = {}

    # 1) PrimeKG triples
    for cat, cfg in CATEGORIES.items():
        n = cfg["quota"].get("primekg", 0)
        if not n:
            continue
        if cat in SINGLE_HOP:
            items, pools[cat] = gen_single_hop(cat, n, adj, filt, priority, rng, usage)
            diff = "easy"
        else:
            items, pools[cat] = gen_cross_domain(n, adj, filt, priority, rng, usage)
            diff = "medium"
        for q, a, ctx, ents in items:
            by_cat[cat].append(("primekg", dict(question=q, answer=a, contexts=ctx, entities=ents, difficulty=diff)))

    # 2) MedQA subset linked to PrimeKG
    medqa_quota = {cat: cfg["quota"].get("medqa", 0) for cat, cfg in CATEGORIES.items()}
    if Path(args.medqa).exists() and sum(medqa_quota.values()):
        items, cands, _, _ = analyse_medqa(adj, args.medqa)
        by_id = {c["medqa_id"]: c for c in cands}
        sel_path = Path(args.medqa_selection)
        if sel_path.exists():
            with open(sel_path, encoding="utf-8-sig") as f:
                label_to_cat = {cfg["label"]: cat for cat, cfg in CATEGORIES.items()}
                picked = {r["medqa_id"].strip(): label_to_cat.get(r["category"].strip(), r["category"].strip())
                          for r in csv.DictReader(f) if r.get("medqa_id", "").strip()}
            print(f"MedQA selection read from {sel_path} ({len(picked)} questions)")
        else:
            picked = select_medqa(cands, medqa_quota, rng)
            with open(sel_path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["medqa_id", "category", "answer", "kg_triples", "question"])
                for mid, cat in picked.items():
                    w.writerow([mid, CATEGORIES[cat]["label"], items[mid].get("answer", ""),
                                " | ".join(verbalise(*t) for t in by_id[mid]["triples"]),
                                " ".join(items[mid].get("question", "").split())[:300]])
            print(f"MedQA selection written to {sel_path} ({len(picked)} questions) - review it")
        for mid, cat in picked.items():
            if mid not in items or cat not in CATEGORIES:
                print(f"  ! {mid}: unknown id or category, skipped")
                continue
            q, ref = medqa_text(items[mid])
            triples = by_id[mid]["triples"] if mid in by_id else []
            ents = sorted({c for _, c, _ in triples} | {t for _, _, t in triples})
            by_cat[cat].append(("medqa", dict(question=q, answer=ref, contexts=[verbalise(*t) for t in triples],
                                              entities=ents, medqa_id=mid)))
            for e in ents:
                usage[e.lower()] += 1
    elif sum(medqa_quota.values()):
        print(f"! {args.medqa} not found: MedQA slots left for manual filling")

    # 3) PrimeKG node descriptions -> entity definitions
    n_def = sum(cfg["quota"].get("primekg_node_description", 0) for cfg in CATEGORIES.values())
    features = find_features_file(args.features)
    if n_def and features and features.exists():
        defs = load_definitions(features, args.kg)
        kg_diseases = {d for rel in ("indication", "disease_phenotype_positive", "disease_protein") for d in adj[rel]}
        pool = [d for d in kg_diseases if d.lower() in defs and filt("disease_phenotype", d)
                and d.lower() not in def_excluded and display(d).lower() not in def_excluded]
        pools["entity_definition"] = len(pool)
        for cat, cfg in CATEGORIES.items():
            n = cfg["quota"].get("primekg_node_description", 0)
            for d in ordered_candidates(pool, priority, rng, usage)[:n]:
                name, text = defs[d.lower()]
                by_cat[cat].append(("primekg_node_description", dict(
                    question=f"What is {display(d)}?", answer=definition_answer(d, text),
                    contexts=[f"{d} --[definition]--> {text}"], entities=[d])))
                usage[d.lower()] += 1
    elif n_def:
        print("! disease_features file not found (PrimeKG node descriptions): definition slots left for manual filling")

    # 4) assemble: generated items, then placeholders for Harrison + any shortfall
    samples, template_rows, report = [], [], []
    for cat, cfg in CATEGORIES.items():
        rows, counts, short = [], Counter(), Counter()
        for source in SOURCES:
            want = cfg["quota"].get(source, 0)
            got = [kw for src, kw in by_cat[cat] if src == source][:want]
            counts[source] = want if source == "harrison" else len(got)
            short[source] = want - len(got) if source != "harrison" else 0
            rows += [(source, kw) for kw in got] + [(source, None)] * (want - len(got))
        for k, (source, kw) in enumerate(rows, 1):
            sid = f"{cfg['code']}-{k:03d}"
            if kw:
                samples.append(make_sample(sid, cat, source, **kw))
            else:
                samples.append(make_sample(sid, cat, source))
                note = HARRISON_NOTE if source == "harrison" else f"automatic {source} generation fell short - fill manually"
                template_rows.append({"id": sid, "category": cfg["label"], "source": source, "question": "",
                                      "answer": "", "key_entities": "", "harrison_ref": "",
                                      "difficulty": DIFFICULTY[source], "notes": note})
        report.append((cfg["label"], len(rows), counts, short, pools.get(cat, "")))

    n_done = save_dataset(samples, args.work, args.out)
    with open(args.template, "w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel shows "–" correctly
        w = csv.DictWriter(f, fieldnames=["id", "category", "source", "question", "answer", "key_entities",
                                          "harrison_ref", "difficulty", "notes"])
        w.writeheader()
        w.writerows(template_rows)

    print(f"\n{'Category':<20}{'Total':>6}{'KG':>5}{'MedQA':>7}{'Def':>5}{'Harrison':>10}{'KG pool':>9}")
    for label, total, counts, short, pool in report:
        flag = "".join(f"  <- {n} {src} short" for src, n in short.items() if n > 0)
        print(f"{label:<20}{total:>6}{counts['primekg']:>5}{counts['medqa']:>7}"
              f"{counts['primekg_node_description']:>5}{counts['harrison']:>10}{str(pool):>9}{flag}")
    by_source = Counter(s["metadata"]["source_type"] for s in samples if s["metadata"]["status"] == "generated")
    print(f"\nTotal: {len(samples)} samples; generated {sum(by_source.values())} "
          f"({', '.join(f'{k} {v}' for k, v in by_source.items())}); "
          f"{len(template_rows)} slot(s) to write in {args.template}")
    print(f"Wrote {args.out} ({n_done} completed records)")
    print(f"Wrote {args.work} (working file with ids + metadata, used by merge)")

# ===========================================================================
# Merge: manually written (Harrison) rows
# ===========================================================================
def build_index(adj):
    """name(lower) -> canonical name, and canonical name -> [(rel, center, target), ...]."""
    names, triples = {}, defaultdict(list)
    for rel, graph in adj.items():
        for c, targets in graph.items():
            names.setdefault(c.lower(), c)
            for t in targets:
                names.setdefault(t.lower(), t)
                triples[c].append((rel, c, t))
                triples[t].append((rel, c, t))
    return names, triples


def match_entity(ent, names, keys):
    e = ent.strip().lower()
    for variant in (e, f"{e} (disease)"):  # PrimeKG often suffixes ambiguous disease names
        if variant in names:
            return names[variant], "exact"
    close = difflib.get_close_matches(e, keys, n=1, cutoff=0.9)
    return (names[close[0]], "fuzzy") if close else (None, "none")


def suggest(ent, keys, names, n=3):
    """Closest PrimeKG names for an entity that was not found (printed so key_entities can be fixed)."""
    return [names[k] for k in difflib.get_close_matches(ent.strip().lower(), keys, n=n, cutoff=0.6)]


def mentioned(name, text):
    """Whole-word mention. Gene symbols (all caps, e.g. ACE) must match case, so 'ACE inhibitors' is not gene ACE."""
    if name.isupper() and len(name) <= 10:
        return re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])(?! inhibitor)", text) is not None
    return re.search(r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)", text.lower()) is not None


def cmd_merge(args):
    with open(args.work, encoding="utf-8") as f:
        samples = json.load(f)
    by_id = {s["id"]: s for s in samples}

    print(f"Loading KG from {args.kg} ...")
    adj_all = build_adjacency(load_kg(args.kg))
    names, triples = build_index(adj_all)
    treats = defaultdict(set)  # drug (lower) -> diseases it is indicated for (lower)
    for d, drugs in adj_all["indication"].items():
        for drug in drugs:
            treats[drug.lower()].add(d.lower())
    keys = list(names)

    merged, report = 0, []
    with open(args.curated, encoding="utf-8-sig") as f:  # -sig: Excel's "CSV UTF-8" adds a BOM
        for row in csv.DictReader(f):
            q, a = row.get("question", "").strip(), row.get("answer", "").strip()
            sid = row.get("id", "").strip()
            if not (q and a):
                continue
            if sid not in by_id:
                print(f"  ! {sid}: id not found in dataset, skipped")
                continue
            ents = [e.strip() for e in row.get("key_entities", "").split(";") if e.strip()]
            matched, notes = [], []
            for e in ents:
                canon, how = match_entity(e, names, keys)
                if canon:
                    matched.append(canon)
                    if how == "fuzzy":
                        notes.append(f"'{e}' fuzzy-matched to '{canon}'")
                else:
                    hint = suggest(e, keys, names)
                    notes.append(f"'{e}' not found in PrimeKG" + (f" (closest: {', '.join(hint)})" if hint else ""))

            # supporting triples: the other end is mentioned in the answer, or links two key entities.
            # Key-entity phrases are blanked out first, so "fever" inside "rheumatic fever" is not a mention.
            matched_l = {m.lower() for m in matched}
            a_rest = a
            for phrase in sorted(set(ents) | set(matched) | {display(m) for m in matched}, key=len, reverse=True):
                if len(phrase.split()) > 1:
                    a_rest = re.sub(re.escape(phrase), " ", a_rest, flags=re.I)
            ctx = []
            for m in matched:
                for rel, c, t in triples[m]:
                    other = t if c == m else c
                    if c.lower() == t.lower() or display(c).lower() == t.lower():
                        continue  # self-loop, e.g. "Burkitt lymphoma --[phenotype_present]--> Burkitt lymphoma"
                    if rel == "drug_effect" and c.lower() not in matched_l:
                        continue  # side effects only of drugs the QA is about (not e.g. pyridoxine via "neuropathy")
                    if rel == "drug_effect" and any(t.lower() in d or d in t.lower() for d in treats.get(c.lower(), ())):
                        continue  # drug's own indication listed as its "side effect" (SIDER artefact)
                    # multi-word names count when mentioned anywhere; single words only outside key-entity phrases
                    hit = mentioned(other, a) if len(other.split()) > 1 else mentioned(other, a_rest)
                    if hit or other.lower() in matched_l:
                        s = verbalise(rel, c, t)
                        if s not in ctx:
                            ctx.append(s)

            if ents and len(matched) == len(ents) and ctx:
                support = "yes"
            elif matched and ctx:
                support = "partial"
            else:
                support = "no"

            s = by_id[sid]
            s.update({"question": q, "answer": a, "contexts": ctx,
                      "difficulty": row.get("difficulty", "").strip() or s.get("difficulty")})
            if s["metadata"]["source_type"] == "harrison":
                s["source"] = HARRISON_WITH_KG if ctx else SOURCE_LABEL["harrison"]
            s["metadata"].update({
                "key_entities": ents, "kg_supported": support,
                "harrison_ref": row.get("harrison_ref", "").strip() or None,
                "status": "curated_needs_review",
                "curation_notes": "; ".join(filter(None, notes + [
                    n for n in [row.get("notes", "").strip()] if n != HARRISON_NOTE and "fell short" not in n])) or None,
            })
            merged += 1
            report.append((sid, support, len(ctx), notes))

    n_done = save_dataset(samples, args.work, args.out)

    print(f"\nMerged {merged} curated sample(s):")
    for sid, sup, n, notes in report:
        print(f"  {sid:<9} kg_supported={sup:<8} contexts={n:<3} {' | '.join(notes)}")
    pending = sum(1 for s in samples if s["metadata"]["status"] == "pending_curation")
    print(f"\nStill pending curation: {pending}")
    print(f"Wrote {args.out} ({n_done} completed records) and {args.work}")

# ===========================================================================
# MedQA: priority lists + curation candidates
# ===========================================================================
# Generic names that would match almost any vignette, and drug names that are everyday words
MEDQA_STOP = {"disease", "disorder", "syndrome", "infection", "neoplasm", "carcinoma", "cancer", "tumor",
              "lesion", "injury", "inflammation", "deficiency", "anomaly", "malformation", "pain",
              "water", "oxygen", "iron", "calcium", "sodium", "potassium", "magnesium", "zinc", "copper",
              "alcohol", "ethanol", "glucose", "sucrose", "nitrogen", "sulfur", "air", "carbon dioxide"}
MEDQA_MAX_NGRAM = 6
ASK_TREATMENT = re.compile(r"treatment|management|therapy|pharmacotherap|drug|medication|prescribe", re.I)
ASK_ADVERSE = re.compile(r"adverse|side effect|toxicity|responsible for|interact|"
                         r"caus(e|ed|ing) (of )?(this|these|the patient)", re.I)
MEDQA_MAX_ANSWER_WORDS = 5  # longer answers are management sentences, not entities


def norm_text(text):
    return re.sub(r"[^a-z0-9'\- ]+", " ", text.lower()).split()


def build_name_index(adj):
    """normalised name -> {node_type: canonical KG name} for diseases, drugs and phenotypes."""
    index = defaultdict(dict)

    def add(name, ntype):
        for variant in {name, display(name)}:
            key = " ".join(norm_text(variant))
            if len(key) >= 4 and key not in MEDQA_STOP and len(key.split()) <= MEDQA_MAX_NGRAM:
                index[key].setdefault(ntype, name)

    for rel, (ctype, ttype, _) in RELATION_SPECS.items():
        for c, targets in adj[rel].items():
            add(c, ctype)
            for t in targets:
                add(t, ttype)
    return index


def find_entities(text, index):
    """Greedy longest-match of KG names in text -> {node_type: [canonical names in order]}."""
    toks, found, i = norm_text(text), defaultdict(list), 0
    while i < len(toks):
        for n in range(min(MEDQA_MAX_NGRAM, len(toks) - i), 0, -1):
            hit = index.get(" ".join(toks[i:i + n]))
            if hit:
                for ntype, name in hit.items():
                    if name not in found[ntype]:
                        found[ntype].append(name)
                i += n
                break
        else:
            i += 1
    return found


FAMILY = re.compile(r"\b(father|mother|brother|sister|sibling|grand(father|mother|parent)|uncle|aunt|"
                    r"cousin|son|daughter|parents?|family history|relatives?)\b", re.I)
HISTORY = re.compile(r"\b(history of|previously|diagnosed with .{0,40}(years?|months?|weeks?|days?) ago|"
                     r"(years?|months?|weeks?) ago,? (she|he) was diagnosed|in the past)\b", re.I)
NEGATION = re.compile(r"\b(denies|denied|no|not|without|negative for|absence of|neither|nor|free of)\b", re.I)
ASK_START = re.compile(r"\b(Which|What|How|Why|Of the following|In addition)\b")


def ask_sentence(q):
    """The actual question: from the last 'Which/What/...' to the end (lab tables often precede it)."""
    starts = list(ASK_START.finditer(q))
    return q[starts[-1].start():].strip() if starts else re.split(r"(?<=[.!?])\s+", q.strip())[-1]
ACTION_ANSWER = re.compile(r"^(discontinu|stop|reduc|decreas|increas|switch|replac|hold|withhold|avoid|continu|taper)", re.I)
ASK_THERAPY = re.compile(r"treat|management|therap|most appropriate|first-line|initial|prevent|recommend", re.I)
ASK_DIAGNOSIS = re.compile(r"diagnos|most likely (cause|explanation|condition)|associated with", re.I)
ASK_SIDE_EFFECT = re.compile(r"adverse|side effect|toxicity|complication|risk|monitor|laborator|abnormalit|"
                             r"associated with|caus(e|ed|ing)", re.I)
ASK_EXCLUDE_TREATMENT = re.compile(r"contraindicat|avoid|mechanism|similar|caus|responsible|interact|"
                                   r"contribut|discontinu|predispose", re.I)


# symptoms shared by too many conditions/drugs to identify anything on their own
GENERIC_SYMPTOMS = {"fever", "headache", "nausea", "vomiting", "fatigue", "malaise", "pain", "abdominal pain",
                    "chills", "dizziness", "weakness", "anorexia", "lethargy", "chest pain", "dyspnea"}


def analyse_medqa(adj, path):
    """
    Parse MedQA and link each question to PrimeKG.
    Returns (items, candidates, disease_score, drug_score):
      items       medqa_id -> raw record (question, answer, options, answer_idx)
      candidates  list of dicts with category, triples linking question and answer, cross flag
    Diseases mentioned only in a family-history sentence are ignored (e.g. "her father has ...").
    """
    index = build_name_index(adj)
    ind, phen, eff = adj["indication"], adj["disease_phenotype_positive"], adj["drug_effect"]
    items, cands, dis_score, drug_score = {}, [], Counter(), Counter()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            mid = f"medqa-{line_no:05d}"
            items[mid] = item
            q = item.get("question", "")
            ans = clean_answer(item.get("answer", ""))
            ask = ask_sentence(q)
            sentences = re.split(r"(?<=[.!?])\s+", q.strip())
            in_q, history, prominent = defaultdict(list), set(), set()
            for pos, sent in enumerate(sentences):
                if FAMILY.search(sent) or NEGATION.search(sent):  # "her father has...", "denies fever..."
                    continue
                for ntype, names in find_entities(sent, index).items():
                    in_q[ntype] += [n for n in names if n not in in_q[ntype]]
                    if HISTORY.search(sent):
                        history.update(names)
                    if pos < 3 or pos == len(sentences) - 1:  # opening sentences or the question itself
                        prominent.update(names)
            in_a = find_entities(ans, index) if len(ans.split()) <= MEDQA_MAX_ANSWER_WORDS else defaultdict(list)
            for d in in_q["disease"]:
                dis_score[d] += 1
            for d in in_q["drug"]:
                drug_score[d] += 1
            for d in in_a["disease"]:
                dis_score[d] += 3   # the answer is the diagnosis
            for d in in_a["drug"]:
                drug_score[d] += 3  # the answer is the drug

            cat, triples, entity, history_link = None, [], None, False
            if in_a["drug"] and not ACTION_ANSWER.search(ans.strip()):
                drug = entity = in_a["drug"][0]
                effects = [p for p in in_q["effect/phenotype"] if p in eff.get(drug, ())]
                diseases = [d for d in in_q["disease"] if drug in ind.get(d, ())]
                treat_first = re.search(r"\btreat|therap|management|mortality|survival|"
                                        r"(lower|reduc|decreas)\w* (her |his |the )?risk|prevent", ask, re.I)
                if ASK_SIDE_EFFECT.search(ask) and effects and not treat_first:
                    cat, triples = "drug_side_effect", [("drug_effect", drug, p) for p in effects]
                elif ASK_TREATMENT.search(ask) and not ASK_EXCLUDE_TREATMENT.search(ask) and diseases:
                    current = [d for d in diseases if d not in history]
                    cat, triples = "disease_drug", [("indication", (current or diseases)[0], drug)]
                    history_link = not current  # treated disease only named as past history
            elif in_a["disease"]:
                dis = entity = in_a["disease"][0]
                hits = [p for p in in_q["effect/phenotype"] if p in phen.get(dis, ())]
                if ASK_DIAGNOSIS.search(ask) and len(hits) >= 3:
                    cat, triples = "disease_phenotype", [("disease_phenotype_positive", dis, p) for p in hits]
            elif in_a["effect/phenotype"]:
                entity = in_a["effect/phenotype"][0]
                drugs = [d for d in in_q["drug"] if entity in eff.get(d, ())]
                if drugs and ASK_SIDE_EFFECT.search(ask):
                    cat, triples = "drug_side_effect", [("drug_effect", drugs[0], entity)]
            if not cat:
                continue
            specific = [t for t in triples if t[2].lower() not in GENERIC_SYMPTOMS]
            if cat in ("disease_phenotype", "drug_side_effect") and not specific:
                continue  # only generic symptoms link question and answer
            linked = {c for _, c, _ in triples} | {t for _, _, t in triples}
            other_diseases = [d for d in in_q["disease"] if d not in linked]
            cands.append({"medqa_id": mid, "category": cat, "cross": bool(other_diseases),
                          "history_link": history_link, "therapy_ask": bool(ASK_THERAPY.search(ask)),
                          "n_specific": len(specific),
                          "prominent": cat != "disease_drug" or triples[0][1] in prominent,
                          "answer_entity": entity, "triples": triples,
                          "diseases_in_question": in_q["disease"], "ask": ask})
    return items, cands, dis_score, drug_score


def clean_answer(ans):
    return " ".join(str(ans).replace('"', " ").split()).strip()


def medqa_text(item):
    """question = vignette + options (MedQA is multiple choice); answer = the correct option."""
    q = " ".join(item.get("question", "").split()).strip().strip('"').strip()
    opts = item.get("options") or {}
    if opts:
        q += "\nOptions:\n" + "\n".join(f"{k}. {clean_answer(v)}" for k, v in sorted(opts.items()))
    letter = item.get("answer_idx")
    ans = clean_answer(item.get("answer", "")).rstrip(".")
    ref = f"The correct answer is {letter}. {ans}." if letter else f"The correct answer is {ans}."
    return q, ref


def select_medqa(cands, quotas, rng):
    """Pick MedQA questions per category: most PrimeKG triples first, no repeated answer or disease."""
    picked, seen = {}, set()
    order = sorted(cands, key=lambda c: (-c["n_specific"], -len(c["triples"]), rng.random()))
    picked_targets = []
    for cat in ("cross_domain", "disease_drug", "disease_phenotype", "drug_side_effect"):
        need = quotas.get(cat, 0)
        for c in order:
            if need <= 0:
                break
            if c["medqa_id"] in picked:
                continue
            if cat == "cross_domain" and not (c["cross"] and c["prominent"]):
                continue
            if cat != "cross_domain" and (c["category"] != cat or c["history_link"]):
                continue
            if cat == "disease_drug" and not (c["therapy_ask"] and c["prominent"]):
                continue  # "which drug is she taking?", or the linked disease is only a side finding
            targets = {t[2].lower() for t in c["triples"]}
            if any(len(targets & p) / len(targets | p) >= 0.5 for p in picked_targets):
                continue  # same topic as an earlier pick (e.g. two NMS questions)
            keys = {c["answer_entity"].lower()} | {d.lower() for d in c["diseases_in_question"][:2]}
            if keys & seen:
                continue
            picked[c["medqa_id"]] = cat
            picked_targets.append(targets)
            seen |= keys
            need -= 1
    return picked


def cmd_medqa(args):
    print(f"Loading KG from {args.kg} ...")
    adj = build_adjacency(load_kg(args.kg))
    items, cands, dis_score, drug_score = analyse_medqa(adj, args.medqa)
    diseases = [d for d, sc in dis_score.most_common() if sc >= args.min_score]
    drugs = [d for d, sc in drug_score.most_common() if sc >= args.min_score]

    out_dir = Path(args.out_dir)
    for name, names in (("medqa_diseases.txt", diseases), ("medqa_drugs.txt", drugs)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            f.write(f"# from {Path(args.medqa).name}, ranked by score (answer x3 + question x1), min {args.min_score}\n")
            f.write("\n".join(names) + "\n")
    with open(out_dir / "medqa_candidates.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["medqa_id", "category", "cross_domain", "answer", "answer_kg_entity", "kg_triples",
                    "diseases_in_question", "question"])
        for c in sorted(cands, key=lambda c: (c["category"], -len(c["triples"]))):
            item = items[c["medqa_id"]]
            w.writerow([c["medqa_id"], CATEGORIES[c["category"]]["label"], "yes" if c["cross"] else "",
                        item.get("answer", ""), c["answer_entity"],
                        " | ".join(verbalise(*t) for t in c["triples"]),
                        "; ".join(c["diseases_in_question"]), " ".join(item.get("question", "").split())])

    print(f"\nMedQA questions read: {len(items)}")
    print(f"Diseases matched (score >= {args.min_score}): {len(diseases)}   e.g. {', '.join(diseases[:8])}")
    print(f"Drugs matched    (score >= {args.min_score}): {len(drugs)}   e.g. {', '.join(drugs[:8])}")
    print(f"Questions linked to PrimeKG: {len(cands)}")
    for cat, n in Counter(c["category"] for c in cands).most_common():
        print(f"  {CATEGORIES[cat]['label']:<20}{n:>5}  (cross-domain: {sum(1 for c in cands if c['category'] == cat and c['cross'])})")
    print(f"Wrote medqa_diseases.txt, medqa_drugs.txt, medqa_candidates.csv to {out_dir}")

# ===========================================================================
# PrimeKG node descriptions (entity definitions)
# ===========================================================================
DEF_COLUMNS = ("mondo_definition", "umls_description", "orphanet_definition")


def find_features_file(path):
    if path:
        return Path(path)
    for name in ("disease_features.tab", "disease_features.csv", "disease_features.tsv",
                 "disease_feature.tab", "disease_feature.csv"):
        if (DATA_DIR / name).exists():
            return DATA_DIR / name
    return None


def load_definitions(features_path, kg_path):
    """KG disease name (lower) -> (KG name, definition) from PrimeKG disease_features."""
    df = None
    for sep in (",", "\t"):
        try:
            cand = pd.read_csv(features_path, sep=sep, dtype=str, low_memory=False)
        except Exception:
            continue
        if any(c in cand.columns for c in DEF_COLUMNS):
            df = cand
            break
    if df is None:
        raise SystemExit(f"{features_path}: no definition columns ({', '.join(DEF_COLUMNS)}) found")

    # node_index -> KG name, so definitions attach to the exact node name used in kg_subset.csv
    idx_name = {}
    if "node_index" in df.columns:
        try:
            kg = pd.read_csv(kg_path, usecols=["x_index", "x_type", "x_name"], dtype=str)
            kg = kg[kg["x_type"] == "disease"].drop_duplicates("x_index")
            idx_name = dict(zip(kg["x_index"], kg["x_name"]))
        except ValueError:
            pass  # kg_subset.csv without x_index: fall back to the names in the features file

    defs = {}
    for _, row in df.iterrows():
        text = next((row[c] for c in DEF_COLUMNS if c in df.columns and isinstance(row[c], str)
                     and len(row[c].strip()) > 30), None)
        if not text:
            continue
        text = " ".join(text.replace('"', "").split())
        sents = re.split(r"(?<=[.!?])\s+", text)
        text = sents[0] if len(sents) == 1 or len(sents[0]) + len(sents[1]) > 350 else " ".join(sents[:2])
        names = [idx_name.get(row.get("node_index"))] + [row.get(c) for c in ("mondo_name", "group_name_bert", "node_name")]
        for name in filter(lambda x: isinstance(x, str) and x.strip(), names):
            defs.setdefault(name.strip().lower(), (name.strip(), text))
    return defs


def definition_answer(name, text):
    n = capitalise(display(name))
    if re.match(r"(a|an|the)\s", text, re.I):
        return f"{n} is {text[0].lower()}{text[1:]}"
    return f"{n}: {text}"

# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg", default=str(DATA_DIR / "kg_subset.csv"), help="PrimeKG-format CSV")
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="build dataset + curation template")
    g.add_argument("--priority", nargs="+", help="file(s) of diseases/drugs to prioritise, one per line (from the medqa command)")
    g.add_argument("--whitelist", help="strict disease whitelist, one per line (only these are used)")
    g.add_argument("--blocklist", help="default data/pair_blocklist.txt if present; lines 'center || target' drop an edge, "
                        "a line with just a name drops that entity, 'def: name' skips it for definitions")
    g.add_argument("--priority-only", action="store_true",
                   help="use only entities listed in --priority (diseases; drugs too if a drug list is given)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default=str(DATA_DIR / "ground_truth_dataset.json"), help="dataset (completed records)")
    g.add_argument("--work", default=str(DATA_DIR / "ground_truth_work.json"), help="working file with ids + metadata")
    g.add_argument("--template", default=str(DATA_DIR / "curation_template.csv"))
    g.add_argument("--medqa", default=str(DATA_DIR / "medqa_train.jsonl"))
    g.add_argument("--medqa-selection", default=str(DATA_DIR / "medqa_selected.csv"),
                   help="MedQA ids + category to use; written on first run, edit and re-run to change")
    g.add_argument("--features", help="PrimeKG disease_features file (default: data/disease_features.csv or .tab)")

    m = sub.add_parser("merge", help="merge filled curation template into the dataset")
    m.add_argument("--curated", default=str(DATA_DIR / "curation_template.csv"))
    m.add_argument("--work", default=str(DATA_DIR / "ground_truth_work.json"), help="working file from generate")
    m.add_argument("--out", default=str(DATA_DIR / "ground_truth_dataset.json"), help="dataset (completed records)")

    q = sub.add_parser("medqa", help="priority lists + MedQA questions linked to PrimeKG")
    q.add_argument("--medqa", default=str(DATA_DIR / "medqa_train.jsonl"))
    q.add_argument("--min-score", type=int, default=3, help="min score to enter the priority lists")
    q.add_argument("--out-dir", default=str(DATA_DIR))

    args = ap.parse_args()
    {"generate": cmd_generate, "merge": cmd_merge, "medqa": cmd_medqa}[args.command](args)


if __name__ == "__main__":
    main()
