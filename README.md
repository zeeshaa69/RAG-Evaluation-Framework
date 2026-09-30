# rag-eval

**Reproducible, offline-first evaluation of Retrieval-Augmented Generation (RAG) systems.**

`rag-eval` measures a RAG pipeline end to end and records exactly what was measured, how, and with which
versions, so two runs can be compared honestly. It is a measurement tool, not a chatbot. It needs no API keys,
no paid services and no network access: every built-in retriever, generator and metric is deterministic and
runs locally.

```
Dataset → Retriever → Retrieved Context → Generator → Evaluation → Metrics → Report
```

## Table of contents

- [Why RAG evaluation matters](#why-rag-evaluation-matters)
- [Metrics](#metrics)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick start (offline demo)](#quick-start-offline-demo)
- [CLI usage](#cli-usage)
- [Python usage](#python-usage)
- [Dataset format](#dataset-format)
- [Experiment format](#experiment-format)
- [Reproducibility](#reproducibility)
- [Dashboard](#dashboard)
- [Testing](#testing)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)

## Why RAG evaluation matters

A RAG system can fail in several independent places, and a fluent answer hides which one broke:

- the retriever never surfaced the right document (**retrieval failure**),
- it surfaced the right document but ranked noise above it (**ranking failure**),
- the generator ignored, misread or contradicted the context (**grounding failure**),
- the generator answered from memory with no supporting evidence (**hallucination**).

Changing a chunk size, an embedding model or a prompt can improve one of these and degrade another. Without
fixed datasets, versioned metrics and stored configurations, "it feels better" is the only signal available.
`rag-eval` separates retrieval quality from answer quality, keeps the evidence behind every score, and makes
runs comparable.

## Metrics

Every metric result carries four things: the **metric name**, the **score**, the **evidence** behind it and the
**evaluation method**. Metrics that are approximations are flagged `heuristic` everywhere (CLI, reports,
dashboard). A `null` score means "not applicable to this sample" (for example, no expected answer) and is
excluded from averages; it is never silently turned into 0 or 1.

### Retrieval metrics (exact computations against your relevance labels)

| Metric | Definition |
|---|---|
| `precision@k` | relevant documents in the top-k ÷ k (divides by k even if fewer were returned) |
| `recall@k` | relevant documents in the top-k ÷ total relevant documents |
| `hit_rate@k` | 1 if any relevant document is in the top-k, else 0 |
| `mrr@k` | 1 ÷ rank of the first relevant document within the top-k (0 if none) |
| `ndcg@k` | DCG ÷ ideal DCG, linear gain = relevance grade, log2(rank+1) discount |

Retrieved lists are de-duplicated by document id (first occurrence wins) before the cut-off is applied.
Labels may be binary or graded (see [Dataset format](#dataset-format)). These metrics are only as good as the
labels.

### Answer metrics

| Metric | Needs expected answer | Kind | What it does |
|---|---|---|---|
| `exact_match` | yes | formula | equality after SQuAD-style normalization |
| `token_f1` | yes | formula | token-level F1 (precision/recall in the evidence) |
| `contains_expected` | yes | heuristic | normalized expected answer occurs as a token span in the answer |
| `semantic_similarity` | yes | heuristic* | cosine similarity of embeddings of answer and expected answer |
| `context_relevance` | no | heuristic | mean coverage of the question's content terms by each retrieved chunk |
| `answer_relevance` | no | heuristic* | embedding similarity of question and answer (abstentions not scored) |
| `citation_coverage` | no | heuristic | share of answer sentences with at least one `[doc_id]` citation to a retrieved document |

\* The default embedder is a deterministic hashed character n-gram vectoriser. It captures surface
similarity, **not** neural semantics. Any object with `name` and `embed(text)` can be plugged in, and the
optional `embeddings` extra provides a `sentence-transformers` embedder for locally cached models.

### Faithfulness / hallucination

`faithfulness` and `contradiction_rate` come from a transparent, sentence-level pipeline. Each claim-bearing
answer sentence is compared with the retrieved context sentences and labelled:

- `supported`: at least 60% of its content terms and all of its numbers occur in one context sentence,
- `contradicted`: it overlaps strongly with a context sentence but disagrees on a number or on negation,
- `unsupported`: neither.

The per-sentence label, the matched context sentence and the reason are returned as evidence, so a person can
audit every decision. **These are lexical heuristics.** They miss paraphrases (a faithful paraphrase can be
labelled `unsupported`) and cannot check reasoning across sentences or world knowledge. Treat them as a
triage signal, not as proof of hallucination.

## Architecture

```
src/rag_eval/
  datasets.py      load / validate / version datasets, quality statistics
  retrievers/      BM25, TF-IDF, character n-gram, baselines, registry for custom retrievers
  generators.py    extractive & baseline generators, local Ollama adapter, callable adapter
  metrics/         retrieval.py, answer.py, faithfulness.py, embeddings.py
  failures.py      turns metric evidence into labelled failure cases
  experiment.py    ExperimentConfig + run_experiment (the pipeline)
  store.py         on-disk run storage (manifest, config, summary, per-sample results)
  compare.py       paired comparison of two runs with seeded bootstrap intervals
  report.py        Markdown / HTML / text rendering
  api.py           RagEval SDK facade
  cli.py serve.py  command line and local dashboard server
dashboard/         React + TypeScript dashboard (plain SVG charts, no chart dependency)
examples/demo/     offline dataset and experiment configs
```

Only the Python standard library is required at runtime.

## Installation

Requires Python 3.11+.

```bash
git clone <this repository> rag-eval && cd rag-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # add ,embeddings for sentence-transformers support
```

The dashboard needs Node 20.19+ only if you want to build it (`cd dashboard && npm install && npm run build`).

## Quick start (offline demo)

The repository ships a small hand-written corpus (32 short passages, 34 labelled questions including
multi-document, graded-relevance and unanswerable questions) and six experiment configs. No keys needed.

```bash
rag-eval dataset validate examples/demo/dataset
rag-eval experiment run examples/demo/experiments/bm25-extractive.json
rag-eval experiment run examples/demo/experiments/first-k-baseline.json
rag-eval experiment compare bm25-extractive first-k-baseline
rag-eval report generate bm25-extractive --format html --out report.html
```

The demo dataset is for demonstrating the tooling. It is **not a benchmark**: it is tiny, and its scores say
nothing about real-world RAG quality. The `control-number-fault` config wraps the extractive generator so that
it deliberately changes a number in its answer; it exists to check that the evaluator notices known faults,
not as a model to rank.

## CLI usage

```
rag-eval dataset validate PATH [--strict] [--json]     exit 1 on errors (or warnings with --strict)
rag-eval dataset quality PATH                          statistics and label-quality signals
rag-eval experiment run CONFIG [--results-dir runs] [--no-save]
rag-eval experiment list [--results-dir runs]
rag-eval experiment compare A B [--format text|md|json] [--seed N]
rag-eval report generate RUN [--format md|html] [--out FILE]
rag-eval dashboard serve [--dashboard-dir dashboard/dist] [--port 8765]
rag-eval dashboard export OUT_DIR                      static site with a frozen snapshot of results
```

`A`, `B` and `RUN` accept a full run id, a unique id prefix, or an experiment name (latest run with that
name). Exit codes: `0` success, `1` validation failed, `2` usage/runtime error.

## Python usage

```python
from rag_eval import RagEval, CallableGenerator

ev = RagEval(results_dir="runs")
print(ev.validate_dataset("examples/demo/dataset").ok)

baseline = ev.run("examples/demo/experiments/bm25-extractive.json")
mine = ev.run(
    {"name": "my-system", "dataset": "examples/demo/dataset", "top_k": 3},
    retriever=my_retriever,  # any object with index() / retrieve()
    generator=CallableGenerator(my_fn, model_id="my-model"),  # fn(question, contexts) -> str
)

print(mine.metric_means["recall@3"])
print(ev.format_comparison(ev.compare(baseline, mine)))
```

`examples/python_api.py` is a complete runnable version. Custom components can also be registered by name
(`register_retriever`, `register_generator`) so they can be used from JSON/TOML configs. To evaluate a local
model served by [Ollama](https://ollama.com), use `{"generator": {"type": "ollama", "params": {"model": "..."}}}`;
this talks only to `http://localhost:11434` by default.

## Dataset format

A dataset is a directory (or a single JSON file with `documents` and `samples` arrays):

```
my-dataset/
  dataset.json     optional: {"name": "...", "description": "..."}
  corpus.jsonl     {"id": "d1", "text": "...", "metadata": {...}}
  samples.jsonl    one sample per line
```

```json
{"id": "q1", "question": "Who created Git?", "expected_answer": "Linus Torvalds",
 "relevant_docs": ["git"], "metadata": {"type": "factoid"}}
```

- `id` and `question` are required; `expected_answer` may be `null` or absent (reference metrics are then
  "not applicable").
- `relevant_docs` is a list of ids, a list of `{"id": ..., "grade": n}` objects (graded relevance for NDCG),
  or an `{id: grade}` object. Every id must exist in the corpus.
- `rag-eval dataset validate` reports malformed JSON (with file and line), missing fields, duplicate ids,
  unknown document references and empty files as **errors**, and missing labels, empty documents and
  duplicate questions as **warnings**.
- The dataset **version** is a content hash of corpus and samples, stored with every run.

## Experiment format

JSON or TOML. Relative dataset paths are resolved against the config file's directory.

```json
{
  "name": "bm25-extractive",
  "description": "BM25 + single-sentence extractive generator",
  "dataset": "../dataset",
  "retriever": {"type": "bm25", "params": {"k1": 1.5, "b": 0.75}},
  "generator": {"type": "extractive", "params": {"max_sentences": 1}},
  "top_k": 5,
  "k_values": [1, 3, 5],
  "metrics": ["precision", "recall", "hit_rate", "mrr", "ndcg", "exact_match", "token_f1",
              "contains_expected", "semantic_similarity", "context_relevance",
              "answer_relevance", "citation_coverage", "faithfulness"],
  "seed": 0
}
```

`k_values` and `metrics` are optional (defaults: all metrics, k in {1, 3, 5, top_k} ≤ top_k). Built-in
retrievers: `bm25`, `tfidf`, `char-ngram`, `first-k` (query-blind baseline), `none`. Built-in generators:
`extractive`, `lead`, `fault-injection` (evaluator control), `ollama`.

## Reproducibility

Every run is stored in `runs/<run_id>/`:

| File | Contents |
|---|---|
| `manifest.json` | run id, UTC timestamp, config hash, dataset name + version + corpus/sample hashes, retriever and generator (type, parameters, model id), embedder, `top_k`, k values, metrics, seed, evaluation version, framework version, Python and platform |
| `config.json` | the experiment configuration as executed |
| `summary.json` | aggregate metric means/std/n, method, heuristic flag, failure counts |
| `samples.jsonl` | per sample: question, answer, retrieved documents, every metric with evidence, failures |
| `dataset_quality.json` | dataset statistics at run time |

All built-in components are deterministic: re-running the same config on the same dataset reproduces the same
metrics (covered by a test). `EVAL_VERSION` is bumped whenever a metric's meaning changes, and `compare`
warns when dataset or evaluation versions differ between two runs. Runs are never overwritten.

Comparison reports deltas as `B − A` from the stored values, per-sample paired win/loss/tie counts, and a
seeded paired-bootstrap 95% interval. With few samples the interval is wide; do not read significance into
small differences.

## Dashboard

```bash
cd dashboard && npm install && npm run build && cd ..
rag-eval dashboard serve            # http://127.0.0.1:8765
```

Views: experiments, metrics (per-k curves and answer metrics, heuristics hatched), dataset quality,
retrieval examples (retrieved documents with relevance labels), failure cases with evidence, and side-by-side
experiment comparison. The dashboard only displays stored results; it computes nothing new except paired
win/loss counts. `rag-eval dashboard export OUT` writes a static copy for hosting.

## Testing

```bash
pytest                                # Python test suite
ruff check . && mypy                  # lint and strict type checking
cd dashboard && npm test && npm run build   # frontend unit tests, type check and production build
```

The suite covers each retrieval and answer metric with hand-computed expectations and edge cases (empty
retrieval, duplicate documents, multiple correct documents, missing expected answers, very long context,
malformed datasets), the faithfulness pipeline, failure analysis, storage, comparison, the CLI, the local
server and a fake local Ollama endpoint. `DEVELOPMENT_CHECKLIST.md` records the verification actually run.

## Limitations

- Lexical heuristics are not judgments of truth. Faithfulness, contradiction, context/answer relevance and
  containment are approximations with false positives and false negatives.
- The default "semantic" similarity is character n-gram overlap, not a neural embedding.
- Retrieval metrics depend entirely on relevance labels; missing labels make correct retrievals look wrong.
- Extractive answers are compared with short expected answers, so `exact_match` and `token_f1` are low for
  verbose generators; prefer `contains_expected` and reading the evidence there.
- The bundled dataset is tiny and hand-written; it demonstrates the tooling and proves nothing about
  real-world performance.
- Sentence splitting is rule-based and English-oriented. No support yet for multilingual text, multi-turn
  conversations or images.
- Latency and cost are not measured.
- The Ollama adapter has been exercised against a local fake server in tests, not against a real model.

## Roadmap

- LLM-as-judge adapters for local models, always reported as a separate, labelled method
- Pluggable claim-level entailment (local NLI models) for faithfulness
- Chunking experiments and corpus ingestion helpers
- Latency and throughput tracking
- Per-metadata slice breakdowns in reports and the dashboard
- Run tagging and regression gates for CI (`rag-eval experiment gate`)

## License

MIT. See [LICENSE](LICENSE).
