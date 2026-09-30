# Development checklist

Everything below was actually executed in this repository (Python 3.11.15, Node 22.22.2, npm 10.9.7,
pytest 9.1.1, mypy 2.3.1, ruff 0.16.9, vite 7.3.6) on 2026-09-30. Output is quoted from the real runs.
Re-run the commands to re-verify; nothing here is an expected value.

## Verification results

| # | Check | Command | Result |
|---|---|---|---|
| 1 | Full Python test suite | `pytest` | **207 passed** |
| 2 | Coverage | `pytest --cov=rag_eval` | 97% total (1516 statements, 43 missed). Metric modules `retrieval.py`, `answer.py`, `embeddings.py`: 100%; `faithfulness.py`: 94%. Uncovered: `__main__.py`, the optional `SentenceTransformerEmbedder` (excluded by pragma), a few CLI/IO error branches |
| 3 | Lint | `ruff check .` | All checks passed |
| 4 | Formatting | `ruff format --check .` | 36 files already formatted |
| 5 | Python type check (strict) | `mypy` | Success: no issues found in 34 source files |
| 6 | Frontend unit tests | `cd dashboard && npm test` | 4 passed (1 file) |
| 7 | Frontend type check + production build | `cd dashboard && npm run build` (`tsc --noEmit && vite build`) | type check clean; 37 modules transformed; `dist/index.html`, one CSS and one JS asset (242.56 kB JS, 75.30 kB gzip) |
| 8 | Dataset validation | `rag-eval dataset validate examples/demo/dataset` | exit 0; 34 samples, 32 documents, version `cd651c4689a1`; 0 errors, 4 warnings (the 2 intentionally unanswerable questions have no labels and no expected answer) |
| 9 | Example experiments | `rag-eval experiment run examples/demo/experiments/<each>.json` (6 configs) | all 6 exit 0 and stored under `runs/` |
| 10 | Comparison | `rag-eval experiment compare bm25-extractive first-k-baseline` | exit 0; table of measured values, deltas, paired win/loss/tie counts, bootstrap interval |
| 11 | Report generation | `rag-eval report generate bm25-extractive --format html --out report.html` | file written (Markdown output also exercised in tests) |
| 12 | Clean install | fresh venv, `pip install .` (non-editable), then `rag-eval dataset validate` and `rag-eval experiment run ... --no-save` from `/tmp` | works; package imported from site-packages, no dev dependencies, no network |
| 13 | Dashboard in a real browser | `rag-eval dashboard serve`, headless Chromium, all 6 tabs visited and screenshotted | 0 page errors / console errors (after adding an inline favicon; the first run showed one 404 for `/favicon.ico`) |
| 14 | Determinism | test `test_runs_are_deterministic` | identical samples and summary for two runs of the same config |
| 15 | README example validity | test `test_readme_experiment_example_is_valid` | the JSON in the README's "Experiment format" section parses as a real config |

## Evaluator control (does the pipeline detect known faults?)

`control-number-fault` wraps the extractive generator and changes one number in its answer. Comparing it to
`bm25-extractive` on the same retrieval results: 13 answers were altered and all 13 were flagged
`contradicted_answer` by the faithfulness heuristic (13/13 on this dataset). This checks the detector on an
easy, synthetic fault type only. It says nothing about real model hallucinations, which are subtler.

## Problems found during verification and fixed

- Sentence splitter separated a trailing `[doc_id]` citation from its sentence, which made
  `citation_coverage` report 0.5 for correctly cited single-sentence answers. Fixed in `text.split_sentences`;
  regression test added.
- The fault-injection generator could not find numbers written with thousands separators (`8,849`), so some
  faults were silently not injected. Found by `test_fault_injection_modes[number]`; fixed.
- The first `wrong_answer` failure rule used token F1, which flagged correct but verbose extractive answers
  (17 of 34 flagged). Replaced with a rule requiring both no containment of the expected answer and expected-token
  recall under 0.5, and marked as heuristic.
- `ruff format` reformatted the Python block in README.md; README re-formatted to keep the format check green.
- mypy strict initially reported 15 errors (mostly `Any` returns and test typing) and ruff reported 64
  (mostly line length); all fixed or configured (per-file ignore for HTML templates in `report.py`, missing-stub
  override for the optional `sentence_transformers`).

## Requirement coverage

| Requirement | Where |
|---|---|
| No paid/mandatory AI provider | stdlib-only runtime; built-ins are deterministic; optional local Ollama adapter |
| Dataset management (question, expected answer, relevant docs, metadata) | `datasets.py`, README "Dataset format" |
| Retrieval metrics: Precision@K, Recall@K, Hit Rate, MRR, NDCG | `metrics/retrieval.py`, `tests/test_retrieval_metrics.py` |
| Answer metrics: exact match, token overlap, semantic similarity, context relevance, answer relevance, citation coverage | `metrics/answer.py`, `tests/test_answer_metrics.py` |
| Faithfulness with metric/score/evidence/method; heuristics flagged | `metrics/faithfulness.py`, `MetricResult`, `tests/test_faithfulness.py` |
| Experiment runner and stored results | `experiment.py`, `store.py`, `tests/test_experiment.py` |
| Comparison with measured values | `compare.py`, CLI `experiment compare`, dashboard Compare tab |
| CLI (`dataset validate`, `experiment run/compare`, `report generate`) | `cli.py`, `tests/test_cli.py` |
| Python SDK | `api.py` (`RagEval`), `examples/python_api.py` |
| Dashboard (experiments, metrics, charts, dataset quality, retrieval examples, failure cases, comparison) | `dashboard/` |
| Failure analysis (missed doc, irrelevant context, no evidence, contradiction) | `failures.py`, dashboard Failure cases tab |
| Reproducibility record | `manifest.json` (config, dataset version, model id, evaluation version, timestamp, parameters) |
| Edge-case tests: empty retrieval, duplicates, multiple correct docs, no expected answer, long context, malformed dataset | `tests/test_retrieval_metrics.py`, `test_answer_metrics.py`, `test_faithfulness.py`, `test_datasets.py` |
| Offline demo | `examples/demo/` |

## Known gaps (not verified, not claimed)

- The Ollama adapter was tested only against a local fake HTTP server, never against a real model.
- The optional `sentence-transformers` embedder was not installed or run.
- The dashboard has unit tests for its comparison logic only; UI behaviour was checked manually in a headless
  browser, not with automated end-to-end tests.
- `npm install` reported advisories from `npm audit` in dev tooling dependencies; they were not investigated.
- Not tested on Windows or macOS, or on Python versions other than 3.11.
- Faithfulness and the other heuristics have not been validated against human judgments.
