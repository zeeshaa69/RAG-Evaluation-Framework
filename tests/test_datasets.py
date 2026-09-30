from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval.datasets import DatasetError, dataset_quality, inspect_dataset, load_dataset
from tests.conftest import DEMO, write_jsonl


def codes(path: Path) -> set[str]:
    return {i.code for i in inspect_dataset(path)[1].issues}


def make(tmp_path: Path, docs: list[object], samples: list[object]) -> Path:
    d = tmp_path / "d"
    d.mkdir()
    write_jsonl(d / "corpus.jsonl", docs)
    write_jsonl(d / "samples.jsonl", samples)
    return d


def test_loads_valid_dataset(tiny_dataset: Path) -> None:
    ds = load_dataset(tiny_dataset)
    assert len(ds.samples) == 2 and len(ds.documents) == 3
    assert ds.samples[0].relevant_docs == {"d1": 1}
    assert ds.get_document("d3") is not None and ds.get_document("nope") is None


def test_version_is_content_hash(tiny_dataset: Path, tmp_path: Path) -> None:
    v1 = load_dataset(tiny_dataset).version
    assert v1 == load_dataset(tiny_dataset).version
    (tiny_dataset / "samples.jsonl").write_text(
        (tiny_dataset / "samples.jsonl").read_text().replace("Tokyo", "Kyoto")
    )
    assert load_dataset(tiny_dataset).version != v1


def test_single_file_format(tmp_path: Path) -> None:
    f = tmp_path / "ds.json"
    f.write_text(
        json.dumps(
            {
                "name": "single",
                "documents": [{"id": "a", "text": "hello"}],
                "samples": [
                    {
                        "id": "s",
                        "question": "hi?",
                        "relevant_docs": ["a"],
                        "expected_answer": "hello",
                    }
                ],
            }
        )
    )
    ds = load_dataset(f)
    assert ds.name == "single" and ds.samples[0].expected_answer == "hello"


def test_graded_relevance_formats(tmp_path: Path) -> None:
    d = make(
        tmp_path,
        [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}],
        [
            {
                "id": "s",
                "question": "q?",
                "expected_answer": "x",
                "relevant_docs": [{"id": "a", "grade": 3}, "b"],
            },
            {"id": "t", "question": "q2?", "expected_answer": "x", "relevant_docs": {"a": 2}},
        ],
    )
    ds = load_dataset(d)
    assert ds.samples[0].relevant_docs == {"a": 3, "b": 1}
    assert ds.samples[1].relevant_docs == {"a": 2}


def test_no_expected_answer_is_warning_not_error(tmp_path: Path) -> None:
    d = make(
        tmp_path,
        [{"id": "a", "text": "x"}],
        [{"id": "s", "question": "q?", "relevant_docs": ["a"]}],
    )
    ds, report = inspect_dataset(d)
    assert report.ok and "no_expected_answer" in {i.code for i in report.warnings}
    assert ds is not None and ds.samples[0].expected_answer is None


def test_blank_expected_answer_normalized_to_none(tmp_path: Path) -> None:
    d = make(
        tmp_path,
        [{"id": "a", "text": "x"}],
        [{"id": "s", "question": "q?", "expected_answer": "  ", "relevant_docs": ["a"]}],
    )
    assert load_dataset(d).samples[0].expected_answer is None


# ---- malformed datasets -----------------------------------------------------------------


def test_malformed_json_line_reports_location(tmp_path: Path) -> None:
    d = make(tmp_path, [{"id": "a", "text": "x"}], [])
    (d / "samples.jsonl").write_text('{"id": "s", "question": "q?"}\n{not json}\n')
    _, report = inspect_dataset(d)
    bad = [i for i in report.errors if i.code == "malformed_json"]
    assert bad and bad[0].location == "samples.jsonl:2"


@pytest.mark.parametrize(
    ("row", "code"),
    [
        ({"question": "q?"}, "missing_id"),
        ({"id": "s"}, "missing_question"),
        ({"id": "s", "question": "   "}, "missing_question"),
        ({"id": "s", "question": "q?", "expected_answer": 5}, "bad_answer"),
        ({"id": "s", "question": "q?", "metadata": []}, "bad_metadata"),
        ({"id": "s", "question": "q?", "relevant_docs": "a"}, "bad_relevant_docs"),
        ({"id": "s", "question": "q?", "relevant_docs": [{"id": "a", "grade": -1}]}, "bad_grade"),
        ({"id": "s", "question": "q?", "relevant_docs": [{"id": "a", "grade": True}]}, "bad_grade"),
        ({"id": "s", "question": "q?", "relevant_docs": ["ghost"]}, "unknown_relevant_doc"),
        ([1, 2], "bad_record"),
    ],
)
def test_bad_sample_rows(tmp_path: Path, row: object, code: str) -> None:
    d = make(tmp_path, [{"id": "a", "text": "x"}], [row])
    assert code in codes(d)
    with pytest.raises(DatasetError):
        load_dataset(d)


@pytest.mark.parametrize(
    ("row", "code"),
    [
        ({"text": "x"}, "missing_id"),
        ({"id": "", "text": "x"}, "missing_id"),
        ({"id": "a"}, "missing_text"),
        ({"id": "a", "text": 5}, "missing_text"),
        ({"id": "a", "text": "x", "metadata": "no"}, "bad_metadata"),
    ],
)
def test_bad_document_rows(tmp_path: Path, row: object, code: str) -> None:
    d = make(tmp_path, [row], [{"id": "s", "question": "q?"}])
    assert code in codes(d)


def test_duplicate_ids(tmp_path: Path) -> None:
    d = make(
        tmp_path,
        [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}],
        [{"id": "s", "question": "q?"}, {"id": "s", "question": "r?"}],
    )
    c = codes(d)
    assert {"duplicate_doc_id", "duplicate_sample_id"} <= c


def test_duplicate_question_and_empty_doc_warn(tmp_path: Path) -> None:
    d = make(
        tmp_path,
        [{"id": "a", "text": "  "}],
        [{"id": "s", "question": "Why?"}, {"id": "t", "question": " why? "}],
    )
    _, report = inspect_dataset(d)
    assert report.ok
    assert {"duplicate_question", "empty_document"} <= {i.code for i in report.warnings}


def test_missing_files_and_paths(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "missing_file" in codes(empty)
    _, report = inspect_dataset(tmp_path / "nope")
    assert not report.ok
    ds, _ = inspect_dataset(tmp_path / "nope")
    assert ds is None


def test_empty_corpus_and_samples_are_errors(tmp_path: Path) -> None:
    d = make(tmp_path, [], [])
    assert {"empty_corpus", "empty_samples"} <= codes(d)


def test_single_file_malformed(tmp_path: Path) -> None:
    f = tmp_path / "x.json"
    f.write_text("{oops")
    ds, report = inspect_dataset(f)
    assert ds is None and "malformed_json" in {i.code for i in report.errors}
    f.write_text("[1]")
    assert inspect_dataset(f)[0] is None
    f.write_text('{"documents": 3, "samples": null}')
    assert "missing_field" in codes(f)


def test_non_utf8_file(tmp_path: Path) -> None:
    d = make(tmp_path, [{"id": "a", "text": "x"}], [{"id": "s", "question": "q?"}])
    (d / "corpus.jsonl").write_bytes(b"\xff\xfe\x00bad")
    assert "file_unreadable" in codes(d)


def test_bad_dataset_json_metadata(tmp_path: Path) -> None:
    d = make(tmp_path, [{"id": "a", "text": "x"}], [{"id": "s", "question": "q?"}])
    (d / "dataset.json").write_text("{bad")
    assert "malformed_json" in codes(d)


def test_load_error_message_mentions_first_problem(tmp_path: Path) -> None:
    d = make(
        tmp_path,
        [{"id": "a", "text": "x"}],
        [{"id": "s", "question": "q?", "relevant_docs": ["z"]}],
    )
    with pytest.raises(DatasetError, match="not in corpus"):
        load_dataset(d)


# ---- quality --------------------------------------------------------------------------


def test_quality_report(tiny_dataset: Path) -> None:
    q = dataset_quality(load_dataset(tiny_dataset))
    assert q["num_samples"] == 2 and q["num_documents"] == 3
    assert q["documents_never_relevant"] == 1  # d3 is a distractor
    assert q["samples_with_expected_answer"] == 2
    assert q["answer_support_mean"] == 1.0


def test_quality_flags_unsupported_labels_and_duplicates(tmp_path: Path) -> None:
    d = make(
        tmp_path,
        [{"id": "a", "text": "Cats purr."}, {"id": "b", "text": "cats   purr."}],
        [
            {
                "id": "s",
                "question": "q?",
                "expected_answer": "Submarines dive deeply",
                "relevant_docs": ["a"],
            }
        ],
    )
    q = dataset_quality(load_dataset(d))
    assert q["duplicate_documents"] == [["a", "b"]]
    assert q["low_answer_support"][0]["sample_id"] == "s"


def test_demo_dataset_is_valid() -> None:
    ds, report = inspect_dataset(DEMO / "dataset")
    assert ds is not None and report.ok
    assert len(ds.samples) >= 30
    assert any(len(s.relevant_docs) > 1 for s in ds.samples)
    assert any(s.expected_answer is None for s in ds.samples)
