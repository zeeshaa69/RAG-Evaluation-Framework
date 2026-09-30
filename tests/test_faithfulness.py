from __future__ import annotations

from rag_eval.metrics.faithfulness import faithfulness
from rag_eval.types import MetricResult
from tests.conftest import rd

CTX = [
    rd("d1", "Mount Everest is 8,849 meters above sea level. It lies in the Himalayas."),
    rd("d2", "Water boils at 100 degrees Celsius. Water is not a metal."),
]


def labels(results: list[MetricResult]) -> list[str]:
    return [s["label"] for s in results[0].evidence["sentences"]]


def test_supported_sentence() -> None:
    res = faithfulness("Mount Everest is 8,849 meters above sea level. [d1]", CTX)
    assert res[0].score == 1.0 and res[1].score == 0.0
    assert labels(res) == ["supported"]
    assert res[0].evidence["sentences"][0]["best_doc_id"] == "d1"


def test_number_mismatch_is_contradiction() -> None:
    res = faithfulness("Mount Everest is 9,000 meters above sea level.", CTX)
    assert labels(res) == ["contradicted"]
    assert res[1].score == 1.0 and res[0].score == 0.0
    assert "9000" in res[0].evidence["sentences"][0]["reason"]


def test_negation_flip_is_contradiction() -> None:
    res = faithfulness("Water does not boil at 100 degrees Celsius.", CTX)
    assert labels(res) == ["contradicted"]


def test_negation_preserved_is_supported() -> None:
    res = faithfulness("Water is not a metal.", CTX)
    assert labels(res) == ["supported"]


def test_invented_claim_is_unsupported() -> None:
    res = faithfulness("The committee ratified the treaty in a secret vote.", CTX)
    assert labels(res) == ["unsupported"]
    assert res[0].score == 0.0 and res[1].score == 0.0


def test_mixed_answer_scores_fraction() -> None:
    ans = "Mount Everest is 8,849 meters above sea level. The treaty was ratified in secret."
    res = faithfulness(ans, CTX)
    assert res[0].score == 0.5
    assert res[0].evidence["supported"] == 1 and res[0].evidence["unsupported"] == 1


def test_empty_context_makes_everything_unsupported() -> None:
    res = faithfulness("Paris is the capital of France.", [])
    assert labels(res) == ["unsupported"]
    assert res[0].score == 0.0


def test_empty_answer_and_abstention_not_applicable() -> None:
    assert faithfulness("", CTX)[0].score is None
    r = faithfulness("I don't know based on the provided context.", CTX)
    assert r[0].score is None and "abstention" in r[0].evidence["reason"]


def test_answer_with_only_stopwords_is_skipped() -> None:
    assert faithfulness("It is the.", CTX)[0].score is None


def test_all_results_are_flagged_heuristic_with_method_and_note() -> None:
    for r in faithfulness("Mount Everest is tall.", CTX):
        assert r.heuristic and r.method and "Heuristic" in r.note


def test_paraphrase_limitation_is_documented_by_behaviour() -> None:
    # A faithful paraphrase with different vocabulary is NOT recognised as supported.
    res = faithfulness("The tallest peak rises nearly nine kilometres.", CTX)
    assert labels(res) == ["unsupported"]


def test_long_context() -> None:
    filler = " ".join(f"Filler sentence number {i} mentions nothing useful." for i in range(3000))
    ctx = [rd("big", filler + " Paris is the capital of France.")]
    res = faithfulness("Paris is the capital of France.", ctx)
    assert res[0].score == 1.0
    assert res[0].evidence["num_context_sentences"] > 3000
