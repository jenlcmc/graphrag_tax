from pathlib import Path

from evaluation.datasets.sara_v3 import SARAV3Dataset


def _write_case(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_sara_dataset_load_and_parse(tmp_path):
    data_dir = tmp_path / "dataset"
    root = data_dir / "sara_v3"
    (root / "cases").mkdir(parents=True)
    (root / "splits").mkdir(parents=True)

    (root / "splits" / "test").write_text("tax_case_1\ns63_a_pos\n", encoding="utf-8")

    _write_case(
        root / "cases" / "tax_case_1.pl",
        """
% Text
% Alice was paid $1200 in 2019.

% Question
% How much tax does Alice have to pay in 2019? $0

% Facts
:- true.

% Test
:- true.
""".strip(),
    )

    _write_case(
        root / "cases" / "s63_a_pos.pl",
        """
% Text
% In 2017, Alice was paid $33200.

% Question
% Under section 63(a), Alice's taxable income in 2017 is equal to $26948. Entailment

% Facts
:- true.

% Test
:- true.
""".strip(),
    )

    dataset = SARAV3Dataset()
    cases = dataset.load(data_dir, limit=None)

    assert len(cases) == 2
    assert cases[0].id == "tax_case_1"
    assert cases[0].question.endswith("?")
    assert cases[0].rubric == "$0"

    assert cases[1].id == "s63_a_pos"
    assert cases[1].rubric == "Entailment"
    assert any(ref.startswith("26 USC §63") for ref in cases[1].relevant_ids)


def test_sara_score_numeric_and_label():
    dataset = SARAV3Dataset()

    class Case:
        def __init__(self, rubric, expected_type, relevant_ids=None):
            self.rubric = rubric
            self.metadata = {"expected_type": expected_type}
            self.question = "placeholder"
            self.relevant_ids = relevant_ids or []

    numeric_case = Case("$26948", "numeric", ["26 USC §63"])
    label_case = Case("Entailment", "label", ["26 USC §152"])

    numeric = dataset.score(
        "Step 1: income 33200. Step 2: deduction 6252. "
        "33200 - 6252 = 26948 under 26 USC §63. Final Answer: 26948.",
        numeric_case,
        lambda *_: {},
    )
    label = dataset.score(
        "Under 26 USC §152, this is entailment. Final Answer: Entailment.",
        label_case,
        lambda *_: {},
    )

    assert numeric["answer_correct"] == 1.0
    assert numeric["calculation_steps_present"] is True
    assert numeric["citation_fact_precision"] == 1.0
    assert numeric["earned"] == 1.0

    assert label["answer_correct"] == 1.0
    assert label["citation_fact_precision"] == 1.0
    assert label["earned"] == 1.0
