from pathlib import Path

from evaluation.datasets.sara_v3 import SARAV3Dataset


def _write_case(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_sara_dataset_load_and_parse(tmp_path):
    data_dir = tmp_path / "dataset"
    root = data_dir / "sara_v3"
    (root / "cases").mkdir(parents=True)
    (root / "splits").mkdir(parents=True)

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
:- tax("Alice",2019,0).
:- halt.
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

    _write_case(
        root / "cases" / "label_from_test_pos.pl",
        """
% Text
% Alice may be a dependent under section 152.

% Question
% Alice qualifies as a dependent under section 152.

% Facts
:- true.

% Test
:- s152_a("Alice","Bob",2019,_,_).
:- halt.
""".strip(),
    )

    _write_case(
        root / "cases" / "numeric_from_test.pl",
        """
% Text
% Alice earned wages in 2019.

% Question
% How much tax does Alice owe in 2019?

% Facts
:- true.

% Test
:- tax("Alice",2019,1234).
:- halt.
""".strip(),
    )

    _write_case(
        root / "cases" / "s152_c_2_B_pos.pl",
        """
% Text
% Alice has a brother, Bob, who was born January 31st, 2014.

% Question
% Bob bears a relationship to Alice under section 152(c)(2)(B). Entailment

% Facts
:- [statutes/prolog/init].
brother_(span("brother",12,18)).
patient_(span("brother",12,18),span("Alice",0,4)).
agent_(span("brother",12,18),span("Bob",21,23)).
start_(span("brother",12,18),span(20140131,39,56)).
birth_(span("born",34,37)).
agent_(span("born",34,37),span("Bob",21,23)).
start_(span("born",34,37),span(20140131,39,56)).

% Test
:- s152_c_2_B("Bob","Alice",_,_,_).
:- halt.
""".strip(),
    )

    _write_case(
        root / "cases" / "helper_goal_pos.pl",
        """
% Text
% Alice is treated as related to Bob under a rule in 2019.

% Question
% Alice bears a relationship to Bob under section 152(d)(2)(D) for the year 2019. Entailment

% Facts
:- true.

% Test
goal :- s152_d_2_D("Alice","Bob",Start_relationship,End_relationship),
    var(End_relationship),
    first_day_year(2019,First_day),
    is_before(Start_relationship,First_day).
:- goal.
:- halt.
""".strip(),
    )

    _write_case(
        root / "cases" / "helper_goal_neg.pl",
        """
% Text
% Alice is not treated as related to Bob under a rule in 2018.

% Question
% Alice bears a relationship to Bob under section 152(d)(2)(D) for the year 2018. Contradiction

% Facts
:- true.

% Test
goal :- \\+ (s152_d_2_D("Alice","Bob",Start_relationship,End_relationship),
    var(End_relationship),
    first_day_year(2018,First_day),
    is_before(Start_relationship,First_day)).
:- goal.
:- halt.
""".strip(),
    )

    (root / "splits" / "test").write_text(
        "\n".join(
            [
                "tax_case_1",
                "s63_a_pos",
                "label_from_test_pos",
                "numeric_from_test",
                "s152_c_2_B_pos",
                "helper_goal_pos",
                "helper_goal_neg",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = SARAV3Dataset()
    cases = dataset.load(data_dir, limit=None)

    assert len(cases) == 7
    assert cases[0].id == "tax_case_1"
    assert cases[0].question.endswith("?")
    assert cases[0].rubric == "$0"
    assert cases[0].metadata["expected_type"] == "numeric"
    assert cases[0].metadata["test_goal"] == 'tax("Alice",2019,0)'

    assert cases[1].id == "s63_a_pos"
    assert cases[1].rubric == "Entailment"
    assert any(ref.startswith("26 USC §63") for ref in cases[1].relevant_ids)
    assert cases[1].metadata["expected_type"] == "label"

    assert cases[2].id == "label_from_test_pos"
    assert cases[2].rubric == "Entailment"
    assert cases[2].metadata["expected_type"] == "label"
    assert cases[2].metadata["test_goal"] == 's152_a("Alice","Bob",2019,_,_)'

    assert cases[3].id == "numeric_from_test"
    assert cases[3].rubric == "1234"
    assert cases[3].metadata["expected_type"] == "numeric"
    assert cases[3].metadata["test_goal_numbers"][-1] == "1234"

    assert cases[4].id == "s152_c_2_B_pos"
    assert cases[4].rubric == "Entailment"
    assert cases[4].metadata["expected_type"] == "label"
    assert cases[4].metadata["test_goal"] == 's152_c_2_B("Bob","Alice",_,_,_)'
    assert any(ref.startswith("26 USC §152") for ref in cases[4].relevant_ids)

    assert cases[5].id == "helper_goal_pos"
    assert cases[5].rubric == "Entailment"
    assert cases[5].metadata["expected_type"] == "label"
    assert cases[5].metadata["test_goal"] == "goal"
    assert cases[5].metadata["test_goal_negated"] is False

    assert cases[6].id == "helper_goal_neg"
    assert cases[6].rubric == "Contradiction"
    assert cases[6].metadata["expected_type"] == "label"
    assert cases[6].metadata["test_goal"] == "goal"
    assert cases[6].metadata["test_goal_negated"] is True


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


def test_sara_score_string_answer_type():
    dataset = SARAV3Dataset()

    class Case:
        def __init__(self, rubric, expected_type, relevant_ids=None):
            self.rubric = rubric
            self.metadata = {"expected_type": expected_type}
            self.question = "placeholder"
            self.relevant_ids = relevant_ids or []

    string_case = Case("Head of Household", "string", ["26 USC §2"])
    result = dataset.score(
        "Under 26 USC §2, Final Answer: Head of Household",
        string_case,
        lambda *_: {},
    )

    assert result["answer_correct"] == 1.0
    assert result["citation_fact_precision"] == 1.0
    assert result["earned"] == 1.0


def test_sara_score_label_true_false_mapping():
    dataset = SARAV3Dataset()

    class Case:
        def __init__(self, rubric, expected_type, relevant_ids=None, test_goal="", test_goal_negated=False):
            self.rubric = rubric
            self.metadata = {
                "expected_type": expected_type,
                "test_goal": test_goal,
                "test_goal_negated": test_goal_negated,
            }
            self.question = "placeholder"
            self.relevant_ids = relevant_ids or []

    entail_case = Case(
        "Entailment",
        "label",
        ["26 USC §152"],
        test_goal='s152_c_2_B("Bob","Alice",_,_,_)',
        test_goal_negated=False,
    )
    contradiction_case = Case(
        "Contradiction",
        "label",
        ["26 USC §152"],
        test_goal='s152_c_2_B("Bob","Alice",_,_,_)',
        test_goal_negated=True,
    )

    true_result = dataset.score(
        "Under 26 USC §152(c)(2)(B), this is supported. Final Answer: True",
        entail_case,
        lambda *_: {},
    )
    false_result = dataset.score(
        "Under 26 USC §152(c)(2)(B), this is not satisfied. Final Answer: False",
        contradiction_case,
        lambda *_: {},
    )

    assert true_result["answer_correct"] == 1.0
    assert true_result["predicted_label"] == "entailment"
    assert true_result["test_goal_label"] == "entailment"
    assert true_result["expected_label_consistent_with_test"] is True

    assert false_result["answer_correct"] == 1.0
    assert false_result["predicted_label"] == "contradiction"
    assert false_result["test_goal_label"] == "contradiction"
    assert false_result["expected_label_consistent_with_test"] is True
