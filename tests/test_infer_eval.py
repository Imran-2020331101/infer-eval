import pytest

from infer_eval.grader import grade, parse_answers, reference_response, trap_response
from infer_eval.scenarios import SCENARIOS, simulate_continuous, simulate_static


@pytest.mark.parametrize("sid", list(SCENARIOS))
def test_reference_scores_perfectly(sid):
    s = SCENARIOS[sid]
    r = grade(s, reference_response(s))
    assert r.score == pytest.approx(1.0), r.render()


@pytest.mark.parametrize("sid", list(SCENARIOS))
def test_trap_is_penalised_and_diagnosed(sid):
    s = SCENARIOS[sid]
    wrong = trap_response(s)
    if wrong is None:
        pytest.skip("no trap")
    r = grade(s, wrong)
    assert r.numeric < 1.0, r.render()
    assert r.diagnoses, r.render()


def test_missing_answers_score_zero_numeric():
    r = grade(SCENARIOS["kv-capacity"], "I think it is about 40 sequences.")
    assert r.numeric == 0.0


def test_parser_handles_markdown_and_commas():
    got = parse_answers("**ANSWER: a = 1,234.5**\n`ANSWER: b = 2e3`")
    assert got == {"a": 1234.5, "b": 2000.0}


def test_known_values():
    ans = SCENARIOS["kv-capacity"].solve()[0]
    assert ans["kv_bytes_per_token"] == 327_680
    ans = SCENARIOS["mamba2-state"].solve()[0]
    assert ans["kv_at_128k_gb"] == pytest.approx(17.18, rel=1e-3)


def test_continuous_never_slower_than_static():
    from infer_eval.scenarios import _lengths
    ls = _lengths()
    assert simulate_continuous(ls, 32) <= simulate_static(ls, 32)
