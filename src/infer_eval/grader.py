"""Grade a free-form model response against a scenario.

Score = 0.7 * numeric + 0.3 * rubric (numeric only if the scenario has no
rubric). Numeric answers are parsed from `ANSWER: key = value` lines. If a
wrong answer matches a known trap the report names the likely mistake.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .scenarios import DISCUSSION, SCENARIOS, Scenario

_NUM = r"[-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?"
_ANSWER_RE = re.compile(rf"ANSWER\s*:\s*(\w+)\s*=\s*({_NUM})")


def parse_answers(text: str) -> Dict[str, float]:
    text = text.replace("*", "").replace("`", "")
    out: Dict[str, float] = {}
    for key, val in _ANSWER_RE.findall(text):
        try:
            out[key] = float(val.replace(",", ""))
        except ValueError:
            continue
    return out


def _close(got: float, want: float, rel: float) -> bool:
    if rel == 0:
        return got == want
    return abs(got - want) <= rel * abs(want)


@dataclass
class Report:
    scenario: str
    score: float
    numeric: float
    rubric: Optional[float]
    answer_detail: List[str] = field(default_factory=list)
    rubric_detail: List[str] = field(default_factory=list)
    diagnoses: List[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"## {self.scenario}  score={self.score:.2f}  (numeric={self.numeric:.2f}"
                 + (f", rubric={self.rubric:.2f})" if self.rubric is not None else ")")]
        lines += [f"  {d}" for d in self.answer_detail + self.rubric_detail]
        lines += [f"  DIAGNOSIS: {d}" for d in self.diagnoses]
        return "\n".join(lines)


def grade(scenario: Scenario, response: str) -> Report:
    answers, _, traps = scenario.solve()
    got = parse_answers(response)

    detail, diagnoses, hits = [], [], 0
    for key, want in answers.items():
        tol = scenario.tolerances[key]
        if key not in got:
            detail.append(f"MISS {key}: no ANSWER line (expected ~{want:.6g})")
            continue
        if _close(got[key], want, tol):
            hits += 1
            detail.append(f"OK   {key} = {got[key]:.6g} (expected {want:.6g}, tol {tol:.1%})")
        else:
            detail.append(f"FAIL {key} = {got[key]:.6g} (expected {want:.6g}, tol {tol:.1%})")
            for label, wrong in traps.items():
                if key in wrong and _close(got[key], wrong[key], max(tol, 0.01)):
                    diagnoses.append(f"{key} matches the '{label}' mistake")
    numeric = hits / len(answers)

    rubric_score: Optional[float] = None
    rubric_detail: List[str] = []
    if scenario.rubric:
        positive = sum(r.weight for r in scenario.rubric if not r.pitfall)
        earned = 0.0
        for r in scenario.rubric:
            matched = any(re.search(p, response, re.I) for p in r.any_of)
            if r.pitfall:
                if matched:
                    earned -= r.weight
                    rubric_detail.append(f"PITFALL -{r.weight:.1f} {r.name}")
            elif matched:
                earned += r.weight
                rubric_detail.append(f"OK   +{r.weight:.1f} {r.name}")
            else:
                rubric_detail.append(f"MISS  0.0 {r.name}")
        rubric_score = max(0.0, min(1.0, earned / positive))

    score = numeric if rubric_score is None else 0.7 * numeric + 0.3 * rubric_score
    return Report(scenario.id, score, numeric, rubric_score, detail, rubric_detail, diagnoses)


def reference_response(scenario: Scenario) -> str:
    """The expert write-up: derivation + discussion + machine-readable answers."""
    answers, steps, _ = scenario.solve()
    body = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    tail = "\n".join(f"ANSWER: {k} = {v:.10g}" for k, v in answers.items())
    return f"{body}\n\n{DISCUSSION.get(scenario.id, '')}\n\n{tail}\n"


def trap_response(scenario: Scenario) -> Optional[str]:
    """A plausible-but-wrong response, used to prove the grader catches it."""
    answers, _, traps = scenario.solve()
    if not traps:
        return None
    label, wrong = next(iter(traps.items()))
    merged = {**answers, **wrong}
    tail = "\n".join(f"ANSWER: {k} = {v:.10g}" for k, v in merged.items())
    return f"(Using the shortcut: {label})\n\n{tail}\n"


def get(scenario_id: str) -> Scenario:
    return SCENARIOS[scenario_id]
