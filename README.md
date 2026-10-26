# infer-eval

A small framework for the work an LLM-inference evaluation role involves: **designing realistic systems-engineering scenarios, writing executable reference solutions, and grading model outputs** on LLM inference, GPU performance and model-serving architecture.

Stdlib-only, Python 3.9+.

## Why it is built this way

| Role responsibility | How it shows up here |
|---|---|
| Design realistic scenarios | `scenarios.py`: 7 problems (KV-cache sizing under GQA+TP, decode roofline, W4A16 quantisation, TP all-reduce latency, Mamba2 state vs. KV, continuous batching, fused RMSNorm kernel traffic) |
| Write reference solutions | Each scenario has a *computed* solver. Change a parameter (model, GPU, TP degree) and the key regenerates, so variants resist memorisation. |
| Define "correct" vs "excellent" | Correct = numeric answers within per-key tolerance. Excellent = a weighted rubric for the reasoning (e.g. "names a mitigation for latency-bound all-reduce", "notes SSM prefix caching needs state checkpoints"). Pitfall rubric items subtract. |
| Grade model output | `grader.py` parses `ANSWER: key = value` lines, checks rubric regexes, and **diagnoses specific mistakes**: each scenario carries "trap" answers from a plausible wrong method (ignoring GQA, forgetting TP sharding, ignoring group-scale overhead, ...). |
| Written communication | The reference write-ups (`infer-eval reference <id>`) are the explanation a grader would hand back as feedback. |

## Use

```bash
pip install -e '.[dev]'
infer-eval list
infer-eval show tp-allreduce            # prompt to give a model
infer-eval reference tp-allreduce       # expert derivation + discussion
infer-eval grade tp-allreduce resp.txt  # grade a model's response (or "-" for stdin)
infer-eval demo                         # reference vs. plausible-wrong, every scenario
pytest
```

Sample diagnosis output:

```
## kv-capacity  score=0.15  (numeric=0.00, rubric=0.50)
  FAIL kv_bytes_per_token = 2.62144e+06 (expected 327680, tol 0.1%)
  DIAGNOSIS: kv_bytes_per_token matches the 'ignored GQA (used query heads)' mistake
```

## Scenarios

| id | Level | Topics |
|---|---|---|
| kv-capacity | 2 | vLLM, KV cache, GQA, tensor parallel |
| decode-roofline | 2 | roofline, memory-bound decode |
| quant-w4a16 | 3 | quantisation, group-scale overhead |
| continuous-batching | 3 | scheduling (includes a reference simulator) |
| fused-rmsnorm | 3 | CUDA, kernel fusion, bandwidth model |
| tp-allreduce | 4 | NCCL ring cost, latency-bound small messages |
| mamba2-state | 4 | Mamba2 recurrent state vs. attention KV, prefix caching |

## Adding a scenario

1. Write a solver returning `(answers, steps, traps)` in `scenarios.py`.
2. Add a prompt ending in `ANSWER_FORMAT`, per-key tolerances, and rubric items.
3. Add reference commentary to `DISCUSSION`. `tests/` then automatically checks that the reference scores 1.0 and the trap is penalised and diagnosed.

## Design notes and limits

- Units are decimal (1 GB = 1e9 B) and stated in each prompt; ambiguous units are the most common source of "wrong" answers that aren't.
- Hardware numbers (H100 3.35 TB/s, 450 GB/s NVLink, etc.) are public spec-sheet values, and the all-reduce alpha (3 us/hop) is a stated modelling assumption, not a measurement. Treat results as first-order bounds.
- Rubric matching is regex-based, which is deterministic but shallow. The natural extension is an LLM judge for the rubric half only, keeping numeric checking programmatic.
# infer-eval
