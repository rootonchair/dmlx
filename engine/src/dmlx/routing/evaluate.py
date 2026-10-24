"""Offline evaluation for workflow routers.

The keyword router lives only here as a migration baseline. Production routing
uses capability-bounded providers and manual fallback.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from dmlx.intelligence.base import Decision
from dmlx.intelligence.local_llm import LocalLLMDecisionProvider
from dmlx.intelligence.openai_compatible import OpenAICompatibleProvider
from dmlx.routing.router import WorkflowRouter


class KeywordBaseline:
    name = "legacy-keyword-baseline"
    edit = re.compile(r"^(make|add|remove|change|keep|same|more|less|without|with|swap|fix|crop)\b", re.I)
    fresh = re.compile(r"^(new|another|generate|create|draw|paint|imagine|start)\b", re.I)

    def decide(self, state: dict, candidates: dict[str, str]) -> Decision:
        text = state["message"].strip()
        if "image.edit" in candidates and self.edit.search(text) and not self.fresh.search(text):
            intent = "image.edit"
        else:
            intent = "image.generate" if "image.generate" in candidates else next(iter(candidates))
        return Decision(intent, 1.0, {name: float(name == intent) for name in candidates}, self.name)


def load_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(router: WorkflowRouter, examples: list[dict], *, minimum_selective_accuracy: float = 0.9) -> dict:
    correct = Counter()
    totals = Counter()
    tag_correct = Counter()
    tag_totals = Counter()
    predictions = []
    brier_total = 0.0
    input_hits = 0
    input_count = 0
    for example in examples:
        history = example.get("history") or []
        current_result_id = example.get("current_result_id")
        if current_result_id is None and example.get("has_current"):
            current_result_id = next(
                (item.get("result_id") for item in reversed(history) if item.get("result_id")), "current")
        state = {
            "message": example["message"],
            "current_result_id": current_result_id,
            "attachments": example.get("attachments") or [],
            "history": history,
        }
        decision = router.decide(state, example["candidates"])
        label = example["label"]
        hit = decision.intent == label
        totals[label] += 1
        correct[label] += int(hit)
        for tag in example.get("tags", []):
            tag_totals[tag] += 1
            tag_correct[tag] += int(hit)
        brier_total += sum(
            (decision.probabilities.get(name, 0.0) - float(name == label)) ** 2
            for name in example["candidates"]
        )
        prediction = {"id": example["id"], "expected": label, **decision.as_dict(), "correct": hit}
        if "expected_input_result_ids" in example:
            expected_inputs = example["expected_input_result_ids"]
            input_hit = list(decision.input_result_ids) == expected_inputs
            prediction["expected_input_result_ids"] = expected_inputs
            prediction["input_selection_correct"] = input_hit
            input_hits += int(input_hit)
            input_count += 1
        predictions.append(prediction)
    count = len(examples)
    hits = sum(item["correct"] for item in predictions)
    covered = [item for item in predictions if item["intent"] != "ambiguous"]
    threshold_curve = []
    for threshold in (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        selected = [
            item for item in predictions
            if item["intent"] != "ambiguous" and item["confidence"] >= threshold
        ]
        threshold_curve.append({
            "threshold": threshold,
            "coverage": round(len(selected) / count, 4) if count else 0,
            "selective_accuracy": round(
                sum(item["correct"] for item in selected) / len(selected), 4
            ) if selected else 0,
            "count": len(selected),
        })
    eligible_thresholds = [
        row for row in threshold_curve
        if row["count"] and row["selective_accuracy"] >= minimum_selective_accuracy
    ]
    recommended = max(eligible_thresholds, key=lambda row: (row["coverage"], -row["threshold"]), default=None)
    calibration_bins = []
    calibration_error = 0.0
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        upper = lower + 0.2
        bucket = [
            item for item in predictions
            if ((lower <= item["confidence"] <= upper) if upper == 1.0
                else (lower <= item["confidence"] < upper))
        ]
        if not bucket:
            continue
        accuracy = sum(item["correct"] for item in bucket) / len(bucket)
        confidence = sum(item["confidence"] for item in bucket) / len(bucket)
        calibration_error += len(bucket) / count * abs(accuracy - confidence) if count else 0
        calibration_bins.append({
            "range": [round(lower, 1), round(upper, 1)], "count": len(bucket),
            "accuracy": round(accuracy, 4), "mean_confidence": round(confidence, 4),
        })
    return {
        "count": count,
        "accuracy": round(hits / count, 4) if count else 0,
        "per_class_recall": {name: round(correct[name] / total, 4) for name, total in sorted(totals.items())},
        "per_tag_accuracy": {name: round(tag_correct[name] / total, 4) for name, total in sorted(tag_totals.items())},
        "coverage": round(len(covered) / count, 4) if count else 0,
        "selective_accuracy": round(sum(item["correct"] for item in covered) / len(covered), 4) if covered else 0,
        "multiclass_brier": round(brier_total / count, 4) if count else 0,
        "input_selection_accuracy": round(input_hits / input_count, 4) if input_count else None,
        "input_selection_count": input_count,
        "expected_calibration_error": round(calibration_error, 4),
        "calibration_bins": calibration_bins,
        "threshold_curve": threshold_curve,
        "recommended_threshold": recommended["threshold"] if recommended else None,
        "minimum_selective_accuracy": minimum_selective_accuracy,
        "predictions": predictions,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evals/workflow-routing.jsonl"))
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--local-llm-model-path", type=Path)
    parser.add_argument("--local-llm-revision")
    parser.add_argument("--minimum-selective-accuracy", type=float, default=0.9)
    args = parser.parse_args(argv)
    if args.baseline:
        router = WorkflowRouter(KeywordBaseline())
    elif args.base_url and args.model:
        router = WorkflowRouter(OpenAICompatibleProvider(args.base_url, args.model, args.api_key))
    elif args.local_llm_model_path and args.local_llm_revision:
        router = WorkflowRouter(LocalLLMDecisionProvider(
            args.local_llm_model_path, revision=args.local_llm_revision))
    else:
        parser.error(
            "choose --baseline, provide --base-url and --model, or provide "
            "--local-llm-model-path and --local-llm-revision"
        )
    print(json.dumps(evaluate(
        router, load_examples(args.dataset),
        minimum_selective_accuracy=args.minimum_selective_accuracy,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
