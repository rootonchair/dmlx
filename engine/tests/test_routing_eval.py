from pathlib import Path

from dmlx.intelligence.base import Decision
from dmlx.routing.evaluate import KeywordBaseline, evaluate, load_examples
from dmlx.routing.router import WorkflowRouter


def test_routing_eval_dataset_covers_required_classes():
    path = Path(__file__).parents[1] / "evals" / "workflow-routing.jsonl"
    examples = load_examples(path)
    labels = {item["label"] for item in examples}
    assert len(examples) >= 20
    assert {"image.generate", "image.edit", "image.variation", "image.upscale", "image.restore", "image.enhance", "video.generate"} <= labels


def test_baseline_evaluator_reports_per_class_and_tags():
    examples = [{
        "id": "one", "message": "make it blue", "has_current": True, "attachments": [],
        "candidates": ["image.generate", "image.edit"], "label": "image.edit", "tags": ["english"],
    }]
    report = evaluate(WorkflowRouter(KeywordBaseline()), examples)
    assert report["accuracy"] == 1
    assert report["per_class_recall"] == {"image.edit": 1}
    assert report["per_tag_accuracy"] == {"english": 1}
    assert report["coverage"] == 1
    assert report["selective_accuracy"] == 1
    assert report["multiclass_brier"] == 0
    assert report["expected_calibration_error"] == 0
    assert report["calibration_bins"][-1]["range"] == [0.8, 1.0]
    assert report["recommended_threshold"] == 0.0
    assert report["threshold_curve"][0] == {
        "threshold": 0.0, "coverage": 1.0, "selective_accuracy": 1.0, "count": 1,
    }


def test_threshold_recommendation_abstains_when_quality_target_is_not_met():
    examples = [{
        "id": "wrong", "message": "make it blue", "has_current": True, "attachments": [],
        "candidates": ["image.generate", "image.edit"], "label": "image.generate", "tags": [],
    }]
    report = evaluate(WorkflowRouter(KeywordBaseline()), examples)
    assert report["recommended_threshold"] is None


def test_evaluator_scores_ordered_history_input_selection():
    class Provider:
        name = "fixture"

        def decide(self, _state, candidates):
            return Decision(
                "image.edit", 1, {name: float(name == "image.edit") for name in candidates},
                self.name, input_result_ids=("img_2", "img_1"))

    examples = [{
        "id": "ordered", "message": "second then first", "has_current": True,
        "history": [{"sequence": 0, "result_id": "img_1"}, {"sequence": 1, "result_id": "img_2"}],
        "candidates": ["image.generate", "image.edit"], "label": "image.edit",
        "expected_input_result_ids": ["img_2", "img_1"], "tags": ["history"],
    }]
    report = evaluate(WorkflowRouter(Provider()), examples)
    assert report["input_selection_accuracy"] == 1
    assert report["input_selection_count"] == 1
    assert report["predictions"][0]["input_selection_correct"] is True
