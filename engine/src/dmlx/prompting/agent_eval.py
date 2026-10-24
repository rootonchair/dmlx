"""Score the planning agent (and optionally the router it replaces) on chats from a JSONL dataset.

    uv run --extra local-llm python -m dmlx.prompting.agent_eval \\
        [--router-model-path PATH --router-revision REV] [--out report.json]

Images are synthetic placeholders, so this scores routing (workflow and input images); the prompts the
agent writes are saved for review but not scored. History entries:
    ["gen", name, prompt]              a message and the image generated from it
    ["edit", name, prompt, [sources]]  a message and the edit it produced
    ["upload", name, text]             a message with an uploaded image attached
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

from dmlx.prompting import agent, local_vlm
from dmlx.routing.router import DEFAULT_DESCRIPTIONS

CANDIDATES = ["image.edit", "image.generate"]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def placeholder(directory: Path, name: str) -> Path:
    from PIL import Image, ImageDraw

    path = directory / f"{name}.png"
    if not path.exists():
        digest = hashlib.sha256(name.encode()).digest()
        image = Image.new("RGB", (512, 512), tuple(digest[:3]))
        ImageDraw.Draw(image).ellipse((128, 128, 384, 384), fill=tuple(digest[3:6]))
        image.save(path)
    return path


def state_for(case: dict) -> tuple[dict, dict[str, str]]:
    """The route state the app would send, and a name -> result id map."""
    ids, history, number = {}, [], 0
    for entry in case["history"]:
        kind, name = entry[0], entry[1]
        ids[name] = f"{'up' if kind == 'upload' else 'img'}_{name}"
        if kind == "upload":
            history.append({"sequence": len(history), "kind": "user", "text": entry[2], "image_number": None,
                            "operation": None, "result_id": None, "input_ids": [ids[name]]})
            continue
        sources = [ids[s] for s in (entry[3] if kind == "edit" else [])]
        history.append({"sequence": len(history), "kind": "user", "text": entry[2], "image_number": None,
                        "operation": None, "result_id": None, "input_ids": []})
        number += 1
        history.append({"sequence": len(history), "kind": "image", "text": entry[2], "image_number": number,
                        "operation": "image.edit" if kind == "edit" else "image.generate",
                        "result_id": ids[name], "input_ids": sources})
    for name in case["attachments"]:
        ids[name] = f"up_{name}"
    images = [item for item in history if item["kind"] == "image"]
    uploads = [i for item in history if item["kind"] == "user" for i in item["input_ids"]]
    current = images[-1]["result_id"] if images else (uploads[-1] if uploads else None)
    state = {"message": case["message"], "current_result_id": current,
             "current_prompt": images[-1]["text"] if images else None,
             "attachments": [ids[n] for n in case["attachments"]], "history": history}
    return state, ids


def correct(case: dict, ids: dict[str, str], intent: str, inputs: list[str]) -> tuple[bool, bool]:
    ok_intent = intent in case["labels"]
    accepted = [[ids[n] for n in option] for option in case["inputs"]]
    ok_inputs = intent == "ambiguous" or list(inputs) in accepted
    return ok_intent, ok_intent and ok_inputs


def summarize(rows: list[dict], key: str) -> dict:
    scored = [r for r in rows if key in r]
    by_tag: dict[str, list[bool]] = {}
    for r in scored:
        for tag in r["tags"]:
            by_tag.setdefault(tag, []).append(r[key]["ok"])
    seconds = sorted(r[key]["seconds"] for r in scored)
    return {
        "cases": len(scored),
        "workflow_accuracy": round(sum(r[key]["ok_intent"] for r in scored) / max(len(scored), 1), 3),
        "workflow_and_images_accuracy": round(sum(r[key]["ok"] for r in scored) / max(len(scored), 1), 3),
        "fell_back": sum(r[key].get("fell_back", False) for r in scored),
        "per_tag": {tag: round(sum(v) / len(v), 3) for tag, v in sorted(by_tag.items())},
        "median_seconds": seconds[len(seconds) // 2] if seconds else None,
        "max_seconds": seconds[-1] if seconds else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evals/agent-routing.jsonl"))
    parser.add_argument("--router-model-path", type=Path)
    parser.add_argument("--router-revision")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    path = local_vlm.locate()
    if path is None:
        parser.error("the prompt assistant is not downloaded")
    cases = load(args.dataset)
    workflows = {name: DEFAULT_DESCRIPTIONS[name] for name in CANDIDATES}
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        images = Path(tmp)
        assistant = local_vlm.LocalVLMPromptProvider(path)
        for case in cases:
            state, ids = state_for(case)
            by_id = {rid: name for name, rid in ids.items()}
            started = time.monotonic()
            result = agent.plan(assistant, state, workflows, image_path=lambda rid: placeholder(images, by_id[rid]))
            seconds = round(time.monotonic() - started, 2)
            intent, inputs = (result.intent, list(result.inputs)) if result else ("fallback", [])
            ok_intent, ok = correct(case, ids, intent, inputs)
            row = {"id": case["id"], "tags": case["tags"], "message": case["message"], "agent": {
                "intent": intent, "inputs": inputs, "prompt": result.prompt if result else None,
                "instruction": result.instruction if result else None,
                "ask": result.ask if result else None, "seconds": seconds, "fell_back": result is None,
                "ok_intent": ok_intent, "ok": ok}}
            rows.append(row)
            print(f"{case['id']:24} agent  {intent:15} {'ok' if ok else 'MISS'} {seconds:5.1f}s", flush=True)
        assistant.unload()
        if args.router_model_path and args.router_revision:
            from dmlx.intelligence.local_llm import LocalLLMDecisionProvider

            router = LocalLLMDecisionProvider(args.router_model_path, revision=args.router_revision)
            for row, case in zip(rows, cases):
                state, ids = state_for(case)
                started = time.monotonic()
                if not state["current_result_id"] and not state["attachments"]:
                    intent, inputs = "image.generate", []
                else:
                    decision = router.decide(state, workflows)
                    intent = decision.intent
                    inputs = list(state["attachments"]) or list(decision.input_result_ids) or (
                        [state["current_result_id"]] if intent == "image.edit" else [])
                seconds = round(time.monotonic() - started, 2)
                ok_intent, ok = correct(case, ids, intent, inputs)
                row["router"] = {"intent": intent, "inputs": inputs, "seconds": seconds, "ok_intent": ok_intent, "ok": ok}
                print(f"{case['id']:24} router {intent:15} {'ok' if ok else 'MISS'} {seconds:5.1f}s", flush=True)
            router.unload()
    report = {"agent": summarize(rows, "agent")}
    if any("router" in r for r in rows):
        report["router"] = summarize(rows, "router")
    report["rows"] = rows
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
