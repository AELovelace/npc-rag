"""First-pass Decider 2B QLoRA training on one NVIDIA GPU; saves adapters, never deploys them."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npc_rag.decider_policy import DECIDER_OPTIONS, DECIDER_QUESTION, policy_for_layout
from npc_rag.decider_prompt import build_decision_item
from build_classifier_seed import normalize
from classifier_augmentation import augment_rows, augmentation_stats

LABELS = ("game", "general")
DEFAULT_REVISION = "ff99089b6810691bd79134ec6febb5d9afa6e619"  # Verified Mapika 2B v11 trainable weights.


def load_dataset(path, *, task='routing'):
    """Validate labels, family boundaries and exclusion of the fixed evaluation messages before downloads."""
    if task == 'participation':
        from participation_data import load_dataset as load_participation
        return load_participation(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Dataset is empty")
    excluded = set()
    for name in ("eval_messages.json", "eval_mixed_messages.json"):
        data = json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))
        excluded.update(normalize(s) for label in LABELS for s in data[label])
    ids, messages = set(), set()
    groups, families, splits = defaultdict(set), defaultdict(set), defaultdict(list)
    for row in rows:
        message = row["message"]
        if not isinstance(message, str) or not message.strip() or len(message) > 500:
            raise ValueError("Each message must contain 1-500 characters")
        normalized = normalize(message)
        if row["id"] in ids or normalized in messages or normalized in excluded:
            raise ValueError("Duplicate ID/message or fixed-evaluation overlap")
        if row["label"] not in LABELS or row["split"] not in ("train", "validation", "test"):
            raise ValueError("Unknown label or split")
        if not row.get("family_id") or not row.get("split_group"):
            raise ValueError("Every row needs family_id and split_group")
        ids.add(row["id"]); messages.add(normalized)
        groups[row["split_group"]].add(row["split"])
        families[row["family_id"]].add(row["split"])
        splits[row["split"]].append(row)
    if any(len(s) != 1 for s in list(groups.values()) + list(families.values())):
        raise ValueError("Related examples cross data splits")
    if any(set(r["label"] for r in splits[s]) != set(LABELS) for s in ("train", "validation", "test")):
        raise ValueError("Every split must contain both labels")
    return dict(splits)


def load_development(path, splits, *, task='routing'):
    if task == 'participation':
        from participation_data import load_development as load_participation
        return load_participation(path, splits)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reserved, families, ids, groups = set(), set(), set(), set()
    for group in splits.values():
        for row in group:
            reserved.add(normalize(row['message'])); families.add(row['family_id']); ids.add(row['id'])
            groups.add(row['split_group'])
    for name in ("eval_messages.json", "eval_mixed_messages.json"):
        data = json.loads((ROOT / 'data' / name).read_text(encoding='utf-8'))
        reserved.update(normalize(s) for label in LABELS for s in data[label])
    for row in rows:
        key = normalize(row['message'])
        if key in reserved or row['family_id'] in families or row['split_group'] in groups or row['id'] in ids:
            raise ValueError('Development overlaps training families/messages or fixed evaluation')
        if row['label'] not in LABELS or row['split'] != 'development' or not 0 < len(row['message']) <= 500:
            raise ValueError('Invalid development row')
        reserved.add(key); ids.add(row['id'])
    if not 150 <= len(rows) <= 300 or {r['label'] for r in rows} != set(LABELS):
        raise ValueError('Development requires 150-300 rows and both labels')
    return rows  # This set alone selects checkpoints; fixed evaluation is only an exclusion list here.


class KeepOrder:
    def shuffle(self, values):
        pass  # Evaluation preserves production's GAME-then-CHAT option order.


def encode_rows(rows, tokenizer, max_tokens, *, shuffle_options=False, seed=417, layout='state_first', task='routing'):
    rng = random.Random(seed) if shuffle_options else KeepOrder()
    labels = task_labels(task)
    encoded = []
    for row in rows:
        if task == 'participation':
            from npc_rag.participation_policy import build_participation_item
            item = build_participation_item(row['state'], tokenizer, max_tokens, layout=layout,
                                            rng=rng, gold=labels.index(row['label']))
        else:
            item = build_decision_item(row['message'], tokenizer, max_tokens, layout=layout,
                                       rng=rng, gold=labels.index(row['label']))
        item.update(id=row["id"], expected=row["label"])
        encoded.append(item)
    return encoded  # Labels select the loss target; no labels, provenance or gold answer tokens enter input_ids.


def slot_logits(model, ids, slot, letters):
    """Read only the A/B output weights at the decision slot, avoiding full-vocabulary logits."""
    import torch
    import torch.nn.functional as F
    base = model.get_base_model()
    hidden = base.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    weights = base.lm_head.weight.index_select(0, letters)
    logits = F.linear(hidden[:, slot, :].float(), weights.float())
    cap = getattr(base.config, "final_logit_softcapping", None)
    return cap * torch.tanh(logits / cap) if cap else logits


def task_labels(task):
    if task == 'participation':
        from npc_rag.participation_policy import LABELS as participation_labels
        return participation_labels
    if task != 'routing':
        raise ValueError('Unknown training task')
    return LABELS


def task_schema(task, layout):
    if task == 'participation':
        from npc_rag.participation_policy import QUESTION, OPTIONS, policy_for_layout as participation_policy
        return {'policy': participation_policy(layout), 'question': QUESTION, 'options': list(OPTIONS),
                'state_encoding': 'npc_rag.participation_policy.encode_state', 'task': task}
    task_labels(task)
    return {'policy': policy_for_layout(layout), 'question': DECIDER_QUESTION, 'options': list(DECIDER_OPTIONS),
            'state_encoding': 'json.dumps({"message": message}, ensure_ascii=False)', 'task': task}


def report_rows(rows, *, task='routing'):
    by_class = {}
    for label in task_labels(task):
        subset = [r for r in rows if r["expected"] == label]
        correct = sum(r["predicted"] == label for r in subset)
        by_class[label] = {"total": len(subset), "correct": correct,
                           "recall": correct / len(subset) if subset else None}
    result = {"total": len(rows), "accuracy": sum(r["expected"] == r["predicted"] for r in rows) / len(rows),
              "by_class": by_class, "gate_passed": all(v["recall"] is not None and v["recall"] >= .9 for v in by_class.values())}
    if task == 'participation':
        unwanted = sum(r['expected'] == 'quiet' and r['predicted'] == 'join' for r in rows)
        missed = sum(r['expected'] == 'join' and r['predicted'] == 'quiet' for r in rows)
        joins = sum(r['predicted'] == 'join' for r in rows)
        result.update(unwanted_interruptions=unwanted, missed_joins=missed,
                      join_precision=(joins - unwanted) / joins if joins else None)
    else:
        result.update(game_missed_as_chat=sum(r['expected'] == 'game' and r['predicted'] == 'general' for r in rows),
                      chat_sent_to_game=sum(r['expected'] == 'general' and r['predicted'] == 'game' for r in rows))
    return result  # Report both error directions; a silent classifier cannot pass by avoiding all interruptions.


def evaluate(model, encoded, letters, dtype, *, task='routing'):
    import torch
    import torch.nn.functional as F
    model.eval()
    rows, losses = [], []
    with torch.no_grad():
        for item in encoded:
            ids = torch.tensor([item["ids"]], device="cuda", dtype=torch.long)
            with torch.autocast("cuda", dtype=dtype):
                logits = slot_logits(model, ids, item["slots"][0], letters)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite evaluation logits")
            probs = logits.float().softmax(-1)[0].cpu().tolist()
            prediction = max(range(2), key=lambda i: probs[i])
            target = torch.tensor(item["golds"], device="cuda")
            losses.append(F.cross_entropy(logits.float(), target).item())
            rows.append({"id": item["id"], "expected": item["expected"],
                         "predicted": task_labels(task)[item["perms"][0][prediction]],
                         "probabilities_by_option_order": probs, "logits": logits.float()[0].cpu().tolist()})
    result = report_rows(rows, task=task)
    result.update(loss=sum(losses) / len(losses), rows=rows)
    model.train()
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run(args, splits):
    if args.local_files_only:
        os.environ['HF_HUB_OFFLINE'] = '1'  # Also cover PEFT's optional metadata checks during adapter saving.
    import torch
    import torch.nn.functional as F
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError("This first-pass runner requires NVIDIA CUDA; AMD training is not configured")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The pinned first-pass setup requires BF16-capable CUDA (the RTX 3080 supports it)")
    if args.output.exists():
        raise ValueError("Output already exists. Choose a new --output directory to preserve previous runs")
    args.output.mkdir(parents=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    cache = ROOT / "data" / "models" / "training"
    if args.local_files_only and not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        raise ValueError('Offline training requires an exact 40-character model revision')
    revision = args.revision if args.local_files_only else HfApi().model_info(args.model, revision=args.revision).sha
    config_path = hf_hub_download(args.model, "decider_config.json", revision=revision, cache_dir=cache,
                                  local_files_only=args.local_files_only)
    decision_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if decision_config.get("layout", "plain") != "plain":
        raise ValueError("This runner supports plain Decider checkpoints only")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, cache_dir=cache, local_files_only=args.local_files_only)
    letters_list = [tokenizer.encode(x, add_special_tokens=False) for x in ("A", "B")]
    if any(len(x) != 1 for x in letters_list) or letters_list[0] == letters_list[1]:
        raise ValueError("Expected distinct single-token A/B option letters")
    letters = torch.tensor([x[0] for x in letters_list], device="cuda", dtype=torch.long)
    splits = {**splits, 'development': load_development(args.dev_dataset, splits, task=args.task)}
    selected = {key: rows[:8] if args.smoke_test else rows for key, rows in splits.items()}
    encoded = {key: encode_rows(rows, tokenizer, args.max_tokens, shuffle_options=key == "train", seed=args.seed, layout=args.prompt_layout, task=args.task)
               for key, rows in selected.items() if key != "test"}  # Test examples are not scored during checkpoint selection.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, cache_dir=cache, dtype=dtype, attn_implementation="sdpa", local_files_only=args.local_files_only,
        device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype))
    if model.config.model_type != "qwen3_5_text":
        raise ValueError("Expected the text-only Qwen3.5 Decider 2B architecture")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=args.rank, lora_alpha=2 * args.rank,
                                           lora_dropout=.05, target_modules="all-linear", bias="none"))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No adapter parameters are trainable")
    metadata = {"model": args.model, "revision": revision, **task_schema(args.task, args.prompt_layout),
                "prompt_layout": args.prompt_layout,
                "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "dataset": str(args.dataset), "seed": args.seed, "rank": args.rank,
                "development_sha256": hashlib.sha256(args.dev_dataset.read_bytes()).hexdigest(),
                "selection": "development worst-class recall, accuracy, negative loss",
                "epochs": args.epochs, "learning_rate": args.learning_rate, "accumulation": args.accumulation,
                "max_tokens": args.max_tokens, "smoke_test": args.smoke_test,
                "gpu": torch.cuda.get_device_name(0), "trainable_parameters": sum(p.numel() for p in trainable),
                "packages": {name: version(name) for name in ("torch", "transformers", "peft", "bitsandbytes", "decider-ai")},
                "split_counts": {k: len(v) for k, v in selected.items()},
                "calibrated_after_training": False, "deployment_ready": False}
    write_json(args.output / "run.json", metadata)
    print(json.dumps({"gpu": metadata["gpu"], "revision": revision, "trainable_parameters": metadata["trainable_parameters"],
                      "max_prompt_tokens": max(len(i["ids"]) for group in encoded.values() for i in group)}, indent=2), flush=True)
    baseline = evaluate(model, encoded["development"], letters, dtype, task=args.task)
    write_json(args.output / "baseline-development.json", baseline)
    write_json(args.output / "baseline-validation.json", evaluate(model, encoded['validation'], letters, dtype, task=args.task))
    print("Baseline development:", json.dumps({k: v for k, v in baseline.items() if k != "rows"}), flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=.01)
    accumulation = 4 if args.smoke_test else args.accumulation
    epochs = 1 if args.smoke_test else args.epochs
    steps_total = epochs * math.ceil(len(encoded["train"]) / accumulation)
    warmup = max(1, round(steps_total * .05))
    step, best, best_epoch, history = 0, None, None, []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(epochs):
        augmented = augment_rows(selected['train'], args.seed + epoch) if args.task == 'routing' else selected['train']
        if args.task == 'routing':
            stats = augmentation_stats(augmented) if not args.smoke_test else {'smoke_subset': True}
        else:
            stats = {'task': args.task, 'augmented': False}
        write_json(args.output / f'epoch-{epoch + 1}-augmentation.json', {'stats': stats, 'rows': augmented})
        items = encode_rows(augmented, tokenizer, args.max_tokens, shuffle_options=True, seed=args.seed + epoch, layout=args.prompt_layout, task=args.task)
        random.Random(args.seed + epoch).shuffle(items)
        model.train()
        for start in range(0, len(items), accumulation):
            chunk = items[start:start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0
            for item in chunk:
                ids = torch.tensor([item["ids"]], device="cuda", dtype=torch.long)
                target = torch.tensor(item["golds"], device="cuda")
                with torch.autocast("cuda", dtype=dtype):
                    logits = slot_logits(model, ids, item["slots"][0], letters)
                    loss = F.cross_entropy(logits.float(), target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                (loss / len(chunk)).backward()  # The final partial chunk has the same mean-loss scale.
                loss_sum += loss.item()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            if args.smoke_test and float(norm) == 0:
                raise RuntimeError("Smoke test found no adapter gradient")
            scale = min((step + 1) / warmup, 1.0) * max(0.1, 1.0 - step / steps_total)
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * scale
            optimizer.step(); step += 1
            progress = {"epoch": epoch + 1, "step": step, "total_steps": steps_total,
                        "loss": round(loss_sum / len(chunk), 5),
                        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            with (args.output / "training-metrics.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(progress) + "\n")  # Preserve loss and memory measurements even if a later step fails.
            if step == 1 or step % 5 == 0 or args.smoke_test:
                print(json.dumps(progress), flush=True)
        validation = evaluate(model, encoded["validation"], letters, dtype, task=args.task)
        write_json(args.output / f"epoch-{epoch + 1}-validation.json", validation)
        development = evaluate(model, encoded['development'], letters, dtype, task=args.task)
        write_json(args.output / f'epoch-{epoch + 1}-development.json', development)
        score = (min(v["recall"] for v in development["by_class"].values()), development["accuracy"], -development["loss"])
        history.append({"epoch": epoch + 1, "validation": {k: v for k, v in validation.items() if k != "rows"},
                        'development': {k: v for k, v in development.items() if k != 'rows'}})
        print('Development:', json.dumps(history[-1]['development']), flush=True)
        if best is None or score > best:
            best, best_epoch = score, epoch + 1
            model.save_pretrained(args.output / "best-adapter", safe_serialization=True)
            tokenizer.save_pretrained(args.output / "best-adapter")
            write_json(args.output / "best-adapter" / "routing_policy.json", metadata)
            write_json(args.output / 'best-adapter' / 'decider_config.json',
                       {**decision_config, 'temperature': 1.0, 'temperature_by_type': {},
                        'calibration_status': 'uncalibrated_after_finetuning', 'routing_policy': metadata['policy'],
                        'routing_prompt_order': args.prompt_layout, 'schema_first': args.prompt_layout == 'schema_first',
                        'schema_first_trained': args.prompt_layout == 'schema_first'})
        write_json(args.output / "history.json", history)
    model.load_adapter(str(args.output / "best-adapter"), adapter_name="selected", is_trainable=False)
    model.set_adapter("selected")  # Verify the saved adapter can be reloaded before reporting completion.
    if args.smoke_test:
        reloaded = evaluate(model, encoded["validation"], letters, dtype, task=args.task)
        if any(max(abs(a - b) for a, b in zip(old["probabilities_by_option_order"], new["probabilities_by_option_order"])) > 1e-3
               for old, new in zip(validation["rows"], reloaded["rows"])):
            raise RuntimeError("Reloaded smoke adapter changed decision probabilities unexpectedly")
        write_json(args.output / "reloaded-validation.json", reloaded)
        print("GPU smoke test passed: finite forward/backward, nonzero gradients, and matching saved/reloaded adapter.", flush=True)
    else:
        test = evaluate(model, encode_rows(splits["test"], tokenizer, args.max_tokens, layout=args.prompt_layout, task=args.task), letters, dtype, task=args.task)
        write_json(args.output / "final-test.json", test)
        print("Final synthetic test:", json.dumps({k: v for k, v in test.items() if k != "rows"}), flush=True)
    write_json(args.output / "completion.json", {"completed": True, "smoke_test": args.smoke_test,
               "seconds": round(time.perf_counter() - started, 2), "steps": step,
               "selected_epoch": best_epoch,
               "development_improved_over_baseline": best > (
                   min(v["recall"] for v in baseline["by_class"].values()), baseline["accuracy"], -baseline["loss"]),
               "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "deployment_ready": False})
    print("Adapter:", args.output / "best-adapter", flush=True)
    print("Experimental adapter only. Merge, recalibrate/evaluate and quantize before considering deployment.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('routing', 'participation'), default='routing')
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dev-dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="Mapika/decider-2b")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument('--prompt-layout', choices=('state_first', 'schema_first'),
                        help='Defaults: routing state_first; participation schema_first')
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--seed", type=int, default=417)
    parser.add_argument("--check-data", action="store_true", help="validate dataset without GPU packages, downloads or training")
    parser.add_argument("--smoke-test", action="store_true", help="two optimizer steps on eight training rows, with an eight-row validation check")
    parser.add_argument('--local-files-only', action='store_true', help='Use cached weights/tokenizer at an exact revision without network access')
    args = parser.parse_args()
    participation = args.task == 'participation'
    args.dataset = args.dataset or ROOT / ('data/participation/synthetic-pilot-v1.jsonl' if participation else 'data/classifier/synthetic-pilot-v2.jsonl')
    args.dev_dataset = args.dev_dataset or ROOT / ('data/participation/development-v1.jsonl' if participation else 'data/classifier/realistic-dev-v2.jsonl')
    args.output = args.output or ROOT / 'runs' / (('participation-' if participation else 'classifier-') + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args.prompt_layout = args.prompt_layout or ('schema_first' if participation else 'state_first')
    if args.max_tokens is None:
        args.max_tokens = 768 if participation else 384  # Conversation context needs a measured, explicit larger budget.
    if args.epochs < 1 or args.rank < 1 or args.accumulation < 1 or args.max_tokens < 128 or not 0 < args.learning_rate < 1:
        parser.error("Invalid training hyperparameter")
    splits = load_dataset(args.dataset, task=args.task)
    development = load_development(args.dev_dataset, splits, task=args.task)
    print(json.dumps({"counts": {s: dict(Counter(r["label"] for r in rows)) for s, rows in splits.items()},
                      "development_rows": len(development),
                      "synthetic_draft_rows": sum(r.get("review_status") != "approved" for rows in splits.values() for r in rows)}, indent=2), flush=True)
    if not args.check_data:
        run(args, splits)


if __name__ == "__main__":
    main()
