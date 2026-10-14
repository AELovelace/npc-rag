"""Keep synthetic variants and evaluation examples out of the wrong training partitions."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from train_classifier import load_dataset, load_development, report_rows
from classifier_augmentation import augment_rows, check_shortcut


def fixture_rows():
    rows = []
    for split in ("train", "validation", "test"):
        for label in ("game", "general"):
            rows.append({"id": f"{split}-{label}", "message": f"Fixture message {split} {label}",
                         "label": label, "split": split, "family_id": split, "split_group": split})
    return rows


def write_rows(tmp_path, rows):
    path = tmp_path / "dataset.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_checked_in_pilot_has_balanced_family_safe_partitions():
    data = load_dataset(ROOT / "data/classifier/synthetic-pilot-v2.jsonl")
    assert {key: len(rows) for key, rows in data.items()} == {"train": 800, "validation": 100, "test": 100}
    for rows in data.values():
        assert sum(r["label"] == "game" for r in rows) == len(rows) // 2


@pytest.mark.parametrize("field", ["family_id", "split_group"])
def test_related_rows_cannot_cross_splits(tmp_path, field):
    rows = fixture_rows()
    rows[2][field] = rows[0][field]
    with pytest.raises(ValueError, match="cross data splits"):
        load_dataset(write_rows(tmp_path, rows))


def test_case_and_punctuation_do_not_hide_duplicate_messages(tmp_path):
    rows = fixture_rows()
    rows[1]["message"] = rows[0]["message"].upper() + "!!!"
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(write_rows(tmp_path, rows))


def test_original_evaluation_cannot_be_imported_as_training(tmp_path):
    rows = fixture_rows()
    evaluation = json.loads((ROOT / "data/eval_messages.json").read_text())
    rows[0]["message"] = evaluation["game"][0]
    with pytest.raises(ValueError, match="fixed-evaluation overlap"):
        load_dataset(write_rows(tmp_path, rows))


def test_success_on_only_one_class_fails_training_report_gate():
    report = report_rows([{"expected": "game", "predicted": "general"},
                          {"expected": "general", "predicted": "general"}])
    assert report["accuracy"] == .5 and not report["gate_passed"]
    assert report["by_class"]["game"]["recall"] == 0


def test_shortcut_ceiling_and_epoch_augmentation():
    data = load_dataset(ROOT / 'data/classifier/synthetic-pilot-v2.jsonl')
    check_shortcut(data['train'])
    for seed in range(417, 437):
        rows = augment_rows(data['train'], seed)
        assert len(rows) == 800
        assert [(r['id'], r['label'], r['family_id']) for r in rows] == [(r['id'], r['label'], r['family_id']) for r in data['train']]
        check_shortcut(rows)
    assert augment_rows(data['train'], 417) == augment_rows(data['train'], 417)
    assert augment_rows(data['train'], 417) != augment_rows(data['train'], 418)
    with pytest.raises(ValueError, match='shortcut'):
        check_shortcut([{'message': 'mechanic?', 'label': 'game'}, {'message': 'you are nice', 'label': 'general'}])


def test_development_separate_from_training_and_fixed_evaluation(tmp_path):
    data = load_dataset(ROOT / 'data/classifier/synthetic-pilot-v2.jsonl')
    rows = load_development(ROOT / 'data/classifier/realistic-dev-v2.jsonl', data)
    assert len(rows) == 200
    fixed = json.loads((ROOT / 'data/eval_messages.json').read_text())
    rows[0]['message'] = fixed['general'][0]
    with pytest.raises(ValueError, match='overlaps'):
        load_development(write_rows(tmp_path, rows), data)


def test_emoji_reactions_are_distinct():
    from build_classifier_seed import normalize
    assert normalize('😅') != normalize('🌻')
