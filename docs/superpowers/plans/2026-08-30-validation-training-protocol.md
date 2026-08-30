# Validation Training Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Add deterministic validation selection, per-step cosine scheduling, early stopping, and one final test report to the training command.

**Architecture:** A focused NumPy split module returns training and validation masks. The CLI constructs a post-split training scene, checkpoints selection state, chooses best.pt using validation only, restores it, and emits one test report.

**Tech Stack:** Python 3.11, NumPy, PyTorch, PyYAML, pytest, Ruff.

**Spec:** docs/superpowers/specs/2026-08-30-validation-training-protocol-design.md

## Global Constraints

- Validation candidates are only seen-class pixels from the original train_mask; test data never influences selection.
- Normalization, pseudo-RGB statistics, tile eligibility, and class-aware sampling only use the post-split training mask.
- Preserve the existing network and loss APIs.
- New checkpoints persist scheduler and selection state; old checkpoints without selection state remain loadable.
- Do not modify .vscode/ or docs/figures/.

---

### Task 1: Implement deterministic stratified masks

**Files:**
- Create: src/hsi_lidar_ovseg/data/splits.py
- Modify: src/hsi_lidar_ovseg/data/__init__.py
- Create: tests/test_splits.py

**Interfaces:**
- Produces: split_training_mask(labels, train_mask, seen_ids, validation_fraction, seed) -> tuple[np.ndarray, np.ndarray].
- Consumed by: CLI training orchestration.

- [ ] **Step 1: Write the failing tests**

~~~python
def test_split_training_mask_is_reproducible_and_stratified() -> None:
    labels = np.array([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=np.int64)
    original = np.ones_like(labels, dtype=np.bool_)

    train_a, validation_a = split_training_mask(labels, original, (1, 2), 0.25, 7)
    train_b, validation_b = split_training_mask(labels, original, (1, 2), 0.25, 7)

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    assert not np.any(train_a & validation_a)
    assert [int(np.sum(validation_a & (labels == item))) for item in (1, 2)] == [1, 1]


def test_split_training_mask_keeps_singleton_seen_class_in_training() -> None:
    labels = np.array([[1, 1, 2, 3]], dtype=np.int64)
    train, validation = split_training_mask(labels, np.ones_like(labels, bool), (1, 2), 0.5, 19)

    assert bool(train[0, 2]) and not bool(validation[0, 2])
    assert bool(train[0, 3]) and not bool(validation[0, 3])
~~~

The first test catches a nondeterministic or non-stratified split. The second catches removal of the sole training label for a class.

- [ ] **Step 2: Confirm RED**

Run: D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_splits.py -q

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the module**

Implement the public function with shape/type checks, a DataError for invalid fraction, seed, mask shape, or no seen IDs, and per-class RNG SeedSequence((seed, class_id)). For a class with count >= 2, select:

~~~python
validation_count = min(count - 1, max(1, round(count * validation_fraction)))
~~~

Set only selected coordinates in the validation mask, clear them from a copied training mask, and export the function from data/__init__.py.

- [ ] **Step 4: Confirm GREEN**

Run: D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_splits.py -q

Expected: all tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add src/hsi_lidar_ovseg/data/splits.py src/hsi_lidar_ovseg/data/__init__.py tests/test_splits.py
git commit -m "feat: add deterministic validation split"
~~~

### Task 2: Add configuration and checkpoint selection state

**Files:**
- Modify: src/hsi_lidar_ovseg/config.py:221-256
- Modify: src/hsi_lidar_ovseg/engine/checkpoint.py:34-128
- Modify: tests/test_config.py
- Modify: tests/test_checkpoint.py

**Interfaces:**
- Produces: validated validation_fraction, early_stopping_patience, early_stopping_min_delta, cosine_eta_min.
- Produces: optional TrainingState.selection_state containing best_score and epochs_without_improvement.
- Consumed by: scheduler and selection logic in task 4.

- [ ] **Step 1: Write failing tests**

~~~python
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("validation_fraction", 0.0),
        ("validation_fraction", 1.0),
        ("early_stopping_patience", 0),
        ("early_stopping_min_delta", -0.1),
    ],
)
def test_train_config_rejects_invalid_validation_controls(field: str, value: float | int) -> None:
    with pytest.raises(ConfigError):
        TrainConfig(**{field: value})


def test_checkpoint_round_trip_preserves_selection_state(tmp_path: Path) -> None:
    state = _state(model, optimizer)
    state.selection_state = {"best_score": 0.4, "epochs_without_improvement": 3}
    save_checkpoint(path, state)

    assert load_checkpoint(path, model, optimizer, _identity()).selection_state == state.selection_state
~~~

Add a legacy-payload test that deletes selection_state from a serialized payload and expects None after loading. These tests catch rejected invalid protocol config, lost resume state, and a backward-incompatible decoder.

- [ ] **Step 2: Confirm RED**

Run: D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_config.py tests/test_checkpoint.py -q

Expected: missing configuration fields and missing selection_state failure.

- [ ] **Step 3: Implement validated state**

Add defaults 0.1, 20, 0.0, and 1e-6 to TrainConfig. Require fraction in (0, 1), positive patience, nonnegative delta, and 0 <= cosine_eta_min < min(learning_rate, backbone_learning_rate).

Add selection_state: dict[str, float | int] | None to TrainingState; serialize it; treat absence in old payloads as None; require finite best_score and nonnegative integer epochs_without_improvement when supplied.

- [ ] **Step 4: Confirm GREEN**

Run: D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_config.py tests/test_checkpoint.py -q

Expected: all targeted tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add src/hsi_lidar_ovseg/config.py src/hsi_lidar_ovseg/engine/checkpoint.py tests/test_config.py tests/test_checkpoint.py
git commit -m "feat: persist validation selection state"
~~~

### Task 3: Construct and wire a per-step cosine schedule

**Files:**
- Modify: src/hsi_lidar_ovseg/cli.py:262-449
- Modify: tests/test_training_smoke.py

**Interfaces:**
- Produces: _cosine_scheduler(optimizer, config, steps_per_epoch) -> CosineAnnealingLR, with T_max=epochs * steps_per_epoch and eta_min=cosine_eta_min.
- Consumed by: existing Trainer.train_step, which already advances a provided scheduler after each optimizer step.

- [ ] **Step 1: Write the failing construction test**

~~~python
def test_cosine_scheduler_uses_total_training_steps() -> None:
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-4)
    scheduler = _cosine_scheduler(optimizer, TrainConfig(epochs=3, cosine_eta_min=1e-6), 5)

    assert scheduler.T_max == 15
    assert scheduler.eta_min == 1e-6
~~~

This catches an incorrect scheduling horizon or missing scheduler factory without testing PyTorch's implementation.

- [ ] **Step 2: Confirm RED**

Run: D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_training_smoke.py::test_cosine_scheduler_uses_total_training_steps -q

Expected: import failure because _cosine_scheduler does not exist.

- [ ] **Step 3: Build scheduler after DataLoader**

Add _cosine_scheduler in cli.py, then move Trainer construction in _train_command to after Dataset/DataLoader construction. Create the scheduler through that helper, pass it to Trainer, restore it in load_checkpoint, and persist its state for every checkpoint.

- [ ] **Step 4: Confirm GREEN**

Run: D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_training_smoke.py tests/test_checkpoint.py -q

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add src/hsi_lidar_ovseg/cli.py tests/test_training_smoke.py tests/test_checkpoint.py
git commit -m "feat: add cosine learning-rate schedule"
~~~

### Task 4: Select on validation, early stop, and test once

**Files:**
- Modify: src/hsi_lidar_ovseg/cli.py:341-449
- Modify: tests/test_cli.py
- Modify: README.md
- Modify: configs/base.yaml
- Modify: configs/houston2013.yaml
- Modify: configs/trento.yaml
- Modify: configs/muufl.yaml

**Interfaces:**
- Consumes: split masks, selection controls, scheduler, and checkpoint selection state.
- Produces: last.pt, validation-selected best.pt, and test_metrics.json.
- Ensures: the test mask is used only after restoring best.pt.

- [ ] **Step 1: Write the failing offline CLI integration test**

Extend the existing CPU fixture with epochs: 3, validation_fraction: 0.25, early_stopping_patience: 1, and cosine_eta_min: 0.000001. Assert:

~~~python
assert (output_dir / "last.pt").is_file()
assert (output_dir / "best.pt").is_file()
metrics = json.loads((output_dir / "test_metrics.json").read_text(encoding="utf-8"))
assert "miou" in metrics
state = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=True)
assert state["scheduler_state"] is not None
assert state["selection_state"] is not None
~~~

The fixture must contain at least two labeled pixels per seen class, so validation selection is observable. This catches missing final test output and missing resumable protocol state.

- [ ] **Step 2: Confirm RED**

Run: D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_cli.py::test_cli_runs_one_offline_cpu_training_epoch -q

Expected: failure because test_metrics.json and selection state do not exist.

- [ ] **Step 3: Implement orchestration**

1. Call split_training_mask and create a SceneArrays training view with the split training mask.
2. Fit normalization and create the training Dataset from that view.
3. After each epoch, evaluate only validation_mask and update finite best_score/patience state.
4. Save selection state to last.pt; save best.pt only when validation exceeds best_score + early_stopping_min_delta; break at configured patience.
5. Restore best.pt, compute metrics once on the original scene’s test_mask, and write test_metrics.json.

- [ ] **Step 4: Update defaults and documentation**

Set epochs: 100 and explicit new controls in all four YAML files. Document the stratified 90/10 split, validation-selected checkpoints, early stopping, and final-only test report.

- [ ] **Step 5: Confirm GREEN**

Run: D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_cli.py tests/test_training_smoke.py -q

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

~~~powershell
git add src/hsi_lidar_ovseg/cli.py tests/test_cli.py README.md configs/base.yaml configs/houston2013.yaml configs/trento.yaml configs/muufl.yaml
git commit -m "feat: select checkpoints on validation metrics"
~~~

### Task 5: Verify the completed branch

**Files:**
- Verify: all files changed in tasks 1–4.

- [ ] **Step 1: Run complete tests**

Run: D:\miniconda\envs\hsi-lidar\python.exe -m pytest -q

Expected: zero failures.

- [ ] **Step 2: Run static checks**

Run: D:\miniconda\envs\hsi-lidar\python.exe -m ruff check .

Expected: All checks passed!

Run: D:\miniconda\envs\hsi-lidar\python.exe -m ruff format --check .

Expected: all files formatted.

- [ ] **Step 3: Validate YAML and build**

Run: D:\miniconda\envs\hsi-lidar\python.exe -c "import sys; from pathlib import Path; sys.path.insert(0, 'src'); from hsi_lidar_ovseg.config import load_config; [load_config(path, check_files=False) for path in Path('configs').glob('*.yaml')]; print('validated')"

Expected: validated.

Run: D:\miniconda\envs\hsi-lidar\python.exe -m build --no-isolation

Expected: a source distribution and wheel are built.

- [ ] **Step 4: Commit the plan and final work**

~~~powershell
git add docs/superpowers/plans/2026-08-30-validation-training-protocol.md
git commit -m "docs: add validation training implementation plan"
~~~
