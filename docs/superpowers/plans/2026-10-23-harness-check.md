# Harness Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The device tells, live and without Cabana, whether the comma harness puts the camera's steering messages on the camera bus or needs a pin swap.

**Architecture:** A pure classifier (`harness_check.py`) tallies frames per `(bus, address)` from the `can` stream and returns a verdict dict. A listen-only process `harnessd` feeds it for 15-60 s per drive and writes the result to the `HarnessCheck` param. A Developer panel item renders it (`harness_status.py`), and a HUD overlay (`harness_alert.py`) shows a 30 s red card when a pin swap is needed.

**Tech Stack:** Python 3.12, openpilot `messaging`/`Params`, raylib (`pyray`) UI, pytest.

Spec: `docs/superpowers/specs/2026-10-23-harness-check-design.md`

## Global Constraints

- Listen-only: never transmit on CAN.
- Steering messages: `LKAS` 0x50, `LKAS_ALT` 0x110 (HDA2); `LFA` 0x12A, `LFA_ALT` 0xCB (non-HDA2).
- Camera bus = `2 + 4 * (max active bus // 4)` (opendbc `CanBusBase` rule).
- Ignore `can` entries with `src >= 64` (panda TX echo +128, blocked +192).
- A message counts on a bus only with `MIN_FRAMES = 20` frames.
- Timing: provisional result every 2 s, final at 15 s, inconclusive keeps listening until 60 s.
- Params: `HarnessCheckEnabled {PERSISTENT | BACKUP, BOOL, "1"}`, `HarnessCheck {PERSISTENT, JSON}`.
- Alert: 30 s, only for `swap_needed` / `swap_likely`, only for a result written during this drive; independent of Drive Notifications.
- The device HTML renderer supports only `h1`-`h6`, `p`, `b`, `ul`, `li`, `br`; each tag starts a new block, so no inline bold. Escape dynamic text with `comm_issue_watch._safe`.
- On-screen text is ASCII (">" not an arrow glyph).
- Pure modules (`harness_check.py`, `harness_status.py`) must not import openpilot runtime pieces (messaging, Params, pyray) so their tests run on Windows.

Windows test command (from the repo root, scratchpad venv with numpy + pytest):
`PYTHONPATH="<repo>" python -m pytest -p no:cacheprovider --noconftest -q <test files>`

---

### Task 1: Classifier

**Files:**
- Create: `openpilot/sunnypilot/selfdrive/harness/__init__.py` (empty)
- Create: `openpilot/sunnypilot/selfdrive/harness/harness_check.py`
- Create: `openpilot/sunnypilot/selfdrive/harness/tests/__init__.py` (empty)
- Test: `openpilot/sunnypilot/selfdrive/harness/tests/test_harness_check.py`

**Interfaces:**
- Produces: `HarnessTally.ingest(t_ns: int, src: int, address: int, length: int) -> None`, `HarnessTally.seconds -> float`,
  `classify(tally, decision: dict | None, now: float) -> dict`, `is_final(seconds: float, verdict: str) -> bool`,
  `alert_text(result: dict | None, drive_start: float) -> str | None`,
  `decision_from(fingerprint: str, brand: str, lka_steer: bool) -> dict`, `Verdict` (StrEnum), `PROVISIONAL_PERIOD_S`,
  `FINAL_S`, `GIVE_UP_S`, `MIN_FRAMES`.
- Result dict keys: `verdict, title, detail, action, camera_bus, can_fd, seconds, t, buses, decision, disagrees, final`
  (`final` defaults to False; harnessd sets it). `buses` = list of `{"bus", "frames", "ids", "steering": {name: {"frames", "hz"}}}`.

- [ ] **Step 1: Write the failing tests** (verdict per row, second panda, stray frames, non-CAN-FD, TX echo ignored,
  decision/disagreement, timing, alert, result contract) in `test_harness_check.py`.
- [ ] **Step 2: Run them, expect ImportError** — `pytest .../harness/tests -q` → `ModuleNotFoundError: ...harness_check`.
- [ ] **Step 3: Implement `harness_check.py`** (see the module in the commit; constants above, verdict order exactly as in the spec).
- [ ] **Step 4: Run tests, expect all pass.**
- [ ] **Step 5: Commit** `feat(harness): classifier for which bus the steering messages arrive on`.

### Task 2: Result page

**Files:**
- Create: `openpilot/selfdrive/ui/sunnypilot/harness_status.py`
- Test: `openpilot/selfdrive/ui/sunnypilot/tests/test_harness_status.py`

**Interfaces:**
- Consumes: the result dict from Task 1 (`classify`).
- Produces: `harness_status_html(result: dict | None) -> str`.

- [ ] **Step 1: Write failing tests:** `None`/`{}` → "No harness check yet"; each verdict renders its title, detail and action under "What to do";
  provisional result says "Still listening"; per-bus list marks "(camera bus)" and lists steering messages with Hz;
  decision lines (recognized HDA2, recognized + disagrees, not recognized); dynamic text containing `<` is escaped.
- [ ] **Step 2: Run, expect ImportError.**
- [ ] **Step 3: Implement** using `_safe` from `openpilot.selfdrive.ui.sunnypilot.comm_issue_watch`, one block per line (no inline tags).
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** `feat(harness): result page for the Developer panel`.

### Task 3: harnessd service and params

**Files:**
- Create: `openpilot/sunnypilot/selfdrive/harness/harnessd.py`
- Modify: `openpilot/common/params_keys.h` (after `HideVEgoUI`-adjacent sunnypilot block, alphabetical near `GreenLightAlert`)
- Modify: `openpilot/system/manager/process_config.py` (gate next to `dash_re_enabled`; process next to `dashred`)

**Interfaces:**
- Consumes: `HarnessTally`, `classify`, `decision_from`, `is_final`, `PROVISIONAL_PERIOD_S`.
- Produces: `HarnessCheck` param (result dict with `final`), process `harnessd`.

```python
def harness_check_enabled(started, params, CP: car.CarParams) -> bool:
  return params.get_bool("HarnessCheckEnabled")

  # harness check: which bus the camera's steering messages arrive on (listen-only)
  PythonProcess("harnessd", "openpilot.sunnypilot.selfdrive.harness.harnessd", and_(only_onroad, harness_check_enabled), enabled=not PC),
```

- [ ] **Step 1: Implement `harnessd.py`**: drain `can`, `tally.ingest(msg.logMonoTime, c.src, c.address, len(c.dat))`; every
  `PROVISIONAL_PERIOD_S` of data time classify with `read_decision(params)` and `time.time()`, set `final = is_final(...)`,
  `params.put("HarnessCheck", result)`; on final, close the socket and sleep forever. Exceptions in a cycle → `cloudlog.exception`, continue.
- [ ] **Step 2: Register params and process** as above.
- [ ] **Step 3: Verify** `python -c "import ast; ast.parse(open(f).read())"` for each changed .py; grep `params_keys.h` for both keys.
- [ ] **Step 4: Commit** `feat(harness): harnessd listen-only service and params`.

### Task 4: Developer panel item

**Files:**
- Create: `openpilot/selfdrive/ui/sunnypilot/layouts/settings/harness_check.py`
- Modify: `openpilot/selfdrive/ui/sunnypilot/layouts/settings/developer.py` (construct `HarnessCheckItems`, add its items before Dash Research, call `update_state()`)
- Modify: `openpilot/selfdrive/ui/sunnypilot/ui_state.py` (`self.harness_check = self.params.get_bool("HarnessCheckEnabled")` next to `hard_brake_flash`)

**Interfaces:**
- Consumes: `harness_status_html`.
- Produces: `HarnessCheckItems.items`, `HarnessCheckItems.update_state()`, `ui_state.harness_check`.

- [ ] **Step 1: Implement `HarnessCheckItems`**: toggle (`HarnessCheckEnabled`), "Harness Check Result" VIEW button visible when the toggle is on, opening `HtmlModalSP(text=harness_status_html(...))`.
- [ ] **Step 2: Wire into `DeveloperLayoutSP`** and `ui_state`.
- [ ] **Step 3: Verify** syntax via `ast.parse`; `pytest` the Windows-runnable suites still pass.
- [ ] **Step 4: Commit** `feat(harness): Developer panel toggle and result viewer`.

### Task 5: Onroad alert

**Files:**
- Create: `openpilot/selfdrive/ui/sunnypilot/onroad/overlays/harness_alert.py`
- Modify: `openpilot/selfdrive/ui/sunnypilot/onroad/overlays/__init__.py` (import; append `HarnessAlert` to `HUD_OVERLAY_TYPES`)

**Interfaces:**
- Consumes: `alert_text`, `ui_state.harness_check`, `ui_state.started`, `HarnessCheck` param.
- Produces: `HarnessAlert` with `update()` / `render(rect)` like the other HUD overlays.

- [ ] **Step 1: Implement**: on the first started frame record `drive_start = time.time()`; read the param at most once a second until
  `alert_text` returns a title; show for `SHOW_S = 30` s with a fade; red rounded card with the title and "Settings > Developer > Harness Check".
- [ ] **Step 2: Register** in `HUD_OVERLAY_TYPES` (last, drawn on top).
- [ ] **Step 3: Verify** syntax via `ast.parse`.
- [ ] **Step 4: Commit** `feat(harness): onroad alert when a pin swap is needed`.

### Task 6: Spec sync

- [ ] Update the spec where implementation settled details (HTML builder lives UI-side in `harness_status.py`; timing is the pure `is_final()`; elapsed time is measured from CAN timestamps). Commit `docs: harness check spec matches implementation`.
