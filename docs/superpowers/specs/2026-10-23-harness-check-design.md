# Harness Check: on-device CAN pin-swap detection

Date: 2026-10-23
Status: approved design

## Problem

On Hyundai/Kia/Genesis CAN-FD cars, openpilot can only steer if the comma harness puts the
camera-side steering messages on the camera bus (bus 2, or bus 6 with a second panda). Some trims
need a few CAN pins swapped on the stock comma harness. Today the only way to check is to open a
route in Cabana and look at which bus `LKAS` / `LKAS_ALT` (HDA2) or `LFA` / `LFA_ALT` (non-HDA2)
arrive on. We want the device to answer that by itself, live, the first time the car is turned on.

openpilot already makes this exact decision when it fingerprints the car
(`opendbc/car/hyundai/interface.py`): `lka_steering = 0x50 in fingerprint[cam_can] or 0x110 in
fingerprint[cam_can]`, where `cam_can = 2 + offset` and `offset = 4 * (max active bus // 4)`.
If the harness is wrong, that test silently fails and the car is set up as non-HDA2.

## Decisions

- Data source: live CAN while the car is on (no rlog scanning).
- Surfaces: a Developer panel item with the full breakdown, plus an automatic onroad alert when the
  result says a pin swap is needed.
- Architecture: a new listen-only process `harnessd`, a pure-Python classifier, a settings item and
  a HUD overlay. No changes to `card`, `selfdrived` or cereal.
- Toggleable: `HarnessCheckEnabled`, default on. Off stops `harnessd` and hides the alert.

## Messages

| Name | Address | Meaning when seen on the camera bus |
|---|---|---|
| `LKAS` | 0x50 (80) | HDA2: the camera's LKA steering message |
| `LKAS_ALT` | 0x110 (272) | HDA2 (alternate) |
| `LFA` | 0x12A (298) | non-HDA2: the camera steers the MDPS directly |
| `LFA_ALT` | 0xCB (203) | non-HDA2 (alternate) |

Only frames the car sent count: `can` entries with `src >= 64` are panda TX echoes (+128) or
blocked frames (+192) and are ignored (Cabana treats them the same way).

## Units

### 1. Classifier: `openpilot/sunnypilot/selfdrive/harness/harness_check.py`

Pure Python, no openpilot runtime imports, so it is unit-testable anywhere.

- `HarnessTally`: `ingest(t_ns, src, address, length)` counts frames per `(bus, address)` with when
  each was first and last seen, the longest frame, and the CAN time span (`seconds`, from the frames'
  own `logMonoTime`). `hz(key)` is a message's rate over its own span, so a camera that starts late
  still reads its true rate.
- `Stamp(t, mono, boot)`: wall clock (only shown as a date), `time.monotonic()` and the kernel boot id
  (`/proc/sys/kernel/random/boot_id`, `""` where absent).
- `classify(tally, decision, stamp) -> dict`: returns the result written to the `HarnessCheck`
  param.
  - `offset = 4 * (max active bus // 4)`; `camera_bus = 2 + offset`.
  - A message counts as present on a bus only with at least `MIN_FRAMES = 20` frames.
  - CAN-FD = any frame longer than 8 bytes.
  - Verdict, first match wins:
    1. not CAN-FD → `inconclusive` ("This car's CAN is not CAN-FD; the check only applies to
       Hyundai/Kia/Genesis CAN-FD cars.")
    2. `LKAS` or `LKAS_ALT` on the camera bus → `ok_hda2`
    3. `LKAS` or `LKAS_ALT` only on other buses → `swap_needed` (names the buses)
    4. `LFA` or `LFA_ALT` on the camera bus → `non_hda2` (correct for a non-HDA2 car; openpilot
       will treat the car as non-HDA2)
    5. `LFA` or `LFA_ALT` only on other buses → `swap_likely`
    6. otherwise → `inconclusive` ("No steering messages seen yet.")
  - Result fields: `verdict`, `title`, `detail`, `action`, `camera_bus`, `can_fd`, `buses` (per bus:
    frames, ids, and for each steering message with at least `MIN_FRAMES` frames its count and rate in
    Hz), `seconds`, `final`, `decision` (openpilot's own fingerprint decision, see below),
    `disagrees`, `t`, `mono`, `boot`.
- `decision` input: `{"recognized": bool, "car": str, "hda2": bool | None}`, built by `harnessd`
  from `CarParams` (`brand == "hyundai"` and `HyundaiFlags.CANFD_LKA_STEER_MSG`). The result
  notes when openpilot's decision disagrees with what the bus shows.
- `HarnessSession`: the write cadence. `due()` every `PROVISIONAL_PERIOD_S` of CAN time until a final
  result has been written; `result(decision, stamp)` classifies and sets `final` (`is_final()`);
  `finish(result)` is called only after the params write succeeded, so a failed write is retried;
  `changed` reports verdict transitions for logging.
- `settled(result, drive_start, boot)` / `alert_text(result, drive_start, boot)`: a result belongs to
  this drive only if its `boot` matches and its `mono` is at or after the drive's start; the alert
  needs a final result with a swap verdict.
- The popup body is built UI-side in `openpilot/selfdrive/ui/sunnypilot/harness_status.py`
  (`harness_status_html(result)`), next to `scene_status.py`, using `comm_issue_watch._safe`. The
  settings viewer has no tables, so the per-bus breakdown is a list.

### 2. Service: `openpilot/sunnypilot/selfdrive/harness/harnessd.py`

- Process `harnessd`, gate `and_(only_onroad, harness_check_enabled)`, `enabled=not PC`.
- Subscribes to `can`, feeds every frame to the session's `HarnessTally`.
- Writes a provisional result to `HarnessCheck` every 2 s of CAN time; the final result at 15 s. If
  still `inconclusive` at 15 s, keeps listening and re-checks every 2 s until 60 s, then writes it final.
- Reads `CarParams` on every write (it can still hold the previous drive's car for the first seconds).
- Logs each verdict change; after the final write it closes the socket and idles (sleep loop) until
  the manager stops it.
- Never transmits. Exceptions inside one cycle are logged with `cloudlog.exception` and the loop
  continues.

### 3. Settings: `openpilot/selfdrive/ui/sunnypilot/layouts/settings/harness_check.py`

`HarnessCheckItems`, added to the Developer panel next to Dash Research:
- toggle "Harness Check" (`HarnessCheckEnabled`)
- "Harness Check Result" — VIEW opens `HtmlModalSP(harness_status_html(params.get("HarnessCheck")))`;
  visible only when the toggle is on.

### 4. Onroad alert: `openpilot/selfdrive/ui/sunnypilot/onroad/overlays/harness_alert.py`

- `HarnessAlert`, added to `HUD_OVERLAY_TYPES`; independent of Drive Notifications.
- Records `time.monotonic()` when the drive starts; reads `HarnessCheck` at most once a second
  while onroad and the toggle is on, and stops reading once this drive's result is final.
- Shows a red card for 30 s when this drive's final result has verdict `swap_needed` or
  `swap_likely`: the result's `title`, plus "Settings > Developer > Harness Check" (ASCII; the font
  may lack an arrow glyph). Nothing for other verdicts or provisional results: in the first seconds
  the ADAS ECU's LFA can arrive before the camera's LKAS.

### 5. Wiring

- `openpilot/common/params_keys.h`: `HarnessCheckEnabled {PERSISTENT | BACKUP, BOOL, "1"}`,
  `HarnessCheck {PERSISTENT, JSON}`.
- `openpilot/system/manager/process_config.py`: gate function and `PythonProcess`.
- `openpilot/selfdrive/ui/sunnypilot/layouts/settings/developer.py`: add the items.
- `openpilot/selfdrive/ui/sunnypilot/ui_state.py`: `harness_check` toggle flag.

### 6. Radar survey: `openpilot/sunnypilot/selfdrive/harness/radar_survey.py`

Added 2026-10-23 so a car's radar layout can be read without SSH. Pure, fed by the same tally and window.

- Ranges (from the upstream `hyundai-radar-tracks` research): `500` (0x500, 64 slots, 8 B), `602` (0x602, 22, 8 B),
  `210` (0x210, 16, 32 B), `3A5` (0x3A5, 32, 24 B), `3D0` aux (0x3D0, 5, 32 B), `235` camera objects (0x235, 20, 32 B).
- A range counts on a bus when at least half its addresses (minimum 2) have `MIN_FRAMES` frames. Each entry: bus, addresses
  present of expected, frame sizes, median rate, and the latest raw frame of the range's first address (hex).
- Verdict: `supported_210`, `size_match_3a5` (size matches, payload unconfirmed: the 2026 Sportage sample did not decode),
  `unknown_layout` (210/3A5 at another size), `legacy` (500/602), `none`. Camera and aux ranges are listed but never radar.
- The tally records each ID's frame size when first seen and the latest frame for the six range-start addresses only.
- Shown as a "Radar" section on the result page; no alert.

## Testing

- `harness/tests/test_harness_check.py` (pure, runs on Windows): every verdict row; second panda
  (camera bus 6); single stray frames below `MIN_FRAMES`; non-CAN-FD traffic; TX echo / blocked
  frames ignored; decision disagreement note; result shape (contract) and HTML rendering of every
  verdict plus `None`/empty input.
- `harnessd` timing logic lives in the pure `HarnessSession` and is tested the same way, as are
  `settled`/`alert_text` (provisional, earlier drive, earlier boot) and `harness_status_html`.
- The overlay and settings item follow the existing overlay/settings test patterns; those need the
  Linux openpilot environment.

## Out of scope

- Scanning saved rlogs.
- Showing the wiring diagram itself or naming specific pins.
- Changing how openpilot fingerprints or steers.
