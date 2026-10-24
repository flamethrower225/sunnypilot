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

- `HarnessTally`: `ingest(src, address, length)` counts frames per `(bus, address)`, total frames
  and distinct addresses per bus, and the longest frame seen. `elapsed` is tracked by the caller.
- `classify(tally, seconds, decision) -> dict`: returns the result written to the `HarnessCheck`
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
  - Result fields: `verdict`, `title`, `detail`, `action`, `camera_bus`, `buses` (per bus: frames,
    ids, and for each of the four messages the frame count and rate in Hz), `seconds`, `final`,
    `decision` (openpilot's own fingerprint decision, see below), `t` (wall-clock seconds).
- `decision` input: `{"recognized": bool, "car": str, "hda2": bool | None}`, built by `harnessd`
  from `CarParams` (`brand == "hyundai"` and `HyundaiFlags.CANFD_LKA_STEER_MSG`). The result
  notes when openpilot's decision disagrees with what the bus shows.
- `result_html(result) -> str`: the popup body (verdict, action, per-bus table, decision, time).

### 2. Service: `openpilot/sunnypilot/selfdrive/harness/harnessd.py`

- Process `harnessd`, gate `and_(only_onroad, harness_check_enabled)`, `enabled=not PC`.
- Subscribes to `can`, feeds every frame to `HarnessTally`.
- Writes a provisional result to `HarnessCheck` every 2 s; the final result at 15 s. If still
  `inconclusive` at 15 s, keeps listening and re-checks every 2 s until 60 s, then writes it final.
- After the final write it closes the socket and idles (sleep loop) until the manager stops it.
- Never transmits. Exceptions inside one cycle are logged with `cloudlog.exception` and the loop
  continues.

### 3. Settings: `openpilot/selfdrive/ui/sunnypilot/layouts/settings/harness_check.py`

`HarnessCheckItems`, added to the Developer panel next to Dash Research:
- toggle "Harness Check" (`HarnessCheckEnabled`)
- "Harness Check Result" — VIEW opens `HtmlModalSP(result_html(params.get("HarnessCheck")))`;
  visible only when the toggle is on.

### 4. Onroad alert: `openpilot/selfdrive/ui/sunnypilot/onroad/overlays/harness_alert.py`

- `HarnessAlert`, added to `HUD_OVERLAY_TYPES`; independent of Drive Notifications.
- Reads `HarnessCheck` at most once a second while onroad and the toggle is on.
- Shows a red card for 30 s when this drive's result (`t` at or after the drive started) has
  verdict `swap_needed` or `swap_likely`: the result's `title`, plus "Settings → Developer →
  Harness Check". Nothing for other verdicts.

### 5. Wiring

- `openpilot/common/params_keys.h`: `HarnessCheckEnabled {PERSISTENT | BACKUP, BOOL, "1"}`,
  `HarnessCheck {PERSISTENT, JSON}`.
- `openpilot/system/manager/process_config.py`: gate function and `PythonProcess`.
- `openpilot/selfdrive/ui/sunnypilot/layouts/settings/developer.py`: add the items.
- `openpilot/selfdrive/ui/sunnypilot/ui_state.py`: `harness_check` toggle flag.

## Testing

- `harness/tests/test_harness_check.py` (pure, runs on Windows): every verdict row; second panda
  (camera bus 6); single stray frames below `MIN_FRAMES`; non-CAN-FD traffic; TX echo / blocked
  frames ignored; decision disagreement note; result shape (contract) and HTML rendering of every
  verdict plus `None`/empty input.
- `harnessd` timing logic is factored into a pure `HarnessSchedule` and tested the same way.
- The overlay and settings item follow the existing overlay/settings test patterns; those need the
  Linux openpilot environment.

## Out of scope

- Scanning saved rlogs.
- Showing the wiring diagram itself or naming specific pins.
- Changing how openpilot fingerprints or steers.
