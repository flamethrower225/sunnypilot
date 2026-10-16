# Elantra digital dash: CAN reverse engineering

Tools and a plan for working out what drives the 2021–23 Elantra (CN7) instrument cluster, and what openpilot
can change on it. Everything here only listens to CAN; nothing transmits.

## On the device, no SSH needed

Settings → Developer → **Dash Research** turns on `dashred`, a background service that runs while the car is
on. It writes its findings to params every few seconds, and they stay there after the car is off.

- **Dash Research Findings** shows what the service has found:
  - **Free text check:** multi-frame (ISO-TP) traffic outside diagnostics, which is the shape text takes, plus
    any head unit → cluster (M-CAN) IDs that reach the buses the harness sees
  - **Camera-only messages:** substitutable with a safety change. Any missing from the DBC are flagged.
  - **Bus 1 contents**
  - **Orphans:** cluster inputs no ECU sends on this car, after 30 s of listening
  - **Presence checks:** `Navi_HU`, `LKAS12` and the climate messages
- **Mark Dash Action** (car on, parked): pick what you're about to do, tap Select, and do it within a second.
  The service compares the 2 s before with the 3.5 s after.
- **Dash Action Results** lists the bits that changed for each mark. `[CLU]` marks a signal the cluster reads.
  Counter and checksum bits are filtered out.

Start with the two turn-signal reference marks. Each should report `CGW1` bit 19 (`CF_Gway_TurnSigLh`), which
confirms the timing works before you trust the other results.

## How to narrow the search

The car has hundreds of messages. Four filters cut that down to a short list before you record anything:

1. **Does the cluster read it?** Each signal in the Hyundai DBC lists the ECUs that receive it, and `CLU` is
   the cluster. 149 messages carry cluster inputs: 77 on C-CAN and 72 on M-CAN.
2. **Who sends it?** The comma harness sits between the car and the front camera (MFC). The panda receives
   the car side as **bus 0** and the camera side as **bus 2**, and forwards traffic between them. Whatever
   it forwards or openpilot sends is not received back. So an address seen only on bus 2 comes from the
   camera, and one seen only on bus 0 comes from the car.
3. **What does the panda safety code allow?** openpilot can only send what
   `opendbc_repo/opendbc/safety/modes/hyundai.h` allows. That list can be extended (see below).
4. **What changes when you act?** Record with labels, then compare a quiet stretch against the moment of
   each action. Counter and checksum bits are filtered out, so the bits left over are tied to the action.

The first three filters give four tiers (`analyze candidates` lists them):

| Tier | What | Count | How to change it |
|---|---|---|---|
| 1 | openpilot already transmits it | 8 messages, 53 dash signals | edit `hyundaican.py` / carcontroller |
| 2 | camera sends it, the panda forwards it untouched | 2 known (`LKAS12`, `HDA11_MFC`), more expected | add it to the safety TX list, then send our own |
| 3 | an ECU on the car side of C-CAN sends it | 67 | read-only: a second sender would collide |
| 4 | M-CAN, between the head unit and the cluster | 72 | needs extra hardware (see CarPlay) |

The most promising dash signals:

- **Tier 1, already reachable:**
  - `LKAS11`: lane lines, the car-ahead icon and its color, the "keep hands on wheel" message.
  - `LFAHDA_MFC`: the HDA and LFA icons. `HDA_VSetReq` is a speed the cluster displays.
  - `SCC11`: set speed, the gap bars, the lead-car icon. `Navi_SCC_Curve_*` and `Navi_SCC_Camera_*` drive
    the curve and speed-camera icons that navigation-based cruise normally shows.
  - `SCCInfoDisplay` and `DriverAlertDisplay`: the canned cruise messages.
- **Tier 2:**
  - `LKAS12` drives the speed-limit sign the camera reads off the road (`CF_Lkas_TsrSpeed_Display_Clu`)
    and the driver attention status. Taking it over would let openpilot show its own speed limit, from
    maps, in the cluster's sign.
  - `HDA11_MFC` is almost entirely undocumented.

Stay away from the AEB/FCA and `TakeOverReq` fields for anything decorative. The cluster treats them as
collision warnings, and borrowing them could hide a real one.

## Tools

On the device, over SSH (openpilot running, ignition on):

```bash
cd /data/openpilot
# record everything; files land in /data/dash_re/
python -m openpilot.sunnypilot.tools.dash_re.capture record --name climate
# live view of the camera bus; add --addr 0x53e to follow one message
python -m openpilot.sunnypilot.tools.dash_re.capture watch --bus 2
```

`record` stops after 30 minutes or 200 MB by default (`--max-minutes`, `--max-mb`).

**Labels:** type the label first, then press Enter at the moment you act. The comparison looks at the
2 seconds before and after Enter, skipping the first 0.3 s.

Copy captures to a laptop with the repo checked out:

```bash
scp -P 22 comma@<device-ip>:/data/dash_re/*.dashcap.gz .
```

On the laptop:

```bash
A="python -m openpilot.sunnypilot.tools.dash_re.analyze"
$A candidates                          # the tier list above, with every dash signal
$A inventory FILE --dash-only          # origin, rate, counters, checksums, bits the DBC doesn't explain
$A diff FILE --label climate           # bits that changed at each label
$A isotp FILE 0x4f2 --bus 1            # reassemble multi-frame payloads (e.g. CarPlay -> cluster)
```

`FILE` can be a capture or an openpilot `rlog`/`qlog`. Drive logs record all CAN, and the bookmark button
on the device becomes a label.

## Recording routine

Use a separate capture for each session, so the labels stay readable.

1. **Baseline** (`--name baseline`), parked, about 2 minutes, no labels. Run `inventory` on it to find
   counters, checksums and the camera's full set of addresses. Any camera-only address missing from the
   DBC is printed at the end and becomes a new tier-2 candidate.
2. **Dash states** (`--name dash`). Label each change:
   - LKA button
   - Driving Assist button
   - each steering-wheel cluster view
   - driver attention warning setting on or off in the head unit settings
   - TSR on or off

   Then drive past posted speed-limit signs and label each sign. Run `diff` to see which camera bits move.
3. **Climate** (`--name climate`). Label each change:
   - temperature up and down
   - fan speed
   - A/C on and off
   - recirculation
   - defrost
   - auto

   This shows whether the climate controls (`DATC11`–`DATC14`, `FATC11`) reach the C-CAN the harness can
   see. If nothing moves, climate is on a bus the harness can't reach.
4. **CarPlay** (`--name carplay`). Start CarPlay navigation and media, label each turn and each track
   change, and check whether anything shows up (see below).

## Your two goals

**Climate change → dash notification.** Possible if the climate messages are on C-CAN (session 3 will
tell us), with one catch. openpilot can read the climate state on bus 0 and react, but the cluster only
shows notifications it already knows: its built-in icons, popups and chimes. It has no free-text field
on C-CAN. A climate change could trigger an existing icon or popup, but it can't show "72°F". Free text
only travels on M-CAN.

**CarPlay data on the dash.** The head unit sends CarPlay, Android Auto, media and navigation data to
the cluster over **M-CAN**, using:

- `TP_HU_CARPLAY_CLU` (0x4F2), `TP_HU_ANDAUTO_CLU` (0x4F4), `TP_HU_MP_CLU` (media text) and
  `TP_HU_FM_CLU`: ISO-TP channels, so their payloads span several frames and need reassembling.
- `HU_CLU_PE_11`–`HU_CLU_PE_13`: turn-by-turn distances.

The harness isn't on M-CAN, so this needs hardware:

- **Reading:** wire a CAN tap on M-CAN (at the cluster or head unit connector) to a spare panda CAN port.
  `inventory` and `isotp` already handle it.
- **Writing** (our own text or turn arrows on the dash): put an inline gateway on M-CAN that blocks the
  head unit's frames and substitutes ours, the same way the harness relay handles the camera. The safety
  code would also need an M-CAN TX list.

Whether this cluster does anything with CarPlay data is still open. The DBC is generic across Hyundai
models, so the first M-CAN capture settles it.

## Changing the safety code

Tier 2 messages need entries in the panda safety TX list, `HYUNDAI_TX_MSGS` / `HYUNDAI_LONG_TX_MSGS` in
`opendbc_repo/opendbc/safety/modes/hyundai.h`. For example, to take over `LKAS12`:

```c
{0x53E, 0, 6, .check_relay = true},   /* LKAS12 Bus 0 */
```

`check_relay = true` stops the panda forwarding the camera's copy, so the cluster only sees ours.
sunnypilot still reads the camera's original on bus 2 for its speed-limit input.

Things to check before shipping a change like this:

- Check `LKAS12`'s real length in a capture first; the DBC says 6 bytes.
- Run the opendbc safety tests (`opendbc_repo/opendbc/safety/tests/test_hyundai.py`) after the change.
- The device builds the panda firmware from this repo and reflashes it on boot when it changes, so a
  safety change ships like any other code.
