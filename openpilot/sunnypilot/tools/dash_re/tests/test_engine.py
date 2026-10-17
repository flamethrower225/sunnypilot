import random

from openpilot.sunnypilot.tools.dash_re.engine import DashReEngine, events_html, findings_html, results_html

NS = 1_000_000_000
CGW1, CLU11, LKAS12, DATC11, UNKNOWN_CAM, TP_HU_CARPLAY_CLU = 0x541, 0x4F1, 0x53E, 0x44, 0x5F0, 0x4F2
TURN_SIG_LH_BIT = 19  # CF_Gway_TurnSigLh, 19|2@1+ in CGW1


def drive(engine: DashReEngine, seconds: float, signal_on_at: float | None = None, hz: int = 50, t0: float = 0.) -> None:
  for i in range(int(seconds * hz)):
    t = int((t0 + i / hz) * NS)
    sig = signal_on_at is not None and t >= signal_on_at * NS
    engine.ingest(t, 0, CGW1, bytes([0, 0, (1 << 3) if sig else 0, 0, 0, 0, 0, 0]))
    engine.ingest(t, 0, CLU11, bytes([0, 0, 0, (i % 16) << 4]))
    engine.ingest(t, 0, DATC11, bytes(8))
    engine.ingest(t, 2, LKAS12, bytes([0, 0, 65, 0, 0, 0]))
    engine.ingest(t, 2, UNKNOWN_CAM, bytes([i % 7, 1, 2, 3]))
    engine.ingest(t, 1, 0x500, bytes(8))
    engine.ingest(t, 130, 0x340, bytes(8))  # openpilot's own TX echo: ignored
    if i % 25 == 0:  # an ISO-TP payload twice a second, like head unit -> cluster text
      engine.ingest(t, 0, TP_HU_CARPLAY_CLU, bytes([0x10, 14]) + b"Main S")
      engine.ingest(t + 1000, 0, TP_HU_CARPLAY_CLU, bytes([0x21]) + b"t\0\0\0\0\0\0")


def test_findings():
  e = DashReEngine()
  drive(e, 40)
  f = e.findings()
  assert f["listening_s"] == 40 and f["orphans_checked"]
  assert f["buses"]["0"]["ids"] == 4 and f["buses"]["2"]["ids"] == 2 and "130" not in f["buses"]

  camera = {r["addr"]: r for r in f["camera_only"]}
  assert set(camera) == {LKAS12, UNKNOWN_CAM}
  assert camera[LKAS12]["dash"] and camera[LKAS12]["in_dbc"]
  assert not camera[UNKNOWN_CAM]["in_dbc"] and abs(camera[UNKNOWN_CAM]["hz"] - 50) < 1

  assert [r["addr"] for r in f["bus1"]] == [0x500]
  assert [(r["addr"], r["bus"]) for r in f["mcan_seen"]] == [(TP_HU_CARPLAY_CLU, 0)]
  assert [(r["addr"], r["pairs"]) for r in f["isotp_like"]] == [(TP_HU_CARPLAY_CLU, 80)]

  orphans = {r["name"] for r in f["orphans"]}
  assert "HUD11" in orphans and "CGW1" not in orphans and "DATC11" not in orphans
  assert "LKAS11" not in orphans  # openpilot sends it: tier 1, not an orphan

  assert f["checks"]["LKAS12 (0x53E) camera speed sign"] == [2]
  assert f["checks"]["climate messages"]["DATC11"] == [0] and f["checks"]["climate messages"]["DATC12"] == []
  assert f["checks"]["Navi_HU (0x544) nav speed limit from head unit"] == []

  html = findings_html(f)
  assert "NOT IN DBC" in html and "TP_HU_CARPLAY_CLU" in html and "HUD11" in html


def test_orphans_wait_for_enough_listening():
  e = DashReEngine()
  drive(e, 5)
  f = e.findings()
  assert not f["orphans_checked"] and f["orphans"] == []
  assert "needs 30 s" in findings_html(f)


def test_mark_finds_turn_signal_and_ignores_counters():
  e = DashReEngine()
  drive(e, 3)
  e.mark("Left turn signal ON (reference)", int(3.0 * NS))
  drive(e, 2, signal_on_at=3.6, t0=3.0)
  assert e.process_marks() == []  # after-window not complete yet
  drive(e, 3, signal_on_at=3.6, t0=5.0)
  (r,) = e.process_marks()
  assert r["label"] == "Left turn signal ON (reference)" and not r.get("error")
  (c,) = r["changes"]
  assert (c["bus"], c["addr"], c["bit"], c["before"], c["after"]) == (0, CGW1, TURN_SIG_LH_BIT, "0", "1")
  assert c["signal"] == "CF_Gway_TurnSigLh" and c["dash"]
  assert e.results == [r]
  assert "CF_Gway_TurnSigLh" in results_html(e.results) and "[CLU]" in results_html(e.results)


def test_mark_without_data_reports_error():
  e = DashReEngine()
  e.mark("LKA button", int(1 * NS))
  drive(e, 6, t0=1.0)  # data starts at the mark: no before-window
  (r,) = e.process_marks()
  assert r["error"] and r["changes"] == []
  assert "not enough CAN data" in results_html([r])


def test_html_empty_states_and_escaping():
  assert "No findings yet" in findings_html(None)
  assert "No marked actions yet" in results_html([])
  html = results_html([{"label": "<script>", "time": "t", "total": 0, "changes": []}])
  assert "<script>" not in html and "‹script›" in html and "&lt;" not in html


def detector_drive(e: DashReEngine, seconds: float = 40., hz: int = 50) -> None:
  rng = random.Random(0)
  next_irregular = 0.
  for i in range(int(seconds * hz)):
    s = i / hz
    t = int(s * NS)
    e.ingest(t, 0, CLU11, bytes([0, 0, 0, (i % 16) << 4]))
    e.ingest(t, 0, CGW1, bytes([0, 0, (1 << 3) if s >= 20 else 0, 0, 0, 0, 0, 0]))  # signal on at 20 s
    # LKAS12 sent 8 long (DBC says 6), bit 0 toggles every second: no DBC signal covers it
    e.ingest(t, 2, LKAS12, bytes([(i // 50) % 2, 0, 65, 0, 0, 0, 0, 0]))
    e.ingest(t, 0, 0x5F1, bytes(rng.getrandbits(8) for _ in range(8)))  # random binary: must not read as text
    if i % 10 == 0:
      e.ingest(t, 0, 0x5F2, b"Main St\0")  # plain single-frame text, resent
    if s >= 15:
      e.ingest(t, 0, 0x5F3, bytes(8))  # switched on at 15 s
    if s < 10:
      e.ingest(t, 0, 0x5F4, bytes(8))  # stops at 10 s
    if s >= next_irregular:
      e.ingest(t, 0, 0x5F5, bytes(8))  # sent on change: irregular gaps
      next_irregular = s + rng.choice([0.02, 0.02, 0.02, 1.5, 3.0])


def test_detectors():
  e = DashReEngine()
  detector_drive(e)
  f = e.findings()

  # rare-bit event log caught the turn signal, and nothing from counters, the toggling bit or random noise
  (ev,) = [x for x in f["events"] if x["addr"] == CGW1]
  assert (ev["bit"], ev["value"], ev["signal"], ev["dash"]) == (TURN_SIG_LH_BIT, 1, "CF_Gway_TurnSigLh", True)
  assert not [x for x in f["events"] if x["addr"] in (CLU11, 0x5F1)]
  assert "CF_Gway_TurnSigLh" in events_html(f)

  text = {r["addr"]: r["samples"] for r in f["text"]}
  assert text == {0x5F2: ["Main St"]}

  assert [r["addr"] for r in f["late"]] == [0x5F3] and f["late"][0]["first_s"] == 15
  assert [r["addr"] for r in f["stopped"]] == [0x5F4]
  assert 0x5F5 in [r["addr"] for r in f["irregular"]]
  assert not {CLU11, CGW1, 0x5F1} & {r["addr"] for r in f["irregular"]}

  assert {0x5F1, 0x5F2, 0x5F3, 0x5F4, 0x5F5} <= {r["addr"] for r in f["unknown"]}
  assert [(r["name"], r["dbc_len"], r["seen"]) for r in f["length_mismatch"]] == [("LKAS12", 6, [8])]
  assert [(r["name"], r["bits"]) for r in f["undocumented"]] == [("LKAS12", [0])]

  html = findings_html(f)
  for snippet in ("Main St", "first seen at 15 s", "silent for", "DBC 6 bytes, seen [8]", "LKAS12 on bus 2: bits 0"):
    assert snippet in html


def test_mark_reports_messages_that_appear():
  e = DashReEngine()
  for i in range(400):  # 8 s at 50 Hz, a new message starts right after the mark at 3 s
    t = int(i / 50 * NS)
    e.ingest(t, 0, CLU11, bytes([0, 0, 0, (i % 16) << 4]))
    if i >= 165:
      e.ingest(t, 0, 0x5F3, bytes(8))
    if i == 150:
      e.mark("Media: next track", t)
  (r,) = e.process_marks()
  (tr,) = r["traffic"]
  assert (tr["addr"], tr["kind"], tr["hz_before"]) == (0x5F3, "appeared", 0.0)
  assert "appeared" in results_html([r])


def test_control_messages_that_look_like_isotp_are_not_text():
  # like MDPS12 on the Elantra: 100 Hz data whose first byte is sometimes 0x1_ then 0x2_ by chance
  e = DashReEngine()
  rng = random.Random(3)
  for i in range(10_000):
    e.ingest(int(i / 100 * NS), 0, 0x251, bytes(rng.getrandbits(8) for _ in range(8)))
  f = e.findings()
  assert e.stats[(0, 0x251)].isotp_pairs > 0  # the coincidence happens...
  assert f["isotp_like"] == [] and f["text"] == []  # ...but is not reported


def test_real_transport_channel_text_is_found_and_described():
  e = DashReEngine()
  text = "Main St".encode("utf-16-le")
  for i in range(40):
    t = int(i * 0.5 * NS)
    e.ingest(t, 0, 0x49B, bytes([0x10, len(text)]) + text[:6])
    e.ingest(t + 1000, 0, 0x49B, bytes([0x21]) + text[6:13])
    e.ingest(t + 2000, 0, 0x49B, bytes([0x22]) + text[13:] + bytes(6))
  f = e.findings()
  assert [r["addr"] for r in f["isotp_like"]] == [0x49B]
  assert {r["addr"]: r["samples"] for r in f["text"]} == {0x49B: ["Main St"]}
  (m,) = f["mcan_seen"]
  assert m["name"] == "TP_HU_NAVI_CLU" and m["isotp"] and m["last"].startswith("22")
  assert f["checks"]["TP_HU_NAVI_CLU (0x49B) head unit to cluster navigation"] == [0]
  html = findings_html(f)
  assert "formatted as a transport channel" in html and "0x49B, head unit to cluster navigation: bus 0" in html


def test_fast_control_messages_with_printable_bytes_are_not_text():
  # like LKAS11 at 100 Hz on the Elantra, whose bytes read "DFAM<", "DVAU<", ... with a counter in them
  e = DashReEngine()
  for i in range(3000):
    e.ingest(int(i / 100 * NS), 2, 0x340, bytes([0x44, 0x46 + (i % 2) * 0x10, 0x41, 0x4D, 0x3C, 0x20, 0x41, i % 16]))
    if i % 50 == 0:  # slow, resent text on another ID still counts
      e.ingest(int(i / 100 * NS), 0, 0x5F2, b"Main St\0")
  f = e.findings()
  assert {r["addr"]: r["samples"] for r in f["text"]} == {0x5F2: ["Main St"]}
