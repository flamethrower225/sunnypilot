from openpilot.sunnypilot.tools.dash_re.engine import DashReEngine, findings_html, results_html

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
  assert "<script>" not in html and "&lt;script&gt;" in html
