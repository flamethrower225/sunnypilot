import gzip

from openpilot.sunnypilot.tools.dash_re import analysis
from openpilot.sunnypilot.tools.dash_re.analysis import Frame, Marker
from openpilot.sunnypilot.tools.dash_re.dbcinfo import HYUNDAI_DBC, HYUNDAI_MCAN_DBC, all_cluster_candidates, parse_dbc, tier

NS = 1_000_000_000
LKAS12, LFAHDA_MFC, CLU11, TP_HU_CARPLAY_CLU = 0x53E, 0x485, 0x4F1, 0x4F2


def synthetic_drive(seconds: float = 10.0, hz: int = 50, action_t: float = 5.0) -> list[Frame]:
  """Camera sends LKAS12 with a rolling counter in byte 0 and a 'checksum' in byte 7; at action_t the
  speed-sign field (byte 2) changes from 0 to 65. The car side sends CLU11 with a 4-bit counter."""
  frames = []
  for i in range(int(seconds * hz)):
    t = int(i * NS / hz)
    sign = 65 if t >= action_t * NS else 0
    body = bytes([i % 256, 0, sign, 0, 0, 0, 0])
    frames.append(Frame(t, 2, LKAS12, body + bytes([(sum(body) * 37 + i * 101) % 256])))
    frames.append(Frame(t, 0, CLU11, bytes([0, 0, 0, (i % 16) << 4])))
    frames.append(Frame(t, 128, 0x340, bytes(8)))  # openpilot's own TX echo
  return frames


def test_dbc_receivers_and_tiers():
  dbc = parse_dbc(HYUNDAI_DBC)
  lkas12 = dbc[LKAS12]
  assert "CF_Lkas_TsrSpeed_Display_Clu" in {s.name for s in lkas12.cluster_signals}
  assert tier(dbc[0x340]) == 1  # LKAS11: openpilot sends it
  assert tier(dbc[0x420]) == 1 and tier(dbc[0x420], op_long=False) == 3  # SCC11 only with openpilot long
  assert tier(lkas12) == 2
  assert tier(dbc[0x44]) == 3  # DATC11 (climate)
  mcan = parse_dbc(HYUNDAI_MCAN_DBC)
  assert mcan[TP_HU_CARPLAY_CLU].name == "TP_HU_CARPLAY_CLU"
  assert any(t == 4 and m.addr == TP_HU_CARPLAY_CLU for t, m in all_cluster_candidates())


def test_motorola_bits():
  dbc = parse_dbc(HYUNDAI_MCAN_DBC)
  byte0 = next(s for s in dbc[TP_HU_CARPLAY_CLU].signals if s.name.startswith("Byte0"))
  assert sorted(byte0.bits) == list(range(8))  # 7|8@0 covers all of byte 0
  byte1 = next(s for s in dbc[TP_HU_CARPLAY_CLU].signals if s.name.startswith("Byte1"))
  assert sorted(byte1.bits) == list(range(8, 16))


def test_origins_ignore_tx_echo():
  origins = analysis.origins(synthetic_drive(1))
  assert origins[LKAS12] == {2} and analysis.origin_label(origins[LKAS12]) == "camera"
  assert origins[CLU11] == {0}
  assert 0x340 not in origins


def test_inventory_finds_counters_and_checksum():
  stats = analysis.inventory(synthetic_drive())
  lk = stats[(2, LKAS12)]
  assert abs(lk.hz - 50) < 0.5
  assert (0, 8) in lk.counters
  assert lk.checksum_bytes == [7]
  assert (28, 4) in stats[(0, CLU11)].counters
  # byte 2 bits that change for 65 = 0b1000001 are bits 16 and 22, and the DBC covers them
  assert {16, 22} <= set(lk.changing_bits)
  assert analysis.undocumented_changing_bits(lk, parse_dbc(HYUNDAI_DBC)[LKAS12]) == []


def test_marker_diff_points_at_the_changed_field():
  frames = synthetic_drive()
  ((marker, changes),) = analysis.marker_diffs(frames, [Marker(int(4.8 * NS), "sign appears")], window_s=1.0, settle_s=0.3)
  top = {(c.addr, c.bit) for c in changes if c.score == 1.0}
  assert top == {(LKAS12, 16), (LKAS12, 22)}
  assert all(c.before == "0" and c.after == "1" for c in changes if c.score == 1.0)
  # counters and checksum never show up
  assert not [c for c in changes if c.bit < 8 or c.bit >= 56]


def test_capture_round_trip(tmp_path):
  path = tmp_path / ("x" + analysis.CAPTURE_SUFFIX)
  frames = synthetic_drive(0.1) + [Frame(5, 0, 0x7FF, b"")]
  markers = [Marker(3, "climate 72 F")]
  with gzip.open(path, "wt") as f:
    for fr in frames:
      analysis.write_frame(f, fr)
    for m in markers:
      analysis.write_marker(f, m)
  assert analysis.read_capture(path) == (frames, markers)


def test_isotp_reassembly_and_text():
  text = "Main St".encode("utf-16-le")  # 14 bytes: first frame + 2 consecutive frames
  frames = [
    Frame(1, 1, TP_HU_CARPLAY_CLU, bytes([0x10, len(text)]) + text[:6]),
    Frame(2, 1, TP_HU_CARPLAY_CLU, bytes([0x21]) + text[6:13]),
    Frame(3, 1, TP_HU_CARPLAY_CLU, bytes([0x22]) + text[13:] + bytes(6)),
    Frame(4, 1, TP_HU_CARPLAY_CLU, bytes([0x03, 0x41, 0x42, 0x43, 0, 0, 0, 0])),  # single frame "ABC"
    Frame(5, 1, 0x4F3, bytes([0x02, 1, 2])),  # other address
  ]
  payloads = analysis.isotp_payloads(frames, TP_HU_CARPLAY_CLU)
  assert [p for _, p in payloads] == [text, b"ABC"]
  assert analysis.printable(text) == "Main St"
  assert analysis.printable(b"ABC") == "ABC"


def test_isotp_drops_payload_with_missing_frame():
  frames = [
    Frame(1, 1, TP_HU_CARPLAY_CLU, bytes([0x10, 20]) + bytes(6)),
    Frame(2, 1, TP_HU_CARPLAY_CLU, bytes([0x22]) + bytes(7)),  # sequence 1 was lost
  ]
  assert analysis.isotp_payloads(frames, TP_HU_CARPLAY_CLU) == []
