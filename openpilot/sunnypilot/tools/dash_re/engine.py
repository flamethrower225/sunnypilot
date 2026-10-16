"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Incremental CAN findings for the on-device dash research service: fed one frame at a time, it keeps running
statistics and a short rolling buffer, and answers two questions on demand:

  findings()   what this car puts on each bus, measured against what the cluster is known to read
  mark()       which bits changed right after the driver did something (a labeled action)
"""
import time
from collections import deque
from dataclasses import dataclass, field
from html import escape

from openpilot.sunnypilot.tools.dash_re import analysis
from openpilot.sunnypilot.tools.dash_re.analysis import Frame, Marker
from openpilot.sunnypilot.tools.dash_re.dbcinfo import HYUNDAI_DBC, HYUNDAI_MCAN_DBC, Message, parse_dbc, tier

DIAG_RANGE = range(0x700, 0x800)  # UDS request/response addresses; multi-frame by nature, not display data
NAVI_HU = 0x544
CLIMATE = {0x40: "DATC14", 0x42: "DATC12", 0x43: "DATC13", 0x44: "DATC11", 0x383: "FATC11"}
LKAS12 = 0x53E

ORPHAN_MIN_S = 30.  # how long to listen before calling a cluster input absent
MARK_BEFORE_S = 2.
MARK_AFTER_S = 3.5
MARK_SETTLE_S = 0.5  # the driver taps the screen, then acts
MAX_RESULTS = 20
TOP_CHANGES = 12

# labels offered in the UI; anything the driver does that the dash reacts to, plus known references
MARK_LABELS = [
  "Left turn signal ON (reference)",
  "Left turn signal OFF (reference)",
  "LKA button",
  "Driving Assist button",
  "Cluster view change (wheel button)",
  "High beams ON",
  "High beams OFF",
  "Climate temp UP",
  "Climate temp DOWN",
  "Climate fan UP",
  "Climate fan DOWN",
  "A/C toggle",
  "Recirculation toggle",
  "Defrost toggle",
  "Climate AUTO",
  "Media: next track",
  "Media: volume change",
  "CarPlay nav: start route",
  "Phone call start/end",
  "Speed limit sign passed",
  "Other",
]


@dataclass
class KeyStats:
  count: int = 0
  t_first: int = 0
  t_last: int = 0
  last_v: int = 0
  changed_mask: int = 0
  lengths: set[int] = field(default_factory=set)
  isotp_pairs: int = 0  # first frame followed by consecutive frame #1
  pending_first: bool = False


class DashReEngine:
  def __init__(self, ring_s: float = MARK_BEFORE_S + MARK_SETTLE_S + MARK_AFTER_S + 2.):
    self.dbc: dict[int, Message] = parse_dbc(HYUNDAI_DBC)
    self.mcan: dict[int, Message] = parse_dbc(HYUNDAI_MCAN_DBC)
    self.stats: dict[tuple[int, int], KeyStats] = {}
    self.ring: deque[Frame] = deque()
    self.ring_ns = int(ring_s * 1e9)
    self.t_start: int | None = None
    self.t_now = 0
    self.pending_marks: list[Marker] = []
    self.results: list[dict] = []

  # *** input ***

  def ingest(self, t: int, bus: int, addr: int, dat: bytes) -> None:
    if bus >= analysis.PANDA_TX_OFFSET:
      return
    if self.t_start is None:
      self.t_start = t
    self.t_now = max(self.t_now, t)
    key = (bus, addr)
    st = self.stats.get(key)
    if st is None:
      st = self.stats[key] = KeyStats(t_first=t)
    v = int.from_bytes(dat[:8].ljust(8, b"\0"), "little")
    if st.count:
      st.changed_mask |= st.last_v ^ v
    st.count += 1
    st.t_last = t
    st.last_v = v
    st.lengths.add(len(dat))
    if dat and addr not in DIAG_RANGE:
      pci = dat[0] >> 4
      if pci == 1 and len(dat) == 8:
        st.pending_first = True
      else:
        if pci == 2 and (dat[0] & 0xF) == 1 and st.pending_first:
          st.isotp_pairs += 1
        st.pending_first = False

    self.ring.append(Frame(t, bus, addr, dat))
    while self.ring and self.ring[0].t < t - self.ring_ns:
      self.ring.popleft()

  def mark(self, label: str, t: int) -> None:
    self.pending_marks.append(Marker(t, label))

  # *** marks ***

  def process_marks(self) -> list[dict]:
    """Resolve marks whose after-window has fully arrived. Returns the new results."""
    done = []
    for m in list(self.pending_marks):
      if self.t_now < m.t + int((MARK_SETTLE_S + MARK_AFTER_S) * 1e9):
        continue
      self.pending_marks.remove(m)
      done.append(self._resolve(m))
    self.results = (done[::-1] + self.results)[:MAX_RESULTS]
    return done

  def _resolve(self, m: Marker) -> dict:
    series = analysis.build_series(self.ring)
    if not series or self.ring[0].t > m.t - int(MARK_BEFORE_S * 1e9):
      return {"label": m.label, "time": _clock(), "error": "not enough CAN data around the mark (was the car on?)", "changes": []}
    stats = analysis.inventory(series)
    ignore = {k: st.noise_bits for k, st in stats.items()}
    settle = m.t + int(MARK_SETTLE_S * 1e9)
    changes = analysis.diff_windows(series, (m.t - int(MARK_BEFORE_S * 1e9), m.t), (settle, settle + int(MARK_AFTER_S * 1e9)), ignore)
    out = []
    for c in changes[:TOP_CHANGES]:
      msg = self.dbc.get(c.addr)
      sig = next((s for s in msg.signals if c.bit in s.bits), None) if msg else None
      out.append({"bus": c.bus, "addr": c.addr, "name": msg.name if msg else "?", "bit": c.bit, "before": c.before,
                  "after": c.after, "ones_after": c.ones_after, "score": c.score, "signal": sig.name if sig else "",
                  "dash": bool(sig and "CLU" in sig.receivers)})
    return {"label": m.label, "time": _clock(), "total": len(changes), "changes": out}

  # *** findings ***

  def _buses(self, addr: int) -> set[int]:
    return {b for (b, a) in self.stats if a == addr}

  def _hz(self, st: KeyStats) -> float:
    dt = (st.t_last - st.t_first) / 1e9
    return round((st.count - 1) / dt, 1) if dt > 0 else 0.0

  def findings(self) -> dict:
    duration = (self.t_now - self.t_start) / 1e9 if self.t_start is not None else 0.0
    per_bus: dict[int, dict] = {}
    for (bus, _), st in self.stats.items():
      b = per_bus.setdefault(bus, {"ids": 0, "frames": 0})
      b["ids"] += 1
      b["frames"] += st.count

    def entry(bus: int, addr: int) -> dict:
      st = self.stats[(bus, addr)]
      msg = self.dbc.get(addr)
      return {"addr": addr, "bus": bus, "name": msg.name if msg else "?", "hz": self._hz(st), "in_dbc": msg is not None,
              "dash": bool(msg and (msg.cluster_signals or msg.camera_originated)), "changing_bits": bin(st.changed_mask).count("1")}

    camera_only = [entry(2, a) for (b, a) in sorted(self.stats, key=lambda k: k[1]) if b == 2 and self._buses(a) == {2}]
    bus1 = [entry(1, a) for (b, a) in sorted(self.stats, key=lambda k: k[1]) if b == 1]

    orphans = []
    if duration >= ORPHAN_MIN_S:
      for addr, msg in sorted(self.dbc.items()):
        if msg.cluster_signals and not (self._buses(addr) & {0, 2}) and tier(msg) == 3:
          orphans.append({"addr": addr, "name": msg.name, "sender": msg.sender, "signals": [s.name for s in msg.cluster_signals][:6]})

    mcan_seen = [{"addr": a, "bus": b, "name": self.mcan[a].name} for (b, a) in sorted(self.stats) if a in self.mcan and a not in self.dbc]
    isotp_like = [{"addr": a, "bus": b, "pairs": st.isotp_pairs, "name": (self.dbc.get(a) or self.mcan.get(a) or Message(a, "?", 0, "")).name}
                  for (b, a), st in sorted(self.stats.items()) if st.isotp_pairs >= 2]

    checks = {
      "Navi_HU (0x544) nav speed limit from head unit": sorted(self._buses(NAVI_HU)),
      "LKAS12 (0x53E) camera speed sign": sorted(self._buses(LKAS12)),
      "climate messages": {name: sorted(self._buses(a)) for a, name in CLIMATE.items()},
    }
    return {"updated": _clock(), "listening_s": round(duration), "buses": {str(k): v for k, v in sorted(per_bus.items())},
            "camera_only": camera_only, "bus1": bus1, "orphans": orphans, "orphans_checked": duration >= ORPHAN_MIN_S,
            "mcan_seen": mcan_seen, "isotp_like": isotp_like, "checks": checks}


def _clock() -> str:
  return time.strftime("%Y-%m-%d %H:%M:%S")


# *** HTML for the settings viewer ***
# The settings HTML viewer supports headings, paragraphs, lists and line breaks only: no inline formatting or entities.

def _ul(lines: list[str], empty: str = "none") -> str:
  return "<ul>" + "".join(f"<li>{line}</li>" for line in lines) + "</ul>" if lines else f"<p>{empty}</p>"


def _ids(rows: list[dict], extra=lambda r: "") -> list[str]:
  return [f"{r['addr']:#05x} {escape(r['name'])}{extra(r)}" for r in rows]


def findings_html(f: dict | None) -> str:
  if not f:
    return " ".join(["<h2>No findings yet</h2><p>Turn on Dash Research, then turn the car on.",
                     "Findings update every few seconds while the car is on, and stay here after it is off.</p>"])
  buses = ", ".join(f"bus {b}: {v['ids']} IDs" for b, v in f["buses"].items()) or "no CAN received"
  checks = f["checks"]

  def where(buses):
    return ", ".join(f"bus {b}" for b in buses) if buses else "not seen"

  if f["orphans_checked"]:
    orphans = _ul(_ids(f["orphans"], lambda r: f" ({escape(', '.join(r['signals']))})"))
  else:
    orphans = "<p>needs 30 s of listening</p>"
  check_lines = [f"Navi_HU, nav speed limit from the head unit: {where(checks['Navi_HU (0x544) nav speed limit from head unit'])}",
                 f"LKAS12, camera speed sign: {where(checks['LKAS12 (0x53E) camera speed sign'])}"]
  check_lines += [f"{escape(n)}, climate: {where(b)}" for n, b in checks["climate messages"].items()]
  parts = [
    "<h2>Dash research findings</h2>",
    f"<p>Updated {escape(f['updated'])}, listened {f['listening_s']} s. {escape(buses)}.</p>",
    "<h3>Free text check</h3>",
    "<p>Multi-frame (ISO-TP) traffic outside diagnostics, the shape text takes on CAN:</p>",
    _ul(_ids(f["isotp_like"], lambda r: f" on bus {r['bus']}, {r['pairs']} sequences"), "none: no sign of text on these buses"),
    "<p>Head unit to cluster (M-CAN) IDs seen on reachable buses:</p>",
    _ul(_ids(f["mcan_seen"], lambda r: f" on bus {r['bus']}")),
    "<h3>Camera-only messages (substitutable with a safety change)</h3>",
    _ul(_ids(f["camera_only"], lambda r: f", {r['hz']} Hz" + ("" if r["in_dbc"] else ", NOT IN DBC"))),
    "<h3>Bus 1</h3>",
    _ul(_ids(f["bus1"], lambda r: f", {r['hz']} Hz")),
    "<h3>Cluster inputs nobody sends on this car (orphans)</h3>",
    orphans,
    "<h3>Checks</h3>",
    _ul(check_lines),
  ]
  return "".join(parts)


def results_html(results: list[dict] | None) -> str:
  if not results:
    return " ".join(["<h2>No marked actions yet</h2><p>With the car on and parked, tap MARK next to Mark Dash Action, pick what",
                     "you are about to do, tap Select, then do it within a second. The result appears here a few seconds later.</p>"])
  parts = ["<h2>Marked actions</h2><p>Newest first. [CLU] marks a signal the cluster reads.</p>"]
  for r in results:
    parts.append(f"<h3>{escape(r['label'])}</h3><p>{escape(r['time'])}</p>")
    if r.get("error"):
      parts.append(f"<p>{escape(r['error'])}</p>")
      continue
    lines = []
    for c in r["changes"]:
      after = c["after"] if c["after"] != "toggling" else f"1 for {round(100 * c.get('ones_after', 0))}% of the time"
      signal = escape(c["signal"]) or "(undocumented)"
      lines.append(f"bus {c['bus']} {c['addr']:#05x} {escape(c['name'])} bit {c['bit']}: {c['before']} -> {after}, {signal}"
                   + (" [CLU]" if c["dash"] else ""))
    more = r["total"] - len(r["changes"])
    if more > 0:
      lines.append(f"+{more} more")
    parts.append(_ul(lines, "no bits changed"))
  return "".join(parts)
