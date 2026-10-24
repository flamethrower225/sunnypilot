"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Incremental CAN findings for the on-device dash research service: fed one frame at a time, it keeps running
statistics and a short rolling buffer, and answers three questions on demand:

  findings()   what this car puts on each bus, measured against what the cluster is known to read
  mark()       which bits and messages changed right after the driver did something (a labeled action)
  events       rare bit flips logged as they happen, so actions nobody marked still show up
"""
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field

from openpilot.sunnypilot.tools.dash_re import analysis
from openpilot.sunnypilot.tools.dash_re.analysis import Frame, Marker
from openpilot.sunnypilot.tools.dash_re.dbcinfo import HYUNDAI_DBC, HYUNDAI_MCAN_DBC, Message, parse_dbc, tier

DIAG_RANGE = range(0x700, 0x800)  # UDS request/response addresses; multi-frame by nature, not display data
NAVI_HU = 0x544
TP_HU_NAVI_CLU = 0x49B  # M-CAN: head unit -> cluster navigation transport
CLIMATE = {0x40: "DATC14", 0x42: "DATC12", 0x43: "DATC13", 0x44: "DATC11", 0x383: "FATC11"}
LKAS12 = 0x53E

ORPHAN_MIN_S = 30.  # how long to listen before calling a cluster input absent
MARK_BEFORE_S = 2.
MARK_AFTER_S = 3.5
MARK_SETTLE_S = 0.5  # the driver taps the screen, then acts
MAX_RESULTS = 20
TOP_CHANGES = 12

# rare-bit event log: a bit that has flipped at most this often is "rare"; noisier bits are ignored
RARE_TOGGLES = 6
EVENT_WARMUP_S = 3.  # ignore a message's first seconds, when every bit looks new
MAX_EVENTS = 80

# text scan: printable runs inside a single payload (ASCII or UTF-16LE), and reassembled ISO-TP payloads
TEXT_ASCII = re.compile(rb"[\x20-\x7e]{6,}")
TEXT_UTF16 = re.compile(rb"(?:[\x20-\x7e]\x00){4,}")
TEXT_MIN_REPEATS = 3
TEXT_MAX_HZ = 20.  # text gets sent on change or slowly; a printable run in a faster control message is a coincidence
ISOTP_SHAPE_MIN = 0.9  # a transport channel formats every frame; ordinary data only looks like it by chance (~20%)
ISOTP_MIN_FRAMES = 20  # text gets resent; random bytes that happen to look printable rarely repeat
MAX_TEXT_SAMPLES = 24

LATE_S = 10.  # an ID first seen this long after listening started appeared because of something
STOPPED_S = 5.  # an ID silent this long while others keep going has stopped
IRREGULAR_CV = 1.0  # inter-arrival coefficient of variation above which a message is event-driven
RATE_CHANGE = 2.0  # mark: a message whose rate changes by this factor around the action

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


def _text_runs(dat: bytes) -> list[str]:
  """Printable runs that look like words: at least 4 letters and 3 different characters."""
  out = [m.decode("ascii") for m in TEXT_ASCII.findall(dat)] + [m.decode("utf-16-le") for m in TEXT_UTF16.findall(dat)]
  return [t.strip() for t in out if sum(c.isalpha() for c in t) >= 4 and len(set(t)) >= 3]


@dataclass
class KeyStats:
  count: int = 0
  t_first: int = 0
  t_last: int = 0
  last_v: int = 0
  changed_mask: int = 0
  noisy_mask: int = 0  # bits that flipped more than RARE_TOGGLES times
  toggles: list[int] = field(default_factory=lambda: [0] * 64)
  lengths: set[int] = field(default_factory=set)
  isotp_pairs: int = 0  # first frame followed by consecutive frame #1
  isotp_shaped: int = 0  # frames whose first byte is a valid ISO-TP header
  pending_first: bool = False
  isotp_buf: bytearray = field(default_factory=bytearray)
  isotp_expect: int = 0
  isotp_seq: int = 0
  texts: Counter = field(default_factory=Counter)
  last_len: int = 0
  dt_sum: float = 0.  # inter-arrival statistics, seconds
  dt_sq_sum: float = 0.


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
    self.events: deque[dict] = deque(maxlen=MAX_EVENTS)

  def _name(self, addr: int) -> str:
    msg = self.dbc.get(addr) or self.mcan.get(addr)
    return msg.name if msg else "?"

  def _signal(self, addr: int, bit: int):
    msg = self.dbc.get(addr)
    return next((s for s in msg.signals if bit in s.bits), None) if msg else None

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
      dt = (t - st.t_last) / 1e9
      st.dt_sum += dt
      st.dt_sq_sum += dt * dt
      diff = st.last_v ^ v
      if diff:
        st.changed_mask |= diff
        self._toggles(st, key, t, diff, v)
    st.count += 1
    st.t_last = t
    st.last_v = v
    st.last_len = len(dat)
    st.lengths.add(len(dat))
    if dat and addr not in DIAG_RANGE:
      self._isotp(st, dat)
      for text in _text_runs(dat):
        if text in st.texts or len(st.texts) < MAX_TEXT_SAMPLES:
          st.texts[text] += 1

    self.ring.append(Frame(t, bus, addr, dat))
    while self.ring and self.ring[0].t < t - self.ring_ns:
      self.ring.popleft()

  def _toggles(self, st: KeyStats, key: tuple[int, int], t: int, diff: int, v: int) -> None:
    rare = diff & ~st.noisy_mask
    warm = t - st.t_first >= EVENT_WARMUP_S * 1e9
    while diff:
      low = diff & -diff
      b = low.bit_length() - 1
      diff ^= low
      st.toggles[b] += 1
      if st.toggles[b] > RARE_TOGGLES:
        st.noisy_mask |= low
      elif warm and rare & low:
        sig = self._signal(key[1], b)
        self.events.appendleft({"time": _clock(), "t": t, "bus": key[0], "addr": key[1], "name": self._name(key[1]), "bit": b,
                                "value": (v >> b) & 1, "signal": sig.name if sig else "", "dash": bool(sig and "CLU" in sig.receivers)})

  @staticmethod
  def isotp_shaped(st: KeyStats) -> bool:
    return st.count >= ISOTP_MIN_FRAMES and st.isotp_shaped >= ISOTP_SHAPE_MIN * st.count

  @staticmethod
  def _isotp(st: KeyStats, dat: bytes) -> None:
    pci, low = dat[0] >> 4, dat[0] & 0xF
    if (pci == 0 and 0 < low < len(dat)) or (pci == 1 and len(dat) == 8 and ((low << 8) | (dat[1] if len(dat) > 1 else 0)) > 7) or \
       pci == 2 or (pci == 3 and low <= 2):
      st.isotp_shaped += 1
    if pci == 0 and 0 < (dat[0] & 0xF) < len(dat):  # single frame
      payload = bytes(dat[1:1 + (dat[0] & 0xF)])
      st.pending_first = False
    elif pci == 1 and len(dat) == 8:  # first frame
      st.pending_first = True
      st.isotp_expect = ((dat[0] & 0xF) << 8) | dat[1]
      st.isotp_buf = bytearray(dat[2:])
      st.isotp_seq = 1
      return
    elif pci == 2 and st.isotp_expect and (dat[0] & 0xF) == st.isotp_seq % 16:  # consecutive frame
      if st.isotp_seq == 1 and st.pending_first:
        st.isotp_pairs += 1
      st.pending_first = False
      st.isotp_buf += dat[1:]
      st.isotp_seq += 1
      if len(st.isotp_buf) < st.isotp_expect:
        return
      payload = bytes(st.isotp_buf[:st.isotp_expect])
      st.isotp_expect = 0
    else:
      st.pending_first = False
      st.isotp_expect = 0
      return
    # only trust payloads on IDs that are formatted as a transport channel throughout, not data that looks like it by chance
    if st.isotp_pairs and DashReEngine.isotp_shaped(st):
      for text in _text_runs(payload):
        st.texts[text] += TEXT_MIN_REPEATS  # reassembled text counts as confirmed

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
    before = (m.t - int(MARK_BEFORE_S * 1e9), m.t)
    settle = m.t + int(MARK_SETTLE_S * 1e9)
    after = (settle, settle + int(MARK_AFTER_S * 1e9))
    changes = analysis.diff_windows(series, before, after, ignore)
    out = []
    for c in changes[:TOP_CHANGES]:
      sig = self._signal(c.addr, c.bit)
      out.append({"bus": c.bus, "addr": c.addr, "name": self._name(c.addr), "bit": c.bit, "before": c.before,
                  "after": c.after, "ones_after": c.ones_after, "score": c.score, "signal": sig.name if sig else "",
                  "dash": bool(sig and "CLU" in sig.receivers)})

    # messages that start, stop or change rate around the action (event-driven messages often carry no steady bits)
    traffic = []
    for (bus, addr), s in sorted(series.items()):
      b0, b1, a0, a1 = s.t.searchsorted([*before, *after])
      n_b, n_a = int(b1 - b0), int(a1 - a0)
      hz_b, hz_a = n_b / MARK_BEFORE_S, n_a / MARK_AFTER_S
      if n_b == 0 and n_a > 0:
        kind = "appeared"
      elif n_a == 0 and n_b > 1:
        kind = "stopped"
      elif min(hz_a, hz_b) > 0 and max(hz_a, hz_b) / min(hz_a, hz_b) >= RATE_CHANGE and max(n_a, n_b) >= 4:
        kind = "rate change"
      else:
        continue
      traffic.append({"bus": bus, "addr": addr, "name": self._name(addr), "kind": kind, "hz_before": round(hz_b, 1), "hz_after": round(hz_a, 1)})
    return {"label": m.label, "time": _clock(), "total": len(changes), "changes": out, "traffic": traffic[:TOP_CHANGES]}

  # *** findings ***

  def _buses(self, addr: int) -> set[int]:
    return {b for (b, a) in self.stats if a == addr}

  def _hz(self, st: KeyStats) -> float:
    dt = (st.t_last - st.t_first) / 1e9
    return round((st.count - 1) / dt, 1) if dt > 0 else 0.0

  @staticmethod
  def _cv(st: KeyStats) -> float:
    n = st.count - 1
    if n < 10:
      return 0.
    mean = st.dt_sum / n
    var = max(st.dt_sq_sum / n - mean * mean, 0.)
    return var ** 0.5 / mean if mean > 0 else 0.

  def _undocumented(self) -> list[dict]:
    """Bits that change on messages the cluster reads (or the camera sends) but that no DBC signal covers."""
    series = analysis.build_series(self.ring)
    noise = {k: st.noise_bits for k, st in analysis.inventory(series).items()}
    out = []
    for (bus, addr), st in sorted(self.stats.items()):
      msg = self.dbc.get(addr)
      if msg is None or not (msg.cluster_signals or msg.camera_originated):
        continue
      bits = [b for b in range(64) if (st.changed_mask >> b) & 1 and b not in msg.covered_bits and b not in noise.get((bus, addr), set())]
      if bits:
        out.append({"addr": addr, "bus": bus, "name": msg.name, "bits": bits[:16]})
    return out

  def findings(self) -> dict:
    duration = (self.t_now - self.t_start) / 1e9 if self.t_start is not None else 0.0
    t_start = self.t_start or 0
    per_bus: dict[int, dict] = {}
    for (bus, _), st in self.stats.items():
      totals = per_bus.setdefault(bus, {"ids": 0, "frames": 0})
      totals["ids"] += 1
      totals["frames"] += st.count

    def entry(bus: int, addr: int) -> dict:
      st = self.stats[(bus, addr)]
      msg = self.dbc.get(addr)
      return {"addr": addr, "bus": bus, "name": self._name(addr), "hz": self._hz(st), "in_dbc": msg is not None,
              "dash": bool(msg and (msg.cluster_signals or msg.camera_originated)), "changing_bits": bin(st.changed_mask).count("1")}

    keys = sorted(self.stats, key=lambda k: (k[1], k[0]))
    camera_only = [entry(2, a) for (b, a) in keys if b == 2 and self._buses(a) == {2}]
    bus1 = [entry(1, a) for (b, a) in keys if b == 1]
    unknown = [entry(b, a) for (b, a) in keys if a not in self.dbc and a not in self.mcan and a not in DIAG_RANGE]

    orphans = []
    if duration >= ORPHAN_MIN_S:
      for addr, msg in sorted(self.dbc.items()):
        if msg.cluster_signals and not (self._buses(addr) & {0, 2}) and tier(msg) == 3:
          orphans.append({"addr": addr, "name": msg.name, "sender": msg.sender, "signals": [s.name for s in msg.cluster_signals][:3]})

    def last_bytes(st: KeyStats) -> str:
      return st.last_v.to_bytes(8, "little")[:st.last_len].hex(" ")

    mcan_seen = [{"addr": a, "bus": b, "name": self.mcan[a].name, "hz": self._hz(st), "last": last_bytes(st),
                  "changing_bits": bin(st.changed_mask).count("1"), "isotp": self.isotp_shaped(st)}
                 for (b, a) in keys if a in self.mcan and a not in self.dbc and (st := self.stats[(b, a)])]
    isotp_like = [{"addr": a, "bus": b, "pairs": st.isotp_pairs, "name": self._name(a)}
                  for (b, a) in keys if (st := self.stats[(b, a)]).isotp_pairs >= 2 and self.isotp_shaped(st)]
    text = []
    for (b, a) in keys:
      st = self.stats[(b, a)]
      if not self.isotp_shaped(st) and self._hz(st) > TEXT_MAX_HZ:
        continue  # e.g. LKAS11 at 100 Hz, whose bytes sometimes read "DFAM<"
      samples = [t for t, n in st.texts.most_common() if n >= TEXT_MIN_REPEATS][:4]
      if samples:
        text.append({"addr": a, "bus": b, "name": self._name(a), "samples": samples})

    late = [dict(entry(b, a), first_s=round((st.t_first - t_start) / 1e9)) for (b, a) in keys
            if (st := self.stats[(b, a)]).t_first - t_start >= LATE_S * 1e9]
    stopped = [dict(entry(b, a), silent_s=round((self.t_now - st.t_last) / 1e9)) for (b, a) in keys
               if self.t_now - (st := self.stats[(b, a)]).t_last >= STOPPED_S * 1e9]
    irregular = [dict(entry(b, a), cv=round(self._cv(st), 1)) for (b, a) in keys
                 if self._cv(st := self.stats[(b, a)]) >= IRREGULAR_CV]
    length_mismatch = [{"addr": a, "bus": b, "name": self.dbc[a].name, "dbc_len": self.dbc[a].length, "seen": sorted(st.lengths)}
                       for (b, a) in keys if a in self.dbc and (st := self.stats[(b, a)]).lengths != {self.dbc[a].length}]

    checks = {
      "Navi_HU (0x544) nav speed limit from head unit": sorted(self._buses(NAVI_HU)),
      "LKAS12 (0x53E) camera speed sign": sorted(self._buses(LKAS12)),
      "TP_HU_NAVI_CLU (0x49B) head unit to cluster navigation": sorted(self._buses(TP_HU_NAVI_CLU)),
      "climate messages": {name: sorted(self._buses(a)) for a, name in CLIMATE.items()},
    }
    return {"updated": _clock(), "listening_s": round(duration), "buses": {str(k): v for k, v in sorted(per_bus.items())},
            "camera_only": camera_only, "bus1": bus1, "orphans": orphans, "orphans_checked": duration >= ORPHAN_MIN_S,
            "mcan_seen": mcan_seen, "isotp_like": isotp_like, "text": text, "unknown": unknown, "late": late, "stopped": stopped,
            "irregular": irregular, "length_mismatch": length_mismatch, "undocumented": self._undocumented(), "checks": checks,
            "events": list(self.events)}


def escape(text: str) -> str:
  """The settings viewer parses < as a tag and shows entities literally, so swap in look-alike characters."""
  return str(text).replace("&", "＆").replace("<", "‹").replace(">", "›")


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
                     "Findings update every 10 seconds while the car is on, and stay here after it is off.</p>"])
  buses = ", ".join(f"bus {b}: {v['ids']} IDs" for b, v in f["buses"].items()) or "no CAN received"
  checks = f["checks"]

  def where(buses):
    return ", ".join(f"bus {b}" for b in buses) if buses else "not seen"

  if f["orphans_checked"]:
    orphans = _ul(_ids(f["orphans"], lambda r: f" ({escape(', '.join(r['signals']))})"))
  else:
    orphans = "<p>needs 30 s of listening</p>"
  check_lines = [f"Navi_HU, nav speed limit from the head unit: {where(checks['Navi_HU (0x544) nav speed limit from head unit'])}",
                 f"LKAS12, camera speed sign: {where(checks['LKAS12 (0x53E) camera speed sign'])}",
                 "TP_HU_NAVI_CLU 0x49B, head unit to cluster navigation: "
                 + where(checks.get("TP_HU_NAVI_CLU (0x49B) head unit to cluster navigation", []))]
  check_lines += [f"{escape(n)}, climate: {where(b)}" for n, b in checks["climate messages"].items()]
  parts = [
    "<h2>Dash research findings</h2>",
    f"<p>Updated {escape(f['updated'])}, listened {f['listening_s']} s. {escape(buses)}.</p>",
    "<h3>Free text check</h3>",
    "<p>Multi-frame (ISO-TP) traffic outside diagnostics, the shape text takes on CAN:</p>",
    _ul(_ids(f["isotp_like"], lambda r: f" on bus {r['bus']}, {r['pairs']} sequences"), "none: no sign of text on these buses"),
    "<p>Head unit to cluster (M-CAN) IDs seen on reachable buses:</p>",
    _ul(_ids(f["mcan_seen"], lambda r: f" on bus {r['bus']}, {r.get('hz', 0)} Hz, {r.get('changing_bits', 0)} changing bits"
             + (", formatted as a transport channel" if r.get("isotp") else "") + (f", last: {r['last']}" if r.get("last") else ""))),
    "<p>Readable text found in payloads (repeated, or reassembled from multi-frame):</p>",
    _ul([f"{r['addr']:#05x} {escape(r['name'])} on bus {r['bus']}: " + " | ".join(escape(t) for t in r["samples"])
         for r in f.get("text", [])], "none"),
    "<h3>Messages in no DBC</h3>",
    _ul(_ids(f.get("unknown", []), lambda r: f" on bus {r['bus']}, {r['hz']} Hz, {r['changing_bits']} changing bits")),
    "<h3>Undocumented bits on dash messages</h3>",
    "<p>Bits that change but no DBC signal explains, on messages the cluster reads or the camera sends:</p>",
    _ul([f"{r['addr']:#05x} {escape(r['name'])} on bus {r['bus']}: bits {', '.join(map(str, r['bits']))}"
         for r in f.get("undocumented", [])]),
    "<h3>Event-driven messages</h3>",
    "<p>Appeared after listening started (something switched them on):</p>",
    _ul(_ids(f.get("late", []), lambda r: f" on bus {r['bus']}, first seen at {r['first_s']} s")),
    "<p>Stopped while other traffic continued:</p>",
    _ul(_ids(f.get("stopped", []), lambda r: f" on bus {r['bus']}, silent for {r['silent_s']} s")),
    "<p>Irregular timing (sent on change rather than on a schedule):</p>",
    _ul(_ids(f.get("irregular", []), lambda r: f" on bus {r['bus']}, {r['hz']} Hz average")),
    "<h3>Camera-only messages (substitutable with a safety change)</h3>",
    _ul(_ids(f["camera_only"], lambda r: f", {r['hz']} Hz" + ("" if r["in_dbc"] else ", NOT IN DBC"))),
    "<h3>Bus 1</h3>",
    _ul(_ids(f["bus1"], lambda r: f", {r['hz']} Hz")),
    "<h3>Length differs from the DBC</h3>",
    _ul([f"{r['addr']:#05x} {escape(r['name'])} on bus {r['bus']}: DBC {r['dbc_len']} bytes, seen {r['seen']}"
         for r in f.get("length_mismatch", [])]),
    "<h3>Checks</h3>",
    _ul(check_lines),
    "<h3>Cluster inputs nobody sends on this car (orphans)</h3>",
    orphans,
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
    traffic = [f"bus {c['bus']} {c['addr']:#05x} {escape(c['name'])}: {c['kind']}, {c['hz_before']} -> {c['hz_after']} Hz"
               for c in r.get("traffic", [])]
    if traffic:
      parts.append("<p>Messages that started, stopped or changed rate:</p>" + _ul(traffic))
  return "".join(parts)


def events_html(f: dict | None) -> str:
  events = (f or {}).get("events") or []
  if not events:
    return " ".join(["<h2>No events yet</h2><p>While the car is on, any bit that rarely changes is logged here the moment it flips,",
                     "so actions you did not mark still show up. Noisy bits (counters, sensors) are skipped automatically.</p>"])
  lines = []
  for e in events:
    where = f"{escape(e['time'])} bus {e['bus']} {e['addr']:#05x} {escape(e['name'])} bit {e['bit']}"
    lines.append(f"{where} -> {e['value']}, {escape(e['signal']) or '(undocumented)'}" + (" [CLU]" if e["dash"] else ""))
  return "<h2>Event log</h2><p>Rare bit flips, newest first. [CLU] marks a signal the cluster reads.</p>" + _ul(lines)
