"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
from types import SimpleNamespace

import numpy as np
import pyray as rl

from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui import DeveloperUiState
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_scope import LANE_DISTANCES, MAX_LATERAL, MAX_RANGE, PANEL_H, PANEL_W, \
  ScopeFrame, ring_points, ring_ranges, scope_frame, scope_rect
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_tracks import BOARD_TTC_MAX, FEED_HOLD, MAX_POINTS, NO_TRACK, Lane, RadarFeed, \
  TrackClass, TrackSmoother, classify_motion, display_zone, in_threat_corridor, is_threat, lane_of, lead_track_ids, make_track, \
  match_leads, parse_tracks, rank_threats, select_threats, service_fresh, time_to_collision, ttc_zone
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.threat_board import HEADER_H, PANEL_W as BOARD_W, ROW_H, SLOT_H, board_rect, \
  format_ttc
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.track_brackets import BOX_HEIGHT, BOX_WIDTH, MAX_TRACKS, MIN_BOX_H, MIN_BOX_W, \
  pad_box, project_box, select_tracks, tagged_tracks

INF = math.inf
V_EGO = 26.8  # m/s, 60 mph
HUD = rl.Rectangle(30, 30, 2100, 1020)


def track(track_id, d_rel, y_rel=0.0, v_rel=0.0, v_ego=V_EGO):
  return make_track(track_id, d_rel, y_rel, v_rel, v_ego)


def points(*rows):
  return [SimpleNamespace(trackId=i, dRel=d, yRel=y, vRel=v) for i, d, y, v in rows]


def lead(present=True, radar=True, track_id=NO_TRACK, d_rel=40.0, y_rel=0.0, v_rel=0.0):
  return SimpleNamespace(present=present, radar=radar, radarTrackId=track_id, dRel=d_rel, yRel=y_rel, vRel=v_rel)


class FakeSM:
  """Just enough of a SubMaster for RadarFeed."""

  def __init__(self, started=1):
    self.valid = {"radarTracks": True, "radarState": True}
    self.alive = {"radarTracks": True, "radarState": True}
    self.recv_frame = {"radarTracks": started, "radarState": started, "carState": started}
    self.data = {
      "radarTracks": SimpleNamespace(points=[]),
      "radarState": SimpleNamespace(leadOne=lead(present=False), leadTwo=lead(present=False)),
      "carState": SimpleNamespace(vEgo=V_EGO),
    }

  def __getitem__(self, service):
    return self.data[service]


class TestClassification(OpenpilotTestCase):
  @parameterized.expand([
    ("same_speed", V_EGO, TrackClass.MOVING),
    ("slow_mover", 5.0, TrackClass.MOVING),
    ("just_moving", 1.01, TrackClass.MOVING),
    ("sign", 0.0, TrackClass.STATIONARY),
    ("crawling", 0.99, TrackClass.STATIONARY),
    ("slightly_backward", -0.99, TrackClass.STATIONARY),
    ("oncoming", -20.0, TrackClass.ONCOMING),
    ("just_oncoming", -1.01, TrackClass.ONCOMING),
  ])
  def test_motion(self, _, v_abs, expected):
    assert classify_motion(v_abs) == expected

  def test_absolute_speed_comes_from_ego_speed(self):
    sign = track(1, 50.0, 0.0, -V_EGO)
    assert sign.cls == TrackClass.STATIONARY and abs(sign.v_abs) < 1e-9
    assert track(2, 50.0, 0.0, -V_EGO - 20.0).cls == TrackClass.ONCOMING
    assert track(3, 50.0, 0.0, 2.0).cls == TrackClass.MOVING

  def test_classification_at_a_standstill(self):
    assert track(1, 20.0, 0.0, 0.0, v_ego=0.0).cls == TrackClass.STATIONARY
    assert track(2, 20.0, 0.0, 5.0, v_ego=0.0).cls == TrackClass.MOVING

  @parameterized.expand([
    ("center", 0.0, Lane.CENTER),
    ("left_edge_is_ours", 1.8, Lane.CENTER),
    ("right_edge_is_ours", -1.8, Lane.CENTER),
    ("left", 1.81, Lane.LEFT),
    ("right", -1.81, Lane.RIGHT),
    ("far_left", 7.0, Lane.LEFT),
  ])
  def test_lane(self, _, y_rel, expected):
    assert lane_of(y_rel) == expected

  def test_lane_tags(self):
    assert [str(lane) for lane in (Lane.LEFT, Lane.CENTER, Lane.RIGHT)] == ["L", "C", "R"]

  def test_y_rel_is_left_positive_and_projects_negated(self):
    t = track(1, 30.0, 3.0)
    assert t.lane == Lane.LEFT and t.y_right == -3.0


class TestTimeToCollision(OpenpilotTestCase):
  def test_closing(self):
    assert time_to_collision(30.0, -6.0) == 5.0

  @parameterized.expand([
    ("opening", 50.0, 3.0),
    ("matched", 50.0, 0.0),
    ("barely_closing", 50.0, -0.3),
    ("jitter", 50.0, -0.1),
  ])
  def test_not_closing(self, _, d_rel, v_rel):
    assert time_to_collision(d_rel, v_rel) == INF

  def test_track_exposes_closing(self):
    t = track(1, 30.0, 0.0, -6.0)
    assert t.closing and t.closing_speed == 6.0 and t.ttc == 5.0
    t = track(2, 30.0, 0.0, 2.0)
    assert not t.closing and t.closing_speed == 0.0 and t.ttc == INF

  @parameterized.expand([
    ("critical", 1.0, GapZone.CRITICAL),
    ("critical_edge", 2.49, GapZone.CRITICAL),
    ("close", 2.5, GapZone.CLOSE),
    ("close_edge", 3.99, GapZone.CLOSE),
    ("caution", 4.0, GapZone.CAUTION),
    ("caution_edge", 5.99, GapZone.CAUTION),
    ("good", 6.0, GapZone.GOOD),
    ("good_far", 19.0, GapZone.GOOD),
    ("open", INF, GapZone.OPEN),
  ])
  def test_zone(self, _, ttc, expected):
    assert ttc_zone(ttc) == expected


class TestParse(OpenpilotTestCase):
  def test_parses_and_classifies(self):
    tracks = parse_tracks(points((7, 40.0, 0.2, -1.0), (8, 66.0, 6.0, -V_EGO), (9, 80.0, -3.5, -V_EGO - 15.0)), V_EGO)
    assert [t.track_id for t in tracks] == [7, 8, 9]
    assert [t.cls for t in tracks] == [TrackClass.MOVING, TrackClass.STATIONARY, TrackClass.ONCOMING]
    assert [t.lane for t in tracks] == [Lane.CENTER, Lane.LEFT, Lane.RIGHT]
    assert abs(tracks[0].ttc - 40.0) < 1e-9

  def test_empty(self):
    assert parse_tracks([], V_EGO) == []

  @parameterized.expand([
    ("nan_distance", (1, math.nan, 0.0, 0.0)),
    ("nan_lateral", (1, 20.0, math.nan, 0.0)),
    ("inf_speed", (1, 20.0, 0.0, INF)),
    ("behind_the_bumper", (1, -2.0, 0.0, 0.0)),
    ("at_the_bumper", (1, 0.0, 0.0, 0.0)),
  ])
  def test_garbage_is_skipped(self, _, row):
    assert parse_tracks(points(row, (2, 30.0, 0.0, 0.0)), V_EGO) == [track(2, 30.0)]

  def test_point_count_is_capped(self):
    many = points(*[(i, 10.0 + i, 0.0, 0.0) for i in range(200)])
    assert len(parse_tracks(many, V_EGO)) == MAX_POINTS


class TestThreats(OpenpilotTestCase):
  def test_ranking_is_ttc_first_then_distance(self):
    tracks = [
      track(1, 20.0, 0.0, 1.0),        # opening, nearest
      track(2, 60.0, 0.0, -6.0),       # ttc 10
      track(3, 90.0, 3.6, 0.5),        # opening, farthest
      track(4, 30.0, 0.0, -10.0),      # ttc 3
      track(5, 35.0, -3.6, -2.0),      # ttc 17.5
      track(6, 10.0, 0.0, 0.0),        # matched speed, no ttc
    ]
    assert [t.track_id for t in rank_threats(tracks)] == [4, 2, 5, 6, 1, 3]

  def test_equal_ttc_goes_to_the_nearer_one(self):
    tracks = [track(1, 60.0, 0.0, -10.0), track(2, 30.0, 0.0, -5.0)]
    assert [t.track_id for t in rank_threats(tracks)] == [2, 1]

  def test_ranking_leaves_input_alone(self):
    tracks = [track(1, 60.0, 0.0, 0.0), track(2, 30.0, 0.0, -5.0)]
    rank_threats(tracks)
    assert [t.track_id for t in tracks] == [1, 2]

  @parameterized.expand([
    ("our_lane_moving", track(1, 30.0, 0.0, -5.0), True),
    ("neighbor_moving", track(1, 30.0, 3.6, -5.0), True),
    ("neighbor_edge", track(1, 30.0, 5.4, -5.0), True),
    ("two_lanes_over", track(1, 30.0, 6.0, -5.0), False),
    ("our_lane_oncoming", track(1, 60.0, 0.0, -V_EGO - 20.0), True),
    ("neighbor_oncoming", track(1, 60.0, 3.6, -V_EGO - 20.0), False),
    ("our_lane_stationary", track(1, 60.0, 0.0, -V_EGO), False),
    ("neighbor_stationary", track(1, 60.0, 3.6, -V_EGO), False),
  ])
  def test_corridor(self, _, t, expected):
    assert in_threat_corridor(t) == expected

  def test_is_threat_needs_a_short_ttc(self):
    assert is_threat(track(1, 30.0, 0.0, -6.0))          # 5 s
    assert not is_threat(track(1, 40.0, 0.0, -5.0))      # 8 s
    assert not is_threat(track(1, 30.0, 0.0, 1.0))
    assert is_threat(track(1, 40.0, 0.0, -5.0), ttc_max=10.0)

  def test_display_zone(self):
    assert display_zone(track(1, 20.0, 0.0, -10.0)) == GapZone.CRITICAL
    assert display_zone(track(1, 30.0, 0.0, -6.0)) == GapZone.CAUTION
    assert display_zone(track(1, 100.0, 0.0, -5.0)) is None
    assert display_zone(track(1, 20.0, 0.0, -V_EGO)) is None  # a sign is never a threat

  def test_select_threats_drops_noise(self):
    tracks = [
      track(1, 66.0, 6.0, -V_EGO),            # guardrail, ttc 2.5 s but stationary
      track(2, 18.0, 3.7, -4.5),              # closing from the neighbor lane
      track(3, 38.0, 0.0, -0.9),              # lead, ttc 42 s
      track(4, 60.0, 5.4, -V_EGO - 18.0),     # oncoming across the line
      track(5, 52.0, -3.6, 2.0),              # pulling away
    ]
    assert [t.track_id for t in select_threats(tracks)] == [2]

  def test_select_threats_limit_and_order(self):
    tracks = [track(i, 20.0 + 10 * i, 0.0, -2.0 - i) for i in range(8)]
    ranked = select_threats(tracks, limit=4)
    assert len(ranked) == 4
    assert [t.ttc for t in ranked] == sorted(t.ttc for t in ranked)

  def test_select_threats_ttc_cutoff(self):
    slow = track(1, BOARD_TTC_MAX * 1.2, 0.0, -1.0)   # ttc is 1.2x the cutoff
    assert select_threats([slow]) == []
    assert select_threats([track(2, BOARD_TTC_MAX - 1, 0.0, -1.0)]) != []

  def test_stopped_lead_gets_a_row(self):
    stopped = track(7, 40.0, 0.0, -V_EGO)
    assert select_threats([stopped]) == []
    assert select_threats([stopped], lead_ids=(7, NO_TRACK)) == [stopped]


class TestLeads(OpenpilotTestCase):
  def test_ids_only_when_radar_matched(self):
    state = SimpleNamespace(leadOne=lead(track_id=7), leadTwo=lead(track_id=11, radar=False))
    assert lead_track_ids(state) == (7, NO_TRACK)

  def test_absent_lead_has_no_id(self):
    state = SimpleNamespace(leadOne=lead(present=False, track_id=7), leadTwo=lead(track_id=-1))
    assert lead_track_ids(state) == (NO_TRACK, NO_TRACK)

  def test_match(self):
    tracks = [track(7, 40.0), track(11, 70.0, 3.5)]
    one, two = match_leads(tracks, (7, 11))
    assert one is tracks[0] and two is tracks[1]
    assert match_leads(tracks, (NO_TRACK, 99)) == (None, None)
    assert match_leads([], (7, 11)) == (None, None)

  def test_a_track_id_of_minus_one_never_matches(self):
    assert match_leads([track(NO_TRACK, 10.0)], (NO_TRACK, NO_TRACK)) == (None, None)


class TestSmoother(OpenpilotTestCase):
  DT = 0.05

  def test_first_frame_passes_through(self):
    out = TrackSmoother(self.DT).update([track(1, 40.0, 1.0, -2.0)], V_EGO)
    assert (out[0].d_rel, out[0].y_rel, out[0].v_rel) == (40.0, 1.0, -2.0)

  def test_jitter_is_smoothed(self):
    s = TrackSmoother(self.DT)
    for _ in range(40):
      s.update([track(1, 40.0, 0.0, -2.0)], V_EGO)
    out = s.update([track(1, 41.5, 0.5, -2.0)], V_EGO)[0]
    assert 40.0 < out.d_rel < 41.5 and 0.0 < out.y_rel < 0.5

  def test_converges(self):
    s = TrackSmoother(self.DT)
    s.update([track(1, 40.0)], V_EGO)
    for _ in range(100):
      out = s.update([track(1, 43.0)], V_EGO)[0]
    assert abs(out.d_rel - 43.0) < 0.05

  def test_jump_snaps(self):
    s = TrackSmoother(self.DT)
    for _ in range(10):
      s.update([track(1, 40.0)], V_EGO)
    assert s.update([track(1, 70.0)], V_EGO)[0].d_rel == 70.0

  def test_class_follows_the_smoothed_speed(self):
    s = TrackSmoother(self.DT)
    out = s.update([track(1, 40.0, 0.0, -V_EGO)], V_EGO)[0]
    assert out.cls == TrackClass.STATIONARY

  def test_forgets_vanished_tracks(self):
    s = TrackSmoother(self.DT)
    s.update([track(1, 40.0), track(2, 50.0)], V_EGO)
    assert [t.track_id for t in s.update([track(2, 50.0)], V_EGO)] == [2]
    # track 1 comes back as new, so it starts from its raw value
    assert s.update([track(1, 44.0), track(2, 50.0)], V_EGO)[0].d_rel == 44.0


class TestRadarFeed(OpenpilotTestCase):
  def _feed(self, sm, now=0.0, started=1):
    feed = RadarFeed(0.05)
    feed.update(sm, started, now)
    return feed

  def test_reads_tracks_and_ego_speed(self):
    sm = FakeSM()
    sm.data["radarTracks"].points = points((7, 40.0, 0.0, -1.0), (8, 66.0, 6.0, -V_EGO))
    feed = self._feed(sm)
    assert feed.available and feed.v_ego == V_EGO
    assert [t.cls for t in feed.tracks] == [TrackClass.MOVING, TrackClass.STATIONARY]

  def test_invalid_radar_hides_everything(self):
    sm = FakeSM()
    sm.data["radarTracks"].points = points((7, 40.0, 0.0, -1.0))
    sm.valid["radarTracks"] = False
    feed = self._feed(sm)
    assert not feed.available and feed.tracks == []

  def test_stale_radar_hides_everything(self):
    sm = FakeSM()
    sm.data["radarTracks"].points = points((7, 40.0, 0.0, -1.0))
    sm.alive["radarTracks"] = False
    assert not self._feed(sm).available

  def test_radar_from_before_the_drive_is_ignored(self):
    sm = FakeSM(started=3)
    sm.data["radarTracks"].points = points((7, 40.0, 0.0, -1.0))
    sm.recv_frame["radarTracks"] = 2
    assert not self._feed(sm, started=3).available
    assert not service_fresh(sm, "radarTracks", 3)
    assert service_fresh(sm, "radarTracks", 2)

  def test_radarless_car_stays_hidden(self):
    feed = self._feed(FakeSM())
    assert not feed.available and feed.tracks == []

  def test_an_empty_road_doesnt_blink_the_overlays_away(self):
    sm = FakeSM()
    feed = RadarFeed(0.05)
    sm.data["radarTracks"].points = points((7, 40.0, 0.0, -1.0))
    feed.update(sm, 1, 0.0)
    sm.data["radarTracks"].points = []
    feed.update(sm, 1, FEED_HOLD - 1.0)
    assert feed.available
    feed.update(sm, 1, FEED_HOLD + 0.5)
    assert not feed.available

  def test_lead_is_matched_to_its_track(self):
    sm = FakeSM()
    sm.data["radarTracks"].points = points((7, 40.0, 0.0, -1.0), (11, 70.0, 3.5, 1.5))
    sm.data["radarState"] = SimpleNamespace(leadOne=lead(track_id=7), leadTwo=lead(track_id=11, d_rel=70.0, y_rel=3.5))
    feed = self._feed(sm)
    assert feed.lead_ids == (7, 11)
    assert feed.lead_one.track_id == 7 and feed.lead_two.track_id == 11
    assert feed.vision_leads == (None, None)

  def test_vision_lead_has_no_track(self):
    sm = FakeSM()
    sm.data["radarTracks"].points = points((7, 40.0, 0.0, -1.0))
    sm.data["radarState"] = SimpleNamespace(leadOne=lead(radar=False, d_rel=55.0, y_rel=-0.5, v_rel=-2.0), leadTwo=lead(present=False))
    feed = self._feed(sm)
    assert feed.lead_ids == (NO_TRACK, NO_TRACK) and feed.lead_one is None
    assert feed.vision_leads[0].d_rel == 55.0 and feed.vision_leads[0].y_rel == -0.5 and feed.vision_leads[1] is None

  def test_stale_radar_state_means_no_leads(self):
    sm = FakeSM()
    sm.data["radarTracks"].points = points((7, 40.0, 0.0, -1.0))
    sm.data["radarState"] = SimpleNamespace(leadOne=lead(track_id=7), leadTwo=lead(present=False))
    sm.valid["radarState"] = False
    feed = self._feed(sm)
    assert feed.available and feed.lead_one is None and feed.lead_ids == (NO_TRACK, NO_TRACK)


class TestScopeGeometry(OpenpilotTestCase):
  def test_panel_position(self):
    panel = scope_rect(HUD, rocket_fuel=False)
    assert (panel.x, panel.y, panel.width, panel.height) == (HUD.x + 40, HUD.y + 290, PANEL_W, PANEL_H)
    assert scope_rect(HUD, rocket_fuel=True).x == HUD.x + 80

  def test_panel_stays_in_the_left_column(self):
    panel = scope_rect(HUD, rocket_fuel=True)
    assert panel.y >= HUD.y + 260                       # below the MAX box and speed limit sign
    assert panel.x + panel.width <= HUD.x + 520
    assert panel.y + panel.height <= HUD.y + HUD.height - 250   # above the driver monitoring icon

  def test_bottom_developer_ui_shortens_the_panel(self):
    full, short = scope_rect(HUD, False), scope_rect(HUD, False, 60)
    assert full.height - short.height == 60 and full.y == short.y
    frame = scope_frame(short)
    assert frame.height > 250 and frame.origin_y < short.y + short.height

  def test_frame_fits_the_panel(self):
    panel = scope_rect(HUD, rocket_fuel=False)
    f = scope_frame(panel)
    assert abs(f.cx - (panel.x + panel.width / 2)) < 1e-6
    assert panel.y < f.origin_y - f.height and f.origin_y < panel.y + panel.height
    left, _ = f.project(10.0, -MAX_LATERAL)
    right, _ = f.project(10.0, MAX_LATERAL)
    assert left >= panel.x and right <= panel.x + panel.width

  def test_projection_is_monotonic_and_compressed(self):
    f = scope_frame(scope_rect(HUD, rocket_fuel=False))
    ys = [f.project(d, 0.0)[1] for d in (0.0, 10.0, 25.0, 50.0, 100.0, MAX_RANGE)]
    assert ys == sorted(ys, reverse=True)
    assert f.project(0.0, 0.0)[1] == f.origin_y
    assert abs(f.project(MAX_RANGE, 0.0)[1] - (f.origin_y - f.height)) < 1e-6
    # the first 25 m take far more room than the last 25 m
    assert (f.origin_y - f.project(25.0, 0.0)[1]) > 4 * (f.project(95.0, 0.0)[1] - f.project(120.0, 0.0)[1])

  def test_right_is_screen_right(self):
    f = ScopeFrame(100.0, 400.0, 10.0, 300.0)
    assert f.project(20.0, 2.0)[0] == 120.0 and f.project(20.0, -2.0)[0] == 80.0

  def test_range_is_clamped(self):
    f = ScopeFrame(100.0, 400.0, 10.0, 300.0)
    assert f.project(500.0, 0.0)[1] == 100.0

  @parameterized.expand([
    ("ahead", 50.0, 0.0, True),
    ("edge_range", MAX_RANGE, 0.0, True),
    ("too_far", MAX_RANGE + 1, 0.0, False),
    ("behind", -1.0, 0.0, False),
    ("edge_lateral", 50.0, MAX_LATERAL, True),
    ("too_wide", 50.0, -MAX_LATERAL - 0.1, False),
  ])
  def test_visible(self, _, d, y, expected):
    assert ScopeFrame.visible(d, y) == expected

  def test_rings(self):
    assert [label for _, label in ring_ranges(True)] == ["25", "50", "100"]
    assert [r for r, _ in ring_ranges(True)] == [25.0, 50.0, 100.0]
    imperial = ring_ranges(False)
    assert [label for _, label in imperial] == ["100", "200", "300"]
    assert abs(imperial[0][0] - 30.48) < 0.01 and imperial[-1][0] < MAX_RANGE

  def test_ring_points_stay_on_the_circle(self):
    pts = ring_points(25.0)
    assert len(pts) > 8
    assert all(abs(math.hypot(d, y) - 25.0) < 1e-6 for d, y in pts)
    assert all(abs(y) <= MAX_LATERAL for _, y in pts)
    assert max(d for d, _ in pts) == 25.0

  def test_lane_samples_are_even_on_the_compressed_axis(self):
    steps = np.diff(np.sqrt(LANE_DISTANCES / MAX_RANGE))
    assert np.allclose(steps, steps[0]) and LANE_DISTANCES[0] == 0.0 and LANE_DISTANCES[-1] == MAX_RANGE


class TestBrackets(OpenpilotTestCase):
  @staticmethod
  def _pinhole(x, y, height):
    """Camera 1.2 m up, looking down the road. Returns None past the edges like RoadProjector."""
    if x <= 0:
      return None
    sx, sy = 1080 + 1500 * y / x, 400 + 1500 * (1.2 - height) / x
    return (sx, sy) if 0 <= sx <= 2160 and 0 <= sy <= 1080 else None

  def test_box_size_follows_distance(self):
    near = project_box(self._pinhole, 20.0, 0.0)
    far = project_box(self._pinhole, 80.0, 0.0)
    assert near is not None and far is not None
    assert near[2] - near[0] > 3.5 * (far[2] - far[0])
    assert abs((near[2] - near[0]) - 1500 * BOX_WIDTH / 20.0) < 1e-6
    assert abs((near[3] - near[1]) - 1500 * BOX_HEIGHT / 20.0) < 1e-6

  def test_box_stands_on_the_road(self):
    box = project_box(self._pinhole, 40.0, 1.5)
    assert box is not None
    assert box[3] == 400 + 1500 * 1.2 / 40.0
    assert abs((box[0] + box[2]) / 2 - (1080 + 1500 * 1.5 / 40.0)) < 1e-6

  def test_offscreen_has_no_box(self):
    assert project_box(self._pinhole, 5.0, 8.0) is None
    assert project_box(self._pinhole, -3.0, 0.0) is None

  def test_tiny_boxes_get_a_minimum_size(self):
    box = pad_box(1000.0, 500.0, 1006.0, 505.0)
    assert box[2] - box[0] == MIN_BOX_W and box[3] - box[1] == MIN_BOX_H
    assert box[3] == 505.0 and abs((box[0] + box[2]) / 2 - 1003.0) < 1e-9

  def test_big_boxes_are_left_alone(self):
    assert list(pad_box(100.0, 200.0, 400.0, 500.0)) == [100.0, 200.0, 400.0, 500.0]

  def test_selects_the_nearest_eight_within_reach(self):
    tracks = [track(i, 10.0 + 8.0 * i, 0.0, 1.0) for i in range(12)]
    chosen = select_tracks(tracks)
    assert len(chosen) == MAX_TRACKS
    assert [t.track_id for t in chosen] == list(range(8))
    assert [t.d_rel for t in chosen] == sorted(t.d_rel for t in chosen)

  @parameterized.expand([
    ("too_far", track(1, 100.0)),
    ("beyond_radar_range", track(1, 130.0)),
    ("too_wide_left", track(1, 30.0, 8.5)),
    ("too_wide_right", track(1, 30.0, -8.5)),
  ])
  def test_out_of_reach_is_skipped(self, _, t):
    assert select_tracks([t]) == []

  def test_at_the_edge_of_reach(self):
    assert len(select_tracks([track(1, 99.0, 8.0), track(2, 30.0, -8.0)])) == 2

  def test_the_lead_is_skipped(self):
    tracks = [track(7, 40.0), track(8, 50.0, 3.6)]
    assert [t.track_id for t in select_tracks(tracks, skip_id=7)] == [8]
    assert len(select_tracks(tracks, skip_id=NO_TRACK)) == 2

  def test_a_distant_threat_beats_nearby_clutter(self):
    clutter = [track(i, 10.0 + i, 7.0, -V_EGO) for i in range(10)]       # guardrail
    threat = track(50, 70.0, 0.0, -14.0)                                  # ttc 5 s
    chosen = select_tracks(clutter + [threat])
    assert len(chosen) == MAX_TRACKS and threat in chosen

  def test_tags_go_to_the_three_most_threatening(self):
    tracks = [
      track(1, 20.0, 0.0, -10.0),    # ttc 2
      track(2, 40.0, 3.6, -8.0),     # ttc 5
      track(3, 40.0, 0.0, -2.0),     # ttc 20
      track(4, 90.0, 0.0, -1.0),     # ttc 90
      track(5, 30.0, 0.0, 2.0),      # opening, but right there
    ]
    assert tagged_tracks(tracks) == {1, 2, 3}
    assert tagged_tracks(tracks, count=5) == {1, 2, 3, 5}

  def test_distant_traffic_pulling_away_is_not_tagged(self):
    assert tagged_tracks([track(1, 60.0, 0.0, 2.0), track(2, 31.0, 3.6, 1.0)]) == set()
    assert tagged_tracks([track(1, 29.0, 3.6, 1.0)]) == {1}

  def test_stationary_things_are_never_tagged(self):
    tracks = [track(1, 20.0, 0.0, -V_EGO), track(2, 30.0, 6.0, -V_EGO), track(3, 50.0, 0.0, -5.0)]
    assert tagged_tracks(tracks) == {3}


class TestBoardLayout(OpenpilotTestCase):
  def test_right_column_position(self):
    panel = board_rect(HUD, None, 4 * SLOT_H)
    assert panel.width == BOARD_W
    assert panel.x == HUD.x + HUD.width - 40 - BOARD_W and panel.y == HUD.y + 250

  @parameterized.expand([
    ("off", DeveloperUiState.OFF, 0),
    ("none_yet", None, 0),
    ("bottom", DeveloperUiState.BOTTOM, 0),
    ("right", DeveloperUiState.RIGHT, 230),
    ("both", DeveloperUiState.BOTH, 230),
  ])
  def test_developer_ui_shift(self, _, mode, shift):
    assert board_rect(HUD, mode, 100.0).x == HUD.x + HUD.width - 40 - BOARD_W - shift

  def test_height_follows_the_rows(self):
    one = board_rect(HUD, None, ROW_H)
    four = board_rect(HUD, None, 3 * SLOT_H + ROW_H)
    assert four.height - one.height == 3 * SLOT_H
    assert one.height > HEADER_H + ROW_H

  def test_clear_of_the_wheel_button(self):
    assert board_rect(HUD, None, 0.0).y >= HUD.y + 230

  def test_ttc_text(self):
    assert format_ttc(1.26) == "1.3"
    assert format_ttc(9.94) == "9.9"
    assert format_ttc(12.6) == "13"
