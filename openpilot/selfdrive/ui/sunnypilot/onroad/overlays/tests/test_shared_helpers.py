"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The shared, cached helpers the overlays use to stay cheap: they have to give exactly what the per-call versions gave.
"""
from types import SimpleNamespace

import numpy as np
import pyray as rl

from openpilot.cereal import messaging
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
from openpilot.selfdrive.ui.sunnypilot import model_plan as mp
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import clamp, rgba
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays import edge_glow, radar_tracks
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.projection import RoadProjector
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_scope import runs


class FakeRenderer:
  """The parts of ModelRenderer the projector reads, with ModelRenderer's own projection code."""
  _map_to_screen = ModelRenderer._map_to_screen
  _map_line_to_polygon = ModelRenderer._map_line_to_polygon

  def __init__(self):
    xs = np.linspace(0.0, 150.0, 33)
    self._path = SimpleNamespace(raw_points=np.column_stack([xs, 0.002 * xs ** 2, 0.01 * xs]).astype(np.float32))
    self._path_offset_z = 1.22
    self._camera_offset = 0.1
    # a pinhole camera looking down the road: x forward, y right, z down onto a 2160 x 1080 screen
    f, cx, cy = 1000.0, 1080.0, 540.0
    self._car_space_transform = np.array([[cx, f, 0.0], [cy, 0.0, f], [1.0, 0.0, 0.0]], dtype=np.float32)
    self._clip_region = rl.Rectangle(-500, -500, 2160 + 1000, 1080 + 1000)
    self._rect = rl.Rectangle(0, 0, 2160, 1080)


class TestProjection(OpenpilotTestCase):
  def setup_method(self):
    self.p = RoadProjector(FakeRenderer())

  def test_many_matches_one_at_a_time(self):
    xs = np.array([-1.0, 0.0, 2.0, 10.0, 40.0, 140.0, 0.4])
    ys = np.array([0.0, 1.0, -1.5, 3.0, -2.0, 1.0, 400.0])
    hs = np.array([0.0, 0.0, 0.5, 1.6, 0.0, 0.0, 0.0])
    screen, ok = self.p.to_screen_many(xs, ys, hs)
    for i in range(len(xs)):
      one = self.p.to_screen(xs[i], ys[i], hs[i])
      assert (one is not None) == ok[i], i
      if one is not None:
        assert np.allclose(one, screen[i], atol=1e-3)
    assert not ok.all() and ok.any()  # the check covered both cases

  def test_many_keeps_the_input_shape(self):
    xs = np.full((3, 4), 20.0)
    screen, ok = self.p.to_screen_many(xs, 0.0, 0.0)
    assert screen.shape == (3, 4, 2) and ok.shape == (3, 4) and ok.all()

  def test_ribbons_match_ribbon(self):
    xs = np.geomspace(3.0, 140.0, 40)
    ys = self.p.path_y(xs)
    centers, widths = [ys - 1.0, ys + 1.0, ys], [0.36, 0.2, 0.08]
    for poly, center, hw in zip(self.p.ribbons(xs, centers, widths), centers, widths, strict=True):
      assert np.allclose(poly, self.p.ribbon(xs, center, hw), atol=1e-2)


class TestColors(OpenpilotTestCase):
  def test_clamp_is_np_clip(self):
    for value, lo, hi in ((0.5, 0.0, 1.0), (-1.0, 0.0, 1.0), (2.0, 0.0, 1.0), (5.0, 10.0, 3.0), (-5.0, 10.0, 3.0)):
      assert clamp(value, lo, hi) == float(np.clip(value, lo, hi))

  def test_rgba_is_cached_and_clamped(self):
    a = rgba((10, 20, 30), 0.5)
    assert a is rgba((10.4, 20, 30), 0.5) and (a.r, a.g, a.b, a.a) == (10, 20, 30, 127)
    assert rgba((1, 2, 3), 7.0).a == 255 and rgba((1, 2, 3), -1.0).a == 0
    assert rgba(np.array([1.0, 2.0, 3.0]), 1.0) is rgba((1, 2, 3), 1.0)


class FakeSM:
  def __init__(self):
    self.frame = 0
    self.valid = {"radarTracks": True, "radarState": True}
    self.alive = {"radarTracks": True, "radarState": True}
    self.recv_frame = {"radarTracks": 1, "radarState": 1, "carState": 1, "modelV2": -1}
    lead = SimpleNamespace(present=False, radar=False, radarTrackId=-1, dRel=0.0, yRel=0.0, vRel=0.0)
    self.data = {"radarTracks": SimpleNamespace(points=[SimpleNamespace(trackId=1, dRel=30.0, yRel=0.0, vRel=-2.0)]),
                 "radarState": SimpleNamespace(leadOne=lead, leadTwo=lead), "carState": SimpleNamespace(vEgo=20.0)}

  def __getitem__(self, service):
    return self.data[service]


class TestSharedRadarFeed(OpenpilotTestCase):
  def test_advances_once_per_frame(self):
    radar_tracks._shared = None
    sm = FakeSM()
    sm.frame = 1
    feed = radar_tracks.shared_radar_feed(sm, 1, 0.05)
    assert [t.d_rel for t in feed.tracks] == [30.0]
    sm.data["radarTracks"].points[0].dRel = 31.0
    assert radar_tracks.shared_radar_feed(sm, 1, 0.05).tracks[0].d_rel == 30.0   # same frame: no second update
    sm.frame = 2
    moved = radar_tracks.shared_radar_feed(sm, 1, 0.05).tracks[0].d_rel
    assert 30.0 < moved < 31.0   # next frame: smoothed toward the new distance, once
    assert radar_tracks.shared_radar_feed(sm, 1, 0.05) is feed


class TestModelPlan(OpenpilotTestCase):
  def test_converted_once_per_message(self):
    msg = messaging.new_message('modelV2')
    m = msg.modelV2
    m.position.x, m.position.y = [0.0, 10.0, 20.0], [0.0, 0.5]
    m.velocity.x, m.orientationRate.z = [10.0, 10.0, 10.0], [0.1, -0.2, 0.0]

    class SM:
      recv_frame = {'modelV2': 5}

      def __getitem__(self, name):
        return m
    sm = SM()
    plan = mp.model_plan(sm, 0)
    assert plan.size == 3 and np.allclose(plan.lat_acc, [1.0, 2.0, 0.0]) and np.allclose(plan.y, [0.0, 0.5, 0.0])
    assert mp.model_plan(sm, 0) is plan
    assert mp.model_plan(sm, 10).size == 0   # from before this drive


class TestEdgeGlowAndRuns(OpenpilotTestCase):
  def test_bands_are_built_once_per_size(self):
    rect = rl.Rectangle(0, 0, 2000, 1000)
    bands = edge_glow.edge_bands(rect, 80.3, 20)
    assert edge_glow.edge_bands(rect, 79.8, 20) is bands   # under a pixel apart: the same bands
    assert len(bands) == 20 and bands[0].falloff == 1.0 and bands[-1].falloff < 0.01
    assert bands[1].rect.x == 4.0 and bands[0].thickness == 8.0
    assert edge_glow.edge_bands(rect, 120.0, 20) is not bands

  def test_runs(self):
    assert runs(np.array([True, True, False, True, False, True, True, True])) == [(0, 2), (5, 8)]
    assert runs(np.array([], dtype=bool)) == []
