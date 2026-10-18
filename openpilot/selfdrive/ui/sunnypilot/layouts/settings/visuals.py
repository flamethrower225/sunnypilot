"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, multiple_button_item_sp
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.widgets import Widget

CHEVRON_INFO_DESCRIPTION = {
  "enabled": tr_noop("Display useful metrics below the chevron that tracks the lead car " +
                     "only applicable to cars with sunnypilot longitudinal control."),
  "disabled": tr_noop("This feature requires sunnypilot longitudinal control to be available.")
}


class VisualsLayout(Widget):
  def __init__(self):
    super().__init__()

    self._params = Params()
    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._toggle_defs = {
      "BlindSpot": (
        lambda: tr("Show Blind Spot Warnings"),
        tr("Enabling this will display warnings when a vehicle is detected in your " +
           "blind spot as long as your car has BSM supported."),
        None,
      ),
      "TorqueBar": (
        lambda: tr("Steering Arc"),
        tr("Display steering arc on the driving screen when lateral control is enabled."),
        None,
      ),
      "RainbowMode": (
        lambda: tr("Enable Tesla Rainbow Mode"),
        tr("A beautiful rainbow effect on the path the model wants to take. " +
           "It does not affect driving in any way."),
        None,
      ),
      "StandstillTimer": (
        lambda: tr("Enable Standstill Timer"),
        tr("Show a timer on the HUD when the car is at a standstill."),
        None,
      ),
      "RoadNameToggle": (
        lambda: tr("Display Road Name"),
        tr("Displays the name of the road the car is traveling on." +
           "<br>The OpenStreetMap database of the location must be downloaded from " +
           "the OSM panel to fetch the road name."),
        None,
      ),
      "GreenLightAlert": (
        lambda: tr("Green Traffic Light Alert (Beta)"),
        tr("A chime and on-screen alert will play when the traffic light you are waiting for " +
           "turns green and you have no vehicle in front of you." +
           "<br>Note: This chime is only designed as a notification. " +
           "It is the driver's responsibility to observe their environment and make decisions accordingly."),
        None,
      ),
      "LeadDepartAlert": (
        lambda: tr("Lead Departure Alert (Beta)"),
        tr("A chime and on-screen alert will play when you are stopped, and the vehicle in front of you start moving." +
           "<br>Note: This chime is only designed as a notification. " +
           "It is the driver's responsibility to observe their environment and make decisions accordingly."),
        None,
      ),
      "TrueVEgoUI": (
        lambda: tr("Speedometer: Always Display True Speed"),
        tr("For applicable vehicles, always display the true vehicle current speed from wheel speed sensors."),
        None,
      ),
      "HideVEgoUI": (
        lambda: tr("Speedometer: Hide from Onroad Screen"),
        tr("When enabled, the speedometer on the onroad screen is not displayed."),
        None,
      ),
      "ShowTurnSignals": (
        lambda: tr("Display Turn Signals"),
        tr("When enabled, visual turn indicators are drawn on the HUD."),
        None,
      ),
      "RocketFuel": (
        lambda: tr("Real-time Acceleration Bar"),
        tr("Show an indicator on the left side of the screen to display real-time vehicle acceleration and deceleration. " +
           "This displays what the car is currently doing, not what the planner is requesting."),
        None,
      ),
    }
    self._toggles = {}
    for param, (title, desc, callback) in self._toggle_defs.items():
      toggle = toggle_item_sp(
        title=title,
        description=desc,
        param=param,
        initial_state=ui_state.params.get_bool(param),
        callback=callback,
      )
      self._toggles[param] = toggle

    self._chevron_info = multiple_button_item_sp(
      title=lambda: tr("Display Metrics Below Chevron"),
      description="",
      buttons=[lambda: tr("Off"), lambda: tr("Distance"), lambda: tr("Speed"), lambda: tr("Time"), lambda: tr("All")],
      param="ChevronInfo",
      inline=False
    )
    self._lead_gap_badge = multiple_button_item_sp(
      title=lambda: tr("Lead Gap Badge"),
      description=lambda: tr("Show a large badge with the time gap to the car ahead and its speed, colored from green to red " +
                             "as the gap closes. Dial, Strip and Block sit at the bottom center. Lock brackets the lead car and " +
                             "replaces its chevron. Paint tints the path and paints the gap on the road."),
      buttons=[lambda: tr("Off"), lambda: tr("Dial"), lambda: tr("Strip"), lambda: tr("Lock"), lambda: tr("Block"), lambda: tr("Paint")],
      param="LeadGapBadge",
      button_width=250,
      inline=False
    )

    # HUD overlays, each behind its own toggle
    self._overlay_defs = {
      "StopDetection": (
        lambda: tr("Stop Detection"),
        tr("Show the moment the driving model plans a stop, such as for a red light, stop sign or stopped traffic. " +
           "A card shows the distance, whether a car is the reason, and whether openpilot will stop for it, and a stop " +
           "line is drawn on the road, with a dashed BRAKE mark where gentle braking has to start. The model doesn't name " +
           "lights or signs, so a stop with no car ahead is shown as likely a light, sign or crossing."),
      ),
      "DetectionLog": (
        lambda: tr("Detection Log"),
        tr("Show what the model and planner pick up: planned stops, go, new leads, cut-ins, curves, predicted hard " +
           "braking and drive notifications. While driving only the newest shows, for a few seconds. When stopped, the last few are listed with " +
           "the time each was first seen."),
      ),
      "LeadGapHistory": (
        lambda: tr("Gap History"),
        tr("Draw the last 30 seconds of following gap as a line under the lead gap badge. Needs a badge style."),
      ),
      "FollowGhost": (
        lambda: tr("Follow Target Ghost"),
        tr("Outline the spot on the road where openpilot is trying to place your car behind the lead."),
      ),
      "LeadBrakingWake": (
        lambda: tr("Lead Braking Wake"),
        tr("When the car ahead brakes hard, show where the driving model predicts it will be in the next few seconds."),
      ),
      "CurveHologram": (
        lambda: tr("Curve Preview"),
        tr("In curves, light up the path edges by predicted sideways acceleration and show a suggested speed at the sharpest point. " +
           "The speed flashes red while you are well over it."),
      ),
      "RadarScope": (
        lambda: tr("Radar Scope"),
        tr("Show a top-down map of every radar track, the lanes and the lead car on the left side of the screen. " +
           "With a radar that reports only the car ahead, or no radar tracks, it shows the leads openpilot picks. " +
           "The lane beside the car lights up when the blind spot monitor sees something there."),
      ),
      "RadarTrackBrackets": (
        lambda: tr("Radar Track Brackets"),
        tr("Mark the radar tracks ahead in the camera view, colored by how quickly you are closing on them."),
      ),
      "ThreatBoard": (
        lambda: tr("Threat Board"),
        tr("List what is closing on you, ranked by time to collision, and the car ahead even when it isn't, " +
           "on the right side of the screen."),
      ),
      "DisengageHorizon": (
        lambda: tr("Disengage Risk Meter"),
        tr("Show the driving model's predicted chance of a disengagement over the next 2 to 10 seconds, under the speed."),
      ),
      "PaceBorder": (
        lambda: tr("Pace Border"),
        tr("Glow the screen edges when you are not slowing enough for the car ahead or for a planned stop at a light or sign. " +
           "Yellow means ease off, orange brake, red brake hard, and it flashes faster the more braking is missing. " +
           "It also covers a sharp curve ahead, and stays dark while the car is already slowing enough. It pulses green when it is " +
           "time to go, and gives the edge to the hard brake warning while that shows. " +
           "Along the bottom, four lights read like a runway PAPI whenever there is something to slow for: two white and two " +
           "red is on profile, more red means brake more (all red flashes), more white means ease off."),
      ),
      "DriveNotifications": (
        lambda: tr("Drive Notifications"),
        tr("Short advisories at the top right: pass a lead that is well under the speed limit (on the side with an open lane and " +
           "no car in the blind spot), stopped traffic ahead while you are still at speed, a sharp curve you aren't slowing for, " +
           "a new speed limit, speeding well over it, and a break after two hours. With Camera Object Detection on, also red, " +
           "yellow, green and flashing lights, stop signs, a pedestrian or cyclist in your path, brake lights ahead and a car " +
           "merging into your lane. One at a time, each with a cooldown, and each is also listed in the Detection Log."),
      ),
      "SceneDetection": (
        lambda: tr("Camera Object Detection"),
        tr("Run an object detector on the road cameras, next to the driving model, to read traffic lights (with turn arrows " +
           "and flashing signals), stop signs, pedestrians, cyclists and the cars around you, including their brake lights and " +
           "a car drifting into your lane. The light or stop sign shows in a chip at the top left (with how long you have waited " +
           "at a red), names the stop on the stop card, feeds Pace Border and adds notifications. Cars show as rounded squares " +
           "on the radar scope (with brake lamps, and a chevron when merging), people as amber figures. Distances calibrate " +
           "themselves against the radar. It shares the GPU with the driving and driver monitoring models and steps aside " +
           "whenever either slows down; Developer > Camera Detection Status shows how it ran. Advisory only: it doesn't " +
           "change how openpilot drives. Takes effect on the next drive."),
      ),
      "HardBrakeFlash": (
        lambda: tr("Hard Brake Warning"),
        tr("Flash the screen edges when the driving model predicts hard braking ahead."),
      ),
      "DriverFlags": (
        lambda: tr("Driver Monitoring Flags"),
        tr("Show DROWSY, PHONE or DISTRACTED next to the driver monitoring icon when the driver camera detects them. " +
           "DISTRACTED is driver monitoring's own call that you are looking away from the road."),
      ),
    }
    self._overlay_toggles = {}
    for param, (title, desc) in self._overlay_defs.items():
      self._overlay_toggles[param] = toggle_item_sp(
        title=title,
        description=desc,
        param=param,
        initial_state=ui_state.params.get_bool(param),
      )
    self._dev_ui_info = multiple_button_item_sp(
      title=lambda: tr("Developer UI"),
      description=lambda: tr("Display real-time parameters and metrics from various sources."),
      buttons=[lambda: tr("Off"), lambda: tr("Bottom"), lambda: tr("Right"), lambda: tr("Right & Bottom")],
      param="DevUIInfo",
      button_width=350,
      inline=False
    )

    items = list(self._toggles.values()) + [
      self._chevron_info,
      self._lead_gap_badge,
      *self._overlay_toggles.values(),
      self._dev_ui_info,
    ]
    return items

  def _update_state(self):
    super()._update_state()

    for param in self._toggle_defs:
      self._toggles[param].action_item.set_state(self._params.get_bool(param))
    for param in self._overlay_defs:
      self._overlay_toggles[param].action_item.set_state(self._params.get_bool(param))

    self._dev_ui_info.action_item.set_selected_button(ui_state.params.get("DevUIInfo", return_default=True))
    self._lead_gap_badge.action_item.set_selected_button(ui_state.params.get("LeadGapBadge", return_default=True))

    if ui_state.has_longitudinal_control:
      self._chevron_info.set_description(tr(CHEVRON_INFO_DESCRIPTION["enabled"]))
      self._chevron_info.action_item.set_selected_button(ui_state.params.get("ChevronInfo", return_default=True))
      self._chevron_info.action_item.set_enabled(True)
    else:
      self._chevron_info.set_description(tr(CHEVRON_INFO_DESCRIPTION["disabled"]))
      self._chevron_info.action_item.set_enabled(False)
      ui_state.params.put("ChevronInfo", 0)

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    self._scroller.show_event()
    if not ui_state.has_longitudinal_control:
      self._chevron_info.set_description(tr(CHEVRON_INFO_DESCRIPTION["disabled"]))
      self._chevron_info.show_description(True)
