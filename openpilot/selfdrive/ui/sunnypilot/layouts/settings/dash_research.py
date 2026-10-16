"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.tools.dash_re.engine import MARK_LABELS, findings_html, results_html
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.list_view import button_item
from openpilot.system.ui.widgets.option_dialog import MultiOptionDialog
from openpilot.system.ui.sunnypilot.widgets.html_render import HtmlModalSP
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp


class DashResearchItems:
  """Developer panel controls for the dash research service (dashred)."""

  def __init__(self):
    self._mark_dialog: MultiOptionDialog | None = None

    self.toggle = toggle_item_sp(tr("Dash Research"),
                                 tr("While the car is on, listen to CAN (never transmit) and work out which messages the instrument cluster " +
                                    "reads, which ones openpilot could change, and whether free text reaches the reachable buses. " +
                                    "Results are shown below and kept after the car is off."), param="DashReEnabled")
    self.findings_btn = button_item(tr("Dash Research Findings"), tr("VIEW"), tr("What the service has found on each bus so far."),
                                    callback=self._show_findings)
    self.mark_btn = button_item(tr("Mark Dash Action"), tr("MARK"),
                                tr("Car on and parked: pick what you are about to do, tap Select, then do it within a second. " +
                                   "The bits that changed show up under Dash Action Results a few seconds later."),
                                callback=self._show_mark_dialog)
    self.results_btn = button_item(tr("Dash Action Results"), tr("VIEW"), tr("The bits that changed after each marked action."),
                                   callback=self._show_results)
    self.reset_btn = button_item(tr("Reset Dash Research"), tr("RESET"), tr("Delete the saved findings and action results."),
                                 callback=self._confirm_reset)

    self.items: list = [self.toggle, self.findings_btn, self.mark_btn, self.results_btn, self.reset_btn]

  def update_state(self) -> None:
    enabled = ui_state.params.get_bool("DashReEnabled")
    for item in (self.findings_btn, self.mark_btn, self.results_btn, self.reset_btn):
      item.set_visible(enabled)
    self.mark_btn.action_item.set_enabled(ui_state.started)

  @staticmethod
  def _show_findings():
    gui_app.push_widget(HtmlModalSP(text=findings_html(ui_state.params.get("DashReFindings"))))

  @staticmethod
  def _show_results():
    gui_app.push_widget(HtmlModalSP(text=results_html(ui_state.params.get("DashReResults"))))

  def _show_mark_dialog(self):
    def on_select(result: DialogResult):
      if result == DialogResult.CONFIRM and self._mark_dialog is not None and self._mark_dialog.selection:
        # monotonic ns is the clock CAN frames are stamped with (logMonoTime)
        ui_state.params.put("DashReMark", {"id": time.time_ns(), "label": self._mark_dialog.selection, "t": time.monotonic_ns()})
      self._mark_dialog = None

    self._mark_dialog = MultiOptionDialog(tr("What are you about to do?"), MARK_LABELS, callback=on_select)
    gui_app.push_widget(self._mark_dialog)

  @staticmethod
  def _confirm_reset():
    def on_result(result: DialogResult):
      if result == DialogResult.CONFIRM:
        ui_state.params.remove("DashReFindings")
        ui_state.params.remove("DashReResults")

    gui_app.push_widget(ConfirmDialog(tr("Delete all dash research findings and results?"), tr("Delete"), tr("Cancel"), rich=False,
                                      callback=on_result))
