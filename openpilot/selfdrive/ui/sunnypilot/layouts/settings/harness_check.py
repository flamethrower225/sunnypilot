"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.selfdrive.ui.sunnypilot.harness_status import harness_status_html
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets.list_view import button_item
from openpilot.system.ui.sunnypilot.widgets.html_render import HtmlModalSP
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp


class HarnessCheckItems:
  """Developer panel controls for the harness check service (harnessd)."""

  def __init__(self):
    self.toggle = toggle_item_sp(tr("Harness Check"),
                                 tr("While the car is on, listen to CAN (never transmit) for about 15 seconds and check which bus the " +
                                    "camera's steering messages arrive on. If they are on the wrong bus, the comma harness needs a " +
                                    "pin swap and an alert says so."), param="HarnessCheckEnabled")
    self.result_btn = button_item(tr("Harness Check Result"), tr("VIEW"),
                                  tr("Which bus each steering message arrived on during the last drive, and whether that needs a " +
                                     "pin swap."), callback=self._show_result)

    self.items: list = [self.toggle, self.result_btn]

  def update_state(self) -> None:
    self.result_btn.set_visible(ui_state.params.get_bool("HarnessCheckEnabled"))

  @staticmethod
  def _show_result():
    gui_app.push_widget(HtmlModalSP(text=harness_status_html(ui_state.params.get("HarnessCheck"))))
