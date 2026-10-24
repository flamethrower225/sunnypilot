"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The updater's git fetch must not recurse into submodules. On-demand recursion downloads every submodule commit pinned
anywhere in the newly fetched history; upstream sunnypilot history pins opendbc commits that were later deleted, so one
"not our ref" failed the whole Download with "failed to check for update" (seen on a comma 3X, 2026-10-23).
"""
from subprocess import CalledProcessError

import pytest

from openpilot.common.test import OpenpilotTestCase
from openpilot.system.updated import updated
from openpilot.system.updated.updated import Updater, git_fetch_cmd

SUBMODULE_UPDATE = ["git", "submodule", "update", "--init", "--recursive"]


class TestGitFetchCmd(OpenpilotTestCase):
  def test_fetch_does_not_recurse_into_submodules(self):
    assert git_fetch_cmd("master") == ["git", "fetch", "--no-recurse-submodules", "origin", "master"]

  def test_the_target_branch_is_fetched(self):
    assert git_fetch_cmd("hkg-angle-steering-2025")[-2:] == ["origin", "hkg-angle-steering-2025"]


class TestFetchUpdate(OpenpilotTestCase):
  def _fake_git(self, mocker, fail: list[str] | None = None) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], cwd: str | None = None) -> str:
      calls.append(cmd)
      if fail is not None and cmd[:len(fail)] == fail:
        raise CalledProcessError(1, cmd, output="fatal: remote error: upload-pack: not our ref")
      return ""

    mocker.patch.object(updated, "run", side_effect=fake_run)
    for name in ("setup_git_options", "set_consistent_flag", "handle_agnos_update", "finalize_update"):
      mocker.patch.object(updated, name)
    mocker.patch.object(Updater, "target_branch", new_callable=mocker.PropertyMock, return_value="master")
    return calls

  def test_fetches_without_recursion_then_updates_the_pinned_submodules(self, mocker):
    calls = self._fake_git(mocker)
    Updater().fetch_update()

    fetches = [c for c in calls if c[:2] == ["git", "fetch"]]
    assert fetches == [["git", "fetch", "--no-recurse-submodules", "origin", "master"]]
    assert SUBMODULE_UPDATE in calls
    assert calls.index(fetches[0]) < calls.index(SUBMODULE_UPDATE)

  def test_a_failed_fetch_stops_before_the_checkout_is_touched(self, mocker):
    calls = self._fake_git(mocker, fail=["git", "fetch"])
    with pytest.raises(CalledProcessError):
      Updater().fetch_update()
    assert not any(c[:2] == ["git", "checkout"] for c in calls)
    assert SUBMODULE_UPDATE not in calls
