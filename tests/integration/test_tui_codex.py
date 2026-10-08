"""Codex failures remain visible while a Ralph loop runs with details switched off."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from hmz.coganchor import backends
from hmz.runtime import Hmz
from hmz.runtime.kept import Runs
from hmz.tui import Humanize
from tests.integration.doubles_agents import Standins, standins
from tests.integration.doubles_tui import SIZE, screen, shows, typed, until
from tests.integration.test_agents_codex import CODEX

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Standins:
    """A saved Ralph loop in an isolated workspace, driven by the repository's Codex CLI."""

    def nowhere(command: str) -> None:
        return None

    monkeypatch.setattr(os, "defpath", "/usr/bin:/bin")
    monkeypatch.setattr(backends, "elsewhere", nowhere)
    held = standins(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    held.log = workspace / "calls.jsonl"
    held.install("codex", CODEX)
    monkeypatch.chdir(workspace)
    settings = Hmz(workspace).settings
    settings.detailing(on=False)
    settings.remember(
        "ralph_loop",
        {"agent": Runs("codex/gpt-stand-in:high")},
        budget={"duration": "PT1M"},
    )
    return held


async def test_a_codex_routing_failure_is_visible_before_a_loop_stops(
    codex: Standins,
) -> None:
    app = Humanize()
    async with app.run_test(size=SIZE) as pilot:
        await shows(pilot, "ralph_loop", "codex/gpt-stand-in:high")
        assert app.settings.details is False

        await typed(pilot, "routing failure")

        await until(
            pilot,
            lambda: (
                "workspace routing discovery failed" in " ".join(screen(app).split())
            ),
            "the failed turn to be visible",
        )
        assert "the flow is done" not in screen(app)
        ended = " ".join((await shows(pilot, "the flow is done")).split())
        assert ended.count("workspace routing discovery failed") == 3
        assert ended.index("workspace routing discovery failed") < ended.index(
            "the flow is done"
        )

    assert codex.said().count("routing failure") == 3
