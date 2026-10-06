"""A conversation one `hmz exec` kept, carried on by a later one, on the stand-in `claude`.

The stand-in resumes a conversation only from where Claude keeps one for the directory it is
started in. So a later run's session carrying on what an earlier run kept, from a copy of it
and working in another directory, gets a first turn only where humanize brought the copy in
where that run keeps its sessions, and carried it to that directory, before the turn began.
Where a run keeps no sessions of its own, that is the CLI's home, whose conversation is never
put back to a copy of it.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest

from hmz.sdk import Hmz
from tests.integration.doubles_core import (
    AGENT,
    hmz_exec,
    install,
    started,
    write_flow,
)

if TYPE_CHECKING:
    from pathlib import Path

#: A flow that tells its agent something, and writes down where the conversation is kept.
KEPT = """
import dataclasses
import json
import pathlib

from hmz.flows import Agent, AgentCollection, EnvCollection, FlowParams, flow


class Agents(AgentCollection):
    worker: Agent


@flow(agents=Agents, envs=EnvCollection, params=FlowParams)
async def kept(task, *, agents, envs, params, ctx):
    worker = agents["worker"]
    session = await worker.spawn()
    await worker.run("the word is papaya", session=session)
    pathlib.Path(task).write_text(json.dumps(dataclasses.asdict(session.kept)))
"""

#: A flow that carries on the conversation written down at the path it is given, elsewhere.
CARRIED = """
import json
import pathlib

from hmz.flows import Agent, AgentCollection, Env, EnvCollection, FlowParams, KeptSession, flow


class Agents(AgentCollection):
    worker: Agent


class Envs(EnvCollection):
    there: Env


@flow(agents=Agents, envs=Envs, params=FlowParams)
async def carried(task, *, agents, envs, params, ctx):
    worker = agents["worker"]
    kept = KeptSession(**json.loads(pathlib.Path(task).read_text()))
    session = await worker.spawn(carry_on=kept)
    return await worker.run("what was the word?", session=session, env=envs["there"])
"""


#: A flow that tells its agent something, keeps a copy of the conversation then, and goes on.
SNAPSHOT = """
import dataclasses
import json
import pathlib
import shutil

from hmz.flows import Agent, AgentCollection, EnvCollection, FlowParams, flow


class Agents(AgentCollection):
    worker: Agent


@flow(agents=Agents, envs=EnvCollection, params=FlowParams)
async def snapshot(task, *, agents, envs, params, ctx):
    worker = agents["worker"]
    session = await worker.spawn()
    await worker.run("the word is papaya", session=session)
    kept = session.kept
    at = pathlib.Path(task)
    shutil.copytree(kept.directory, at / "copy")
    copied = dataclasses.replace(kept, directory=str(at / "copy"))
    (at / "kept.json").write_text(json.dumps(dataclasses.asdict(copied)))
    await worker.run("the word is mango", session=session)
"""


@pytest.fixture(autouse=True)
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return install(tmp_path, monkeypatch)


def test_a_conversation_one_run_kept_is_carried_on_from_a_copy_by_a_later_one(
    tmp_path: Path, project: Path
) -> None:
    flows = tmp_path / "flows"
    written = tmp_path / "kept.json"
    first = hmz_exec(
        "-f",
        write_flow(flows, "kept", KEPT),
        "-a",
        f"worker={AGENT}",
        "-p",
        "budget.cost=1",
        str(written),
    )
    assert first.returncode == 0, first.stderr
    (told,) = started(tmp_path)
    kept = json.loads(written.read_text())
    assert (kept["harness"], kept["id"]) == ("claude", told["session"])
    # What a flow keeping a snapshot keeps: the conversation copied out of the run.
    copied = tmp_path / "copied"
    shutil.copytree(kept["directory"], copied)
    (transcript,) = copied.glob(f"projects/*/{told['session']}.jsonl")
    was = transcript.read_bytes()
    written.write_text(json.dumps({**kept, "directory": str(copied)}))
    there = tmp_path / "there"
    there.mkdir()

    later = hmz_exec(
        "-f",
        write_flow(flows, "carried", CARRIED),
        "-a",
        f"worker={AGENT}",
        "-e",
        f"there=local/{there}",
        "-p",
        "budget.cost=1",
        str(written),
    )

    assert later.returncode == 0, later.stderr
    assert later.stdout.strip() == "did: what was the word?"
    _, carried = started(tmp_path)
    argv = carried["argv"]
    assert argv[argv.index("--resume") + 1] == told["session"]
    assert "--fork-session" in argv
    assert carried["cwd"] == str(there.resolve())
    assert carried["session"] != told["session"]
    assert transcript.read_bytes() == was, "the copy is left as it was"
    epics = Hmz(project).epics
    _, second = epics.all()
    ran = epics.read(second)
    assert ran is not None
    (session,) = ran.sessions
    assert (session.ident, session.parent) == (carried["session"], told["session"])


def test_a_conversation_nothing_kept_is_refused_by_the_first_turn(
    tmp_path: Path,
) -> None:
    written = tmp_path / "kept.json"
    written.write_text(
        json.dumps({"harness": "claude", "id": "nobody", "directory": str(tmp_path)})
    )

    ran = hmz_exec(
        "-f",
        write_flow(tmp_path / "flows", "carried", CARRIED),
        "-a",
        f"worker={AGENT}",
        "-e",
        f"there=local/{tmp_path}",
        "-p",
        "budget.cost=1",
        str(written),
    )

    assert ran.returncode != 0
    assert "the session cannot be forked: claude: no conversation nobody" in ran.stderr
    assert started(tmp_path) == []


def test_a_copy_of_what_the_clis_home_holds_as_it_went_on_is_not_put_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HUMANIZE_SESSIONS", "off")
    flows = tmp_path / "flows"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    first = hmz_exec(
        "-f",
        write_flow(flows, "snapshot", SNAPSHOT),
        "-a",
        f"worker={AGENT}",
        "-p",
        "budget.cost=1",
        str(snapshot),
    )
    assert first.returncode == 0, first.stderr
    (told,) = started(tmp_path)
    (held,) = (tmp_path / "claude").glob(f"projects/*/{told['session']}.jsonl")
    went_on = held.read_bytes()
    assert b"mango" in went_on
    there = tmp_path / "there"
    there.mkdir()

    later = hmz_exec(
        "-f",
        write_flow(flows, "carried", CARRIED),
        "-a",
        f"worker={AGENT}",
        "-e",
        f"there=local/{there}",
        "-p",
        "budget.cost=1",
        str(snapshot / "kept.json"),
    )

    assert later.returncode != 0
    assert f"another copy of conversation {told['session']} is kept at" in later.stderr
    assert held.read_bytes() == went_on
    assert len(started(tmp_path)) == 1
