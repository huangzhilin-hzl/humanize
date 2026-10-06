"""A conversation one real run kept is carried on by a later one, from a copy, elsewhere.

Which is what `Agent.spawn(carry_on=...)` is for. A flow tells its agent a word and writes
down where the conversation is kept; that is copied where a snapshot of the run would copy it,
and the run's own is removed; a later run, working in a copy of the workspace, carries the copy
on and asks for the word back -- a second process reading the conversation, under an id of its
own. A run holds one copy of a conversation: a second snapshot of it, carried on in the run the
first was, is refused. For Claude Code and Codex, each on its cheapest model, wherever it is
signed in here.

Where this machine supervises no turn, a run keeps its sessions in the CLI's own home, sign-in
and all, which is never copied: the conversation is carried on from there, as it is, the
snapshots are of its own files alone, and a snapshot of what the home holds as it went on is
refused rather than put back.

`HMZ_SYSTEM_MODEL_<HARNESS>` swaps the model a harness runs at, as `MODEL[:EFFORT]`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from hmz.runtime.epic import epics, read
from tests.system.doubles_flows import harness

#: Tells the agent a word, and writes down where its conversation is kept.
KEPT = """
import dataclasses
import json
import pathlib

from hmz.flows import Agent, AgentCollection, EnvCollection, FlowParams, flow


class Agents(AgentCollection):
    worker: Agent


@flow(agents=Agents, envs=EnvCollection, params=FlowParams)
async def kept(task, *, agents, envs, params, ctx):
    asked = json.loads(task)
    worker = agents["worker"]
    session = await worker.spawn()
    await worker.run(
        f"Remember this code word: {asked['word']}. Reply with exactly one word: OK",
        session=session,
    )
    pathlib.Path(asked["kept"]).write_text(json.dumps(dataclasses.asdict(session.kept)))
"""

#: Tells the agent a word, copies its conversation's files, tells it another, copies them again.
SNAPSHOTS = """
import dataclasses
import json
import pathlib
import shutil

from hmz.flows import Agent, AgentCollection, EnvCollection, FlowParams, flow


class Agents(AgentCollection):
    worker: Agent


def snapshot(kept, at):
    for path in pathlib.Path(kept.directory).rglob(f"*{kept.id}*"):
        if path.is_file():
            copy = at / path.relative_to(kept.directory)
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, copy)
    copied = dataclasses.replace(kept, directory=str(at))
    (at / "kept.json").write_text(json.dumps(dataclasses.asdict(copied)))
    (at / "from.txt").write_text(kept.directory)


@flow(agents=Agents, envs=EnvCollection, params=FlowParams)
async def snapshots(task, *, agents, envs, params, ctx):
    asked = json.loads(task)
    worker = agents["worker"]
    session = await worker.spawn()
    for word, at in zip(asked["words"], asked["snapshots"]):
        await worker.run(
            f"Remember this code word: {word}. Reply with exactly one word: OK",
            session=session,
        )
        snapshot(session.kept, pathlib.Path(at))
"""

#: Carries on each conversation written down where it is told, in `there`, in turn, and writes
#: down what each said.
BOTH = """
import json
import pathlib

from hmz.flows import (
    Agent, AgentCollection, Env, EnvCollection, FlowParams, KeptSession, SessionError, flow,
)


class Agents(AgentCollection):
    worker: Agent


class Envs(EnvCollection):
    there: Env


@flow(agents=Agents, envs=Envs, params=FlowParams)
async def both(task, *, agents, envs, params, ctx):
    asked = json.loads(task)
    worker = agents["worker"]
    said = []
    for at in asked["snapshots"]:
        kept = KeptSession(**json.loads((pathlib.Path(at) / "kept.json").read_text()))
        session = await worker.spawn(carry_on=kept)
        try:
            said.append(await worker.run(
                "What is the code word I asked you to remember last? Reply with it alone.",
                session=session,
                env=envs["there"],
            ))
        except SessionError as refused:
            said.append(f"refused: {refused}")
    pathlib.Path(asked["said"]).write_text(json.dumps(said))
"""

#: Carries on the conversation written down at the path it is given, in `there`.
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
    session = await worker.spawn(carry_on=KeptSession(**json.loads(pathlib.Path(task).read_text())))
    return await worker.run(
        "What is the code word I asked you to remember? Reply with the code word alone.",
        session=session,
        env=envs["there"],
    )
"""

BUDGET = "budget.cost=1,budget.duration=600"


def _flow(under: Path, name: str, source: str) -> str:
    at = under / name
    at.mkdir(parents=True)
    (at / "__init__.py").write_text(source)
    return str(at)


def _exec(at: Path, *argv: str) -> list[dict[str, Any]]:
    """Runs `hmz exec --json` in `at`, failing the test where it fails, and answers its events."""
    ran = subprocess.run(
        [sys.executable, "-m", "hmz", "exec", "--json", *argv],
        cwd=at,
        stdout=subprocess.PIPE,
        text=True,
        timeout=900,
        check=False,
    )
    assert ran.returncode == 0, ran.stdout[-4000:]
    return [json.loads(line) for line in ran.stdout.splitlines() if line.strip()]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("cli", ["claude", "codex"])
def test_a_conversation_one_run_kept_is_carried_on_by_a_later_one(
    cli: str, workspace: Path, tmp_path: Path
) -> None:
    spec = harness(cli)
    override = os.environ.get(f"HMZ_SYSTEM_MODEL_{cli.upper()}")
    if override:
        spec = f"{cli}/{override}"
    word = f"tangerine{uuid.uuid4().int % 1000:03d}"
    written = tmp_path / "kept.json"
    flows = tmp_path / "flows"

    _exec(
        workspace,
        *("-f", _flow(flows, "kept", KEPT), "-a", f"worker={spec}", "-p", BUDGET),
        json.dumps({"word": word, "kept": str(written)}),
    )
    kept = json.loads(written.read_text())
    assert kept["harness"] == cli
    copied = Path(kept["directory"])
    if copied.resolve().is_relative_to(tmp_path.resolve()):
        # What a flow keeping a snapshot keeps: the conversation copied out of the run, and
        # the workspace beside it -- and the run's own gone. Only where it is the run's, in
        # this test's home: the CLI's own, where a machine that supervises no turn keeps it,
        # holds its sign-in, and is carried on from in place.
        copied = tmp_path / "copied"
        shutil.copytree(kept["directory"], copied)
        shutil.rmtree(kept["directory"])
        written.write_text(json.dumps({**kept, "directory": str(copied)}))
    there = tmp_path / "there"
    shutil.copytree(workspace, there)
    left = {
        path: path.read_bytes()
        for path in copied.rglob(f"*{kept['id']}*")
        if path.is_file()
    }
    assert left, f"nothing of {kept['id']} under {kept['directory']}"

    said = _exec(
        workspace,
        *("-f", _flow(flows, "carried", CARRIED), "-a", f"worker={spec}"),
        *("-e", f"there=local/{there}", "-p", BUDGET, str(written)),
    )

    answers = [str(one["text"]) for one in said if one["kind"] == "result"]
    assert answers, said
    assert word in answers[-1].lower(), f"{cli} carried on without the word: {answers}"
    first, later = epics(workspace)
    told, carried = read(first), read(later)
    assert told is not None
    assert carried is not None
    (session,) = carried.sessions
    assert session.parent == kept["id"]
    assert session.ident != kept["id"]
    assert [one.ident for one in told.sessions] == [kept["id"]]
    assert {path: path.read_bytes() for path in left} == left, "the copy is as it was"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("cli", ["claude", "codex"])
def test_a_second_snapshot_of_a_conversation_is_refused_by_the_run_holding_the_first(
    cli: str, workspace: Path, tmp_path: Path
) -> None:
    spec = harness(cli)
    override = os.environ.get(f"HMZ_SYSTEM_MODEL_{cli.upper()}")
    if override:
        spec = f"{cli}/{override}"
    number = uuid.uuid4().int % 1000
    words = [f"tangerine{number:03d}", f"pomegranate{number:03d}"]
    snapshots = [str(tmp_path / "early"), str(tmp_path / "late")]
    flows = tmp_path / "flows"
    _exec(
        workspace,
        *(
            "-f",
            _flow(flows, "snapshots", SNAPSHOTS),
            "-a",
            f"worker={spec}",
            "-p",
            BUDGET,
        ),
        json.dumps({"words": words, "snapshots": snapshots}),
    )
    there = tmp_path / "there"
    shutil.copytree(workspace, there)

    written = tmp_path / "said.json"

    _exec(
        workspace,
        *("-f", _flow(flows, "both", BOTH), "-a", f"worker={spec}"),
        *("-e", f"there=local/{there}", "-p", BUDGET),
        json.dumps({"snapshots": snapshots, "said": str(written)}),
    )

    early, late = json.loads(written.read_text())
    kept = Path(snapshots[0], "from.txt").read_text()
    if Path(kept).resolve().is_relative_to(tmp_path.resolve()):
        assert words[0] in early.lower(), (
            f"{cli} carried it on without its word: {early}"
        )
        assert late.startswith("refused: "), late
        assert "another copy of conversation" in late
    else:
        # The CLI's own home holds the conversation as it went on past the first snapshot,
        # and is never put back to it. The second is what it holds, or what it held before
        # the CLI wrote down the end of its last turn.
        assert early.startswith("refused: "), early
        assert "another copy of conversation" in early
