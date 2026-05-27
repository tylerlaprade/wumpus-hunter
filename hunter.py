#!/usr/bin/env python3
"""hunter.py - I/O, game lifecycle, CLI.

Spawns the `/tmp/wumpus` binary on a pseudo-tty, runs one fair life
end-to-end using `strategy.choose_action`, applying the dynamic belief
updates from `strategy` for snatches and missed shots. Optionally repeats
across many independent games for benchmarking.

A "fair fresh-game" run uses no information across games — the belief
state is rebuilt from each new opening observation. After a loss we
answer `SAME SETUP (Y-N) ? N` so the binary discards the hidden map and
starts over with a freshly randomized layout.
"""

from __future__ import annotations

import argparse
import os
import pty
import re
import select
import sys
import time
from dataclasses import dataclass, field
from typing import Final, Literal, cast, get_args

from belief import (
    ARROW_SELF_TEXT,
    Action,
    BAT_SNATCH_TEXT,
    Belief,
    MISSED_TEXT,
    OUT_OF_ARROWS_TEXT,
    LOSE_BANNER_TEXT,
    Observation,
    PIT_DEATH_TEXT,
    PIT_SHRIEK_TEXT,
    Shot,
    WIN_TEXT,
    WUMPUS_DEATH_TEXT,
    current_player,
    filter_belief,
    fmt_shot_path,
    initial_belief,
    parse_observation,
)
from strategy import (
    best_desperation_action,
    choose_action,
    update_on_miss,
    update_on_move,
    update_on_snatch,
)


DEFAULT_BINARY: Final[str] = "/tmp/wumpus"
DEFAULT_TIMEOUT_S: Final[float] = 5.0
QUIET_WINDOW_S: Final[float] = 0.08
SETUP_PROMPT_DEADLINE_S: Final[float] = 3.0
StateKey = tuple[int, int, Belief]

PromptKind = Literal[
    "INSTRUCTIONS (Y-N)",
    "SHOOT OR MOVE (S-M)",
    "WHERE TO",
    "NO. OF ROOMS (1-5)",
    "ROOM #",
    "SAME SETUP (Y-N)",
    "TYPE AN E THEN RETURN",
]
PROMPT_KINDS: Final[tuple[PromptKind, ...]] = get_args(PromptKind)
_PROMPT_RE: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(p) for p in PROMPT_KINDS)
)


# ---------------------------------------------------------------------------
# GameProcess: spawn binary on pty, read until prompt, send line.
# ---------------------------------------------------------------------------
class GameProcess:
    def __init__(self, argv: list[str]) -> None:
        self.argv: list[str] = argv
        self.pid: int | None = None
        self.fd: int | None = None

    def start(self) -> None:
        pid, fd = pty.fork()
        if pid == 0:
            # Any failure in the child must terminate it; otherwise the child
            # falls through to parent code and races on the pty.
            try:
                os.execv(self.argv[0], self.argv)
            except BaseException:
                os._exit(127)
        self.pid = pid
        self.fd = fd

    def stop(self) -> None:
        if self.pid is None:
            return
        try:
            os.kill(self.pid, 9)
        except ProcessLookupError:
            pass
        try:
            _ = os.waitpid(self.pid, 0)
        except ChildProcessError:
            pass
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.pid = None
        self.fd = None

    def read_until_prompt(
        self,
        overall_timeout: float = DEFAULT_TIMEOUT_S,
        quiet_window: float = QUIET_WINDOW_S,
    ) -> str:
        if self.fd is None:
            raise RuntimeError("read_until_prompt called on stopped process")
        deadline = time.monotonic() + overall_timeout
        out = b""
        last_byte_t = time.monotonic()
        while True:
            now = time.monotonic()
            if now > deadline:
                break
            wait = min(quiet_window, max(0.0, deadline - now))
            r, _, _ = select.select([self.fd], [], [], wait)
            if r:
                try:
                    chunk = os.read(self.fd, 4096)
                except OSError:
                    chunk = b""
                if not chunk:
                    break
                out += chunk
                last_byte_t = time.monotonic()
                continue
            trimmed = out.rstrip(b" \t\r\n")
            if trimmed.endswith(b"?"):
                break
            if out and now - last_byte_t > 0.5:
                break
        return out.decode("utf-8", errors="replace")

    def write_line(self, value: str) -> None:
        if self.fd is None:
            raise RuntimeError("write_line called on stopped process")
        _ = os.write(self.fd, (value + "\n").encode("ascii"))


# ---------------------------------------------------------------------------
# Block parser: detect prompts, warnings, terminal events.
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Block:
    raw: str
    obs: Observation | None
    prompt: PromptKind | None
    bat_snatch: bool
    miss: bool
    death_pit: bool
    death_wumpus: bool
    death_arrow: bool
    out_of_arrows: bool
    lose_banner: bool
    victory: bool

    @property
    def terminal(self) -> bool:
        return (
            self.victory
            or self.death_pit
            or self.death_wumpus
            or self.death_arrow
            or self.out_of_arrows
            or self.lose_banner
        )


def _last_prompt(upper_text: str) -> PromptKind | None:
    last: PromptKind | None = None
    for match in _PROMPT_RE.finditer(upper_text):
        # cast: every alternation in _PROMPT_RE is a PromptKind literal.
        last = cast(PromptKind, match.group(0))
    return last


def parse_block(text: str) -> Block:
    u = text.upper()
    return Block(
        raw=text,
        obs=parse_observation(text),
        prompt=_last_prompt(u),
        bat_snatch=BAT_SNATCH_TEXT in u,
        miss=MISSED_TEXT in u,
        death_pit=PIT_DEATH_TEXT in u or PIT_SHRIEK_TEXT in u,
        death_wumpus=WUMPUS_DEATH_TEXT in u,
        death_arrow=ARROW_SELF_TEXT in u,
        out_of_arrows=OUT_OF_ARROWS_TEXT in u,
        lose_banner=LOSE_BANNER_TEXT in u,
        victory=WIN_TEXT in u,
    )


# ---------------------------------------------------------------------------
# Outcome + stats.
# ---------------------------------------------------------------------------
Outcome = Literal["win", "loss-pit", "loss-wumpus", "loss-arrow", "loss-out-of-arrows", "loss-unknown"]


@dataclass(slots=True)
class GameLog:
    outcome: Outcome
    moves: int
    shots: int
    trail: list[str]


@dataclass(slots=True)
class SessionStats:
    games: int = 0
    wins: int = 0
    losses_pit: int = 0
    losses_wumpus: int = 0
    losses_arrow: int = 0
    losses_out_of_arrows: int = 0
    losses_unknown: int = 0
    total_moves: int = 0
    total_shots: int = 0
    last_trail: list[str] = field(default_factory=list)

    def record(self, log: GameLog) -> None:
        self.games += 1
        self.total_moves += log.moves
        self.total_shots += log.shots
        self.last_trail = log.trail
        if log.outcome == "win":
            self.wins += 1
        elif log.outcome == "loss-pit":
            self.losses_pit += 1
        elif log.outcome == "loss-wumpus":
            self.losses_wumpus += 1
        elif log.outcome == "loss-arrow":
            self.losses_arrow += 1
        elif log.outcome == "loss-out-of-arrows":
            self.losses_out_of_arrows += 1
        else:
            self.losses_unknown += 1


# ---------------------------------------------------------------------------
# Hunter: drives one life of the binary using strategy + belief updates.
# ---------------------------------------------------------------------------
class Hunter:
    def __init__(
        self,
        argv: list[str],
        *,
        strict: bool = False,
        quiet: bool = False,
    ) -> None:
        self.argv: list[str] = argv
        self.strict: bool = strict
        self.quiet: bool = quiet
        # I/O echoes by default; --quiet hides them.
        self.log_io: bool = not quiet
        self.proc: GameProcess = GameProcess(argv)
        self.stats: SessionStats = SessionStats()
        self.belief: Belief = frozenset()
        self.visited: set[int] = set()
        self.tried_by_state: dict[StateKey, set[Action]] = {}
        self.trail: list[str] = []
        self.game_moves: int = 0
        self.game_shots: int = 0

    # ---- I/O helpers ----
    def _read(self) -> Block:
        text = self.proc.read_until_prompt()
        if self.log_io:
            _ = sys.stdout.write(text)
            _ = sys.stdout.flush()
        return parse_block(text)

    def _send(self, value: str) -> None:
        if self.log_io:
            _ = sys.stdout.write(f"{value}\n")
            _ = sys.stdout.flush()
        self.proc.write_line(value)

    # ---- session entry point ----
    def play_session(self, *, target_wins: int, max_games: int) -> None:
        self.proc.start()
        try:
            blk = self._drain_to_opening()
            while self.stats.wins < target_wins and self.stats.games < max_games:
                next_blk = self._step(blk)
                if next_blk is None:
                    break
                blk = next_blk
        finally:
            self.proc.stop()

    def _drain_to_opening(self) -> Block:
        """Read through any pre-game prompts and start the next game."""
        blk = self._read()
        if blk.prompt == "INSTRUCTIONS (Y-N)":
            self._send("N")
            blk = self._read()
        while blk.prompt == "TYPE AN E THEN RETURN":
            self._send("E")
            blk = self._read()
        self._begin_game(blk)
        return blk

    def _hard_restart(self) -> Block:
        """Tear the binary down and bring up a fresh process."""
        self.proc.stop()
        self.proc.start()
        return self._drain_to_opening()

    # ---- game lifecycle ----
    def _begin_game(self, blk: Block) -> None:
        if blk.obs is None:
            raise RuntimeError(
                f"could not parse opening observation; got:\n{blk.raw!r}"
            )
        self.belief = initial_belief(blk.obs)
        self.visited = {blk.obs.room}
        self.tried_by_state = {}
        self.trail = [f"start in {blk.obs.room}"]
        self.game_moves = 0
        self.game_shots = 0

    def _step(self, blk: Block) -> Block | None:
        if blk.terminal:
            return self._handle_terminal(blk)
        prompt = blk.prompt
        if prompt == "SHOOT OR MOVE (S-M)":
            return self._on_choose()
        if prompt == "SAME SETUP (Y-N)":
            return self._handle_terminal(blk)
        if prompt == "INSTRUCTIONS (Y-N)":
            self._send("N")
            return self._read()
        if prompt == "TYPE AN E THEN RETURN":
            self._send("E")
            return self._read()
        # Unknown prompt; try to nudge with a blank line.
        if not self.quiet:
            _ = sys.stderr.write(f"[unknown prompt: {prompt!r}]\n")
        self._send("")
        return self._read()

    def _on_choose(self) -> Block | None:
        if not self.belief:
            # Belief collapsed — our model and the binary diverged. Record
            # a loss, hard-restart the binary, and continue with a fresh game.
            self._record_early_loss("loss-unknown", note="empty belief")
            return self._hard_restart()
        state_key = (current_player(self.belief), self.game_shots, self.belief)
        tried = self.tried_by_state.setdefault(state_key, set())
        visited = frozenset(self.visited)
        tried_frozen = frozenset(tried)
        action = choose_action(
            self.belief,
            strict=self.strict,
            visited=visited,
            tried=tried_frozen,
        )
        if action is None and not self.strict:
            action = best_desperation_action(
                self.belief,
                visited=visited,
                tried=tried_frozen,
            )
        if action is None:
            reason = (
                "strict: no guaranteed-safe or guaranteed-winning action"
                if self.strict
                else "no untried action for exact belief state"
            )
            # End the session cleanly so stats reflect that the model/strategy
            # reached a public state it cannot advance from without repeating.
            if not self.quiet:
                _ = sys.stderr.write(f"[{reason}]\n")
            self._record_early_loss("loss-unknown", note=reason)
            return None
        tried.add(action)
        if isinstance(action, Shot):
            self.game_shots += 1
            self.trail.append(f"shoot {fmt_shot_path(action.path)}")
            return self._dispatch_shot(action)
        # Move
        self.game_moves += 1
        self.trail.append(f"move {action.room}")
        return self._dispatch_move(action.room)

    # ---- action dispatch ----
    def _dispatch_move(self, target_room: int) -> Block:
        self._send("M")
        blk = self._read()
        if blk.terminal:
            return blk
        if blk.prompt == "WHERE TO":
            self._send(str(target_room))
            blk = self._read()
        if blk.terminal:
            return blk
        if blk.obs is None:
            return blk
        if blk.bat_snatch:
            self.belief = update_on_snatch(self.belief, target_room, blk.obs)
        else:
            self.belief = update_on_move(self.belief, target_room, blk.obs)
        self.visited.add(blk.obs.room)
        return blk

    def _dispatch_shot(self, shot: Shot) -> Block:
        self._send("S")
        blk = self._read()
        if blk.terminal:
            return blk
        if blk.prompt == "NO. OF ROOMS (1-5)":
            self._send(str(len(shot.path)))
            blk = self._read()
            if blk.terminal:
                return blk
        for room in shot.path:
            if blk.prompt != "ROOM #":
                break
            self._send(str(room))
            blk = self._read()
            if blk.terminal:
                return blk
        if blk.terminal:
            return blk
        if blk.obs is None:
            return blk
        if blk.miss:
            self.belief = update_on_miss(self.belief, shot.path, blk.obs)
        else:
            # Either we'll see victory next (handled above) or the binary
            # printed a fresh obs after the shot resolved with no kill +
            # no migration. Refilter to be safe.
            self.belief = filter_belief(self.belief, blk.obs)
        self.visited.add(blk.obs.room)
        return blk

    # ---- terminal handling ----
    def _outcome_from(self, blk: Block) -> Outcome:
        if blk.victory:
            return "win"
        if blk.death_pit:
            return "loss-pit"
        if blk.death_wumpus:
            return "loss-wumpus"
        if blk.death_arrow:
            return "loss-arrow"
        if blk.out_of_arrows:
            return "loss-out-of-arrows"
        if blk.lose_banner and self.game_shots >= 5:
            return "loss-out-of-arrows"
        return "loss-unknown"

    def _handle_terminal(self, blk: Block) -> Block | None:
        outcome = self._outcome_from(blk)
        self.stats.record(
            GameLog(
                outcome=outcome,
                moves=self.game_moves,
                shots=self.game_shots,
                trail=self.trail,
            )
        )
        if not self.quiet:
            tail = ", ".join(self.trail[-6:])
            print(f"[game {self.stats.games:4d}] {outcome:18s}  trail: …{tail}")

        # Advance to SAME SETUP prompt.
        deadline = time.monotonic() + SETUP_PROMPT_DEADLINE_S
        while blk.prompt != "SAME SETUP (Y-N)":
            if time.monotonic() > deadline or not blk.raw:
                break
            blk = self._read()
        if blk.prompt == "SAME SETUP (Y-N)":
            self._send("N")
            blk = self._read()
            self._begin_game(blk)
            return blk
        # Process wedged; restart binary fresh.
        return self._hard_restart()

    def _record_early_loss(self, outcome: Outcome, *, note: str) -> None:
        """Record a game that ended before reaching a terminal block."""
        self.stats.record(
            GameLog(
                outcome=outcome,
                moves=self.game_moves,
                shots=self.game_shots,
                trail=self.trail + [f"[{note}]"],
            )
        )
        if not self.quiet:
            tail = ", ".join(self.trail[-6:])
            print(
                f"[game {self.stats.games:4d}] {outcome:18s}  trail: …{tail} ({note})"
            )


# ---------------------------------------------------------------------------
# CLI entry.
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CliArgs:
    binary: str
    seed: int | None
    games: int
    target_wins: int
    quiet: bool
    strict: bool


def _parse_cli(argv: list[str] | None) -> CliArgs:
    p = argparse.ArgumentParser(
        description="Wumpus Hunter — fair fresh-game autoplayer for /tmp/wumpus.",
    )
    _ = p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Pass -s SEED to the binary (deterministic opening)",
    )
    _ = p.add_argument(
        "--games",
        type=int,
        default=100,
        help="Max games to play (default: 100)",
    )
    _ = p.add_argument(
        "--target-wins",
        type=int,
        default=1,
        help="Stop after this many wins (default: 1)",
    )
    _ = p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress game I/O; print only the final summary",
    )
    _ = p.add_argument(
        "--strict",
        action="store_true",
        help="Refuse to act without certainty (no risk-min, no speculative shots)",
    )
    # Advanced / debug.
    _ = p.add_argument("--binary", default=DEFAULT_BINARY, help=argparse.SUPPRESS)
    ns = p.parse_args(argv)
    return CliArgs(
        binary=cast(str, ns.binary),
        seed=cast(int | None, ns.seed),
        games=cast(int, ns.games),
        target_wins=cast(int, ns.target_wins),
        quiet=cast(bool, ns.quiet),
        strict=cast(bool, ns.strict),
    )


def build_argv(binary: str, seed: int | None) -> list[str]:
    argv: list[str] = [binary]
    if seed is not None:
        argv += ["-s", str(seed)]
    return argv


def print_summary(stats: SessionStats) -> None:
    print()
    print(f"games   : {stats.games}")
    print(f"wins    : {stats.wins}")
    loss_line = (
        f"losses  : pit={stats.losses_pit} "
        + f"wumpus={stats.losses_wumpus} "
        + f"arrow={stats.losses_arrow} "
        + f"out-of-arrows={stats.losses_out_of_arrows} "
        + f"unknown={stats.losses_unknown}"
    )
    print(loss_line)
    if stats.games:
        print(f"win rate: {stats.wins / stats.games * 100:.1f}%")
    print(f"moves   : {stats.total_moves}")
    print(f"shots   : {stats.total_shots}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_cli(argv)
    hunter = Hunter(
        argv=build_argv(args.binary, args.seed),
        strict=args.strict,
        quiet=args.quiet,
    )
    hunter.play_session(target_wins=args.target_wins, max_games=args.games)
    print_summary(hunter.stats)
    return 0 if hunter.stats.wins >= args.target_wins else 1


if __name__ == "__main__":
    raise SystemExit(main())
