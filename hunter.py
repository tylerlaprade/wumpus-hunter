#!/usr/bin/env python3
"""hunter.py - fair fresh-game autoplayer for /tmp/wumpus.

Companion to Codex's `wumpus_hunter.py`. That file holds the precise
world-enumeration belief model and a strict-fair player. This file
adds the pieces strict-mode doesn't cover for one-life/no-cheating play:

  - Risk-minimizing fallback when no guaranteed-safe move exists, using
    expected-death probability computed over the live world set.
  - Belief updates for events strict-mode skips:
      * wumpus migration after a missed shot
      * bat snatch (target room confirmed-bat, player teleported)
      * entering the wumpus's room and surviving (he moved away)
  - Full game lifecycle: handle every prompt, answer SAME SETUP=N after
    death to discard the failed hidden map, and keep trying fresh games
    until we win.
  - Stats across games (wins, loss reasons, moves, shots).

Usage:
    python3 hunter.py [--target-wins K] [--max-games N] [--seed S] [-v]
"""

from __future__ import annotations

import argparse
import os
import pty
import select
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

import wumpus_hunter as wh
from wumpus_hunter import (
    GRAPH,
    Move,
    Observation,
    Shot,
    World,
    action_to_inputs,
    enumerate_shots,
    filter_worlds,
    guaranteed_safe_moves,
    guaranteed_winning_shots,
    initial_worlds,
    matches_observation,
    observation_for,
)


# ---------------------------------------------------------------------------
# Game messages. WIN/LOSE strings come from Codex's file; SUPER BAT and the
# prompts I confirmed from the binary or the user's transcript.
# ---------------------------------------------------------------------------
WIN_TEXT = "AHA! YOU GOT THE WUMPUS"
WUMPUS_DEATH = "TSK TSK TSK"          # wumpus ate us
PIT_DEATH = "FELL IN PIT"             # also "YYYYIIIIEEEE"
ARROW_SELF = "OUCH! ARROW GOT YOU"    # we shot ourselves
OUT_OF_ARROWS = "OUT OF ARROWS"
LOSE_BANNER = "HA HA HA - YOU LOSE"
BAT_SNATCH = "BAT SNATCH"             # appears inside "ZAP--SUPER BAT SNATCH!"
MISSED = "MISSED"

PROMPT_TOKENS = (
    "INSTRUCTIONS (Y-N)",
    "SHOOT OR MOVE (S-M)",
    "WHERE TO",
    "NO. OF ROOMS (1-5)",
    "ROOM #",
    "SAME SETUP (Y-N)",
    "TYPE AN E THEN RETURN",
)


# ---------------------------------------------------------------------------
# Driver: spawn the binary on a pty so the prompt char "?" flushes promptly.
# ---------------------------------------------------------------------------
class GameProcess:
    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.pid: int | None = None
        self.fd: int | None = None

    def start(self) -> None:
        pid, fd = pty.fork()
        if pid == 0:
            try:
                os.execv(self.argv[0], self.argv)
            except Exception:
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
            os.waitpid(self.pid, 0)
        except ChildProcessError:
            pass
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.pid = None
        self.fd = None

    def read_until_prompt(self, overall_timeout: float = 5.0, quiet_window: float = 0.08) -> str:
        """Read until output ends in `?` and stays quiet briefly, or proc exits."""
        assert self.fd is not None
        deadline = time.monotonic() + overall_timeout
        out = b""
        last_byte_t = time.monotonic()
        while True:
            now = time.monotonic()
            if now > deadline:
                break
            timeout = min(quiet_window, max(0.0, deadline - now))
            r, _, _ = select.select([self.fd], [], [], timeout)
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
        assert self.fd is not None
        os.write(self.fd, (value + "\n").encode("ascii"))


# ---------------------------------------------------------------------------
# Block-level parser: detect prompts, warnings, terminal events.
# ---------------------------------------------------------------------------
@dataclass
class Block:
    raw: str
    upper: str = ""
    obs: Observation | None = None
    prompt: str | None = None
    bat_snatch: bool = False
    miss: bool = False
    death_pit: bool = False
    death_wumpus: bool = False
    death_arrow: bool = False
    out_of_arrows: bool = False
    victory: bool = False

    @property
    def terminal(self) -> bool:
        return (
            self.victory
            or self.death_pit
            or self.death_wumpus
            or self.death_arrow
            or self.out_of_arrows
        )


def parse_block(text: str) -> Block:
    blk = Block(raw=text, upper=text.upper())
    u = blk.upper

    blk.obs = Observation.parse_latest(text)
    if BAT_SNATCH in u:
        blk.bat_snatch = True
    if MISSED in u:
        blk.miss = True
    if WIN_TEXT in u:
        blk.victory = True
    if WUMPUS_DEATH in u:
        blk.death_wumpus = True
    if PIT_DEATH in u or "YYYYIIIIEEEE" in u:
        blk.death_pit = True
    if ARROW_SELF in u:
        blk.death_arrow = True
    if OUT_OF_ARROWS in u:
        blk.out_of_arrows = True

    last_prompt = None
    last_pos = -1
    for tok in PROMPT_TOKENS:
        pos = u.rfind(tok)
        if pos > last_pos:
            last_pos = pos
            last_prompt = tok
    blk.prompt = last_prompt
    return blk


# ---------------------------------------------------------------------------
# Belief updates beyond strict mode.
# ---------------------------------------------------------------------------
def update_on_move(
    worlds: set[World],
    target: int,
    post: Observation,
) -> set[World]:
    """Player moved to `target` (no death, no snatch). Update world set."""
    out: set[World] = set()
    for w in worlds:
        if target in w.pits:
            continue  # would've died; not consistent with being alive
        if target in w.bats:
            continue  # would've been snatched (different code path)
        if target == w.wumpus:
            # Wumpus woke. He stayed (P=0.25 → we'd die) or moved (P=0.75
            # → survived). Since we survived, branch over neighbors.
            for new_w in GRAPH[w.wumpus]:
                nw = World(target, new_w, w.pits, w.bats)
                if matches_observation(nw, post):
                    out.add(nw)
        else:
            nw = World(target, w.wumpus, w.pits, w.bats)
            if matches_observation(nw, post):
                out.add(nw)
    return out


def update_on_snatch(
    worlds: set[World],
    snatch_target: int,
    post: Observation,
) -> set[World]:
    """Player moved to `snatch_target` (a bat room), then teleported."""
    out: set[World] = set()
    new_player = post.room
    for w in worlds:
        if snatch_target not in w.bats:
            continue
        if new_player in w.pits:
            continue  # would've died
        if new_player == w.wumpus:
            # Teleport into wumpus room wakes him; he moved away (we survived).
            for new_w in GRAPH[w.wumpus]:
                nw = World(new_player, new_w, w.pits, w.bats)
                if matches_observation(nw, post):
                    out.add(nw)
        else:
            nw = World(new_player, w.wumpus, w.pits, w.bats)
            if matches_observation(nw, post):
                out.add(nw)
    return out


def update_on_miss(
    worlds: set[World],
    shot_path: tuple[int, ...],
    post: Observation,
) -> set[World]:
    """Shot at `shot_path` missed. Wumpus may have moved one step."""
    out: set[World] = set()
    player_room = post.room
    for w in worlds:
        if w.wumpus in shot_path:
            continue  # path would've killed him; inconsistent with miss
        for new_w in (w.wumpus, *GRAPH[w.wumpus]):
            if new_w == player_room:
                continue  # he'd have eaten us
            nw = World(w.player, new_w, w.pits, w.bats)
            if matches_observation(nw, post):
                out.add(nw)
    return out


# ---------------------------------------------------------------------------
# Risk-minimizing fallback policy.
# ---------------------------------------------------------------------------
PIT_PENALTY = 1.0        # certain death
WUMPUS_STAY = 0.25       # P(wumpus stays when we walk in)
BAT_CHAIN_DEATH = 1 / 8  # p = 2/20 + (1/20 * 1/4) + (2/20 * p)


def death_prob_on_move(worlds: set[World], target: int) -> float:
    """One-step expected loss probability for moving to `target`."""
    if not worlds:
        return 1.0
    risk = 0.0
    for w in worlds:
        if target in w.pits:
            risk += PIT_PENALTY
        elif target == w.wumpus:
            risk += WUMPUS_STAY
        elif target in w.bats:
            risk += BAT_CHAIN_DEATH
    return risk / len(worlds)


Action = Move | Shot
StateKey = tuple[int, frozenset[World]]


def best_risky_move(worlds: set[World], player: int, visited: set[int], tried: set[Action]) -> int:
    nbrs = [room for room in GRAPH[player] if Move(room) not in tried]
    if not nbrs:
        nbrs = list(GRAPH[player])
    scored = sorted(nbrs, key=lambda r: (death_prob_on_move(worlds, r), r in visited, r))
    return scored[0]


def best_speculative_shot(worlds: set[World], player: int, tried: set[Action]) -> Shot | None:
    """If a shot is *expected* better than the best move, return it.

    A shot is judged by P(kill) and P(post-miss death). Only consider when
    the wumpus candidate set is small enough that we have a real chance.
    """
    candidates = {w.wumpus for w in worlds}
    if not (1 < len(candidates) <= 6):
        return None
    # Best shot = the one maximizing the share of worlds in which path hits wumpus.
    best_shot: Shot | None = None
    best_kill = 0.0
    best_loss = 1.0
    for shot in enumerate_shots(player):
        if shot in tried:
            continue
        hit = sum(1 for w in worlds if w.wumpus in shot.path)
        if hit == 0:
            continue
        kill_p = hit / len(worlds)
        # On miss the wumpus may walk into us with ~0.25 per world where
        # player is adjacent to wumpus.
        miss_worlds = [w for w in worlds if w.wumpus not in shot.path]
        miss_death = (
            sum(0.25 for w in miss_worlds if player in GRAPH[w.wumpus])
            / max(1, len(worlds))
        )
        # Expected loss = (1 - kill_p) * 0 (we keep playing) but the miss may kill us
        # AND we burn an arrow. Approximate: expected_loss = miss_death.
        if kill_p > best_kill or (kill_p == best_kill and miss_death < best_loss):
            best_kill = kill_p
            best_loss = miss_death
            best_shot = shot
    if best_shot is None:
        return None
    move_risk = min(death_prob_on_move(worlds, n) for n in GRAPH[player])
    # Shoot only when expected-immediate-death of shot < move's, AND kill is
    # plausible. Avoid burning arrows on tiny kill_p.
    if best_kill >= 0.34 and best_loss <= move_risk + 0.02:
        return best_shot
    return None


# ---------------------------------------------------------------------------
# Decide: layered policy.
# ---------------------------------------------------------------------------
def decide(worlds: set[World], player: int, visited: set[int], tried: set[Action]) -> Move | Shot:
    # 1. Guaranteed-winning shot beats anything.
    winners = guaranteed_winning_shots(worlds)
    if winners:
        # prefer shortest path
        winners.sort(key=lambda s: len(s.path))
        return winners[0]
    # 2. Speculative shot if expected value favors it.
    spec = best_speculative_shot(worlds, player, tried)
    if spec is not None:
        return spec
    # 3. Guaranteed-safe move (Codex's choose_move logic — info-gain ranked).
    safe = guaranteed_safe_moves(worlds)
    if safe:
        return _best_safe_move(worlds, safe, visited, tried)
    # 4. Forced gamble.
    return Move(best_risky_move(worlds, player, visited, tried))


def _best_safe_move(worlds: set[World], safe: list[int], visited: set[int], tried: set[Action]) -> Move:
    """Among guaranteed-safe neighbors, pick the most informative.

    Mirrors Codex's choice in `choose_move`: minimize worst-case partition
    size, break ties by maximizing distinct observation classes.
    """
    import collections

    best_room: int | None = None
    best_key = None
    candidates = [room for room in safe if Move(room) not in tried]
    if not candidates:
        candidates = safe
    for room in candidates:
        partitions = collections.Counter(
            observation_for(w.with_player(room)) for w in worlds
        )
        worst = max(partitions.values())
        distinct = len(partitions)
        key = (room in visited, worst, -distinct, room)
        if best_key is None or key < best_key:
            best_key = key
            best_room = room
    assert best_room is not None
    return Move(best_room)


# ---------------------------------------------------------------------------
# Stats + Hunter loop.
# ---------------------------------------------------------------------------
@dataclass
class Stats:
    games: int = 0
    wins: int = 0
    losses_pit: int = 0
    losses_wumpus: int = 0
    losses_arrow: int = 0
    losses_arrows_out: int = 0
    total_moves: int = 0
    total_shots: int = 0
    # Per-game move log (for verbose summary).
    last_game_log: list[str] = field(default_factory=list)


class Hunter:
    def __init__(
        self,
        *,
        argv: list[str],
        verbose: bool = False,
        quiet: bool = False,
        log_io: bool = False,
    ) -> None:
        self.argv = argv
        self.verbose = verbose
        self.quiet = quiet
        self.log_io = log_io
        self.proc = GameProcess(argv)
        self.stats = Stats()
        self.worlds: set[World] = set()
        self.player: int = 0
        self.visited: set[int] = set()
        self.tried_by_state: dict[StateKey, set[Action]] = {}

    # ----- I/O -----
    def _read(self) -> Block:
        text = self.proc.read_until_prompt()
        if self.log_io:
            sys.stdout.write(text)
            sys.stdout.flush()
        return parse_block(text)

    def _send(self, s: str) -> None:
        if self.log_io:
            sys.stdout.write(f"{s}\n")
            sys.stdout.flush()
        self.proc.write_line(s)

    # ----- session loop -----
    def play(self, *, target_wins: int = 1, max_games: int = 1000) -> None:
        self.proc.start()
        blk = self._read()
        if blk.prompt == "INSTRUCTIONS (Y-N)":
            self._send("N")
            blk = self._read()
        # First in-game block: an Observation block + SHOOT OR MOVE prompt.
        self._begin_game(blk)

        while self.stats.wins < target_wins and self.stats.games < max_games:
            blk = self._step(blk)
            if blk is None:
                break
        self.proc.stop()
        self._print_summary()

    def _begin_game(self, blk: Block) -> None:
        if blk.obs is None:
            raise RuntimeError(f"could not parse opening observation:\n{blk.raw!r}")
        self.player = blk.obs.room
        self.visited = {self.player}
        self.tried_by_state = {}
        self.worlds = initial_worlds(blk.obs)
        self.stats.last_game_log = [f"start in {self.player}"]

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
        if self.verbose:
            print(f"[unknown prompt: {prompt!r}]\n{blk.raw!r}", file=sys.stderr)
        self._send("")
        return self._read()

    def _on_choose(self) -> Block:
        state_key = (self.player, frozenset(self.worlds))
        tried = self.tried_by_state.setdefault(state_key, set())
        action = decide(self.worlds, self.player, self.visited, tried)
        tried.add(action)
        inputs = action_to_inputs(action)
        if isinstance(action, Shot):
            self.stats.total_shots += 1
            self.stats.last_game_log.append(f"shoot {action.path}")
            return self._dispatch_shot(action, inputs)
        else:
            self.stats.total_moves += 1
            self.stats.last_game_log.append(f"move {action.room}")
            return self._dispatch_move(action, inputs)

    def _dispatch_move(self, move: Move, inputs: list[str]) -> Block:
        # inputs == ["M", str(room)]
        self._send(inputs[0])
        blk = self._read()
        if blk.terminal:
            return blk
        if blk.prompt == "WHERE TO":
            self._send(inputs[1])
            blk = self._read()
        if blk.terminal:
            # Death on move (pit, or wumpus walked into us, or arrow self —
            # though arrow_self can't happen on move). Belief no longer
            # needed; lifecycle handler will reset.
            return blk
        # Belief update.
        if blk.bat_snatch:
            if blk.obs is None:
                # Shouldn't happen — snatch always shows new room.
                return blk
            self.worlds = update_on_snatch(self.worlds, move.room, blk.obs)
            self.player = blk.obs.room
            self.visited.add(self.player)
        else:
            if blk.obs is None:
                return blk
            self.worlds = update_on_move(self.worlds, move.room, blk.obs)
            self.player = blk.obs.room
            self.visited.add(self.player)
        return blk

    def _dispatch_shot(self, shot: Shot, inputs: list[str]) -> Block:
        # inputs == ["S", str(len), *path]
        self._send(inputs[0])  # "S"
        blk = self._read()
        if blk.terminal:
            return blk
        if blk.prompt == "NO. OF ROOMS (1-5)":
            self._send(inputs[1])
            blk = self._read()
            if blk.terminal:
                return blk
        for room_str in inputs[2:]:
            if blk.prompt == "ROOM #":
                self._send(room_str)
                blk = self._read()
                if blk.terminal:
                    return blk
            else:
                break
        # Post-shot result.
        if blk.victory:
            return blk
        if blk.death_arrow or blk.death_wumpus or blk.out_of_arrows or blk.death_pit:
            return blk
        if blk.miss and blk.obs is not None:
            self.worlds = update_on_miss(self.worlds, shot.path, blk.obs)
            self.player = blk.obs.room
            self.visited.add(self.player)
        elif blk.obs is not None:
            self.player = blk.obs.room
            self.visited.add(self.player)
            self.worlds = filter_worlds(self.worlds, blk.obs)
        return blk

    def _handle_terminal(self, blk: Block) -> Block | None:
        self.stats.games += 1
        if blk.victory:
            self.stats.wins += 1
            tag = "WIN"
        elif blk.death_pit:
            self.stats.losses_pit += 1
            tag = "LOSS pit"
        elif blk.death_wumpus:
            self.stats.losses_wumpus += 1
            tag = "LOSS wumpus"
        elif blk.death_arrow:
            self.stats.losses_arrow += 1
            tag = "LOSS arrow-self"
        elif blk.out_of_arrows:
            self.stats.losses_arrows_out += 1
            tag = "LOSS out-of-arrows"
        else:
            tag = "LOSS unknown"
        if not self.quiet:
            log_tail = ", ".join(self.stats.last_game_log[-6:])
            print(f"[game {self.stats.games:4d}] {tag:20s}  trail: …{log_tail}")

        # Advance to SAME SETUP prompt, answer N, get a fresh observation.
        deadline = time.monotonic() + 3.0
        while blk.prompt != "SAME SETUP (Y-N)":
            if time.monotonic() > deadline or not blk.raw:
                break
            blk = self._read()
        if blk.prompt == "SAME SETUP (Y-N)":
            self._send("N")
            blk = self._read()
            self._begin_game(blk)
            return blk
        # Process wedged; restart fresh.
        self.proc.stop()
        self.proc.start()
        blk = self._read()
        if blk.prompt == "INSTRUCTIONS (Y-N)":
            self._send("N")
            blk = self._read()
        self._begin_game(blk)
        return blk

    def _print_summary(self) -> None:
        s = self.stats
        print()
        print(f"games   : {s.games}")
        print(f"wins    : {s.wins}")
        print(
            "losses  : "
            f"pit={s.losses_pit} wumpus={s.losses_wumpus} "
            f"arrow-self={s.losses_arrow} out-of-arrows={s.losses_arrows_out}"
        )
        if s.games:
            print(f"win rate: {s.wins / s.games * 100:.1f}%")
        print(f"moves   : {s.total_moves}")
        print(f"shots   : {s.total_shots}")


# ---------------------------------------------------------------------------
def build_argv(binary: str, seed: int | None) -> list[str]:
    argv = [binary]
    if seed is not None:
        argv += ["-s", str(seed)]
    return argv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-wins", type=int, default=1)
    p.add_argument("--max-games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None,
                   help="Pass -s SEED to the binary (note: same-setup=N still reseeds across games)")
    p.add_argument("--binary", default="/tmp/wumpus")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--log-io", action="store_true", help="Print raw game I/O")
    args = p.parse_args(argv)

    h = Hunter(
        argv=build_argv(args.binary, args.seed),
        verbose=args.verbose,
        quiet=args.quiet,
        log_io=args.log_io,
    )
    h.play(target_wins=args.target_wins, max_games=args.max_games)
    return 0 if h.stats.wins >= args.target_wins else 1


if __name__ == "__main__":
    sys.exit(main())
