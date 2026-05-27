"""belief.py — data types, parsing, and world enumeration.

Pure library (no I/O, no `__main__`). Models a single playable cave plus
the set of hidden-state hypotheses consistent with what the player has
observed so far.

The game is the 1972 Yob `Hunt the Wumpus` (fixed dodecahedron map, 1
wumpus + 2 pits + 2 super-bats placed in distinct rooms). All
observations come from textual output. A `World` is one fully-specified
hidden state (player position + wumpus + pits + bats). A belief state is
a `frozenset[World]`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Cave: fixed dodecahedron from the 1972 Yob BASIC source. Verified
# against the user's gameplay transcript (1→{2,5,8}, 8→{1,7,9}, etc.).
# ---------------------------------------------------------------------------
ROOM_COUNT: Final[int] = 20
ROOMS: Final[frozenset[int]] = frozenset(range(1, ROOM_COUNT + 1))

GRAPH: Final[dict[int, tuple[int, int, int]]] = {
    1: (2, 5, 8),
    2: (1, 3, 10),
    3: (2, 4, 12),
    4: (3, 5, 14),
    5: (1, 4, 6),
    6: (5, 7, 15),
    7: (6, 8, 17),
    8: (1, 7, 9),
    9: (8, 10, 18),
    10: (2, 9, 11),
    11: (10, 12, 19),
    12: (3, 11, 13),
    13: (12, 14, 20),
    14: (4, 13, 15),
    15: (6, 14, 16),
    16: (15, 17, 20),
    17: (7, 16, 18),
    18: (9, 17, 19),
    19: (11, 18, 20),
    20: (13, 16, 19),
}


# ---------------------------------------------------------------------------
# Game text. Anything we match against output goes here so the driver and
# strategy can share definitions.
# ---------------------------------------------------------------------------
WIN_TEXT: Final[str] = "AHA! YOU GOT THE WUMPUS"
WUMPUS_DEATH_TEXT: Final[str] = "TSK TSK TSK"
PIT_DEATH_TEXT: Final[str] = "FELL IN PIT"
PIT_SHRIEK_TEXT: Final[str] = "YYYYIIIIEEEE"
ARROW_SELF_TEXT: Final[str] = "OUCH! ARROW GOT YOU"
OUT_OF_ARROWS_TEXT: Final[str] = "OUT OF ARROWS"
LOSE_BANNER_TEXT: Final[str] = "HA HA HA - YOU LOSE"
BAT_SNATCH_TEXT: Final[str] = "BAT SNATCH"
MISSED_TEXT: Final[str] = "MISSED"

SMELL_LINE: Final[str] = "I SMELL A WUMPUS!"
DRAFT_LINE: Final[str] = "I FEEL A DRAFT"
BATS_LINE: Final[str] = "BATS NEARBY!"


# ---------------------------------------------------------------------------
# Observations: one block of game-printed warnings + room + tunnels.
# Drafts and bat warnings are *counted* — the binary prints one per
# adjacent hazard, so room 1 next to two pits prints "I FEEL A DRAFT"
# twice. Wumpus is singular, so smell is a flag.
# ---------------------------------------------------------------------------
_OBS_RE: Final[re.Pattern[str]] = re.compile(
    (
        r"(?P<warnings>(?:(?:I SMELL A WUMPUS!|I FEEL A DRAFT|BATS NEARBY!)\r?\n)*)"
        + r"YOU ARE IN ROOM (?P<room>\d+)\r?\n"
        + r"TUNNELS LEAD TO (?P<a>\d+) (?P<b>\d+) (?P<c>\d+)"
    )
)


@dataclass(frozen=True, slots=True)
class Observation:
    room: int
    tunnels: tuple[int, int, int]
    wumpus_near: bool
    pit_near_count: int
    bat_near_count: int

    def warning_summary(self) -> str:
        parts: list[str] = []
        if self.wumpus_near:
            parts.append("wumpus")
        if self.pit_near_count:
            parts.append(f"{self.pit_near_count} pit")
        if self.bat_near_count:
            parts.append(f"{self.bat_near_count} bat")
        return ", ".join(parts) if parts else "none"


def parse_observation(text: str) -> Observation | None:
    """Parse the LAST complete observation block from `text`.

    The game prints a fresh observation block after each successful action,
    and we always want the most recent one (bat teleport can produce two
    in a single output block).
    """
    matches = list(_OBS_RE.finditer(text.replace("\r\n", "\n")))
    if not matches:
        return None
    match = matches[-1]
    warnings = match.group("warnings").splitlines()
    return Observation(
        room=int(match.group("room")),
        tunnels=(
            int(match.group("a")),
            int(match.group("b")),
            int(match.group("c")),
        ),
        wumpus_near=SMELL_LINE in warnings,
        pit_near_count=warnings.count(DRAFT_LINE),
        bat_near_count=warnings.count(BATS_LINE),
    )


# ---------------------------------------------------------------------------
# Actions.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Move:
    room: int


@dataclass(frozen=True, slots=True)
class Shot:
    path: tuple[int, ...]


Action = Move | Shot


# ---------------------------------------------------------------------------
# World = one fully-specified hidden state.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class World:
    player: int
    wumpus: int
    pits: tuple[int, int]  # sorted ascending
    bats: tuple[int, int]  # sorted ascending

    def with_player(self, room: int) -> World:
        return World(room, self.wumpus, self.pits, self.bats)

    def with_wumpus(self, room: int) -> World:
        return World(self.player, room, self.pits, self.bats)


Belief = frozenset[World]


def observation_for(world: World) -> Observation:
    neighbors = GRAPH[world.player]
    return Observation(
        room=world.player,
        tunnels=neighbors,
        wumpus_near=world.wumpus in neighbors,
        pit_near_count=sum(1 for pit in world.pits if pit in neighbors),
        bat_near_count=sum(1 for bat in world.bats if bat in neighbors),
    )


def matches_observation(world: World, obs: Observation) -> bool:
    return observation_for(world) == obs


# ---------------------------------------------------------------------------
# Initial belief: every assignment of (wumpus, 2 pits, 2 bats) into 5
# distinct non-player rooms that matches the opening observation.
#
# Hazard counts are: 1 wumpus + 2 pits + 2 bats = 5 distinct rooms,
# placed uniformly in the 19 non-player rooms. That gives 19 * C(18,2) *
# C(16,2) = 19 * 153 * 120 = 348,840 raw assignments. Filtering by the
# opening observation typically cuts this to a few thousand or fewer.
# ---------------------------------------------------------------------------
def initial_belief(obs: Observation) -> Belief:
    if obs.tunnels != GRAPH[obs.room]:
        raise ValueError(
            f"room {obs.room} tunnels {obs.tunnels} do not match known cave"
        )

    worlds: set[World] = set()
    available = sorted(r for r in ROOMS if r != obs.room)
    for wumpus in available:
        pit_pool = [r for r in available if r != wumpus]
        for i, pit_a in enumerate(pit_pool):
            for pit_b in pit_pool[i + 1 :]:
                pits = (pit_a, pit_b)  # already sorted
                bat_pool = [r for r in pit_pool if r != pit_a and r != pit_b]
                for j, bat_a in enumerate(bat_pool):
                    for bat_b in bat_pool[j + 1 :]:
                        bats = (bat_a, bat_b)
                        world = World(obs.room, wumpus, pits, bats)
                        if matches_observation(world, obs):
                            worlds.add(world)
    return frozenset(worlds)


def filter_belief(belief: Iterable[World], obs: Observation) -> Belief:
    return frozenset(w for w in belief if matches_observation(w, obs))


# ---------------------------------------------------------------------------
# Belief inspection: aggregate over the world set.
# ---------------------------------------------------------------------------
def possible_wumpus_rooms(belief: Belief) -> frozenset[int]:
    return frozenset(w.wumpus for w in belief)


def possible_pit_rooms(belief: Belief) -> frozenset[int]:
    rooms: set[int] = set()
    for w in belief:
        rooms.update(w.pits)
    return frozenset(rooms)


def possible_bat_rooms(belief: Belief) -> frozenset[int]:
    rooms: set[int] = set()
    for w in belief:
        rooms.update(w.bats)
    return frozenset(rooms)


def fmt_rooms(rooms: Iterable[int]) -> str:
    ordered = sorted(rooms)
    return "{" + ", ".join(str(r) for r in ordered) + "}" if ordered else "{}"


def fmt_shot_path(path: tuple[int, ...]) -> str:
    return "-".join(str(r) for r in path)


def describe_belief(belief: Belief) -> str:
    if not belief:
        return "0 possible worlds"
    return "\n".join(
        [
            f"{len(belief)} possible worlds",
            f"player rooms: {fmt_rooms({w.player for w in belief})}",
            f"possible wumpus rooms: {fmt_rooms(possible_wumpus_rooms(belief))}",
            f"possible pit rooms: {fmt_rooms(possible_pit_rooms(belief))}",
            f"possible bat rooms: {fmt_rooms(possible_bat_rooms(belief))}",
        ]
    )


def current_player(belief: Belief) -> int:
    """Player room, asserting the belief is internally consistent."""
    players = {w.player for w in belief}
    if len(players) != 1:
        raise ValueError(f"belief has multiple player rooms: {sorted(players)}")
    return next(iter(players))
