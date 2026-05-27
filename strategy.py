"""strategy.py - decisions and belief updates for dynamic events.

Pure library. Layered strategy:
  1. Guaranteed-winning shot if one exists in every world.
  2. Speculative shot if expected value beats the best move.
  3. Guaranteed-safe move with information-gain tie-breaking.
  4. Risk-minimized forced move.

Risk model (Codex's derivation; pit/wumpus/bat hazards never share a room
by construction):
  - pit      = 1     (certain death)
  - wumpus   = 1/4   (P=0.25 he stays when we walk in)
  - bat      = 1/8   (closed-form snatch-chain death probability):

      p = 2/20 + (1/20)(1/4) + (2/20)·p
        = 1/10 + 1/80 + p/10
      9p/10 = 9/80  →  p = 1/8

Belief updates the strict solver doesn't cover are exported here as
`update_on_*` and used by the driver.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable
from functools import lru_cache
from typing import Final

from belief import (
    GRAPH,
    Action,
    Belief,
    Move,
    Observation,
    Shot,
    World,
    current_player,
    matches_observation,
    observation_for,
    possible_wumpus_rooms,
)

# ---------------------------------------------------------------------------
# Risk constants.
# ---------------------------------------------------------------------------
PIT_RISK: Final[float] = 1.0
WUMPUS_RISK: Final[float] = 0.25
BAT_RISK: Final[float] = 1.0 / 8.0

MAX_ARROW_LEN: Final[int] = 5

_EMPTY_VISITED: Final[frozenset[int]] = frozenset()
_EMPTY_TRIED: Final[frozenset[Action]] = frozenset()


# ---------------------------------------------------------------------------
# Shot enumeration.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def enumerate_shots(player: int, max_len: int = MAX_ARROW_LEN) -> list[Shot]:
    """All arrow paths the binary will accept from `player`.

    Rules from the 1972 BASIC:
      - Each step must be along a tunnel (`GRAPH`).
      - No U-turns: `path[i] != path[i-2]` (where `path[-1]` is implicitly
        the player room). I.e. arrow position 2 can't be the player room,
        position 3 can't equal position 1, etc.
      - We also exclude any path passing through the player's room
        (re-entering kills you with "OUCH! ARROW GOT YOU!").
    """
    shots: list[Shot] = []

    def walk(path: tuple[int, ...]) -> None:
        if path:
            shots.append(Shot(path))
        if len(path) == max_len:
            return
        current = path[-1] if path else player
        # "Two back" relative to the next step is path[-2] if it exists,
        # otherwise the implicit player room.
        two_back = path[-2] if len(path) >= 2 else player
        for nxt in GRAPH[current]:
            if nxt == two_back:
                continue  # U-turn: position i+2 == position i
            if nxt == player:
                continue  # arrow re-entering player's room kills us
            walk(path + (nxt,))

    walk(())
    shots.sort(key=lambda s: (len(s.path), s.path))
    return shots


# ---------------------------------------------------------------------------
# Guaranteed-fair queries.
# ---------------------------------------------------------------------------
def guaranteed_safe_moves(belief: Belief) -> list[int]:
    if not belief:
        return []
    player = current_player(belief)
    safe: list[int] = []
    for room in GRAPH[player]:
        if all(
            room != w.wumpus and room not in w.pits and room not in w.bats
            for w in belief
        ):
            safe.append(room)
    return safe


def guaranteed_winning_shots(belief: Belief) -> list[Shot]:
    if not belief:
        return []
    player = current_player(belief)
    candidates = possible_wumpus_rooms(belief)
    winners: list[Shot] = []
    for shot in enumerate_shots(player):
        path_set = set(shot.path)
        if candidates <= path_set:
            winners.append(shot)
    return winners


# ---------------------------------------------------------------------------
# Risk computation.
# ---------------------------------------------------------------------------
def death_prob_on_move(belief: Belief, target: int) -> float:
    """Expected probability of losing the game by moving to `target` *now*.

    One-step lookahead. Hazards are placed in distinct rooms initially,
    but the wumpus can migrate into a bat or pit room after a wake — so
    wumpus and bat risks are accumulated, not exclusive. Pit risk caps
    at 1.0 (certain death).
    """
    if not belief:
        return 1.0
    total = 0.0
    for w in belief:
        if target in w.pits:
            total += PIT_RISK
            continue
        risk = 0.0
        if target == w.wumpus:
            risk += WUMPUS_RISK
        if target in w.bats:
            risk += BAT_RISK
        total += risk
    return total / len(belief)


def best_risky_move(
    belief: Belief,
    player: int,
    *,
    visited: frozenset[int] = _EMPTY_VISITED,
    tried: frozenset[Action] = _EMPTY_TRIED,
) -> Move | None:
    nbrs = [room for room in GRAPH[player] if Move(room) not in tried]
    if not nbrs:
        return None
    # Sort by (risk, room) so ties break by lowest room number for
    # determinism.
    ranked = sorted(nbrs, key=lambda r: (death_prob_on_move(belief, r), r in visited, r))
    return Move(ranked[0])


# ---------------------------------------------------------------------------
# Speculative shot.
# ---------------------------------------------------------------------------
def _shot_kill_prob(belief: Belief, shot: Shot) -> float:
    if not belief:
        return 0.0
    path = set(shot.path)
    hits = sum(1 for w in belief if w.wumpus in path)
    return hits / len(belief)


def _shot_miss_death_prob(belief: Belief, shot: Shot, player: int) -> float:
    """Approximate P(wumpus walks into us next turn | shot missed).

    On miss the wumpus wakes and with P=0.75 moves uniformly to a
    neighbor (else stays). For each missed-world: P(wumpus -> player) =
    0.75/3 = 0.25 if `player in GRAPH[w.wumpus]`, else 0.
    """
    if not belief:
        return 0.0
    miss_worlds = [w for w in belief if w.wumpus not in shot.path]
    if not miss_worlds:
        return 0.0
    contrib = sum(0.25 for w in miss_worlds if player in GRAPH[w.wumpus])
    return contrib / len(belief)


def best_speculative_shot(
    belief: Belief,
    *,
    min_kill_prob: float = 0.34,
    tried: frozenset[Action] = _EMPTY_TRIED,
) -> Shot | None:
    """Return a shot whose expected outcome beats any move, or None."""
    if not belief:
        return None
    player = current_player(belief)
    untried_moves = [room for room in GRAPH[player] if Move(room) not in tried]
    if not untried_moves:
        return None

    best: Shot | None = None
    best_kill = 0.0
    best_loss = 1.0
    for shot in enumerate_shots(player):
        if shot in tried:
            continue
        kill_p = _shot_kill_prob(belief, shot)
        if kill_p < min_kill_prob:
            continue
        loss_p = _shot_miss_death_prob(belief, shot, player)
        key_better = (kill_p, -loss_p) > (best_kill, -best_loss)
        if key_better:
            best = shot
            best_kill = kill_p
            best_loss = loss_p

    if best is None:
        return None

    # Only shoot when the shot's expected immediate-death risk doesn't
    # exceed the best move's. (We also burn an arrow; the kill chance
    # makes that worth it.)
    move_risk = min(death_prob_on_move(belief, n) for n in untried_moves)
    if best_loss <= move_risk + 0.02:
        return best
    return None


def best_desperation_action(
    belief: Belief,
    *,
    visited: frozenset[int] = _EMPTY_VISITED,
    tried: frozenset[Action] = _EMPTY_TRIED,
) -> Action | None:
    """Choose a fair action that forces progress in a long live game.

    This is intentionally more aggressive than `choose_action`: it still
    uses only public belief, but if the normal strategy has wandered too long,
    it shoots the path with the highest current kill probability. A miss
    wakes the wumpus and consumes an arrow, so repeated use will eventually
    produce either a win, a changed belief state, or a terminal loss. Returns
    None if this exact belief state has no untried useful shot or move left.
    """
    if not belief:
        raise ValueError("best_desperation_action called with empty belief")

    player = current_player(belief)
    best_shot: Shot | None = None
    best_key: tuple[float, float, int, tuple[int, ...]] | None = None
    for shot in enumerate_shots(player):
        if shot in tried:
            continue
        kill_p = _shot_kill_prob(belief, shot)
        if kill_p <= 0:
            continue
        miss_death_p = _shot_miss_death_prob(belief, shot, player)
        key = (kill_p, -miss_death_p, -len(shot.path), tuple(-r for r in shot.path))
        if best_key is None or key > best_key:
            best_key = key
            best_shot = shot
    if best_shot is not None:
        return best_shot
    return best_risky_move(belief, player, visited=visited, tried=tried)


# ---------------------------------------------------------------------------
# Move strategy with info-gain tie-breaker.
# ---------------------------------------------------------------------------
def best_safe_move(
    belief: Belief,
    safe: Iterable[int],
    *,
    visited: frozenset[int] = _EMPTY_VISITED,
    tried: frozenset[Action] = _EMPTY_TRIED,
) -> Move | None:
    """Pick the most informative guaranteed-safe move.

    Criterion: minimize worst-case partition size after the next
    observation (smallest belief subset in the worst case = most
    information). Tie-break by maximizing distinct observation classes,
    then by lowest room number.
    """
    best_room: int | None = None
    best_key: tuple[bool, int, int, int] | None = None
    safe_rooms = list(safe)
    candidates = [room for room in safe_rooms if Move(room) not in tried]
    if not candidates:
        return None
    for room in candidates:
        partitions: collections.Counter[Observation] = collections.Counter(
            observation_for(w.with_player(room)) for w in belief
        )
        worst = max(partitions.values())
        distinct = len(partitions)
        key = (room in visited, worst, -distinct, room)
        if best_key is None or key < best_key:
            best_key = key
            best_room = room
    return Move(best_room) if best_room is not None else None


# ---------------------------------------------------------------------------
# Layered strategy.
# ---------------------------------------------------------------------------
def choose_action(
    belief: Belief,
    *,
    strict: bool = False,
    visited: frozenset[int] = _EMPTY_VISITED,
    tried: frozenset[Action] = _EMPTY_TRIED,
) -> Action | None:
    """Return the action to play, or None if strict mode and no certainty.

    Layers:
      1. Guaranteed-winning shot (shortest path, lexicographic tiebreak).
      2. Guaranteed-safe move (most informative).
      3. (non-strict) Speculative shot if EV favors the best risky move.
      4. (non-strict) Lowest-risk move.
    """
    if not belief:
        return None
    winners = [shot for shot in guaranteed_winning_shots(belief) if shot not in tried]
    if winners:
        return winners[0]  # already sorted shortest-first by enumerate

    safe = guaranteed_safe_moves(belief)
    if safe:
        safe_move = best_safe_move(belief, safe, visited=visited, tried=tried)
        if safe_move is not None:
            return safe_move

    if strict:
        return None

    spec = best_speculative_shot(belief, tried=tried)
    if spec is not None:
        return spec

    return best_risky_move(
        belief,
        current_player(belief),
        visited=visited,
        tried=tried,
    )


# ---------------------------------------------------------------------------
# Belief updates for dynamic events.
#
# The strict belief filter handles "move to a safe room, observe new
# warnings, narrow worlds". The three updates below handle events that
# either resample player position or move the wumpus.
# ---------------------------------------------------------------------------
def update_on_move(belief: Belief, target: int, post: Observation) -> Belief:
    """We moved to `target`, survived, observed `post`.

    Drops worlds where the move would have killed us. For worlds where
    target is the wumpus's room: he woke up; the only surviving branch
    is "he moved to a neighbor of his old room" (one of 3 neighbors,
    each filtered by the observation).
    """
    out: set[World] = set()
    for w in belief:
        if target in w.pits:
            continue  # would have died
        if target in w.bats:
            continue  # would have been snatched (separate update path)
        if target == w.wumpus:
            # He stayed → we died (filtered). He moved → we survived; branch.
            for new_w in GRAPH[w.wumpus]:
                nw = World(target, new_w, w.pits, w.bats)
                if matches_observation(nw, post):
                    out.add(nw)
        else:
            nw = World(target, w.wumpus, w.pits, w.bats)
            if matches_observation(nw, post):
                out.add(nw)
    return frozenset(out)


def update_on_snatch(belief: Belief, snatch_target: int, post: Observation) -> Belief:
    """We moved to `snatch_target`, got snatched by a bat, ended in `post.room`.

    Drops worlds where `snatch_target` isn't actually a bat room. For
    each surviving world, the player jumps to `post.room`; we re-filter.
    If the snatch landed us in the wumpus's room, branch over his
    surviving moves the same way as `update_on_move`.
    """
    out: set[World] = set()
    new_player = post.room
    for w in belief:
        if snatch_target not in w.bats:
            continue
        if new_player in w.pits:
            continue
        if new_player in w.bats:
            continue
        if new_player == w.wumpus:
            for new_w in GRAPH[w.wumpus]:
                nw = World(new_player, new_w, w.pits, w.bats)
                if matches_observation(nw, post):
                    out.add(nw)
        else:
            nw = World(new_player, w.wumpus, w.pits, w.bats)
            if matches_observation(nw, post):
                out.add(nw)
    return frozenset(out)


def update_on_miss(belief: Belief, shot_path: tuple[int, ...], post: Observation) -> Belief:
    """We shot, the arrow missed. Wumpus moved or stayed.

    For each world: if the wumpus was on the shot path, the shot would
    have hit (inconsistent with miss → filter). Otherwise branch over
    `{wumpus} ∪ GRAPH[wumpus]`, dropping the branch that would have
    eaten us (filtered by no-death observation).
    """
    out: set[World] = set()
    player_room = post.room
    path_set = set(shot_path)
    for w in belief:
        if w.wumpus in path_set:
            continue
        candidates = (w.wumpus, *GRAPH[w.wumpus])
        for new_w in candidates:
            if new_w == player_room:
                continue  # he'd have eaten us; observation says we're alive
            nw = World(w.player, new_w, w.pits, w.bats)
            if matches_observation(nw, post):
                out.add(nw)
    return frozenset(out)
