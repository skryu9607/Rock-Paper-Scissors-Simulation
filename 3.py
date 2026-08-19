"""
Bridge 3: Abstract Pmax Policy Realization for Spatial RPS Simulator

Purpose:
    1. Synthesize abstract finite-horizon Pmax policy on population states s=(R,S,P).
    2. Convert abstract action alpha=(alpha_R, alpha_S, alpha_P)
       into executable spatial behavior for individual agents.

Interpretation:
    alpha_R controls how actively Rock agents chase Scissors.
    alpha_S controls how actively Scissors agents chase Paper.
    alpha_P controls how actively Paper agents chase Rock.

Author: Bridge-3 implementation draft
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import random
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np


R, S, P = "R", "S", "P"
TYPES = (R, S, P)

# Winner -> prey mapping
PREY_OF = {
    R: S,  # Rock converts Scissors
    S: P,  # Scissors converts Paper
    P: R,  # Paper converts Rock
}

# Predator -> who can kill me
PREDATOR_OF = {
    R: P,
    S: R,
    P: S,
}

State = Tuple[int, int, int]          # (num_R, num_S, num_P)
Action = Tuple[float, float, float]   # (alpha_R, alpha_S, alpha_P)


# ---------------------------------------------------------------------
# Part 1. Abstract Pmax synthesis
# ---------------------------------------------------------------------

def coexist(s: State) -> bool:
    """True iff all three populations are nonzero."""
    return s[0] > 0 and s[1] > 0 and s[2] > 0


def all_population_states(N: int) -> List[State]:
    """Enumerate all states (R,S,P) with R+S+P=N."""
    states = []
    for r in range(N + 1):
        for sc in range(N - r + 1):
            p = N - r - sc
            states.append((r, sc, p))
    return states


def transition_probs(
    s: State,
    a: Action,
    normalize_if_invalid: bool = True,
) -> Dict[State, float]:
    """
    Abstract transition model.

    a = (alpha_R, alpha_S, alpha_P)
    alpha_R: R converts S, so (R,S,P) -> (R+1,S-1,P)
    alpha_S: S converts P, so (R,S,P) -> (R,S+1,P-1)
    alpha_P: P converts R, so (R,S,P) -> (R-1,S,P+1)

    Note:
        With alpha=1.5 and balanced states, raw probabilities can exceed 1.
        If normalize_if_invalid=True, conversion probabilities are renormalized
        to form a valid distribution. If your previous code uses clipping or a
        different convention, replace this function with that version.
    """
    r, sc, p = s
    N = r + sc + p

    if N <= 1 or not coexist(s):
        return {s: 1.0}

    alpha_R, alpha_S, alpha_P = a
    denom = N * (N - 1)

    prob_R_converts_S = (2.0 * r * sc / denom) * alpha_R if r > 0 and sc > 0 else 0.0
    prob_S_converts_P = (2.0 * sc * p / denom) * alpha_S if sc > 0 and p > 0 else 0.0
    prob_P_converts_R = (2.0 * p * r / denom) * alpha_P if p > 0 and r > 0 else 0.0

    probs = [
        prob_R_converts_S,
        prob_S_converts_P,
        prob_P_converts_R,
    ]

    total = sum(probs)

    if total > 1.0 and normalize_if_invalid:
        probs = [x / total for x in probs]
        total = 1.0

    next_probs: Dict[State, float] = {}

    def add(ns: State, prob: float) -> None:
        if prob <= 0:
            return
        next_probs[ns] = next_probs.get(ns, 0.0) + prob

    p_RS, p_SP, p_PR = probs

    add((r + 1, sc - 1, p), p_RS)
    add((r, sc + 1, p - 1), p_SP)
    add((r - 1, sc, p + 1), p_PR)
    add(s, max(0.0, 1.0 - total))

    return next_probs


def synthesize_pmax_policy(
    N: int,
    horizon: int,
    alpha_values: Tuple[float, float, float] = (0.5, 1.0, 1.5),
) -> Tuple[Dict[Tuple[int, State], Action], Dict[Tuple[int, State], float]]:
    """
    Finite-horizon Pmax synthesis for P[G <= T coexist].

    Returns:
        policy[(remaining_time, state)] = best abstract action
        value[(remaining_time, state)] = optimal coexistence probability

    Important:
        Because this is finite-horizon synthesis, the policy depends on
        remaining_time as well as state.
    """
    states = all_population_states(N)
    actions: List[Action] = list(product(alpha_values, repeat=3))

    V: Dict[Tuple[int, State], float] = {}
    policy: Dict[Tuple[int, State], Action] = {}

    # Base case: zero remaining steps
    for s in states:
        V[(0, s)] = 1.0 if coexist(s) else 0.0
        policy[(0, s)] = (1.0, 1.0, 1.0)

    for t in range(1, horizon + 1):
        for s in states:
            if not coexist(s):
                V[(t, s)] = 0.0
                policy[(t, s)] = (1.0, 1.0, 1.0)
                continue

            best_value = -1.0
            best_action = (1.0, 1.0, 1.0)

            for a in actions:
                trans = transition_probs(s, a)
                q = sum(prob * V[(t - 1, ns)] for ns, prob in trans.items())

                if q > best_value:
                    best_value = q
                    best_action = a

            V[(t, s)] = best_value
            policy[(t, s)] = best_action

    return policy, V


# ---------------------------------------------------------------------
# Part 2. Spatial realization of abstract action
# ---------------------------------------------------------------------

@dataclass
class AgentView:
    """
    Lightweight view of one simulator agent.

    Your actual simulator object does not need to use this dataclass.
    The Bridge3 controller below only assumes each agent has:
        agent.kind or agent.type: "R", "S", or "P"
        agent.x, agent.y
        agent.theta
    """
    kind: str
    x: float
    y: float
    theta: float


def angle_wrap(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def nearest_agent(
    agent,
    agents: Iterable,
    target_kind: str,
    sensing_range: float,
):
    """Return nearest visible agent of target_kind."""
    best = None
    best_d = float("inf")

    for other in agents:
        if other is agent:
            continue
        if get_kind(other) != target_kind:
            continue

        d = distance(agent, other)
        if d < best_d and d <= sensing_range:
            best = other
            best_d = d

    return best


def get_kind(agent) -> str:
    """
    Adapter for simulator object.

    Edit this if your simulator stores type differently.
    """
    if hasattr(agent, "kind"):
        return agent.kind
    if hasattr(agent, "type"):
        return agent.type
    if hasattr(agent, "agent_type"):
        return agent.agent_type
    raise AttributeError("Agent must have kind/type/agent_type equal to 'R', 'S', or 'P'.")


def set_heading(agent, theta: float) -> None:
    """
    Adapter for setting heading.

    Edit this if your simulator uses angle, direction, heading, etc.
    """
    if hasattr(agent, "theta"):
        agent.theta = theta
    elif hasattr(agent, "angle"):
        agent.angle = theta
    else:
        raise AttributeError("Agent must have theta or angle attribute.")


def get_heading(agent) -> float:
    if hasattr(agent, "theta"):
        return agent.theta
    if hasattr(agent, "angle"):
        return agent.angle
    raise AttributeError("Agent must have theta or angle attribute.")


def desired_angle_to(agent, target) -> float:
    return math.atan2(target.y - agent.y, target.x - agent.x)


def turn_toward(
    current_theta: float,
    desired_theta: float,
    max_turn_rate: float,
) -> float:
    err = angle_wrap(desired_theta - current_theta)
    err = max(-max_turn_rate, min(max_turn_rate, err))
    return angle_wrap(current_theta + err)


def random_turn(current_theta: float, max_turn_rate: float) -> float:
    return angle_wrap(current_theta + random.uniform(-max_turn_rate, max_turn_rate))


def count_population_from_agents(agents: Iterable) -> State:
    r = sc = p = 0
    for a in agents:
        k = get_kind(a)
        if k == R:
            r += 1
        elif k == S:
            sc += 1
        elif k == P:
            p += 1
        else:
            raise ValueError(f"Unknown agent type: {k}")
    return (r, sc, p)


def alpha_to_mode(alpha: float, tol: float = 1e-9) -> str:
    """
    Convert abstract multiplier into spatial behavior mode.

    alpha > 1 : chase prey aggressively
    alpha = 1 : neutral random exploration
    alpha < 1 : passive, avoid prey / reduce conversion opportunities
    """
    if alpha > 1.0 + tol:
        return "aggressive"
    if alpha < 1.0 - tol:
        return "passive"
    return "neutral"


class Bridge3PmaxSpatialController:
    """
    Realizes abstract Pmax policy inside the spatial simulator.

    Usage:
        controller = Bridge3PmaxSpatialController(policy, horizon=300)
        controller.apply(agents, step_idx)

    This updates each agent's heading according to the abstract action selected
    at the current population state.
    """

    def __init__(
        self,
        pmax_policy: Dict[Tuple[int, State], Action],
        horizon: int,
        sensing_range: float = 100.0,
        max_turn_rate: float = 0.25,
        passive_avoid_range: float = 120.0,
    ):
        self.pmax_policy = pmax_policy
        self.horizon = horizon
        self.sensing_range = sensing_range
        self.max_turn_rate = max_turn_rate
        self.passive_avoid_range = passive_avoid_range

    def abstract_action(self, agents: List, step_idx: int) -> Action:
        s = count_population_from_agents(agents)
        remaining = max(0, self.horizon - step_idx)

        # Exact finite-horizon lookup
        if (remaining, s) in self.pmax_policy:
            return self.pmax_policy[(remaining, s)]

        # Fallback: use the largest available remaining time for this state
        candidates = [
            (t, a)
            for (t, ss), a in self.pmax_policy.items()
            if ss == s
        ]
        if candidates:
            return max(candidates, key=lambda x: x[0])[1]

        return (1.0, 1.0, 1.0)

    def apply(self, agents: List, step_idx: int) -> Action:
        """
        Apply Bridge-3 controller for one simulator timestep.

        Returns:
            The abstract action used at this population state.
        """
        alpha_R, alpha_S, alpha_P = self.abstract_action(agents, step_idx)

        alpha_by_type = {
            R: alpha_R,
            S: alpha_S,
            P: alpha_P,
        }

        for agent in agents:
            kind = get_kind(agent)
            alpha = alpha_by_type[kind]
            mode = alpha_to_mode(alpha)

            theta = get_heading(agent)

            if mode == "aggressive":
                # Agent actively chases its prey.
                target = nearest_agent(
                    agent,
                    agents,
                    target_kind=PREY_OF[kind],
                    sensing_range=self.sensing_range,
                )
                if target is not None:
                    desired = desired_angle_to(agent, target)
                    set_heading(agent, turn_toward(theta, desired, self.max_turn_rate))
                else:
                    set_heading(agent, random_turn(theta, self.max_turn_rate * 0.5))

            elif mode == "passive":
                # Agent tries to reduce conversion events.
                # Since alpha_i controls "i converts prey", passivity means
                # avoiding its prey rather than chasing it.
                target = nearest_agent(
                    agent,
                    agents,
                    target_kind=PREY_OF[kind],
                    sensing_range=self.passive_avoid_range,
                )
                if target is not None:
                    away = desired_angle_to(target, agent)
                    set_heading(agent, turn_toward(theta, away, self.max_turn_rate))
                else:
                    # Mild random walk
                    set_heading(agent, random_turn(theta, self.max_turn_rate * 0.25))

            else:
                # Neutral behavior: weak random exploration.
                set_heading(agent, random_turn(theta, self.max_turn_rate * 0.5))

        return (alpha_R, alpha_S, alpha_P)


# ---------------------------------------------------------------------
# Part 3. Convenience construction
# ---------------------------------------------------------------------

def build_bridge3_controller(
    N: int = 12,
    horizon: int = 300,
    alpha_values: Tuple[float, float, float] = (0.5, 1.0, 1.5),
    sensing_range: float = 100.0,
    max_turn_rate: float = 0.25,
) -> Tuple[Bridge3PmaxSpatialController, Dict[Tuple[int, State], float]]:
    """
    Build the full Bridge-3 controller:
        abstract Pmax synthesis -> spatial controller.
    """
    pmax_policy, V = synthesize_pmax_policy(
        N=N,
        horizon=horizon,
        alpha_values=alpha_values,
    )

    controller = Bridge3PmaxSpatialController(
        pmax_policy=pmax_policy,
        horizon=horizon,
        sensing_range=sensing_range,
        max_turn_rate=max_turn_rate,
    )

    return controller, V


if __name__ == "__main__":
    controller, V = build_bridge3_controller(N=12, horizon=300)

    test_states = [
        (4, 4, 4),
        (6, 3, 3),
        (8, 2, 2),
        (6, 5, 1),
    ]

    print("Bridge-3 abstract Pmax action examples:")
    for s in test_states:
        a = controller.pmax_policy[(300, s)]
        val = V[(300, s)]
        print(f"s={s}, Pmax action={a}, V={val:.4f}")
