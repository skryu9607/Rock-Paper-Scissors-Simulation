"""Empirical DTMC extraction and PCTL checks for the RSP simulator."""

import math
import random
import numpy as np
from collections import defaultdict, Counter
import time


WIDTH = 300 * 2
HEIGHT = 300 * 2
IMG_WIDTH = 40
IMG_HEIGHT = 40
COLLISION_RADIUS = IMG_WIDTH
SENSE_RANGE = 100.0
BASE_SPEED = 2.0
TURN_RATE = 0.18

PREY_OF = {"R": "S", "S": "P", "P": "R"}
PREDATOR_OF = {"R": "P", "S": "R", "P": "S"}


class Agent:
    def __init__(self, x, y, agent_type):
        self.x = x
        self.y = y
        self.agent_type = agent_type
        self.speed = BASE_SPEED
        self.angle = random.uniform(0, 2 * math.pi)

    @property
    def cx(self):
        return self.x + IMG_WIDTH / 2

    @property
    def cy(self):
        return self.y + IMG_HEIGHT / 2

    def move(self):
        self.x += math.sin(self.angle) * self.speed
        self.y -= math.cos(self.angle) * self.speed
        self.bounce()

    def bounce(self):
        if self.x > WIDTH - IMG_WIDTH:
            self.x = 2 * (WIDTH - IMG_WIDTH) - self.x
            self.angle = -self.angle
        elif self.x < 0:
            self.x = -self.x
            self.angle = -self.angle
        if self.y > HEIGHT - IMG_HEIGHT:
            self.y = 2 * (HEIGHT - IMG_HEIGHT) - self.y
            self.angle = math.pi - self.angle
        elif self.y < 0:
            self.y = -self.y
            self.angle = math.pi - self.angle

    def observe(self, agents):
        prey_type = PREY_OF[self.agent_type]
        pred_type = PREDATOR_OF[self.agent_type]
        obs = {"prey": [], "predator": [], "team": []}
        for other in agents:
            if other is self:
                continue
            d = math.hypot(self.cx - other.cx, self.cy - other.cy)
            if d <= SENSE_RANGE:
                if other.agent_type == prey_type:
                    obs["prey"].append((other, d))
                elif other.agent_type == pred_type:
                    obs["predator"].append((other, d))
                elif other.agent_type == self.agent_type:
                    obs["team"].append((other, d))
        return obs

    def select_action(self, obs, policy="random"):
        if policy == "random":
            return "random"
        elif policy == "aggressive":
            return "chase"
        elif policy == "defensive":
            return "evade"
        elif policy == "balanced":
            if obs["predator"]:
                nearest_d = min(d for _, d in obs["predator"])
                if nearest_d < 0.55 * SENSE_RANGE:
                    return "evade"
            if obs["prey"]:
                return "chase"
            if obs["team"]:
                return "align"
            return "random"
        return "random"

    def apply_action(self, action, obs):
        if action == "chase":
            t = self._nearest(obs["prey"])
            if t:
                self._turn_toward(self._angle_to(t))
            else:
                self._random_turn()
        elif action == "evade":
            p = self._nearest(obs["predator"])
            if p:
                self._turn_toward(self._angle_to(p) + math.pi)
            else:
                self._random_turn()
        elif action == "align":
            if obs["team"]:
                avg = self._circular_mean([a.angle for a, _ in obs["team"]])
                self._turn_toward(avg)
            else:
                self._random_turn()
        else:
            self._random_turn()

    def _nearest(self, items):
        return min(items, key=lambda x: x[1])[0] if items else None

    def _angle_to(self, other):
        dx = other.cx - self.cx
        dy = other.cy - self.cy
        return math.atan2(dx, -dy)
    def _turn_toward(self, desired):
        diff = (desired - self.angle + math.pi) % (2 * math.pi) - math.pi
        diff = max(-TURN_RATE, min(TURN_RATE, diff))
        self.angle += diff

    def _random_turn(self):
        self.angle += random.uniform(-TURN_RATE, TURN_RATE)

    def _circular_mean(self, angles):
        return math.atan2(
            sum(math.sin(a) for a in angles),
            sum(math.cos(a) for a in angles)
        )


def winner(t1, t2):
    if PREY_OF[t1] == t2:
        return t1
    if PREY_OF[t2] == t1:
        return t2
    return t1


def collide_pair(a, b):
    if a.agent_type == b.agent_type:
        return None
    d = math.hypot(a.cx - b.cx, a.cy - b.cy)
    if d >= COLLISION_RADIUS:
        return None

    w = winner(a.agent_type, b.agent_type)
    a.agent_type = w
    b.agent_type = w

    dx, dy = a.x - b.x, a.y - b.y
    tangent = math.atan2(dy, dx)
    angle = 0.5 * math.pi + tangent
    a.angle = 2 * tangent - a.angle
    b.angle = 2 * tangent - b.angle
    a.x += math.sin(angle)
    a.y -= math.cos(angle)
    b.x -= math.sin(angle)
    b.y += math.cos(angle)
    return w


def get_population(agents):
    c = Counter(a.agent_type for a in agents)
    return (c.get("R", 0), c.get("S", 0), c.get("P", 0))


def is_coexist(pop):
    return pop[0] > 0 and pop[1] > 0 and pop[2] > 0


def run_episode_record_transitions(initial_pop=(4, 4, 4), policy="random", max_steps=2000, seed=None):
    """Run one episode and store the population at each timestep."""
    if seed is not None:
        random.seed(seed)

    n_R, n_S, n_P = initial_pop

    agents = []
    for agent_type, count in zip(["R", "S", "P"], [n_R, n_S, n_P]):
        for _ in range(count):
            x = random.uniform(0, WIDTH - IMG_WIDTH)
            y = random.uniform(0, HEIGHT - IMG_HEIGHT)
            agents.append(Agent(x, y, agent_type))

    state_sequence = [get_population(agents)]
    extinction_time = max_steps

    for step in range(1, max_steps + 1):
        if policy != "random":
            action_list = []
            for agent in agents:
                obs = agent.observe(agents)
                action = agent.select_action(obs, policy)
                action_list.append((agent, action, obs))
            for agent, action, obs in action_list:
                agent.apply_action(action, obs)

        for agent in agents:
            agent.move()

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                collide_pair(agents[i], agents[j])

        pop = get_population(agents)
        state_sequence.append(pop)

        if not is_coexist(pop):
            extinction_time = step
            break

    return extinction_time, state_sequence



def extract_empirical_dtmc(initial_pop=(4, 4, 4), policy="random", K=50, max_steps=2000):
    """Estimate an empirical DTMC from simulator rollouts."""
    N = sum(initial_pop)

    S = [(l, m, N - l - m) for l in range(N + 1) for m in range(N - l + 1)]
    S_idx = {s: i for i, s in enumerate(S)}
    nS = len(S)

    trans_count = defaultdict(lambda: defaultdict(int))
    visit_count = defaultdict(int)
    ext_times = []

    for k in range(K):
        et, seq = run_episode_record_transitions(
            initial_pop, policy=policy, max_steps=max_steps, seed=5000 + k
        )
        ext_times.append(et)

        for t in range(len(seq) - 1):
            s = seq[t]
            s_next = seq[t + 1]
            trans_count[s][s_next] += 1
            visit_count[s] += 1

    absorbing_set = set(s for s in S if s[0] == 0 or s[1] == 0 or s[2] == 0)

    T_hat = {}
    states_with_data = set()

    for s in S:
        si = S_idx[s]

        if s in absorbing_set:
            T_hat[si] = [(1.0, si)]
            continue

        total = visit_count[s]
        if total == 0:
            T_hat[si] = [(1.0, si)]
            continue

        states_with_data.add(s)
        transitions = []
        for s_next, count in trans_count[s].items():
            transitions.append((count / total, S_idx[s_next]))

        T_hat[si] = transitions

    return T_hat, visit_count, ext_times, S, S_idx, nS, states_with_data



def pctl_coexist_on_dtmc(T_hat, S, S_idx, nS, T_horizon):
    """Probability of keeping coexistence for T_horizon steps."""
    transient = [s for s in S if s[0] > 0 and s[1] > 0 and s[2] > 0]

    V = np.zeros(nS)
    for s in transient:
        V[S_idx[s]] = 1.0

    for t in range(T_horizon):
        V_new = np.zeros(nS)
        for s in transient:
            si = S_idx[s]
            V_new[si] = sum(p * V[sj] for p, sj in T_hat[si])
        V = V_new

    return V


def expected_extinction_on_dtmc(T_hat, S, S_idx, nS, epsilon=1e-8, max_iter=50000):
    """Expected time to extinction on the empirical DTMC."""
    transient = [s for s in S if s[0] > 0 and s[1] > 0 and s[2] > 0]

    V = np.zeros(nS)
    for iteration in range(max_iter):
        delta = 0.0
        V_new = np.copy(V)
        for s in transient:
            si = S_idx[s]
            val = 1.0 + sum(p * V[sj] for p, sj in T_hat[si])
            delta = max(delta, abs(val - V[si]))
            V_new[si] = val
        V = V_new
        if delta < epsilon:
            break

    return V


def extinction_class_on_dtmc(T_hat, S, S_idx, nS, extinct_idx, epsilon=1e-8, max_iter=50000):
    """Probability that the selected type is extinct in the absorbing state."""
    transient = [s for s in S if s[0] > 0 and s[1] > 0 and s[2] > 0]
    absorbing = [s for s in S if s[0] == 0 or s[1] == 0 or s[2] == 0]

    U = np.zeros(nS)
    for s in absorbing:
        if s[extinct_idx] == 0:
            U[S_idx[s]] = 1.0

    for iteration in range(max_iter):
        delta = 0.0
        U_new = np.copy(U)
        for s in transient:
            si = S_idx[s]
            val = sum(p * U[sj] for p, sj in T_hat[si])
            delta = max(delta, abs(val - U[si]))
            U_new[si] = val
        U = U_new
        if delta < epsilon:
            break

    return U



if __name__ == "__main__":
    N_R = 8
    N_S = 2
    N_P = 2
    INITIAL_POP = (N_R, N_S, N_P)

    N = sum(INITIAL_POP)
    K = 50
    T_HORIZON = 300
    MAX_STEPS = 1000

    s0 = INITIAL_POP
    policies = ["random", "aggressive", "defensive", "balanced"]

    print("=" * 70)
    print("Empirical DTMC Extraction + PCTL Model Checking")
    print(f"N={N}, initial_pop={INITIAL_POP}, K={K} episodes/policy, T={T_HORIZON}, max_steps={MAX_STEPS}")
    print("=" * 70)

    abstract_available = False
    try:
        print("\n[Abstract Model A] Population-level bounds (from rps_pctl.py)...")
        from rps_pctl import (
            pctl_bounded_coexist, expected_extinction_time as eet_abstract,
            S_index as S_idx_abstract
        )
        V_pmax, _ = pctl_bounded_coexist(T_HORIZON, mode="max")
        V_pmin, _ = pctl_bounded_coexist(T_HORIZON, mode="min")
        V_prand, _ = pctl_bounded_coexist(T_HORIZON, mode="random")
        E_pmax = eet_abstract(mode="max")
        E_pmin = eet_abstract(mode="min")
        E_prand = eet_abstract(mode="random")
        s0_ai = S_idx_abstract[s0]
        abstract_available = True

        print(f"  P_max[G≤{T_HORIZON} coexist]  = {V_pmax[s0_ai]:.6f}   E_max  = {E_pmax[s0_ai]:.2f}")
        print(f"  P_rand[G≤{T_HORIZON} coexist] = {V_prand[s0_ai]:.8f} E_rand = {E_prand[s0_ai]:.2f}")
        print(f"  P_min[G≤{T_HORIZON} coexist]  = {V_pmin[s0_ai]:.8f} E_min  = {E_pmin[s0_ai]:.2f}")
    except (ImportError, KeyError):
        print("\n[Abstract Model A] skipped: rps_pctl.py does not contain this initial state.")

    results = {}

    for policy in policies:
        print(f"\n{'-' * 70}")
        print(f"Policy: {policy}")
        print(f"{'-' * 70}")

        t0 = time.time()
        T_hat, visits, ext_times, S, S_idx, nS, states_with_data = \
            extract_empirical_dtmc(INITIAL_POP, policy, K, MAX_STEPS)
        extraction_time = time.time() - t0

        s0_idx = S_idx[s0]

        mean_et_sim = np.mean(ext_times)
        std_et_sim = np.std(ext_times)
        ci95 = 1.96 * std_et_sim / np.sqrt(K)
        p_coexist_sim = sum(1 for et in ext_times if et >= T_HORIZON) / K

        n_transient = len([s for s in S if s[0] > 0 and s[1] > 0 and s[2] > 0])
        print(f"  [SMC] E[ext] = {mean_et_sim:.2f} ± {ci95:.2f},  "
              f"P(coexist@{T_HORIZON}) = {p_coexist_sim:.4f},  "
              f"states observed = {len(states_with_data)}/{n_transient}")

        V_emp = pctl_coexist_on_dtmc(T_hat, S, S_idx, nS, T_HORIZON)
        E_emp = expected_extinction_on_dtmc(T_hat, S, S_idx, nS)

        print(f"  [Empirical DTMC] P[G≤{T_HORIZON} coexist] = {V_emp[s0_idx]:.6f},  "
              f"E[ext] = {E_emp[s0_idx]:.2f}")

        U_R = extinction_class_on_dtmc(T_hat, S, S_idx, nS, 0)
        U_S = extinction_class_on_dtmc(T_hat, S, S_idx, nS, 1)
        U_P = extinction_class_on_dtmc(T_hat, S, S_idx, nS, 2)

        print(f"  [Extinction class] P(R ext)={U_R[s0_idx]:.4f}  "
              f"P(S ext)={U_S[s0_idx]:.4f}  "
              f"P(P ext)={U_P[s0_idx]:.4f}  "
              f"sum={U_R[s0_idx]+U_S[s0_idx]+U_P[s0_idx]:.4f}")

        print(f"  [Time] {extraction_time:.1f}s")

        results[policy] = {
            "mean_et_sim": mean_et_sim,
            "p_coexist_sim": p_coexist_sim,
            "P_coexist_dtmc": V_emp[s0_idx],
            "E_ext_dtmc": E_emp[s0_idx],
            "ext_class": (U_R[s0_idx], U_S[s0_idx], U_P[s0_idx]),
        }

    print("\n" + "=" * 70)
    print("SUMMARY: Empirical DTMC Model Checking")
    print("=" * 70)

    if abstract_available:
        print(f"\n  Abstract Model A bounds:")
        print(f"    E_min = {E_pmin[s0_ai]:.2f}    E_rand = {E_prand[s0_ai]:.2f}    E_max = {E_pmax[s0_ai]:.2f}")
        print(f"    P_min = {V_pmin[s0_ai]:.6f}  P_rand = {V_prand[s0_ai]:.8f}  P_max = {V_pmax[s0_ai]:.6f}")

    print(f"\n  {'Policy':<12} {'E[ext]sim':<11} {'E[ext]DTMC':<12} "
          f"{'P(coex)sim':<12} {'P(coex)DTMC':<13}")
    print(f"  {'-'*12} {'-'*11} {'-'*12} {'-'*12} {'-'*13}")

    for policy in policies:
        r = results[policy]
        print(f"  {policy:<12} {r['mean_et_sim']:<11.2f} {r['E_ext_dtmc']:<12.2f} "
              f"{r['p_coexist_sim']:<12.4f} {r['P_coexist_dtmc']:<13.6f}")

    best_policy = max(results, key=lambda p: results[p]["P_coexist_dtmc"])
    best_r = results[best_policy]

    print(f"\n  Best policy from empirical DTMC: {best_policy}")
    print(f"    P[G≤{T_HORIZON} coexist] = {best_r['P_coexist_dtmc']:.6f}")
    print(f"    E[extinction] = {best_r['E_ext_dtmc']:.2f}")

    if abstract_available:
        print(f"\n  Abstract E_max = {E_pmax[s0_ai]:.2f}")
        print(f"  Empirical best = {best_r['E_ext_dtmc']:.2f}")
