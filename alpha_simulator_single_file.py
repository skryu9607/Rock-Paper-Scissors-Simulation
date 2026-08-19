"""Agent simulator and population-level model checking in one file."""

import math
import random
import numpy as np
from collections import defaultdict, Counter
from itertools import product
import time

# Image setting.

WIDTH = 300
HEIGHT = 300
IMG_WIDTH = 40
IMG_HEIGHT = 40
COLLISION_RADIUS = IMG_WIDTH

# Cyclic dominance
PREY_OF = {"R": "S", "S": "P", "P": "R"}
PREDATOR_OF = {"R": "P", "S": "R", "P": "S"}
# Fixed parameters
SENSE_RANGE = 100.0
BASE_SPEED = 2.0
TURN_RATE = 0.18


class Agent:
    # 2D agent with position, angle, and type (R, S, P).

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
        # app.py style: sin for x, -cos for y
        self.x += math.sin(self.angle) * self.speed
        self.y -= math.cos(self.angle) * self.speed
        self.bounce()

    def bounce(self):
        # Follow the logic of pygame logic.
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
        # Local observation: nearby prey, predators, teammates
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
        '''
        Policy modes : 
          random    : no steering, just random walk (app.py)
          aggressive: always chase prey
          defensive : always evade predator
          cohesive  : evade if threatened, else align with team
          balanced  : context-aware hand-coded policy
        '''
        if policy == "random":
            return "random"
        elif policy == "aggressive":
            return "chase"
        elif policy == "defensive":
            return "evade"
        elif policy == "cohesive":
            if obs["predator"]:
                return "evade"
            return "align"
        elif policy == "balanced":
            if obs["predator"]:
                nearest_pred_d = min(d for _, d in obs["predator"])
                if nearest_pred_d < 0.55 * SENSE_RANGE:
                    return "evade"
            if obs["prey"]:
                return "chase"
            if obs["team"]:
                return "align"
            return "random"
        return "random"

    def apply_action(self, action, obs):
        '''
        Actions  : 
          Chase : toward to the nearest prey
          Evade : run away from the nearest predator
          Align : align with the same type agents' heading angle
          Random : Random turn
        '''
        if action == "chase":
            target = self._nearest(obs["prey"])
            if target:
                self._turn_toward(self._angle_to(target))
            else:
                self._random_turn()
        elif action == "evade":
            pred = self._nearest(obs["predator"])
            if pred:
                self._turn_toward(self._angle_to(pred) + math.pi)
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
        if not items:
            return None
        return min(items, key=lambda x: x[1])[0]

    def _angle_to(self, other):
        return math.atan2(other.cy - self.cy, other.cx - self.cx)

    def _turn_toward(self, desired):
        diff = (desired - self.angle + math.pi) % (2 * math.pi) - math.pi
        diff = max(-TURN_RATE, min(TURN_RATE, diff))
        self.angle = (self.angle + diff + math.pi) % (2 * math.pi) - math.pi

    def _random_turn(self):
        self.angle += random.uniform(-TURN_RATE, TURN_RATE)

    def _circular_mean(self, angles):
        x = sum(math.cos(a) for a in angles)
        y = sum(math.sin(a) for a in angles)
        return math.atan2(y, x)


def winner(t1, t2):
    # Return winner type given cyclic dominance.
    if PREY_OF[t1] == t2:
        return t1
    if PREY_OF[t2] == t1:
        return t2
    return t1  # same type, shouldn't happen


def collide_pair(a, b):
    '''
    Check collision and resolve type conversion.
    Returns the conversion event or None.
    Event: (winner_type, loser_type) ex) ("R", "S") means R dominates S and S is converted to R. 
    '''
    if a.agent_type == b.agent_type:
        return None

    d = math.hypot(a.cx - b.cx, a.cy - b.cy)
    if d >= COLLISION_RADIUS:
        return None

    w = winner(a.agent_type, b.agent_type)
    loser_type = b.agent_type if a.agent_type == w else a.agent_type

    event = (w, loser_type)

    # Both become winner type
    a.agent_type = w
    b.agent_type = w

    # Bounce apart
    dx = a.x - b.x
    dy = a.y - b.y
    tangent = math.atan2(dy, dx)
    angle = 0.5 * math.pi + tangent

    a.angle = 2 * tangent - a.angle
    b.angle = 2 * tangent - b.angle

    a.x += math.sin(angle)
    a.y -= math.cos(angle)
    b.x -= math.sin(angle)
    b.y += math.cos(angle)

    return event


def get_population(agents):
    c = Counter(a.agent_type for a in agents)
    return (c.get("R", 0), c.get("S", 0), c.get("P", 0))


def is_coexist(pop):
    return pop[0] > 0 and pop[1] > 0 and pop[2] > 0


def run_episode(initial_pop=(4, 4, 4), policy="random", max_steps=5000, seed=None):
    '''
    Run one headless episode.

    Returns:
      extinction_time: step at which coexistence broke (or max_steps)
      transition_log: list of (timestep, pop_before, pop_after, events)
        events = list of (winner_type, loser_type) conversions in that step
    '''
    if seed is not None:
        random.seed(seed)

    n_R, n_S, n_P = initial_pop

    agents = []
    for t, n_t in zip(["R", "S", "P"], [n_R, n_S, n_P]):
        for _ in range(n_t):
            x = random.uniform(0, WIDTH - IMG_WIDTH)
            y = random.uniform(0, HEIGHT - IMG_HEIGHT)
            agents.append(Agent(x, y, t))

    N = len(agents)
    transition_log = []
    extinction_time = max_steps

    for step in range(1, max_steps + 1):
        pop_before = get_population(agents)

        # Observe and select action
        if policy != "random":
            actions = []
            for agent in agents:
                obs = agent.observe(agents)
                action = agent.select_action(obs, policy)
                actions.append((agent, action, obs))
            for agent, action, obs in actions:
                agent.apply_action(action, obs)

        # 2. Move
        for agent in agents:
            agent.move()

        # 3. Collisions
        events = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                ev = collide_pair(agents[i], agents[j])
                if ev is not None:
                    events.append(ev)

        pop_after = get_population(agents)

        # Log transition if population changed
        if pop_before != pop_after:
            transition_log.append((step, pop_before, pop_after, events))

        # Check extinction
        if not is_coexist(pop_after):
            extinction_time = step
            break

    return extinction_time, transition_log


def build_population_model(total_pop):
    action_set = [0.5, 1.0, 1.5]
    joint_actions = list(product(action_set, repeat=3))

    states = [(l, m, total_pop - l - m)
              for l in range(total_pop + 1)
              for m in range(total_pop - l + 1)]
    state_index = {s: i for i, s in enumerate(states)}
    absorbing_states = set(s for s in states if s[0] == 0 or s[1] == 0 or s[2] == 0)
    transient_states = [s for s in states if s not in absorbing_states]

    denom = total_pop * (total_pop - 1)
    transition_table = {}

    for action_index, action in enumerate(joint_actions):
        transition_table[action_index] = {}
        alpha_R, alpha_S, alpha_P = action

        for state in states:
            state_id = state_index[state]

            if state in absorbing_states:
                transition_table[action_index][state_id] = [(1.0, state_id)]
                continue

            l, m, n = state
            transitions = []
            prob_sum = 0.0

            if l > 0 and m > 0:
                p = (2 * l * m / denom) * alpha_R
                transitions.append((p, state_index[(l + 1, m - 1, n)]))
                prob_sum += p

            if m > 0 and n > 0:
                p = (2 * m * n / denom) * alpha_S
                transitions.append((p, state_index[(l, m + 1, n - 1)]))
                prob_sum += p

            if l > 0 and n > 0:
                p = (2 * l * n / denom) * alpha_P
                transitions.append((p, state_index[(l - 1, m, n + 1)]))
                prob_sum += p

            if prob_sum > 1.0:
                scale = 0.99 / prob_sum
                transitions = [(p * scale, next_id) for p, next_id in transitions]
                prob_sum *= scale

            if prob_sum < 1.0:
                transitions.append((1.0 - prob_sum, state_id))

            transition_table[action_index][state_id] = transitions

    return {
        "states": states,
        "state_index": state_index,
        "transient_states": transient_states,
        "joint_actions": joint_actions,
        "transition_table": transition_table,
    }


def pctl_bounded_coexist_model(model, horizon, mode="random"):
    states = model["states"]
    state_index = model["state_index"]
    transient_states = model["transient_states"]
    joint_actions = model["joint_actions"]
    transition_table = model["transition_table"]

    values = np.zeros(len(states))
    for state in transient_states:
        values[state_index[state]] = 1.0

    neutral_index = joint_actions.index((1.0, 1.0, 1.0))

    for _ in range(horizon):
        next_values = np.zeros(len(states))
        for state in transient_states:
            state_id = state_index[state]

            if mode == "random":
                next_values[state_id] = sum(
                    p * values[next_id]
                    for p, next_id in transition_table[neutral_index][state_id]
                )
            else:
                action_values = []
                for action_index in range(len(joint_actions)):
                    val = sum(
                        p * values[next_id]
                        for p, next_id in transition_table[action_index][state_id]
                    )
                    action_values.append(val)

                if mode == "max":
                    next_values[state_id] = max(action_values)
                elif mode == "min":
                    next_values[state_id] = min(action_values)
                else:
                    raise ValueError(f"unknown mode: {mode}")

        values = next_values

    return values


def expected_extinction_time_model(model, mode="random", epsilon=1e-8, max_iter=100000):
    states = model["states"]
    state_index = model["state_index"]
    transient_states = model["transient_states"]
    joint_actions = model["joint_actions"]
    transition_table = model["transition_table"]

    values = np.zeros(len(states))
    neutral_index = joint_actions.index((1.0, 1.0, 1.0))

    for _ in range(max_iter):
        delta = 0.0
        next_values = np.copy(values)

        for state in transient_states:
            state_id = state_index[state]

            if mode == "random":
                val = 1.0 + sum(
                    p * values[next_id]
                    for p, next_id in transition_table[neutral_index][state_id]
                )
            else:
                action_values = []
                for action_index in range(len(joint_actions)):
                    val_a = 1.0 + sum(
                        p * values[next_id]
                        for p, next_id in transition_table[action_index][state_id]
                    )
                    action_values.append(val_a)

                if mode == "max":
                    val = max(action_values)
                elif mode == "min":
                    val = min(action_values)
                else:
                    raise ValueError(f"unknown mode: {mode}")

            delta = max(delta, abs(val - values[state_id]))
            next_values[state_id] = val

        values = next_values
        if delta < epsilon:
            break

    return values


# Step 1: Baseline Validation

def step1_baseline_validation(initial_pop=(4, 4, 4), K=200, max_steps=5000):
    '''
    Run random policy K times, measure E[extinction time].
    Compare with population model prediction.
    '''
    n_R, n_S, n_P = initial_pop
    N = n_R + n_S + n_P
    print("=" * 65)
    print("STEP 1: Baseline Validation (random policy = app.py)")
    print(f"  N={N}, initial_pop={initial_pop}, K={K} episodes")
    print("=" * 65)

    ext_times = []
    for k in range(K):
        et, _ = run_episode(initial_pop, policy="random", max_steps=max_steps, seed=1000 + k)
        ext_times.append(et)

    mean_et = np.mean(ext_times)
    std_et = np.std(ext_times)
    ci95 = 1.96 * std_et / np.sqrt(K)

    print(f"\n  Simulator (Monte Carlo, {K} runs):")
    print(f"    E[extinction time] = {mean_et:.2f} +/- {ci95:.2f} (95% CI)")
    print(f"    min = {min(ext_times)}, max = {max(ext_times)}")

    model = build_population_model(N)
    E_rand = expected_extinction_time_model(model, mode="random")
    s0_idx = model["state_index"][initial_pop]
    pop_prediction = E_rand[s0_idx]

    print(f"\n  Population model prediction:")
    print(f"    E[extinction time] = {pop_prediction:.2f} steps")

    print(f"\n  Ratio (simulator / model) = {mean_et / pop_prediction:.2f}")
    print(f"  Note: ratio != 1.0 expected because simulator has spatial structure")
    print(f"        while population model assumes well-mixed encounters.")

    return mean_et, pop_prediction


# 
# Step 2: Effective Aggressiveness Estimation
#

def step2_estimate_alpha_effective(initial_pop=(40, 40, 40), K=300, max_steps=5000):
    '''
    For each hand-coded policy, run K episodes and estimate
    alpha(l,m,n) for each type at each population state.

    alpha(s) = observed_R_beats_S_count(s) / (baseline_rate(s) * time_in_state(s))
    '''
    n_R, n_S, n_P = initial_pop
    N = n_R + n_S + n_P
    policies = ["random", "aggressive", "defensive", "balanced"]

    print("\n" + "=" * 65)
    print("STEP 2: Effective Aggressiveness Estimation")
    print(f"  N={N}, initial_pop={initial_pop}, K={K} episodes per policy")
    print("=" * 65)

    all_results = {}

    for policy in policies:
        print(f"\n  Policy: {policy}")
        print(f"  {'-' * 55}")

        # Accumulators per population state
        time_in_state = defaultdict(int)       # how many steps spent in state s
        RS_count = defaultdict(int)            # RtoS conversions while in state s
        SP_count = defaultdict(int)            # StoP conversions while in state s
        PR_count = defaultdict(int)            # PtoR conversions while in state s

        ext_times = []
        coexist_at_300 = 0
    
        for k in range(K):
            et, transition_log = run_episode(
                initial_pop, policy=policy, max_steps=max_steps, seed=2000 + k
            )
            ext_times.append(et)
            if et > 300:
                coexist_at_300 += 1

            # Reconstruct state-time occupancy
            # Between transition events, population is constant
            current_pop = initial_pop
            last_step = 0

            for step, pop_before, pop_after, events in transition_log:
                # State was pop_before from last_step+1 to step (inclusive of step)
                # But the transition happens at step, so time in pop_before = step - last_step
                duration = step - last_step
                if is_coexist(pop_before) and duration > 0:
                    time_in_state[pop_before] += duration

                    # Count conversion events
                    for winner_type, loser_type in events:
                        if winner_type == "R" and loser_type == "S":
                            RS_count[pop_before] += 1
                        elif winner_type == "S" and loser_type == "P":
                            SP_count[pop_before] += 1
                        elif winner_type == "P" and loser_type == "R":
                            PR_count[pop_before] += 1

                current_pop = pop_after
                last_step = step

            # Time after last transition until extinction or max_steps
            final_duration = et - last_step
            if is_coexist(current_pop) and final_duration > 0:
                time_in_state[current_pop] += final_duration

        mean_et = np.mean(ext_times)
        p_coexist_300 = coexist_at_300 / K

        print(f"    E[extinction time] = {mean_et:.2f}")
        print(f"    P(coexist@300)     = {p_coexist_300:.4f} ({coexist_at_300}/{K})")

        # Estimate alpha for states with enough data
        alpha_effective = {}
        denom = N * (N - 1)

        for s in time_in_state:
            if not is_coexist(s):
                continue
            l, m, n = s
            t_s = time_in_state[s]
            if t_s < 10:  # skip states with too few samples
                continue

            # Baseline rates (well-mixed, per timestep)
            base_RS = 2 * l * m / denom if l > 0 and m > 0 else 0
            base_SP = 2 * m * n / denom if m > 0 and n > 0 else 0
            base_PR = 2 * l * n / denom if l > 0 and n > 0 else 0

            a_R = RS_count[s] / (base_RS * t_s) if base_RS * t_s > 0 else 1.0
            a_S = SP_count[s] / (base_SP * t_s) if base_SP * t_s > 0 else 1.0
            a_P = PR_count[s] / (base_PR * t_s) if base_PR * t_s > 0 else 1.0

            alpha_effective[s] = (a_R, a_S, a_P)

        # Show estimated alpha at key states
        s0 = initial_pop
        show_states = [
            (s0, "initial"),
            ((n_R + 2, max(1, n_S - 1), max(1, n_P - 1)), "R dominant"),
            ((max(1, n_R - 1), n_S + 2, max(1, n_P - 1)), "S dominant"),
            ((max(1, n_R - 1), max(1, n_S - 1), n_P + 2), "P dominant"),
        ]
        show_states = [(s, desc) for s, desc in show_states
                       if sum(s) == N and all(x >= 0 for x in s)]

        print(f"\n    Estimated alpha (R, S, P):")
        for s, desc in show_states:
            if s in alpha_effective:
                aR, aS, aP = alpha_effective[s]
                print(f"      {str(s):<12} ({desc:<14}) alpha_R={aR:.3f}  alpha_S={aS:.3f}  alpha_P={aP:.3f}")
            else:
                print(f"      {str(s):<12} ({desc:<14}) insufficient data")

        all_results[policy] = {
            "mean_et": mean_et,
            "p_coexist_300": p_coexist_300,
            "alpha_effective": alpha_effective,
            "ext_times": ext_times,
        }

    return all_results


# Step 3: Model Checking with Estimated coefficient, alpha

def step3_model_check_with_alpha(all_results, initial_pop=(4, 4, 4), T_horizon=300):
    '''
    Plug estimated alpha into population-level transitions,
    then compute P[G <= T coexist] via value iteration.
    Compare with simulator Monte Carlo (MC).
    '''
    N_pop = sum(initial_pop)

    print("\n" + "=" * 65)
    print("STEP 3: Population Model Checking with alpha")
    print("=" * 65)

    # Build state space (same as rps_pctl.py)
    S = [(l, m, N_pop - l - m)
         for l in range(N_pop + 1)
         for m in range(N_pop - l + 1)]
    S_idx = {s: i for i, s in enumerate(S)}
    nS = len(S)
    transient_states = [s for s in S if s[0] > 0 and s[1] > 0 and s[2] > 0]
    absorbing_states = set(s for s in S if s[0] == 0 or s[1] == 0 or s[2] == 0)

    denom = N_pop * (N_pop - 1)

    def build_transition_matrix(alpha_map, default_alpha=(1.0, 1.0, 1.0)):
        '''
        Build transition matrix using state-dependent alpha.
        For states without data, use default_alpha.
        '''
        T_mat = {}
        for s in S:
            si = S_idx[s]
            if s in absorbing_states:
                T_mat[si] = [(1.0, si)]
                continue

            l, m, n = s
            aR, aS, aP = alpha_map.get(s, default_alpha)

            transitions = []
            prob_sum = 0.0

            if l > 0 and m > 0:
                p = (2 * l * m / denom) * aR
                transitions.append((p, S_idx[(l+1, m-1, n)]))
                prob_sum += p
            if m > 0 and n > 0:
                p = (2 * m * n / denom) * aS
                transitions.append((p, S_idx[(l, m+1, n-1)]))
                prob_sum += p
            if l > 0 and n > 0:
                p = (2 * l * n / denom) * aP
                transitions.append((p, S_idx[(l-1, m, n+1)]))
                prob_sum += p

            # Cap probabilities
            if prob_sum > 1.0:
                scale = 0.99 / prob_sum
                transitions = [(p * scale, sj) for p, sj in transitions]
                prob_sum *= scale

            if 1.0 - prob_sum > 0:
                transitions.append((1.0 - prob_sum, si))

            T_mat[si] = transitions
        return T_mat

    def compute_pctl_coexist(T_mat, T_horizon):
        # P[G <= T coexist] via backward induction with fixed transition matrix.
        V = np.zeros(nS)
        for s in transient_states:
            V[S_idx[s]] = 1.0

        for t in range(T_horizon):
            V_new = np.zeros(nS)
            for s in transient_states:
                si = S_idx[s]
                V_new[si] = sum(p * V[sj] for p, sj in T_mat[si])
            V = V_new
        return V

    def compute_expected_time(T_mat, epsilon=1e-8, max_iter=50000):
        # E[extinction time] via value iteration with fixed transition matrix.
        V = np.zeros(nS)
        for iteration in range(max_iter):
            delta = 0.0
            V_new = np.copy(V)
            for s in transient_states:
                si = S_idx[s]
                val = 1.0 + sum(p * V[sj] for p, sj in T_mat[si])
                delta = max(delta, abs(val - V[si]))
                V_new[si] = val
            V = V_new
            if delta < epsilon:
                break
        return V

    s0 = initial_pop
    s0_idx = S_idx[s0]

    abstract_model = build_population_model(N_pop)
    V_pmax = pctl_bounded_coexist_model(abstract_model, T_horizon, mode="max")
    V_pmin = pctl_bounded_coexist_model(abstract_model, T_horizon, mode="min")
    V_prand = pctl_bounded_coexist_model(abstract_model, T_horizon, mode="random")
    E_pmax = expected_extinction_time_model(abstract_model, mode="max")
    E_pmin = expected_extinction_time_model(abstract_model, mode="min")
    E_prand = expected_extinction_time_model(abstract_model, mode="random")

    print(f"\n  Population model bounds:")
    print(f"    P_max[G<={T_horizon} coexist] = {V_pmax[s0_idx]:.6f}    E_max = {E_pmax[s0_idx]:.2f}")
    print(f"    P_rand[G<={T_horizon} coexist]= {V_prand[s0_idx]:.8f}  E_rand= {E_prand[s0_idx]:.2f}")
    print(f"    P_min[G<={T_horizon} coexist] = {V_pmin[s0_idx]:.8f}  E_min = {E_pmin[s0_idx]:.2f}")

    # For each policy: compare three numbers
    print(f"\n  {'-' * 80}")
    print(f"  {'Policy':<12} {'E[ext] sim':<12} {'E[ext] alpha_eff':<14} {'E[ext] ratio':<14} "
          f"{'P(coex) sim':<14} {'P(coex) alpha_eff':<14}")
    print(f"  {'-' * 80}")

    for policy in all_results:
        res = all_results[policy]
        alpha_map = res["alpha_effective"]

        # Build transition matrix from estimated alpha
        T_mat = build_transition_matrix(alpha_map)

        # Model checking with alpha
        V_alpha = compute_pctl_coexist(T_mat, T_horizon)
        E_alpha = compute_expected_time(T_mat)

        p_coexist_alpha = V_alpha[s0_idx]
        e_time_alpha = E_alpha[s0_idx]

        # Simulator values
        e_time_sim = res["mean_et"]
        p_coexist_sim = res["p_coexist_300"]

        ratio = e_time_sim / e_time_alpha if e_time_alpha > 0 else float('inf')

        print(f"  {policy:<12} {e_time_sim:<12.2f} {e_time_alpha:<14.2f} {ratio:<14.2f} "
              f"{p_coexist_sim:<14.4f} {p_coexist_alpha:<14.8f}")

    # Comparison table with bounds
    print(f"\n  {'-' * 65}")
    print(f"  Where does each policy's alpha land relative to bounds?")
    print(f"  {'-' * 65}")
    print(f"  {'Policy':<12} {'E[extinction]':<15} {'Position'}")
    print(f"  {'-' * 65}")

    for policy in all_results:
        alpha_map = all_results[policy]["alpha_effective"]
        T_mat = build_transition_matrix(alpha_map)
        E_alpha = compute_expected_time(T_mat)
        e = E_alpha[s0_idx]

        e_min = E_pmin[s0_idx]
        e_max = E_pmax[s0_idx]
        e_rand = E_prand[s0_idx]

        if e_max > e_min:
            position = (e - e_min) / (e_max - e_min)
            print(f"  {policy:<12} {e:<15.2f} {position*100:.1f}% along P_mintoP_max  "
                  f"(min={e_min:.1f}, rand={e_rand:.1f}, max={e_max:.1f})")
        else:
            print(f"  {policy:<12} {e:<15.2f} (bounds collapsed)")


#
# Main
#

if __name__ == "__main__":
    # Initial population by type
    N_R = 8
    N_S = 8
    N_P = 8
    INITIAL_POP = (N_R, N_S, N_P)

    print("Formal Synthesis of Safe Coexistence")
    print("Bridge: Agent-Level <-> Population-Level")
    print("=" * 65)

    t0 = time.time()

    # Step 1
    sim_et, model_et = step1_baseline_validation(
        initial_pop=INITIAL_POP, K=300, max_steps=5000
    )

    # Step 2
    all_results = step2_estimate_alpha_effective(
        initial_pop=INITIAL_POP, K=300, max_steps=5000
    )

    # Step 3
    step3_model_check_with_alpha(
        all_results, initial_pop=INITIAL_POP, T_horizon=300
    )

    print(f"\nTotal time: {time.time()-t0:.1f}s")
