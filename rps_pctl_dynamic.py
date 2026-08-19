"""
PCTL model checking for the population-level RPS model.

The state is the population count (R, S, P).
Each type selects an action that changes its encounter rate.
The code computes bounded coexistence probability, expected extinction time,
and extinction class probabilities.
"""

import os
import numpy as np
from itertools import product
import time

# System parameters

N = int(os.environ.get("RPS_TOTAL_POP", "12"))  # total population

# Action set per type. Each value scales the prey encounter rate.
ACTION_SET = [0.5, 1.0, 1.5]
ACTION_LABEL = {0.5: "PAS", 1.0: "NEU", 1.5: "AGG"}

# Joint action space: one action for each type
JOINT_ACTIONS = list(product(ACTION_SET, repeat=3))

# State space

S = [(l, m, N - l - m)
     for l in range(N + 1)
     for m in range(N - l + 1)]

S_index = {s: i for i, s in enumerate(S)}
nS = len(S)

absorbing = [s for s in S if s[0] == 0 or s[1] == 0 or s[2] == 0]
transient = [s for s in S if s[0] > 0 and s[1] > 0 and s[2] > 0]

absorbing_set = set(absorbing)
transient_set = set(transient)

print(f"N = {N}")
print(f"|S| = {nS},  |S_transient| = {len(transient)},  |S_absorbing| = {len(absorbing)}")
print(f"|A| = {len(JOINT_ACTIONS)} joint actions")

# Transition model
# Base rates follow the well-mixed population model.
# Each action scales the corresponding encounter rate.

def get_transitions(state, action):
    """Compute transition probabilities for one state and one joint action."""
    l, m, n = state
    alpha_R, alpha_S, alpha_P = action

    if state in absorbing_set:
        return [(1.0, state)]

    denom = N * (N - 1)
    transitions = []
    prob_sum = 0.0

    # R beats S
    if l > 0 and m > 0:
        p = (2 * l * m / denom) * alpha_R
        transitions.append((p, (l + 1, m - 1, n)))
        prob_sum += p

    # S beats P
    if m > 0 and n > 0:
        p = (2 * m * n / denom) * alpha_S
        transitions.append((p, (l, m + 1, n - 1)))
        prob_sum += p

    # P beats R
    if l > 0 and n > 0:
        p = (2 * l * n / denom) * alpha_P
        transitions.append((p, (l - 1, m, n + 1)))
        prob_sum += p

    # Cap the total transition probability if needed
    if prob_sum > 1.0:
        scale = 0.99 / prob_sum
        transitions = [(p * scale, s) for p, s in transitions]
        prob_sum *= scale

    # Self-loop for no population change
    prob_self = 1.0 - prob_sum
    if prob_self > 0:
        transitions.append((prob_self, state))

    return transitions

# Precompute transitions for all joint actions

print("\nPrecomputing transitions for all state-action pairs...")
t0 = time.time()

# T[ja_index][s_index] stores transition pairs
T = {}
for ja_idx, ja in enumerate(JOINT_ACTIONS):
    T[ja_idx] = {}
    for s in S:
        si = S_index[s]
        raw = get_transitions(s, ja)
        T[ja_idx][si] = [(p, S_index[sp]) for p, sp in raw]

print(f"Done in {time.time()-t0:.2f}s\n")

# Bounded coexistence probability
# Value iteration is used for random, max, and min modes.

def pctl_bounded_coexist(T_horizon, mode="random"):
    """Compute bounded coexistence probability for all states."""
    # Initial value is one for coexistence states
    V = np.zeros(nS)
    for s in transient:
        V[S_index[s]] = 1.0

    neutral_idx = JOINT_ACTIONS.index((1.0, 1.0, 1.0))
    policy = {}

    for t in range(1, T_horizon + 1):
        V_new = np.zeros(nS)

        for s in transient:
            si = S_index[s]

            if mode == "random":
                # Fixed neutral policy
                V_new[si] = sum(p * V[sj] for p, sj in T[neutral_idx][si])

            elif mode == "max":
                # Maximize over joint actions
                best_val = -1.0
                best_ja = None
                for ja_idx in range(len(JOINT_ACTIONS)):
                    val = sum(p * V[sj] for p, sj in T[ja_idx][si])
                    if val > best_val:
                        best_val = val
                        best_ja = JOINT_ACTIONS[ja_idx]
                V_new[si] = best_val
                if t == T_horizon:
                    policy[s] = best_ja

            elif mode == "min":
                # Minimize over joint actions
                worst_val = 2.0
                worst_ja = None
                for ja_idx in range(len(JOINT_ACTIONS)):
                    val = sum(p * V[sj] for p, sj in T[ja_idx][si])
                    if val < worst_val:
                        worst_val = val
                        worst_ja = JOINT_ACTIONS[ja_idx]
                V_new[si] = worst_val
                if t == T_horizon:
                    policy[s] = worst_ja

        V = V_new

    return V, policy


# Expected extinction time

def expected_extinction_time(mode="random", epsilon=1e-8, max_iter=100000):
    """Compute expected extinction time for a selected mode."""
    neutral_idx = JOINT_ACTIONS.index((1.0, 1.0, 1.0))
    V = np.zeros(nS)

    for iteration in range(1, max_iter + 1):
        delta = 0.0
        V_new = np.copy(V)

        for s in transient:
            si = S_index[s]

            if mode == "random":
                val = 1.0 + sum(p * V[sj] for p, sj in T[neutral_idx][si])

            elif mode == "max":
                best = -1.0
                for ja_idx in range(len(JOINT_ACTIONS)):
                    val_a = 1.0 + sum(p * V[sj] for p, sj in T[ja_idx][si])
                    if val_a > best:
                        best = val_a
                val = best

            elif mode == "min":
                worst = float('inf')
                for ja_idx in range(len(JOINT_ACTIONS)):
                    val_a = 1.0 + sum(p * V[sj] for p, sj in T[ja_idx][si])
                    if val_a < worst:
                        worst = val_a
                val = worst

            delta = max(delta, abs(val - V[si]))
            V_new[si] = val

        V = V_new
        if delta < epsilon:
            print(f"  [{mode}] E[T_absorb] converged in {iteration} iterations")
            break

    return V


# Extinction class probabilities

def extinction_class_prob(extinct_idx, mode="random", epsilon=1e-8, max_iter=100000):
    """Compute the probability that one type is the first to go extinct."""
    neutral_idx = JOINT_ACTIONS.index((1.0, 1.0, 1.0))

    U = np.zeros(nS)
    for s in absorbing:
        if s[extinct_idx] == 0:
            U[S_index[s]] = 1.0

    for iteration in range(1, max_iter + 1):
        delta = 0.0
        U_new = np.copy(U)

        for s in transient:
            si = S_index[s]

            if mode == "random":
                val = sum(p * U[sj] for p, sj in T[neutral_idx][si])
            elif mode == "max":
                # Use the neutral policy for extinction-class comparison
                val = sum(p * U[sj] for p, sj in T[neutral_idx][si])
            elif mode == "min":
                val = sum(p * U[sj] for p, sj in T[neutral_idx][si])

            delta = max(delta, abs(val - U[si]))
            U_new[si] = val

        U = U_new
        if delta < epsilon:
            break

    return U


# Main

if __name__ == "__main__":
    s0 = (N // 3, N // 3, N - 2 * (N // 3))
    s0_idx = S_index[s0]
    T_HORIZON = 300

    print(f"Initial state s0 = {s0}  (Rock={s0[0]}, Scissors={s0[1]}, Paper={s0[2]})")
    print(f"Horizon T = {T_HORIZON}")
    print("=" * 65)

    # Property phi1: bounded coexistence

    print("\n[phi1] Bounded Coexistence: P[ G<=T coexist | s0 ]")
    print("-" * 65)

    V_rand, _ = pctl_bounded_coexist(T_HORIZON, mode="random")
    V_max, policy_max = pctl_bounded_coexist(T_HORIZON, mode="max")
    V_min, policy_min = pctl_bounded_coexist(T_HORIZON, mode="min")

    print(f"  P_random  = {V_rand[s0_idx]:.8f}")
    print(f"  P_max     = {V_max[s0_idx]:.8f}")
    print(f"  P_min     = {V_min[s0_idx]:.8f}")
    print(f"  P_max / P_random = {V_max[s0_idx] / V_rand[s0_idx]:.2f}x")

    # Property phi2: expected extinction time

    print(f"\n[phi2] Expected Extinction Time: E[ T_absorb | s0 ]")
    print("-" * 65)

    E_rand = expected_extinction_time(mode="random")
    E_max = expected_extinction_time(mode="max")
    E_min = expected_extinction_time(mode="min")

    print(f"  E_random  = {E_rand[s0_idx]:.2f} steps")
    print(f"  E_max     = {E_max[s0_idx]:.2f} steps")
    print(f"  E_min     = {E_min[s0_idx]:.2f} steps")
    print(f"  E_max / E_random = {E_max[s0_idx] / E_rand[s0_idx]:.2f}x")

    # Property phi3: extinction class probabilities

    print(f"\n[phi3] Extinction Class Probabilities (baseline, no action)")
    print("-" * 65)

    U_R = extinction_class_prob(0, mode="random")
    U_S = extinction_class_prob(1, mode="random")
    U_P = extinction_class_prob(2, mode="random")

    print(f"  P(Rock extinct first)     = {U_R[s0_idx]:.6f}")
    print(f"  P(Scissors extinct first) = {U_S[s0_idx]:.6f}")
    print(f"  P(Paper extinct first)    = {U_P[s0_idx]:.6f}")
    print(f"  Sum = {U_R[s0_idx] + U_S[s0_idx] + U_P[s0_idx]:.6f}")

    # Property phi4: synthesized policy analysis

    print(f"\n[phi4] Synthesized Cooperative Policy pi_max")
    print("-" * 65)
    print(f"  {'State':<15} {'Context':<20} {'R':<6} {'S':<6} {'P':<6}")
    print(f"  {'-'*15} {'-'*20} {'-'*6} {'-'*6} {'-'*6}")

    analysis_states = [
        (s0, "balanced"),
        ((6, 3, 3), "R dominant"),
        ((3, 6, 3), "S dominant"),
        ((3, 3, 6), "P dominant"),
        ((8, 2, 2), "R very dominant"),
        ((2, 8, 2), "S very dominant"),
        ((2, 2, 8), "P very dominant"),
        ((1, 5, 6), "R near extinct"),
        ((5, 1, 6), "S near extinct"),
        ((6, 5, 1), "P near extinct"),
    ]

    for s, desc in analysis_states:
        if s in policy_max:
            a = policy_max[s]
            print(f"  {str(s):<15} {desc:<20} "
                  f"{ACTION_LABEL[a[0]]:<6} {ACTION_LABEL[a[1]]:<6} {ACTION_LABEL[a[2]]:<6}")

    print(f"\n[phi4] Synthesized Adversarial Policy pi_min")
    print("-" * 65)
    print(f"  {'State':<15} {'Context':<20} {'R':<6} {'S':<6} {'P':<6}")
    print(f"  {'-'*15} {'-'*20} {'-'*6} {'-'*6} {'-'*6}")

    for s, desc in analysis_states:
        if s in policy_min:
            a = policy_min[s]
            print(f"  {str(s):<15} {desc:<20} "
                  f"{ACTION_LABEL[a[0]]:<6} {ACTION_LABEL[a[1]]:<6} {ACTION_LABEL[a[2]]:<6}")

    # Sensitivity over initial states

    print(f"\n[Sensitivity] P[G<=T coexist] across initial states")
    print("-" * 65)
    print(f"  {'s0':<15} {'P_min':<12} {'P_random':<12} {'P_max':<12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")

    test_states = [
        (4, 4, 4), (6, 3, 3), (3, 6, 3), (3, 3, 6),
        (8, 2, 2), (2, 8, 2), (2, 2, 8),
        (1, 1, 10), (10, 1, 1), (1, 10, 1),
    ]

    for s in test_states:
        if sum(s) != N:
            continue
        si = S_index[s]
        print(f"  {str(s):<15} {V_min[si]:<12.8f} {V_rand[si]:<12.8f} {V_max[si]:<12.8f}")

    # Summary

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  System: N={N} agents, |S|={nS} states, |A|=27 joint actions")
    print(f"  Initial state: s0={s0}")
    print(f"  Horizon: T={T_HORIZON}")
    print(f"")
    print(f"  {'Metric':<30} {'P_min':<12} {'P_random':<12} {'P_max':<12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'P[G<=T coexist]':<30} {V_min[s0_idx]:<12.6f} {V_rand[s0_idx]:<12.6f} {V_max[s0_idx]:<12.6f}")
    print(f"  {'E[extinction time]':<30} {E_min[s0_idx]:<12.2f} {E_rand[s0_idx]:<12.2f} {E_max[s0_idx]:<12.2f}")
