import numpy as np
# l : Rock , m : Scissors , n : Paper
l0,m0,n0 = 4,4,4
N = l0 + m0 + n0
s0 = (l0,m0,n0)

# State Space
S = [(ll,mm,nn) for ll in range(N+1) for mm in range(N+1) for nn in range(N+1) if ll + mm + nn == N]
print(S)
# Absorbing States
absorbing = [s for s in S if s[0] == 0 or s[1] == 0 or s[2] == 0]

# get Transition

def get_transitions(state):
    l,m,n = state
    transitions = []
    
    denom = N * (N-1) 
    prob_sum = 0.0
    # R win <-> S Lose -> l + 1 , m - 1 , n
    if l > 0 and m > 0:
        prob =  (2 * l * m )/ denom
        transitions.append((prob, (l + 1 ,m - 1, n)))
        prob_sum += prob
    # S win <-> P Lose -> l , m + 1 , n - 1
    if m > 0 and n >0 :
        prob =  (2 * m * n )/ denom
        transitions.append((prob, (l ,m + 1, n - 1)))
        prob_sum += prob
    # P win <-> R Lose -> l - 1 , m  , n + 1  
    if l > 0 and n >0 :
        prob =  (2 * l * n )/ denom
        transitions.append((prob, (l - 1,m , n + 1)))
        prob_sum += prob
    prob_self = 1 - prob_sum
    if prob_self > 0: 
        transitions.append((prob_self, (l, m , n)))
    return transitions
# Task 1 : Expected Time step to Absorb
def value_iteration_expected_time(epsilon = 1e-6):
    V = {s: 0.0 for s in S}
    iteration = 0
    while True:
        max_delta = 0.0
        new_V = V.copy()
        
        for s in S:
            # Expected time from an absorbing state is exactly 0
            if s in absorbing:
                continue 
                
            # Bellman Update: V(s) = 1 + sum( P(s'|s) * V(s') )
            expected_future_time = 0.0
            for prob, next_s in get_transitions(s):
                expected_future_time += prob * V[next_s]
                
            updated_v = 1.0 + expected_future_time
            max_delta = max(max_delta, abs(updated_v - V[s]))
            new_V[s] = updated_v
            
        V = new_V
        iteration += 1
        
        if max_delta < epsilon:
            print(f"[*] Task 1 (Expected Time) converged in {iteration} iterations.")
            break
            
    return V
# ==========================================
# Task 2: Probability of Reaching Each Absorbing Class
# ==========================================
def value_iteration_class_prob(extinct_index, epsilon=1e-6):

    # Initialize U(s): 1.0 if the target type is 0, 0.0 otherwise
    U = {s: 0.0 for s in S}
    for s in absorbing:
        if s[extinct_index] == 0:
            U[s] = 1.0
            
    iteration = 0
    while True:
        max_delta = 0.0
        new_U = U.copy()
        
        for s in S:
            # Absorbing state probabilities are strictly fixed at 1 or 0
            if s in absorbing:
                continue 
                
            # Bellman Update: U(s) = sum( P(s'|s) * U(s') )
            expected_prob = 0.0
            for prob, next_s in get_transitions(s):
                expected_prob += prob * U[next_s]
                
            max_delta = max(max_delta, abs(expected_prob - U[s]))
            new_U[s] = expected_prob
            
        U = new_U
        iteration += 1
        
        if max_delta < epsilon:
            break
            
    return U

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print(f"Initial State: {s0} (Rock: {s0[0]}, Scissors: {s0[1]}, Paper: {s0[2]})\n")

    # 1. Calculate Expected Time to Absorption (Collapse of Coexistence)
    V_time = value_iteration_expected_time()
    print(f"=> Expected time steps until coexistence breaks (any type hits 0): {V_time[s0]:.2f} steps\n")

    # 2. Calculate the probability of which type goes extinct first
    # Index 0: Rock, Index 1: Scissors, Index 2: Paper
    prob_rock_extinct = value_iteration_class_prob(0)
    prob_scissors_extinct = value_iteration_class_prob(1)
    prob_paper_extinct = value_iteration_class_prob(2)

    print("=> Which type is most likely to go extinct FIRST, causing the system collapse?")
    print(f"  Probability that Rock goes extinct first (Scisssors win):     {prob_rock_extinct[s0]:.4f} ( {prob_rock_extinct[s0]*100:.2f}% )")
    print(f"  Probability that Scissors goes extinct first (Paper win): {prob_scissors_extinct[s0]:.4f} ( {prob_scissors_extinct[s0]*100:.2f}% )")
    print(f"  Probability that Paper goes extinct first (Rock win):    {prob_paper_extinct[s0]:.4f} ( {prob_paper_extinct[s0]*100:.2f}% )")
    
    # Validation: The sum of probabilities should be exactly 1.0
    total_prob = prob_rock_extinct[s0] + prob_scissors_extinct[s0] + prob_paper_extinct[s0]
    print(f"\n  (Probability Sum Validation: {total_prob:.4f})")
