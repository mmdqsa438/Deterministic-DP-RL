# -------------------------------------------------------

# Due to the complete model-oriented nature of this implementation, 
# which means that the agent code fragment is equal to the problem-solving code fragment.
# In this implementation, to reduce complexity and save on implementation,
# the definition of the agent class according to the rules of classical object-oriented design is avoided,
# and the agent code is equal to the main code.


# [ NOTE ]
# This is a fully model-base implementation. 
# This means that the agent has access to all the modeled information of the environment. 
# It sees everything, It can read all the actions 
# and the real reward and probability functions of the environment are available.

# -------------------------------------------------------

from Deterministic_Grid_World import standard_GW, GridWorld, print_policy, print_value


# Helper Functions
# ==================

# Max value by avl actions of a state:
# ------------------------------------
def max_value_by_actions( cstste:tuple, valuefunc:dict, gamma:float, gw:GridWorld ):
    
    max_value = -1e3
    
    for act in gw.get_actions(cstste) :
        
        value = 0.0
        for n_state in gw.states :
            value += gw.probability(n_state,cstste,act) * ( gw.rewards.get(n_state) + (gamma * valuefunc.get(n_state)) )
        
        if value > max_value :
            max_value = value
    
    return max_value


# Arg-max optimal value:
# ---------------------------
def max_action_by_value( cstste:tuple, optimalvalue:dict, gamma:float, gw:GridWorld ):
    
    best_action = ""
    best_value  = -1e3
    
    for act in gw.get_actions(cstste):
        
        value = 0.0
        for n_state in gw.states :
            value += gw.probability(n_state,cstste,act) * ( gw.rewards.get(n_state) + (gamma * optimalvalue.get(n_state)) )
            
        if value > best_value :
            best_value  = value
            best_action = act
    
    return best_action




# Main Function logic
# =========================
if __name__ == "__main__" :
    
    cost = -2                                                   # Non-terminal states reward
    gamma = 0.9                                                   # discount factor
    threshold = 1e-3                                              # max Abs Error -> value iteration converge
    
    gw = standard_GW(cost=cost)                                   # Instanciation GW
    value_func = {  state:0.0 for state in gw.states }            # Value func initialiize by -0-
    
    print()
    print("> Standard Grid-World")
    print("> size: 3x4 in cost:",cost)
    print("> Disccount Factor:",gamma)
    print("> Converge threshold:",threshold)
    print()
    
    print("> Value function Initialized")
    print_value(value_func,gw)


    # GW Value-Evaluation 
    # [Why?!]-> try to find the best future-return by different actions of states: mag G(s,a)
    while True :
        
        delta = 0.0  # max change per iteratin.
        
        for state in gw.states :
            
            if gw.is_terminal(state) : # only checking non-terminal states
                continue
            
            old_value = value_func.get(state,0.0)
            value_func[state] = max_value_by_actions( state, value_func, gamma, gw )
            
            delta = max( delta, abs(value_func[state]-old_value))
        
        if delta < threshold :
            break
        
    
    print("> Optimal Val-function found")
    print_value(value_func,gw)
    
    
    optimal_policy = {}
    for state in gw.states:
        
        if gw.is_terminal(state) :
            continue
        
        optimal_policy[state] = max_action_by_value( state, value_func, gamma, gw )
    
    print()
    print("> Optimal Grid's Poliy found")
    print_policy(optimal_policy,gw)
    
    


