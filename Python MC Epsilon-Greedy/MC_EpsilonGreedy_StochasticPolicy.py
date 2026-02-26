
from Stochastic_Grid_world import standard_sticky_GW, GridWorld
from random import choice, randint
from numpy import random



# Helper Functions :
# ====================================================

# Playing an episod by pi.
# ------------------------------
def play_episod( pi: dict, gw: GridWorld, mstep:int ):
    
    stepc = 0
    vstates = []    # visited states
    actions = []
    rewards = [0.0] # starting state's reward is -0-
    
    
    while not gw.game_over():
        
        vst = gw.current_state
        
        act_spc = list( pi[vst].keys() )
        act_prb = list( pi[vst].values() )
        
        idx_lst = [ i for i in range(len(act_spc)) ]
        a_index = random.choice( idx_lst, p=act_prb )
        
        act = act_spc[a_index]       
        rew = gw.transition(act)
        
        vstates.append(vst)
        actions.append(act)
        rewards.append(rew)
        
        stepc += 1
        if stepc > mstep :
            break
    
    if gw.game_over() :
        vstates.append(gw.current_state)
        actions.append("#")
    
    return vstates, actions, rewards


# argmax best action by given state and Q-func:
# ----------------------------------------------
def best_action( state, q_func:dict, epsilon:float, env:GridWorld ):
    
    action_space = env.get_actions(state) 
    action_probs = { act:0.0 for act in action_space }
    action_size  = len(list(action_probs.keys()))
    
    argmax = action_space[0]
    max_value = q_func.get((state,argmax), 0.0 )
    
    for act in action_space[1:]:
        
        value = q_func.get((state,act), 0.0 )
        
        if value > max_value :
            max_value = value
            argmax = act
        
    for act in action_space :
        if act == argmax :
            action_probs[act] = 1 - epsilon + epsilon/action_size
        else :
            action_probs[act] = epsilon/action_size
    
    return action_probs


# Print Policy func Table:
# -----------------------------
def print_policy(V:dict,g:GridWorld):
    
    print("====== Current Policy ======")
    for i in range(g.rows):
        print("============================")
        for j in range(g.cols):
            
            v = V.get( (i,j),{})
            
            if v == {} :
                print("|||||||", end="")
            
            else:
                best = ""
                bstv = -10000
                
                for act in v:
                    if v[act] >= bstv :
                        best = act
                        bstv = v[act]
                
                print(f"   {best}  |",end="")
            
        print()
    print("============================")
    print()




# Monte Carlo action-value improvement.
# =====================================================
def MC_policy_improvement( policy:dict, q_func:dict, g_returns:dict, env:GridWorld, epsilon, gamma=0.9 )-> dict:
        
    # Reset to starting state:
    env.current_state = env.start_state
    
    states, actions, rewards = play_episod( policy, env, 20 )
    
    g = 0.0
    visited_states = []     # vs = [ (state,act), ... ]
    
    
    # G(t) = R(t+1) + YG(t+1) 
    for t in range( len(states)-2, -1, -1 ):
        
        g = rewards[t+1] + gamma*g
        s = states[t]
        a = actions[t]
        key = (s,a)
        
        # MC first Visit
        if key not in visited_states :
            
            visited_states.append(key)
            
            g_returns[key]["count"] += 1
            g_returns[key]["mean"] = ((g_returns[key]["count"]-1)*g_returns[key]["mean"] + g ) / g_returns[key]["count"]
            
            q_func[key] = g_returns[key]["mean"]            # better to calc by count, new-G, prev-mean.
            policy[s] = best_action( s, q_func, epsilon, env )       # Updating probs of actions in current state in the policy.
            
        
    return q_func, g_returns, policy




# Test Space :
#-----------------------
if __name__ == "__main__" :
    
    
    gw = standard_sticky_GW(-0.09)
    
    epsilon = 0.1                       # Epsilon-Greedy factor
    pi = {}                             # P={ state:act }
    action_value_function_Q = {}        # Q={ (state,act):value }
    g_returns = {}                      # R={ (state,act):{} }
    
    
    # initialize Policy, Q-function, Returns-list -> a private method in Agent class!
    for i in range(gw.rows):
        for j in range(gw.cols):
            
            st = (i,j)
            if gw.is_terminal(st):
                continue
            
            act_spc = gw.get_actions(st)
            pi[st] = {}
            prob = 1.0/len(act_spc)
            
            for act in act_spc :
                
                pi[st][act] = prob
                action_value_function_Q[(st,act)] = 0.0
                g_returns[ (st,act) ] = { "mean":0.0, "count":0 }
            
    
    # Output:
    print("> Random Initial Policy")
    print_policy(pi,gw)
    # print("> Zero Initial Q-function")
    # print_q_func(action_value_function_Q)
    
    
    # Find Optimal Policy:
    for i in range(10001):
        if i%1000 == 0 :
            print("> itr:",i)
        action_value_function_Q, g_returns, pi = MC_policy_improvement(pi,action_value_function_Q,g_returns,gw,epsilon)
    
    
    # Output:
    print("\n> Optimal Founded Policy")
    print_policy(pi,gw)
    # print("> Optimal Founded Q-function")
    # print_q_func(action_value_function_Q)
    


