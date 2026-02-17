from Stochastic_Agent import Agent
from Stochastic_Grid_world import GridWorld, standard_sticky_GW
from random import choice

#-------------------------------------

def exper_data( grid_world:GridWorld ):
    
    states  = grid_world.states
    rewards = grid_world.rewards
    actions = set()
    
    for state in grid_world.actions:
        for act in grid_world.actions.get(state,[]):
            actions.add(act)
    
    actions = list(actions)
    return states, actions, rewards


#-------------------------------------

def print_value(V:dict,g:GridWorld):
    for i in range(g.rows):
        print("----------------------------")
        for j in range(g.cols):
            v = V.get( (i,j),0.0)
            
            if v >= 0 :
                print(" %.2f |"%v,end="")
            else:
                print("%.2f |"%v,end="")
            
        print()
    print("----------------------------")
    print()
    
def print_policy(V:dict,g:GridWorld):
    for i in range(g.rows):
        print("========================")
        for j in range(g.cols):
            
            v = V.get( (i,j),"&")
            if v == "&" :
                print("||||||", end="")
            else:
                print(f"  {v}  |",end="")
            
        print()
    print("========================")
    print()


#-------------------------------------
cost = -0.1
env = standard_sticky_GW(cost)
agent = Agent()
print("> COST:",cost)



# exper-base data loading ;
states, action_space, rewards = exper_data(env)


# Value state initialize.
agent.initialize_Vs( env.rows, env.cols )

# Random Policy initialize.
for st in env.actions :
    agent.policy[st] = choice( action_space )



print("> Initial Policy:")
print_policy(agent.policy,env)
print("> Initial Value-function:")
print_value(agent.value_state_function,env)



# Setting data.
agent.exper_states  = states
agent.exper_actions = action_space
agent.exper_rewards = rewards



# Updating into Optimal Policy
agent.Update_policy(env)



# Logging
print("> Final Value function:")
print_value(agent.value_state_function,env)
print("> Final Optimal Policy:")
print_policy(agent.policy,env)

#-------------------------------------
