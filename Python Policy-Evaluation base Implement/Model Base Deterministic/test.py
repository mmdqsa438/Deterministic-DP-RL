from Deterministic_Agent import Agent, save_agent, load_agent
from Deterministic_Grid_World import GridWorld, standard_GW
from random import randint, choice
from pprint import pprint

#-------------------------------------

def model_data( grid_world:GridWorld ):
    
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

env = standard_GW()
agent = Agent()



# Model-base data loading ;
states, action_space, rewards = model_data(env)


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
agent.model_states  = states
agent.model_actions = action_space
agent.model_rewards = rewards



# Updating into Optimal Policy
agent.Update_policy(env)



# Logging
print("> Final Value function:")
print_value(agent.value_state_function,env)
print("> Final Optimal Policy:")
print_policy(agent.policy,env)

#-------------------------------------
