
import pickle
from numpy import random

# Stochastic Grid-world class:
#---------------------------------------------------------------

# stochastic transition function -> if you choose U, maybe go down.
# Deterministic rewards -> you will get same-reward from any way you reach s' form s.
class GridWorld :
    
    def __init__(self, rows, cols, start_state):
        
        self.cols:int = cols
        self.rows:int = rows
        
        # Agent Current State = (i,j)
        self.current_state: tuple = start_state
        
        # States = [ (i,j) ]
        self.states : list = []
        
        # Actions = { (i,j):["U","D","R","L"] }
        self.actions: dict = {}
        
        # Rewards = { (i,j):float }
        self.rewards: dict = {}
        
        self.probabilities: dict = {} 
        
        # Probabilities:  key -> ( state, "action" )     |    value -> { "next-state":prob, "next-state2":prob }
        self.__initialize_states()
        
    
    
    # Current State Actions getter.
    #----------------------------------
    def get_actions(self):
        return self.actions.get(self.current_state)
    
    
    # Environment Transition function: T(s,a)-> r,s'
    #-----------------------------------------------
    def transition(self, action:str ) -> float :
        
        next_state_probs:dict = self.probabilities.get( (self.current_state,action),{} )
        
        if next_state_probs == {} : 
            return 0.0
        
        next_states = list(next_state_probs.keys())
        next_probs  = list(next_state_probs.values())
        
        index_list = [ i for i in range(len(next_states)) ]
        index = random.choice(index_list, p=next_probs)
        self.current_state = next_states[index]
        
        return self.rewards.get(self.current_state,0.0)
    
    
    # Environment Transition Probability function: p(s',r|s,a)-> [0,1]
    #-----------------------------------------------------------------
    def probability(self, to_state, from_state, action ):
        
        i,j = 0,0
        if   action == "U": i -= 1
        elif action == "D": i += 1
        elif action == "R": j += 1
        elif action == "L": j -= 1
        else : return 0.0
        
        if action in self.actions.get(from_state,[]):
            if tuple((from_state[0]+i, from_state[1]+j)) == to_state :
                
                probs:dict = self.probabilities.get((from_state,action),{})
                return probs.get(to_state, 0.0)
            
            else :
                return 0.0
        else : 
            return 0.0
    
    
    # Some God Mode methods prototype
    #---------------------------------
    def is_terminal(self, state ):
        if self.actions.get(state,[]) == [] :
            return True
        else:
            return False
    
    def undo_action(self, action)->None: raise NotImplementedError
    def get_next_state(self,state,action): raise NotImplementedError
    def set_state(self, new_state)->None: raise NotImplementedError
    #=================================
    
    # Game-over check-method.
    #----------------------------
    def game_over(self):
        if self.actions.get(self.current_state,[]) == [] :
            return True
        else:
            return False
    
    
    # Updating grid-world's actions & rewards config.
    #--------------------------------------------
    def set_config( self, actions:dict, rewards: dict, probs: dict )->None:
        self.actions = actions
        self.rewards = rewards
        self.probabilities = probs
    
    
    # Initialize Grid-world States.
    #----------------------------------
    def __initialize_states(self)->None:
        
        for r in range(self.rows):
            for c in range(self.cols):
                self.states.append((r,c))
        
    
#======================================================================================================
#======================================================================================================

# Global functions:
#============================

# saving GW configs into file.
#-----------------------------
def save_env(Gw, filename)->None:
    with open(filename, 'wb') as f:
        pickle.dump(Gw, f)


# Loading GW configs from file.
#-------------------------------
def load_env(filename) -> GridWorld:
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Standard GW creation.
#-------------------------------
def standard_sticky_GW(cost: float):
    r = 3
    c = 4
    start_state = (2,0)
    g = GridWorld(r,c,start_state)
    
    rewards = {}
    for i in range(r):
        for j in range(c):
            rewards[(i,j)] = cost
    rewards[(0,3)] = 1.0
    rewards[(1,3)] = -1
    rewards[(1,1)] = 0
    
    
    actions = {
        (0,0):["D","R"],
        (0,1):["R","L"],
        (0,2):["R","L","D"],
        (1,0):["U","D"],
        (1,2):["R","U","D"],
        (2,0):["U","R"],
        (2,1):["R","L"],
        (2,2):["U","R","L"],
        (2,3):["L","U"]
    }
    
    probs = {
        ( (0,0),"D" ): { (0,0):0.2 , (1,0):0.8 },
        ( (0,0),"R" ): { (0,0):0.2 , (0,1,):0.8 },

        ( (0,1),"L" ): { (0,1):0.2 , (0,0):0.8 },
        ( (0,1),"R" ): { (0,1):0.2 , (0,2):0.8 },

        ( (0,2),"R" ): { (0,2):0.2 , (0,3):0.8 },
        ( (0,2),"L" ): { (0,2):0.2 , (0,1):0.8 },
        ( (0,2),"D" ): { (0,2):0.2 , (1,2):0.8 },
        
        ( (1,0),"U" ): { (1,0):0.2 , (0,0):0.8 },
        ( (1,0),"D" ): { (1,0):0.2 , (2,0):0.8 },
        
        # ( (1,1), ): { (,): , (,): },
        
        ( (1,2),"U" ): { (1,2):0.2 , (0,2):0.4, (1,3):0.4 },
        ( (1,2),"D" ): { (1,2):0.2 , (2,2):0.8 },
        ( (1,2),"R" ): { (1,2):0.2 , (1,3):0.8 },
        
        ( (2,0),"U" ): { (2,0):0.2 , (1,0):0.8 },
        ( (2,0),"R" ): { (2,0):0.2 , (2,1):0.8 },
        
        ( (2,1),"R" ): { (2,1):0.2 , (2,2):0.8 },
        ( (2,1),"L" ): { (2,1):0.2 , (2,0):0.8 },
        
        ( (2,2),"R" ): { (2,2):0.2 , (2,3):0.8 },
        ( (2,2),"U" ): { (2,2):0.2 , (1,2):0.8 },
        ( (2,2),"L" ): { (2,2):0.2 , (2,1):0.8 },
        
        ( (2,3),"L" ): { (2,3):0.2 , (2,2):0.8 },
        ( (2,3),"U" ): { (2,3):0.2 , (1,3):0.8 },
    }
    
    g.set_config(actions,rewards,probs)
    
    return g

# ----------------------------------------------
 


