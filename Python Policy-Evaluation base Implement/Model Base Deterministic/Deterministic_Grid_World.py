import pickle
import numpy as np

# Deterministic Grid-world class:
#---------------------------------------------------------------

# Deterministic transition function -> if you choose U, just move up.
# Deterministic rewards -> you will get same reward from any way you reach s' form s.
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
        
        self.__initialize_states()
        
    
    
    # Current State Actions getter.
    #----------------------------------
    def get_actions(self):
        return self.actions.get(self.current_state)
    
    
    # Environment Transition function: T(s,a)-> r,s'
    #-----------------------------------------------
    def transition(self, action:str ) -> float :
        
        i,j = self.current_state
        
        if action in self.actions.get(self.current_state):
            if action == "U":
                if i > 0 :
                    i -= 1
            elif action == "D":
                if i < self.rows-1 :
                    i += 1
            elif action == "L":
                if j > 0 :
                    j -= 1
            elif action == "R":
                if j < self.cols-1:
                    j += 1
        
        self.current_state = (i,j)
        return self.rewards.get(self.current_state,0)
    
    
    # Environment Transition Probability function: p(s',r|s,a)-> 0/1
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
                return 1.0
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
    
    def undo_action(self, action): pass
    def get_next_state(self,state,action): pass
    def set_state(self, new_state): pass
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
    def set_config( self, actions:dict, rewards: dict ):
        self.actions = actions
        self.rewards = rewards
    
    
    # Initialize Grid-world States.
    #----------------------------------
    def __initialize_states(self):
        
        for r in range(self.rows):
            for c in range(self.cols):
                self.states.append((r,c))
        
    
#======================================================================================================
#======================================================================================================

# Global functions:
#============================

# saving GW configs into file.
#-----------------------------
def save_env(Gw, filename):
    with open(filename, 'wb') as f:
        pickle.dump(Gw, f)


# Loading GW configs from file.
#-------------------------------
def load_env(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Standard GW creation.
#-------------------------------
def standard_GW():
    r = 3
    c = 4
    start_state = (2,0)
    g = GridWorld(r,c,start_state)
    
    rewards = {}
    for i in range(r):
        for j in range(c):
            rewards[(i,j)] = -0.1
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
    g.set_config(actions,rewards)
    return g

