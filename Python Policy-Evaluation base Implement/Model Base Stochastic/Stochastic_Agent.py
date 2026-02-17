
# exper-base RL Agent
class Agent:
    
    def __init__(self,gamma=0.9) -> None :
        
        self.policy:dict = {}                      # stochastic Agent_class policy    : { (i,j): "act" }
        self.value_state_function:dict = {}        # State-value function V(s)  : { (i,j):value  }
        
        self.current_state:tuple                   # [XXX] it maybe creat ambiguity if forgot to update
        
        self.__MAX_ITR = 100                        # max itreration which break training loop.
        self.__GAMMA   = gamma                     # discount factor
        self.__MIN_ERROR = 1e-3                    # threshold
        
        self.exper_states:list = []                # Experiment Visited States  : [ (i,j), (i',j') ]
        self.exper_actions:list = []               # Experiment Done Actions    : [ "act1", "act2" ]
        self.exper_rewards:dict = {}               # Experiment Gained rewards  : { (i,j):reward }
        
    
    
    
    # Policy Probability Disribution : pi(a|s)
    #----------------------------------------
    def policy_probability_disribution( self, action:str, state:tuple) -> bool :
        return  action == self.policy.get(state,"$*(#DHwkh3@#28)")
    
    
    # V(s) Bellman equation implementation.
    # [NOTE]: current-state in this code is the state which going to update it's value.
    #----------------------------------------
    def Value( self, current_state:tuple, probability_func )-> float:
        
        value = 0.0
        for act in self.exper_actions :
            
            pi_a_s = self.policy_probability_disribution( act, current_state )
            
            for next_state in self.exper_states :
                value += pi_a_s * probability_func(next_state, current_state, act ) * ( self.exper_rewards.get(next_state,0) + self.__GAMMA * self.value_state_function.get(next_state,0.0))
        
        return value
    
    
    # Updating value function iterativly.
    #----------------------------------------
    def Update_value(self, probability_func )-> None:
        
        # print("> reached Evaluate-value")
        
        for _ in range(self.__MAX_ITR) :
            # print("> reached Evaluate-value loop")
            max_change = 0
            for st in self.exper_states :
                
                old_value = self.value_state_function.get(st)
                self.value_state_function[st] = self.Value(st,probability_func)
                max_change = max( max_change, abs( self.value_state_function.get(st) - old_value ) )
            
            if max_change < self.__MIN_ERROR :
                break
        
    
    

    # =========== Control Problem ============
    
    # Q(s,a) Bellman equation Action-value function.
    #---------------------------------------------
    def action_Value(self, state, action, probability_func)-> float:
        
        value = 0.0
        for _ in range(self.__MAX_ITR):
            for next_st in self.exper_states:
                value += probability_func(next_st, state, action) * (self.exper_rewards.get(next_st, 0.0) + self.__GAMMA * self.value_state_function.get(next_st, 0.0) )
        
        return value
    
    
    # Improve policy
    #----------------------------------------
    def improve_policy(self,env) -> bool:
        
        policy_sustainability = True
        
        for st in env.actions :
            
            old_act = self.policy.get(st)
            self.policy[st] = self.best_action(st,env)
            
            
            if old_act != self.policy.get(st,"") :
                policy_sustainability = False
        
        return policy_sustainability
    
    
    # Finding Best action.
    #----------------------------------
    def best_action(self, state, env)-> str:
        
        max_value = -100000.0
        best_act  = ""
        value = 0
        
        for act in env.actions.get(state,[]) :
            value = self.action_Value(state,act,env.probability)
            
            if value >= max_value :
                best_act = act
                max_value = value
        
        return best_act
    
    
    
    
    # ======== Finding Optimal Poliy ===========
    
    # Updating iteratively until optimal.
    #---------------------------------------
    def Update_policy(self, env ) -> None:
        
        is_stable = False
        
        for _ in range(self.__MAX_ITR*10):
            for itr in range(self.__MAX_ITR):
                
                old_Vs = self.value_state_function
                self.Update_value(env.probability)
                is_stable = self.improve_policy(env)
                
            if is_stable or self.value_func_singularity(old_Vs):
                return
    
    # Checking Value funtion doesn't change.
    #-------------------------------------------------------------
    def value_func_singularity(self, old_Vs:dict) -> bool:
        
        is_same = True
        
        for st in self.value_state_function :
            if self.value_state_function.get(st,1e-5) != old_Vs.get(st,-1e-6):
                is_same = False
                break
            
        return is_same
    
    
    
    # V(s) initialize-function.
    #-----------------------------------
    def initialize_Vs( self, rows:int, cols:int )-> None :
        
        for i in range(rows) :
            for j in range(cols):
                self.value_state_function[(i,j)] = 0
        




import pickle

# saving Agent_class configs into file.
#-----------------------------
def save_agent(Agent_class:Agent, filename) -> None :
    with open(filename, 'wb') as f:
        pickle.dump(Agent_class, f)


# Loading Agent_class configs from file.
#-------------------------------
def load_agent(filename) -> Agent:
    with open(filename, 'rb') as f:
        return pickle.load(f)


class a:
    a = 0