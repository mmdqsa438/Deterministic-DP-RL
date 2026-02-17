![RL](https://kitrum.com/wp-content/uploads/2023/04/Reinforcement-Learning-1.png)

# Reinforcement Learning: Dynamic Programming Suite

A comprehensive implementation of Dynamic Programming algorithms for Reinforcement Learning, available in both **Python** and **C++**. This project demonstrates policy evaluation, policy improvement, and optimal policy finding using Bellman equations in grid world environments.

## Overview

This repository contains two main implementations:

### Python Implementation
- **Deterministic & Stochastic Environments** - Grid worlds with configurable transition dynamics
- **Agent Classes** - Implementing policy iteration, value iteration, and Bellman equations
- **Visualization Tools** - Print utilities for value functions and policies
- **Model Persistence** - Save/load trained agents and environments using pickle

### C++ Implementation
- **High-Performance Grid World** - Efficient implementation for deterministic environments
- **STL-Based Design** - Using vectors, pairs, and exception handling
- **Memory Efficiency** - Direct index mapping for state-action-reward lookups

## Key Concepts

### Dynamic Programming Algorithms

| Algorithm | Description | Implementation |
|-----------|-------------|----------------|
| **Policy Evaluation** | Iteratively compute V(s) for a given policy | `Evaluate_policy()` / `Update_value()` |
| **Policy Improvement** | Update policy greedily w.r.t current V(s) | `improve_policy()` |
| **Policy Iteration** | Repeat evaluation -> improvement until convergence | `Update_policy()` |

### Bellman Equations

**State-Value Function V(s):**
```
V(s) = sum π(a|s) sum P(s'|s,a) [R(s,a,s') + γ V(s')]
```

**Action-Value Function Q(s,a):**
```
Q(s,a) = sum P(s'|s,a) [R(s,a,s') + γ V(s')]
```

## Project Structure

```
RL-Dynamic-Programming/
│
├── Model Base Deterministic/           # Python - Deterministic
│   ├── Deterministic_Agent.py           # Agent with policy iteration
│   ├── Deterministic_Grid_World.py      # Deterministic environment
│   ├── test.py                           # Example usage
│   └── testbench-*.ipynb                  # Jupyter notebooks
│
├── Model Base Stochastic/               # Python - Stochastic
│   ├── Stochastic_Agent.py                # Agent for stochastic envs
│   ├── Stochastic_Grid_world.py           # Stochastic environment
│   └── test.py                              # Example usage
│
├── C++ Implementation/                  # C++ Version
│   ├── DeterministicGridWorld.hpp         # Class definition
│   ├── DeterministicGridWorld.cpp         # Implementation
│   ├── main.cpp                            # Example usage
│   └── Deterministic-GWjpg.jpg              # Console output demo
│
└── README.md                               # This file
```

## Getting Started

### Python Prerequisites

```bash
pip install matplotlib numpy
```

### C++ Compilation

```bash
g++ -o GridWorld main.cpp DeterministicGridWorld.cpp
./GridWorld
```

### Quick Python Example

```python
from Deterministic_Agent import Agent
from Deterministic_Grid_World import standard_GW

# Create environment and agent
env = standard_GW()
agent = Agent()

# Initialize
agent.initialize_Vs(env.rows, env.cols)
agent.model_states = env.states
agent.model_actions = ["U", "D", "L", "R"]
agent.model_rewards = env.rewards

# Find optimal policy
agent.Update_policy(env)

# Visualize results
print_value(agent.value_state_function, env)
print_policy(agent.policy, env)
```

### Quick C++ Example

```cpp
#include "DeterministicGridWorld.hpp"

// Create standard grid world
GridWorld gw = standard_GW(-0.1);

// Configure actions and rewards
std::vector<std::vector<std::string>> actions = {/* ... */};
std::vector<float> rewards = {/* ... */};
gw.set_config(actions, rewards);

// Use in your RL algorithm
std::vector<std::string> available_actions = gw.get_StateActions();
float reward = gw.transition("U");
```

## Grid World Details

### Standard 3x4 Grid
```
| (0,0) | (0,1) | (0,2) | (0,3) +1 |
| (1,0) | (1,1) | (1,2) | (1,3) -1 |
| (2,0) | (2,1) | (2,2) | (2,3)    |
```

**Rewards:**
- Regular states: `-0.1` (step cost)
- Terminal (0,3): `+1.0` (goal)
- Terminal (1,3): `-1.0` (hole)

**Available Actions:** `"U"` (Up), `"D"` (Down), `"L"` (Left), `"R"` (Right)

## Environment Types

### Deterministic (Python & C++)
- Actions always succeed as intended
- Transition probability: 1.0 for intended move, 0.0 otherwise
- Perfect for learning basic DP algorithms

### Stochastic (Python Only)
- Actions may fail with configurable probabilities
- "Sticky" transitions: 80% success, 20% stay in place
- More realistic and challenging

## Model Persistence

### Python - Save/Load
```python
# Save
save_agent(agent, "optimal_agent.pkl")
save_env(env, "grid_world.pkl")

# Load
loaded_agent = load_agent("optimal_agent.pkl")
loaded_env = load_env("grid_world.pkl")
```

### C++ - State Management
```cpp
// Environment state is preserved in the object
GridWorld gw = standard_GW();
gw.transition("R");  // State changes
std::pair<int,int> current = gw.get_currentState();
```

## Example Results

After running policy iteration on standard 3x4 grid:

**Optimal Value Function V*(s):**
```
----------------------------
 0.80 | 0.85 | 0.91 | 1.00 |
----------------------------
 0.75 | 0.00 | 0.72 | -1.00|
----------------------------
 0.70 | 0.66 | 0.62 | 0.53 |
----------------------------
```

**Optimal Policy π*(s):**
```
========================
  R  |  R  |  R  |  &  |
========================
  U  |     |  U  |  &  |
========================
  R  |  R  |  U  |  U  |
========================
```
(& indicates terminal states)

## Key Parameters

| Parameter | Description | Python Default | C++ Default |
|-----------|-------------|----------------|-------------|
| **γ (GAMMA)** | Discount factor | 0.9 | N/A (user-defined) |
| **θ (threshold)** | Convergence threshold | 1e-3 | N/A |
| **MAX_ITR** | Max iterations | 100 | N/A |
| **cost** | Step penalty | -0.1 | User-defined |

## Running Tests

### Python
```bash
# Deterministic environment
python "Model Base Deterministic/test.py"

# Stochastic environment  
python "Model Base Stochastic/test.py"

# Or explore with Jupyter
jupyter notebook testbench-detAgent.ipynb
```

### C++
```bash
./GridWorld
# Output shown in Deterministic-GWjpg.jpg
```

## Customizing Your Own Grid World

### Python
```python
# Create custom environment
gw = GridWorld(rows=4, cols=5, start_state=(0,0))

# Define rewards
rewards = {(0,0): -0.1, (3,4): 10.0}

# Define available actions per state
actions = {
    (0,0): ["D", "R"],
    (0,1): ["L", "R", "D"],
}

gw.set_config(actions, rewards)
```

### C++
```cpp
// Create custom environment
GridWorld gw(4, 5, -0.1, {0,0});

// Define actions and rewards (parallel vectors)
std::vector<std::vector<std::string>> actions = {
    {"D", "R"},     // state 0
    {"L", "R", "D"} // state 1
    // ... for all states
};

std::vector<float> rewards = {
    -0.1,  // state 0
    -0.1,  // state 1
    // ... for all states
};

gw.set_config(actions, rewards);
```

## Algorithms Explained

### 1. Policy Evaluation
```python
def Evaluate_policy(self, probability_func):
    for _ in range(max_iterations):
        for state in states:
            old_value = V[state]
            V[state] = expected_return(state)
        if max_change < threshold: break
```

### 2. Policy Improvement
```python
def improve_policy(self, env):
    for state in states:
        old_action = π[state]
        π[state] = argmax Q(state, action)
    return old_π == new_π
```

### 3. Policy Iteration
```python
def Update_policy(self, env):
    while not policy_stable:
        self.Evaluate_policy(env.probability)
        policy_stable = self.improve_policy(env)
```

## Future Improvements

- [ ] **Value Iteration** implementation (faster convergence)
- [ ] **GUI Visualization** for grid worlds
- [ ] **Monte Carlo methods** for model-free learning
- [ ] **Temporal Difference learning** (SARSA, Q-learning)
- [ ] **C++ Agent class** with full DP algorithms
- [ ] **Performance benchmarks** between Python and C++
- [ ] **More complex environments** (windy gridworld, cliff walking)

## Contributing

Feel free to fork, experiment, and submit PRs. Areas for contribution:
- Adding new environments
- Implementing additional RL algorithms
- Optimizing C++ performance
- Creating visualization tools

## License

MIT License - free to use in your own projects.

## Contact

**mMahdi** - [@mmdqsa438](https://github.com/mmdqsa438)

**Project Link**: [https://github.com/mmdqsa438/Deterministic-DP-RL](https://github.com/mmdqsa438/Deterministic-DP-RL)

---

*Happy Reinforcement Learning!*
