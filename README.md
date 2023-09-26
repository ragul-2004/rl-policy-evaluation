### EX NO : 02
# <p align="center">POLICY EVALUATION</p>

## AIM :
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT :

The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States :

The environment has 7 states:
* Two Terminal States: **G**: The goal state & **H**: A hole state.
* Five Transition states / Non-terminal States including  **S**: The starting state.

### Actions :

The agent can take two actions:

* R: Move right.
* L: Move left.

### Transition Probabilities :

The transition probabilities for each action are as follows:

* **50%** chance that the agent moves in the intended direction.
* **33.33%** chance that the agent stays in its current state.
* **16.66%** chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards :

The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation :
<p align="center">
<img width="600" src="https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/e7af87e7-fe73-47fa-8bea-2040b7645e44"> </p>


## POLICY EVALUATION FUNCTION :

### Formula :
<img width="350" src="https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/e663bd3d-fc85-41c3-9a5c-dffa57eae250">

### Program :
Developed By : **Nithishkumar P**
</br>
Register No. : **212221230070**
```py
def policy_evaluation(pi, P, gamma=0.9, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V=np.zeros(len(P),dtype=np.float64)
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma*prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V))<theta:
        break
      prev_V=V.copy()
    return V
```

## OUTPUT :
### Policy 1:
![image](https://github.com/Jovita08/rl-policy-evaluation/assets/94174503/36cad0ff-f11d-44a5-abac-96a73ad764eb)
![image](https://github.com/Jovita08/rl-policy-evaluation/assets/94174503/c921ac0e-10bb-4b70-8fa9-63f64d52a14d)
![image](https://github.com/Jovita08/rl-policy-evaluation/assets/94174503/fed6f8e0-4ee8-4735-b1d2-335fea16d37e)

### Policy 2:
![image](https://github.com/NITHISHKUMAR-P/rl-policy-evaluation/assets/93427017/45023493-bbac-4b03-b4f3-b6510d08efd8)
![image](https://github.com/NITHISHKUMAR-P/rl-policy-evaluation/assets/93427017/0252d017-3b34-490e-b0cf-c5b4f9a94052)
![image](https://github.com/NITHISHKUMAR-P/rl-policy-evaluation/assets/93427017/bf576126-ed5b-44af-bf04-9d576dbef1d1)
### Comparison:
![image](https://github.com/Jovita08/rl-policy-evaluation/assets/94174503/c18d1016-be97-442d-9cd6-819c4794496a)
### Conclusion:
![image](https://github.com/Jovita08/rl-policy-evaluation/assets/94174503/41fb8906-0847-4e7c-a302-3d165d19bcbf)

## RESULT :
Thus, a Python program is developed to evaluate the given policy.
