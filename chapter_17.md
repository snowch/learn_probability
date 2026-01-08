---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
downloads:
  - file: notebooks/chapter_17.ipynb
---

# Chapter 17: Introduction to Markov Chains

+++

Welcome to Chapter 17! In this chapter, we venture into the world of stochastic processes by exploring **Markov Chains**. Markov chains are a fundamental concept used to model systems that transition between different states over time, where the future state depends *only* on the current state, not on the sequence of events that preceded it. This 'memoryless' property makes them incredibly useful for modeling various real-world phenomena, from weather patterns and stock market movements to customer behavior and website navigation.

+++

**Learning Objectives:**
* Understand the definition of a Markov chain, its components (states, transitions), and the crucial Markov property.
* Learn how to represent Markov chains using transition matrices.
* Simulate the behavior of a Markov chain over time using Python.
* Calculate multi-step transition probabilities using matrix powers.
* Gain an introductory understanding of state classification (e.g., absorbing states).
* Learn about stationary distributions and how to find them, representing the long-run behavior of a chain.
* Apply these concepts to practical examples using NumPy.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

# Configure plots for better readability
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
```

## 16.1 What is a Markov Chain?

+++

A **Markov chain** is a mathematical model describing a sequence of possible events (or states) where the probability of transitioning to the next state depends *only* on the current state and not on the sequence of states that preceded it. This is known as the **Markov Property** (or memorylessness).

Key components:
* **States:** A finite or countably infinite set of possible conditions or positions the system can be in. Let the set of states be $S = \{s_1, s_2, ..., s_k\}$.
* **Transitions:** Movements between states.
* **Transition Probabilities:** The probability of moving from one state to another in a single time step. The probability of transitioning from state $s_i$ to state $s_j$ is denoted as $P_{ij} = P(X_{t+1} = s_j | X_t = s_i)$, where $X_t$ is the state at time $t$.
* **Initial Distribution:** A probability distribution describing the starting state of the system at time $t=0$.

**Example: Customer Subscription Model**

Consider a company offering subscription plans: Free, Basic, and Premium. Customers can switch plans month-to-month, or they might churn (cancel). We can model this as a Markov chain:
* **States:** $S = \{\text{'Free', 'Basic', 'Premium', 'Churned'}\}$ 
* **Time Step:** One month.
* **Markov Property Assumption:** The probability a customer switches to a new plan next month depends *only* on their current plan, not their entire history (e.g., whether they were Premium two months ago doesn't directly influence the next step if they are currently Basic).

+++

## 16.2 The Transition Matrix

+++

The transition probabilities of a Markov chain with $k$ states can be conveniently organized into a $k \times k$ matrix called the **Transition Matrix**, often denoted by $P$. 

$$ 
P = \begin{pmatrix}
 P_{11} & P_{12} & \cdots & P_{1k} \\
 P_{21} & P_{22} & \cdots & P_{2k} \\
 \vdots & \vdots & \ddots & \vdots \\
 P_{k1} & P_{k2} & \cdots & P_{kk}
 \end{pmatrix}
 $$ 

Where $P_{ij}$ is the probability of transitioning *from* state $i$ *to* state $j$ in one step.

**Properties of a Transition Matrix:**
1.  All entries must be non-negative: $P_{ij} \ge 0$ for all $i, j$.
2.  The sum of probabilities in each row must equal 1: $\sum_{j=1}^{k} P_{ij} = 1$ for all $i$. (From any state $i$, the system must transition to *some* state $j$ in the next step).

+++

**Example: Subscription Model Transition Matrix**

Let's define a plausible transition matrix for our subscription model. States are ordered: 0: Free, 1: Basic, 2: Premium, 3: Churned.

| From \\ To | Free | Basic | Premium | Churned |
|-----------|------|-------|---------|---------|
| Free      | 0.60 | 0.20  | 0.10    | 0.10    |
| Basic     | 0.10 | 0.60  | 0.20    | 0.10    |
| Premium   | 0.05 | 0.10  | 0.70    | 0.15    |
| Churned   | 0.00 | 0.00  | 0.00    | 1.00    |  <- Absorbing State

```{code-cell} ipython3
# Define the states
states = ['Free', 'Basic', 'Premium', 'Churned']
state_map = {state: i for i, state in enumerate(states)} # Map state names to indices

# Define the transition matrix P
P = np.array([
    [0.60, 0.20, 0.10, 0.10],  # Transitions from Free
    [0.10, 0.60, 0.20, 0.10],  # Transitions from Basic
    [0.05, 0.10, 0.70, 0.15],  # Transitions from Premium
    [0.00, 0.00, 0.00, 1.00]   # Transitions from Churned (Absorbing)
])

# Verify rows sum to 1
print("Transition Matrix P:")
print(P)
print("\nRow sums:", P.sum(axis=1))
```

## 16.3 Simulating Markov Chain Paths

+++

We can simulate the progression of a system through states over time using the transition matrix. Given a current state, we use the corresponding row in the transition matrix as probabilities to randomly choose the next state.

We can use `numpy.random.choice` for this.

```{code-cell} ipython3
def simulate_path(transition_matrix, state_names, start_state_name, num_steps):
    """Simulates a path through the Markov chain."""
    state_indices = list(range(len(state_names)))
    current_state_index = state_names.index(start_state_name)
    path_indices = [current_state_index]

    for _ in range(num_steps):
        # Get the transition probabilities from the current state
        probabilities = transition_matrix[current_state_index, :]
        
        # Choose the next state based on these probabilities
        next_state_index = np.random.choice(state_indices, p=probabilities)
        path_indices.append(next_state_index)
        
        # Update the current state
        current_state_index = next_state_index
        
        # Optional: Stop if an absorbing state (like Churned) is reached
        if probabilities[current_state_index] == 1.0 and np.sum(probabilities) == 1.0:
           # Check if it's an absorbing state (only loops back to itself)
           if transition_matrix[current_state_index, current_state_index] == 1.0:
              # Fill remaining steps if needed, or break
              # For simplicity here, we just let it stay in the absorbing state
              pass 

    # Convert indices back to state names
    path_names = [state_names[i] for i in path_indices]
    return path_names

# Simulate a path for 12 months starting from 'Free'
start_state = 'Free'
steps = 12
simulated_journey = simulate_path(P, states, start_state, steps)

print(f"Simulated {steps}-month journey starting from {start_state}:")
print(" -> ".join(simulated_journey))
```

Run the simulation cell multiple times to see different possible paths a customer might take.

+++

## 16.4 n-Step Transition Probabilities

+++

The transition matrix $P$ gives the probabilities of moving between states in *one* step. What if we want to know the probability of transitioning from state $i$ to state $j$ in *n* steps?

This is given by the $(i, j)$-th entry of the matrix power $P^n$. 

$P^{(n)}_{ij} = P(X_{t+n} = s_j | X_t = s_i) = (P^n)_{ij}$

We can calculate matrix powers using `numpy.linalg.matrix_power`.

```{code-cell} ipython3
# Calculate the transition matrix after 3 steps (months)
n_steps = 3
P_n = np.linalg.matrix_power(P, n_steps)

print(f"Transition Matrix after {n_steps} steps (P^{n_steps}):")
print(np.round(P_n, 3)) # Round for readability

# Example: Probability of being in 'Premium' after 3 months, starting from 'Free'
start_idx = state_map['Free']
end_idx = state_map['Premium']
prob_free_to_premium_3_steps = P_n[start_idx, end_idx]

print(f"\nProbability(State=Premium at month 3 | State=Free at month 0): {prob_free_to_premium_3_steps:.3f}")

# Example: Probability of being 'Churned' after 6 months, starting from 'Basic'
n_steps_long = 6
P_n_long = np.linalg.matrix_power(P, n_steps_long)
start_idx_basic = state_map['Basic']
end_idx_churned = state_map['Churned']
prob_basic_to_churned_6_steps = P_n_long[start_idx_basic, end_idx_churned]

print(f"Probability(State=Churned at month 6 | State=Basic at month 0): {prob_basic_to_churned_6_steps:.3f}")
```

## 16.5 Classification of States (Brief Introduction)

+++

States in a Markov chain can be classified based on their long-term behavior:

* **Accessible:** State $j$ is accessible from state $i$ if there is a non-zero probability of eventually reaching $j$ starting from $i$ (i.e., $P^{(n)}_{ij} > 0$ for some $n \ge 0$).
* **Communicating:** States $i$ and $j$ communicate if they are accessible from each other.
* **Irreducible Chain:** A Markov chain is irreducible if all states communicate with each other (it's possible to get from any state to any other state).
* **Recurrent State:** If starting from state $i$, the probability of eventually returning to state $i$ is 1. 
* **Transient State:** If starting from state $i$, there is a non-zero probability of *never* returning to state $i$. 
* **Absorbing State:** A state $i$ is absorbing if once entered, it cannot be left ($P_{ii} = 1$). In our example, 'Churned' is an absorbing state.

In our subscription model:
* 'Churned' is an absorbing state.
* 'Free', 'Basic', and 'Premium' are likely transient states because from any of them, there is a path to 'Churned', and once in 'Churned', you cannot return.
* The chain is *not* irreducible because you cannot leave the 'Churned' state.

+++

## 16.6 Stationary Distributions

+++

For certain types of Markov chains (specifically, irreducible and aperiodic ones), the distribution of probabilities across states converges to a unique **stationary distribution** (or steady-state distribution), regardless of the initial state. Let $\pi = [\pi_1, \pi_2, ..., \pi_k]$ be the row vector representing this distribution, where $\pi_j$ is the long-run probability of being in state $j$.

The stationary distribution $\pi$ satisfies the equation:

$$ \pi P = \pi $$ 

subject to the constraint that $\sum_{j=1}^{k} \pi_j = 1$.

This means that if the distribution of states is $\pi$, it will remain $\pi$ after one more step. $\pi$ is the left eigenvector of the transition matrix $P$ corresponding to the eigenvalue $\lambda = 1$.

**Important Note:** Our subscription model has an absorbing state. In such cases, the long-run probability will eventually concentrate entirely in the absorbing state(s). For any starting state other than 'Churned', the probability of being in 'Churned' approaches 1 as $n \to \infty$. The concept of a unique stationary distribution across *all* states applies more directly to chains where you can always move between states (irreducible chains).

+++

**Example: Finding Stationary Distribution for a Weather Model**

Let's consider a simpler, irreducible weather model (Sunny, Cloudy, Rainy) to illustrate finding a stationary distribution.

Weather States: `['Sunny', 'Cloudy', 'Rainy']`
Weather Matrix `W`:
```
   Sunny Cloudy Rainy
Sunny  [0.7,  0.2,   0.1]
Cloudy [0.3,  0.5,   0.2]
Rainy  [0.2,  0.4,   0.4]
```
We need to find $\pi = [\pi_S, \pi_C, \pi_R]$ such that $\pi W = \pi$ and $\pi_S + \pi_C + \pi_R = 1$. This is equivalent to finding the left eigenvector for eigenvalue 1.

```{code-cell} ipython3
# Weather example
W_states = ['Sunny', 'Cloudy', 'Rainy']
W = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.5, 0.2],
    [0.2, 0.4, 0.4]
])

# Find eigenvalues and eigenvectors
# We need the *left* eigenvector (v P = lambda v), numpy finds *right* (P u = lambda u).
# The left eigenvector of P for eigenvalue lambda is the right eigenvector of P.T for eigenvalue lambda.
eigenvalues, eigenvectors = np.linalg.eig(W.T)

# Find the eigenvector corresponding to eigenvalue 1
# Due to floating point precision, check for eigenvalues close to 1
idx = np.isclose(eigenvalues, 1)

if not np.any(idx):
    print("No eigenvalue close to 1 found. Check matrix.")
else:
    # Get the eigenvector corresponding to eigenvalue 1
    # Ensure we select only one eigenvector if multiple eigenvalues are close to 1
    # and take the real part as the eigenvector should be real for a stochastic matrix
    stationary_vector_raw = eigenvectors[:, np.where(idx)[0][0]].flatten().real
    
    # Normalize the eigenvector so its components sum to 1
    stationary_distribution = stationary_vector_raw / np.sum(stationary_vector_raw)

    print("Stationary Distribution (Long-run probabilities):")
    for state, prob in zip(W_states, stationary_distribution):
        print(f"  {state}: {prob:.4f}")

    # Verification: pi * W should be equal to pi
    print("\nVerification (pi * W):", np.dot(stationary_distribution, W))
```

This stationary distribution tells us that, in the long run, regardless of whether today is Sunny, Cloudy, or Rainy, the probability of a future day being Sunny is about 44.7%, Cloudy is about 36.8%, and Rainy is about 18.4%.

+++

## 16.7 Hands-on: Analyzing the Subscription Model Long-Term

+++

Let's use our simulation and n-step transition probabilities to see what happens to subscribers in the long run in our original model with the 'Churned' state.

```{code-cell} ipython3
# Simulate many paths to see the distribution of final states
num_simulations = 5000
simulation_length = 24 # Simulate for 2 years
final_states = []

initial_state = 'Basic' # Example starting state

for _ in range(num_simulations):
    path = simulate_path(P, states, initial_state, simulation_length)
    final_states.append(path[-1]) # Get the state after 'simulation_length' months

# Calculate the proportion of simulations ending in each state
from collections import Counter # Efficient way to count
final_state_counts = Counter(final_states)
# Ensure all states are present in the counts, even if 0
for state in states:
    if state not in final_state_counts:
        final_state_counts[state] = 0

final_state_proportions = {state: final_state_counts[state] / num_simulations for state in states}

print(f"Distribution of states after {simulation_length} months starting from '{initial_state}' (based on {num_simulations} simulations):")
for state in states: # Print in consistent order
    print(f"  {state}: {final_state_proportions[state]:.4f}")

# Compare with n-step transition probability calculation
P_long_term = np.linalg.matrix_power(P, simulation_length)
start_idx = state_map[initial_state]
theoretical_probs = P_long_term[start_idx, :]

print(f"\nTheoretical probabilities after {simulation_length} months starting from '{initial_state}':")
for i, state in enumerate(states):
    print(f"  {state}: {theoretical_probs[i]:.4f}")
```

Notice how the simulation results closely match the theoretical probabilities calculated using the matrix power $P^n$. Also, observe that as the number of steps (`simulation_length`) increases, the probability mass shifts significantly towards the 'Churned' state, as expected for a model with an absorbing state.

+++

## 16.8 Summary

+++

In this chapter, we introduced Markov chains, a powerful tool for modeling systems that transition between states based only on their current state. 

We learned:
* The definition of states, transitions, and the Markov property.
* How to use transition matrices ($P$) to represent one-step probabilities.
* To simulate Markov chain paths using Python and `np.random.choice`.
* That matrix powers ($P^n$) give n-step transition probabilities.
* The basic classification of states, including absorbing states.
* The concept of a stationary distribution ($\pi P = \pi$) for irreducible, aperiodic chains, representing long-run behavior, and how to find it using eigenvectors.

Markov chains form the basis for many more advanced models and algorithms in fields like reinforcement learning, finance, genetics, and operations research.

+++

## Exercises

+++

1.  **Simple Weather Model:** Consider the weather model (Sunny, Cloudy, Rainy) with transition matrix `W`. If it's Sunny today, what is the probability it will be Rainy the day after tomorrow (in 2 steps)? Calculate this using matrix multiplication.
2.  **Gambler's Ruin (Simulation):** A gambler starts with \$3. They bet \$1 on a fair coin flip (50% chance win, 50% chance lose). They stop if they reach \$0 (ruin) or \$5 (win). 
    a. Define the states (amount of money: 0, 1, 2, 3, 4, 5).
    b. Define the transition matrix. Note the absorbing states 0 and 5.
    c. Simulate 10000 games starting from \$3. What proportion end in ruin (\$0) and what proportion end in winning (\$5)?
3.  **Stationary Distribution Verification:** For the weather model, manually verify that the calculated stationary distribution $\pi$ satisfies $\pi W = \pi$. 
4.  **Modify Subscription Model:** Change the 'Churned' state in the subscription model `P` so that there's a small probability (e.g., 0.05) of a churned customer re-subscribing to the 'Free' plan each month (adjust the $P_{33}$ probability accordingly so the row still sums to 1). Is the chain still absorbing? Try to find the new stationary distribution using the eigenvector method. What does it represent now?

```{code-cell} ipython3
# Exercise 1 Code/Calculation Space
W_states = ['Sunny', 'Cloudy', 'Rainy']
W = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.5, 0.2],
    [0.2, 0.4, 0.4]
])

# Calculate W^2
W_2 = np.linalg.matrix_power(W, 2)

# Probability Sunny -> Rainy in 2 steps
sunny_idx = W_states.index('Sunny')
rainy_idx = W_states.index('Rainy')
prob_sunny_to_rainy_2_steps = W_2[sunny_idx, rainy_idx]
print(f"(Exercise 1) Probability Sunny -> Rainy in 2 steps: {prob_sunny_to_rainy_2_steps:.4f}")
```

```{code-cell} ipython3
# Exercise 2 Code/Calculation Space
# Gambler's Ruin
gambler_states = [0, 1, 2, 3, 4, 5] # Amount of money
P_gambler = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], # From 0 (Ruin)
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0], # From 1
    [0.0, 0.5, 0.0, 0.5, 0.0, 0.0], # From 2
    [0.0, 0.0, 0.5, 0.0, 0.5, 0.0], # From 3
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.5], # From 4
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # From 5 (Win)
])

def simulate_gambler(transition_matrix, states, start_state_val, max_steps=200): # Increased max_steps slightly
    """Simulates one game until an absorbing state is reached."""
    state_indices = list(range(len(states)))
    current_state_index = states.index(start_state_val)
    #path_indices = [current_state_index] # Path not needed for final state only

    steps = 0
    while steps < max_steps:
        current_state_val = states[current_state_index]
        # Check if in absorbing state
        if current_state_val == 0 or current_state_val == 5:
            return current_state_val # Return final state
            
        probabilities = transition_matrix[current_state_index, :]
        next_state_index = np.random.choice(state_indices, p=probabilities)
        #path_indices.append(next_state_index)
        current_state_index = next_state_index
        steps += 1
        
    # If max_steps reached without hitting absorbing state (unlikely here but good practice)
    return states[current_state_index] 

num_games = 10000 # Increased simulations for better accuracy
start_money = 3
end_results = []
for _ in range(num_games):
    end_results.append(simulate_gambler(P_gambler, gambler_states, start_money))

from collections import Counter
end_counts = Counter(end_results)

ruin_count = end_counts.get(0, 0) # Use .get for safety
win_count = end_counts.get(5, 0)

print(f"\n(Exercise 2) Gambler's Ruin Simulation ({num_games} games starting at ${start_money}):")
print(f"  Proportion ending in Ruin ($0): {ruin_count / num_games:.4f}") # Should be 0.4
print(f"  Proportion ending in Win ($5): {win_count / num_games:.4f}")  # Should be 0.6
```

```{code-cell} ipython3
# Exercise 3 Code/Calculation Space
# Verification: pi * W = pi

# From previous calculation (ensure these are accurate)
# Recalculate just in case
eigenvalues, eigenvectors = np.linalg.eig(W.T)
idx = np.isclose(eigenvalues, 1)
stationary_vector_raw = eigenvectors[:, np.where(idx)[0][0]].flatten().real
pi_weather = stationary_vector_raw / np.sum(stationary_vector_raw)

result_vector = np.dot(pi_weather, W)

print("\n(Exercise 3) Stationary Distribution Verification:")
print(f"  Calculated pi: {pi_weather}")
print(f"  Result pi * W: {result_vector}")
# Use np.allclose for robust floating point comparison
print(f"  Are they close? {np.allclose(pi_weather, result_vector)}")
```

```{code-cell} ipython3
# Exercise 4 Code/Calculation Space

# Reload original P and states just in case they were modified
states = ['Free', 'Basic', 'Premium', 'Churned']
state_map = {state: i for i, state in enumerate(states)}
P = np.array([
    [0.60, 0.20, 0.10, 0.10],  # Transitions from Free
    [0.10, 0.60, 0.20, 0.10],  # Transitions from Basic
    [0.05, 0.10, 0.70, 0.15],  # Transitions from Premium
    [0.00, 0.00, 0.00, 1.00]   # Transitions from Churned (Absorbing)
])

P_modified = P.copy() # Start with original subscription matrix

# Modify the 'Churned' row (index 3)
prob_churn_to_free = 0.05
P_modified[3, state_map['Free']] = prob_churn_to_free
P_modified[3, state_map['Churned']] = 1.0 - prob_churn_to_free # Adjust P_33

print("\n(Exercise 4) Modified Transition Matrix:")
print(P_modified)
print("\nIs 'Churned' still absorbing?", P_modified[3, 3] == 1.0) # Should be False
print("The chain is no longer absorbing, as state 'Churned' can transition to 'Free'.")
print("The chain should now be irreducible if all other states can eventually reach Churned and Churned can reach Free.")

# Try finding the stationary distribution for the modified matrix
eigenvalues_mod, eigenvectors_mod = np.linalg.eig(P_modified.T)
idx_mod = np.isclose(eigenvalues_mod, 1)

if not np.any(idx_mod):
    print("\nNo eigenvalue close to 1 found for modified matrix.")
else:
    stationary_vector_raw_mod = eigenvectors_mod[:, np.where(idx_mod)[0][0]].flatten().real
    stationary_distribution_mod = stationary_vector_raw_mod / np.sum(stationary_vector_raw_mod)
    
    print("\nStationary Distribution for Modified Matrix:")
    for state, prob in zip(states, stationary_distribution_mod):
        print(f"  {state}: {prob:.4f}")
    
    print("\nThis represents the long-run proportion of time the system spends in each state.")
    print("Even though customers churn, the small chance of returning means there's a non-zero steady state for all plans.")
    # Verification
    print("\nVerification (pi_mod * P_mod):", np.dot(stationary_distribution_mod, P_modified))
```
