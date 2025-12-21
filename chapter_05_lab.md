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
---

# Chapter 5: Bayes' Theorem and Independence

## Exercise 5.1: Implementing Bayes' Theorem for Disease Test


Let's verify the disease test calculation using Python. Define variables for the prior probability, sensitivity, and specificity, then implement the calculation for $P(D|Pos)$.

```{code-cell} ipython3
# Parameters
p_disease = 0.01        # P(D) - Prior probability (prevalence)
p_pos_given_disease = 0.95 # P(Pos|D) - Sensitivity
p_neg_given_no_disease = 0.95 # P(Neg|D^c) - Specificity

# Derived probabilities
p_no_disease = 1 - p_disease                 # P(D^c)
p_pos_given_no_disease = 1 - p_neg_given_no_disease # P(Pos|D^c) - False Positive Rate

# Calculate P(Pos) using Law of Total Probability
p_pos = (p_pos_given_disease * p_disease) + (p_pos_given_no_disease * p_no_disease)

# Calculate P(D|Pos) using Bayes' Theorem
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos

print(f"Prior P(Disease): {p_disease:.4f}")
print(f"Sensitivity P(Pos|Disease): {p_pos_given_disease:.4f}")
print(f"Specificity P(Neg|No Disease): {p_neg_given_no_disease:.4f}")
print(f"False Positive Rate P(Pos|No Disease): {p_pos_given_no_disease:.4f}")
print("-" * 30)
print(f"Overall P(Pos): {p_pos:.4f}")
print(f"Posterior P(Disease|Pos): {p_disease_given_pos:.4f}")

# What if the test is *negative*? Calculate P(Disease | Neg)
# P(Neg) = P(Neg|D)P(D) + P(Neg|D^c)P(D^c)
p_neg_given_disease = 1 - p_pos_given_disease # P(Neg|D) - False Negative Rate
p_neg = (p_neg_given_disease * p_disease) + (p_neg_given_no_disease * p_no_disease)

# P(D|Neg) = P(Neg|D)P(D) / P(Neg)
p_disease_given_neg = (p_neg_given_disease * p_disease) / p_neg

print("-" * 30)
print(f"Overall P(Neg): {p_neg:.4f}")
print(f"Posterior P(Disease|Neg): {p_disease_given_neg:.4f}")
print(f"Posterior P(No Disease|Neg) = {1 - p_disease_given_neg:.4f}")
```

## Exercise 5.2: Simulating Bayesian Updates

+++

Let's simulate the disease test scenario to build intuition. We'll create a population reflecting the disease prevalence, simulate their test results based on sensitivity/specificity, and then calculate the conditional probability directly from the simulated data.

```{code-cell} ipython3
import numpy as np
import pandas as pd

# Parameters
population_size = 1000000
p_disease = 0.01
p_pos_given_disease = 0.95
p_pos_given_no_disease = 0.05 # 1 - specificity

# Create population
# Assign actual disease status
has_disease = np.random.rand(population_size) < p_disease
num_diseased = np.sum(has_disease)
num_healthy = population_size - num_diseased

# Simulate test results
# Initialize test results array
tests_positive = np.zeros(population_size, dtype=bool)

# For those WITH the disease
tests_positive[has_disease] = np.random.rand(num_diseased) < p_pos_given_disease

# For those WITHOUT the disease
tests_positive[~has_disease] = np.random.rand(num_healthy) < p_pos_given_no_disease

# Create a DataFrame for easier analysis
data = pd.DataFrame({'Has_Disease': has_disease, 'Tests_Positive': tests_positive})
print(data.head())

# Calculate counts from the simulation
true_positives = np.sum(data['Has_Disease'] & data['Tests_Positive'])
false_positives = np.sum(~data['Has_Disease'] & data['Tests_Positive'])
total_positives = np.sum(data['Tests_Positive'])

# Calculate P(Disease | Positive) from simulation data
simulated_p_disease_given_pos = true_positives / total_positives

# Compare with theoretical calculation
print("\n--- Simulation Results ---")
print(f"Population Size: {population_size}")
print(f"Number Actually Diseased: {num_diseased}")
print(f"Number Actually Healthy: {num_healthy}")
print(f"Number Testing Positive: {total_positives}")
print(f"  - True Positives: {true_positives}")
print(f"  - False Positives: {false_positives}")
print("-" * 30)
print(f"Simulated P(Disease | Positive): {simulated_p_disease_given_pos:.4f}")
print(f"Theoretical P(Disease | Positive): {p_disease_given_pos:.4f}")
```

As the `population_size` increases, the simulated probability should converge to the theoretical probability calculated using Bayes' Theorem. This demonstrates how the theorem accurately reflects the underlying frequencies in a large population.

+++

## Exercise 5.3: Testing Independence from Data

+++

Let's simulate rolling two fair dice and check if the events "first die is even" and "sum is 7" are independent.

* Event A: First die is even. $P(A) = 1/2$.
* Event B: Sum is 7. The pairs are (1,6), (2,5), (3,4), (4,3), (5,2), (6,1). $P(B) = 6/36 = 1/6$.
* Event $A \cap B$: First die is even AND sum is 7. The pairs are (2,5), (4,3), (6,1). $P(A \cap B) = 3/36 = 1/12$.

Theoretical Check: Is $P(A \cap B) = P(A)P(B)$?
$1/12 \stackrel{?}{=} (1/2) \times (1/6)$
$1/12 = 1/12$. Yes, they are theoretically independent.

Now, let's check using simulation.

```{code-cell} ipython3
import numpy as np
import pandas as pd

num_rolls = 100000

# Simulate rolls
die1 = np.random.randint(1, 7, size=num_rolls)
die2 = np.random.randint(1, 7, size=num_rolls)
sums = die1 + die2

# Define events
event_A = (die1 % 2 == 0)  # First die is even
event_B = (sums == 7)     # Sum is 7

# Create DataFrame
rolls_data = pd.DataFrame({'Die1': die1, 'Die2': die2, 'Sum': sums, 'A': event_A, 'B': event_B})
print(rolls_data.head())

# Calculate probabilities from simulation
p_A_sim = np.mean(event_A)
p_B_sim = np.mean(event_B)
p_A_intersect_B_sim = np.mean(event_A & event_B)

# Check independence condition
print("\n--- Independence Check from Simulation ---")
print(f"Simulated P(A): {p_A_sim:.4f} (Theoretical: 0.5000)")
print(f"Simulated P(B): {p_B_sim:.4f} (Theoretical: {1/6:.4f})")
print(f"Simulated P(A intersect B): {p_A_intersect_B_sim:.4f} (Theoretical: {1/12:.4f})")
print("-" * 30)
print(f"P(A) * P(B) = {p_A_sim * p_B_sim:.4f}")
print(f"Is P(A intersect B) approx equal to P(A) * P(B)? {'Yes' if np.isclose(p_A_intersect_B_sim, p_A_sim * p_B_sim, atol=0.005) else 'No'}") # Use np.isclose for floating point comparison

# Alternative check: Is P(A|B) approx equal to P(A)?
# P(A|B) = P(A intersect B) / P(B)
if p_B_sim > 0:
    p_A_given_B_sim = p_A_intersect_B_sim / p_B_sim
    print(f"\nSimulated P(A|B): {p_A_given_B_sim:.4f}")
    print(f"Is P(A|B) approx equal to P(A)? {'Yes' if np.isclose(p_A_given_B_sim, p_A_sim, atol=0.01) else 'No'}")
else:
    print("\nCannot calculate P(A|B) as P(B) is zero in simulation.")
```

The simulation results should be close to the theoretical values, confirming the independence of these events. Small discrepancies are expected due to random sampling variation.

+++
