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

# Chapter 7: Common Discrete Distributions

In the previous chapter, we defined discrete random variables and learned how to describe their behavior using Probability Mass Functions (PMFs), Cumulative Distribution Functions (CDFs), expected value, and variance. While we can define custom PMFs for any situation, several specific discrete distributions appear so frequently in practice that they have been studied extensively and given names.

These "common" distributions serve as powerful models for a wide variety of real-world processes. Understanding their properties and when to apply them is crucial for probabilistic modeling. In this chapter, we will explore the most important discrete distributions: Bernoulli, Binomial, Geometric, Negative Binomial, Poisson, and Hypergeometric.

We'll examine the scenarios each distribution models, their key characteristics (PMF, mean, variance), and how to work with them efficiently using Python's `scipy.stats` library. This library provides tools to calculate probabilities (PMF, CDF), generate random samples, and more, significantly simplifying our practical work.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Configure plots
plt.style.use('seaborn-v0_8-whitegrid')
```

## 1. Bernoulli Distribution

The Bernoulli distribution models a single trial with two possible outcomes: "success" (1) or "failure" (0).

**Concrete Example**

Suppose you're conducting a medical screening test for a disease in a high-risk population. Each test either shows positive or negative. From epidemiological data, you know that 30% of individuals in this population test positive.

We model this with a random variable $X$:
- $X = 1$ if the test result is positive (success)
- $X = 0$ if the test result is negative (failure)

The probabilities are:
- $P(X = 1) = 0.3$ (we call this parameter $p$)
- $P(X = 0) = 0.7$ (which equals $1 - p$)

**The Bernoulli PMF**

For any Bernoulli random variable with success probability $p$, the PMF is:

$$ P(X=k) = \begin{cases} p & \text{if } k=1 \\ 1-p & \text{if } k=0 \\ 0 & \text{otherwise} \end{cases} $$

This can also be written compactly as:

$$P(X = k) = p^k (1-p)^{1-k} \text{ for } k \in \{0, 1\}$$

Let's verify this compact formula works for our example where $p = 0.3$:
- When $k = 1$: $P(X=1) = (0.3)^1 (0.7)^0 = 0.3 \times 1 = 0.3$ ✓
- When $k = 0$: $P(X=0) = (0.3)^0 (0.7)^1 = 1 \times 0.7 = 0.7$ ✓

**Key Characteristics**

- **Scenarios**: Coin flip (Heads/Tails), product inspection (Defective/Not Defective), medical test (Positive/Negative), free throw (Make/Miss)
- **Parameter**: $p$, the probability of success ($0 \le p \le 1$)
- **Random Variable**: $X \in \{0, 1\}$

**Mean:** $E[X] = p$

**Variance:** $Var(X) = p(1-p)$

**Visualizing the Distribution**

Let's visualize a Bernoulli distribution with $p = 0.3$ (our medical test example from above):

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Create Bernoulli distribution for visualization (p=0.3)
p_viz = 0.3
bernoulli_viz = stats.bernoulli(p=p_viz)

# Plotting the PMF
k_values_viz = [0, 1]
pmf_values_viz = bernoulli_viz.pmf(k_values_viz)

plt.figure(figsize=(8, 4))
plt.bar(k_values_viz, pmf_values_viz, tick_label=["Failure (0)", "Success (1)"], color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Bernoulli PMF (p={p_viz})")
plt.xlabel("Outcome")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_bernoulli_pmf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Bernoulli PMF](ch07_bernoulli_pmf_generic.svg)

The PMF shows two bars: P(X=0) = 0.7 for a negative test and P(X=1) = 0.3 for a positive test.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values_viz = bernoulli_viz.cdf(k_values_viz)

plt.figure(figsize=(8, 4))
plt.step(k_values_viz, cdf_values_viz, where='mid', color='darkgreen', linewidth=2)
plt.title(f"Bernoulli CDF (p={p_viz})")
plt.xlabel("Outcome")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.ylim(0, 1.1)
plt.xticks([0, 1])
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_bernoulli_cdf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Bernoulli CDF](ch07_bernoulli_cdf_generic.svg)

The CDF shows the cumulative probability: P(X ≤ 0) = 0.7 (just the negative test outcome) and P(X ≤ 1) = 1.0 (both outcomes).

**Understanding PMF and CDF Charts**

Now that we've seen both types of visualizations, let's understand how to read and use them practically:

**PMF (Probability Mass Function) Charts:**
- **What they show:** The height of each bar represents the probability of that exact outcome
- **How to read:** Look at the bar height to find P(X = k) for any specific value k
- **Practical use:** Answer questions like "What's the probability of getting exactly 3 successes?"
- **Key property:** All bar heights must sum to 1.0 (total probability)

**CDF (Cumulative Distribution Function) Charts:**
- **What they show:** The cumulative probability P(X ≤ k) up to and including each value k
- **How to read:** The height at position k tells you the probability of getting k or fewer successes
- **Why step functions?** For discrete distributions, probability accumulates in jumps at each possible value. Between possible values, the CDF stays constant (no additional probability). The step occurs at each value where the distribution has mass.
- **Practical uses:**
  - Find P(X ≤ k) directly by reading the height at k
  - Find P(X > k) by calculating 1 - P(X ≤ k)
  - Find P(a < X ≤ b) by calculating P(X ≤ b) - P(X ≤ a)
- **Key property:** The CDF always increases (or stays flat) and approaches 1.0

**Note on CDF visualization:** The charts use `where='mid'` in the step plot for visual clarity, which centers the step between points. In mathematical terms, discrete CDFs are right-continuous functions (they jump up at each value and include that value in the cumulative probability).

:::{admonition} Example: Medical Diagnostic Test with p = 0.1
:class: tip

Modeling the outcome of a single medical diagnostic test where the probability of a positive result is 0.1.

Let's use `scipy.stats.bernoulli` to calculate probabilities, compute the mean and variance, and generate random samples.

```{code-cell} ipython3
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Using scipy.stats.bernoulli
p_positive = 0.1
bernoulli_rv = stats.bernoulli(p=p_positive)

# PMF: Probability of success (k=1) and failure (k=0)
print(f"P(X=1) (Positive Test): {bernoulli_rv.pmf(1):.2f}")
print(f"P(X=0) (Negative Test): {bernoulli_rv.pmf(0):.2f}")

# Mean and Variance
print(f"Mean (Expected Value): {bernoulli_rv.mean():.2f}")
print(f"Variance: {bernoulli_rv.var():.2f}")
```

```{code-cell} ipython3
# Generate random samples
n_samples = 10
samples = bernoulli_rv.rvs(size=n_samples)
print(f"{n_samples} simulated test outcomes (1=Positive, 0=Negative):")
print(samples)
```

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the PMF
k_values = [0, 1]
pmf_values = bernoulli_rv.pmf(k_values)

plt.figure(figsize=(8, 4))
plt.bar(k_values, pmf_values, tick_label=["Negative (0)", "Positive (1)"], color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Bernoulli PMF (p={p_positive})")
plt.xlabel("Outcome")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_bernoulli_pmf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Bernoulli PMF](ch07_bernoulli_pmf.svg)

The PMF shows the probability of each outcome. With p = 0.1, "Negative" has probability 0.9 and "Positive" has probability 0.1.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values = bernoulli_rv.cdf(k_values)

plt.figure(figsize=(8, 4))
plt.step(k_values, cdf_values, where='mid', color='darkgreen', linewidth=2)
plt.title(f"Bernoulli CDF (p={p_positive})")
plt.xlabel("Outcome")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.ylim(0, 1.1)
plt.xticks([0, 1])
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_bernoulli_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Bernoulli CDF](ch07_bernoulli_cdf.svg)

The CDF shows cumulative probabilities: P(X ≤ 0) = 0.9 and P(X ≤ 1) = 1.0.

:::

**Quick Check Questions**

1. A quality control inspector checks a single product. It's either defective or not defective. Which distribution models this?

2. For a Bernoulli distribution with p = 0.3, what is P(X = 0)?

3. A basketball player has a 75% free throw success rate. If we model a single free throw, what are the mean and variance?

```{admonition} Answers
:class: dropdown

1. **Bernoulli distribution** - Single trial with two possible outcomes.

2. **P(X = 0) = 1 - p = 0.7** - The probability of failure is 1 - p.

3. **Mean = 0.75, Variance = 0.75 × 0.25 = 0.1875** - Use E[X] = p and Var(X) = p(1-p).
```

+++

## 2. Binomial Distribution

The Binomial distribution models the number of successes in a *fixed number* of independent Bernoulli trials, where each trial has the same probability of success.

**Concrete Example**

Suppose you flip a fair coin 10 times. Each flip is a Bernoulli trial with p = 0.5 (probability of heads). How many heads will you get?

We model this with a random variable $X$:
- $X$ = the number of heads in 10 flips
- $X$ can take values 0, 1, 2, ..., 10

The probabilities are:
- $P(X = 0)$ = probability of 0 heads (all tails)
- $P(X = 5)$ = probability of exactly 5 heads
- $P(X = 10)$ = probability of 10 heads (all heads)

**The Binomial PMF**

For $n$ independent trials with success probability $p$:

$$ P(X=k) = \binom{n}{k} p^k (1-p)^{n-k} \quad \text{for } k = 0, 1, \dots, n $$

where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the binomial coefficient (number of ways to choose $k$ successes from $n$ trials).

Let's verify this works for our coin flip example (n=10, p=0.5):
- $P(X=5) = \binom{10}{5} (0.5)^5 (0.5)^5 = 252 \times 0.03125 \times 0.03125 \approx 0.246$ ✓

**Key Characteristics**

- **Scenarios**: Number of heads in coin flips, defective items in a batch, successful free throws, correct guesses on a test, customers who purchase
- **Parameters**:
    - $n$: number of independent trials
    - $p$: probability of success on each trial ($0 \le p \le 1$)
- **Random Variable**: $X \in \{0, 1, 2, ..., n\}$

**Mean:** $E[X] = np$

**Variance:** $Var(X) = np(1-p)$

**Visualizing the Distribution**

Let's visualize a Binomial distribution with $n = 10$ and $p = 0.5$ (our coin flip example):

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Create Binomial distribution for visualization (n=10, p=0.5)
n_viz = 10
p_viz = 0.5
binomial_viz = stats.binom(n=n_viz, p=p_viz)

# Plotting the PMF
k_values_viz = np.arange(0, n_viz + 1)
pmf_values_viz = binomial_viz.pmf(k_values_viz)

plt.figure(figsize=(8, 4))
plt.bar(k_values_viz, pmf_values_viz, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Binomial PMF (n={n_viz}, p={p_viz})")
plt.xlabel("Number of Successes (k)")
plt.ylabel("Probability")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_binomial_pmf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Binomial PMF](ch07_binomial_pmf_generic.svg)

The PMF shows the probability distribution for the number of heads in 10 coin flips. The distribution is symmetric around the mean (np = 5) since p = 0.5.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values_viz = binomial_viz.cdf(k_values_viz)

plt.figure(figsize=(8, 4))
plt.step(k_values_viz, cdf_values_viz, where='mid', color='darkgreen', linewidth=2)
plt.title(f"Binomial CDF (n={n_viz}, p={p_viz})")
plt.xlabel("Number of Successes (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_binomial_cdf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Binomial CDF](ch07_binomial_cdf_generic.svg)

The CDF shows P(X ≤ k), the cumulative probability of getting k or fewer heads.

:::{admonition} Example: Sales Calls with n = 20, p = 0.15
:class: tip

Modeling the number of successful sales calls out of 20, where each call has a 0.15 probability of success.

We'll demonstrate how to use `scipy.stats.binom` to calculate probabilities, compute statistics, and generate random samples.

:::{dropdown} Python Implementation

```{code-cell} ipython3
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Using scipy.stats.binom
n_calls = 20
p_success_call = 0.15
binomial_rv = stats.binom(n=n_calls, p=p_success_call)

# PMF: Probability of exactly k successes
k_successes = 5
print(f"P(X={k_successes} successes out of {n_calls}): {binomial_rv.pmf(k_successes):.4f}")
```

```{code-cell} ipython3
# CDF: Probability of k or fewer successes
k_or_fewer = 3
print(f"P(X <= {k_or_fewer} successes out of {n_calls}): {binomial_rv.cdf(k_or_fewer):.4f}")
print(f"P(X > {k_or_fewer} successes out of {n_calls}): {1 - binomial_rv.cdf(k_or_fewer):.4f}")
print(f"P(X > {k_or_fewer} successes out of {n_calls}) (using sf): {binomial_rv.sf(k_or_fewer):.4f}")
```

```{code-cell} ipython3
# Mean and Variance
print(f"Mean (Expected number of successes): {binomial_rv.mean():.2f}")
print(f"Variance: {binomial_rv.var():.2f}")
print(f"Standard Deviation: {binomial_rv.std():.2f}")
```

```{code-cell} ipython3
# Generate random samples
n_simulations = 1000
samples = binomial_rv.rvs(size=n_simulations)
# print(f"\nSimulated number of successes in {n_calls} calls ({n_simulations} simulations): {samples[:20]}...") # Print first 20
```

:::

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the PMF
k_values = np.arange(0, n_calls + 1)
pmf_values = binomial_rv.pmf(k_values)

plt.figure(figsize=(8, 4))
plt.bar(k_values, pmf_values, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Binomial PMF (n={n_calls}, p={p_success_call})")
plt.xlabel("Number of Successes (k)")
plt.ylabel("Probability")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_binomial_pmf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Binomial PMF](ch07_binomial_pmf.svg)

The PMF shows the probability distribution for the number of successful calls. With n = 20 and p = 0.15, the distribution is centered around np = 3 successes.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values = binomial_rv.cdf(k_values)

plt.figure(figsize=(8, 4))
plt.step(k_values, cdf_values, where='mid', color='darkgreen', linewidth=2)
plt.title(f"Binomial CDF (n={n_calls}, p={p_success_call})")
plt.xlabel("Number of Successes (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_binomial_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Binomial CDF](ch07_binomial_cdf.svg)

The CDF shows P(X ≤ k), the cumulative probability of getting k or fewer successful calls.

:::

:::

**Quick Check Questions**

1. You roll a die 12 times and count how many times you get a 6. Which distribution models this and what are the parameters?

2. For a Binomial distribution with n = 8 and p = 0.25, what is the expected value (mean)?

3. True or False: In a Binomial distribution, each trial must have the same probability of success.

```{admonition} Answers
:class: dropdown

1. **Binomial distribution with n = 12, p = 1/6** - Fixed number of trials (12 rolls), each with same success probability (1/6).

2. **E[X] = np = 8 × 0.25 = 2** - Expected number of successes is np.

3. **True** - The Binomial distribution requires independent trials with constant success probability p.
```

+++

## 3. Geometric Distribution

The Geometric distribution models the number of independent Bernoulli trials needed to get the *first* success.

**Concrete Example**

You're shooting free throws until you make your first basket. Each shot has a 0.4 probability of success. How many shots will it take to make your first basket?

We model this with a random variable $X$:
- $X$ = the trial number on which the first success occurs
- $X$ can take values 1, 2, 3, ... (first shot, second shot, etc.)

The probabilities are:
- $P(X = 1)$ = make it on first shot = 0.4
- $P(X = 2)$ = miss first, make second = $(1-0.4) \times 0.4 = 0.24$
- $P(X = 3)$ = miss first two, make third = $(1-0.4)^2 \times 0.4 = 0.144$

**The Geometric PMF**

For trials with success probability $p$:

$$ P(X=k) = (1-p)^{k-1} p \quad \text{for } k = 1, 2, 3, \dots $$

This means $k-1$ failures followed by one success.

Let's verify for our example (p=0.4):
- $P(X=2) = (0.6)^1 (0.4) = 0.24$ ✓

**Key Characteristics**

- **Scenarios**: Coin flips until first Head, job applications until first offer, attempts to pass an exam, at-bats until first hit
- **Parameter**: $p$, probability of success on each trial ($0 < p \le 1$)
- **Random Variable**: $X \in \{1, 2, 3, ...\}$

**Mean:** $E[X] = \frac{1}{p}$

**Variance:** $Var(X) = \frac{1-p}{p^2}$

:::{admonition} Note
:class: note

`scipy.stats.geom` defines $k$ as the number of *failures before* the first success ($k=0, 1, 2, ...$), which shifts by 1 from our definition. We'll use scipy's definition in code but state results in terms of trial numbers.
:::

**Visualizing the Distribution**

Let's visualize a Geometric distribution with $p = 0.4$ (our free throw example):

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Create Geometric distribution for visualization (p=0.4)
p_viz = 0.4
geom_viz = stats.geom(p=p_viz)

# Plotting the PMF
k_values_viz = np.arange(1, 11)
pmf_values_viz = geom_viz.pmf(k_values_viz - 1)

plt.figure(figsize=(8, 4))
plt.bar(k_values_viz, pmf_values_viz, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Geometric PMF (p={p_viz})")
plt.xlabel("Trial Number (k)")
plt.ylabel("Probability P(X=k)")
plt.xticks(k_values_viz)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_geometric_pmf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Geometric PMF](ch07_geometric_pmf_generic.svg)

The PMF shows exponentially decreasing probabilities - you're most likely to succeed on the first few trials.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values_viz = geom_viz.cdf(k_values_viz - 1)

plt.figure(figsize=(8, 4))
plt.step(k_values_viz, cdf_values_viz, where='mid', color='darkgreen', linewidth=2)
plt.title(f"Geometric CDF (p={p_viz})")
plt.xlabel("Trial Number (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.xticks(k_values_viz)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_geometric_cdf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Geometric CDF](ch07_geometric_cdf_generic.svg)

The CDF shows P(X ≤ k), approaching 1 as k increases (eventually you'll succeed).

:::{admonition} Example: Certification Exam with p = 0.6
:class: tip

Modeling the number of attempts needed to pass a certification exam where the pass probability is 0.6.

Let's use `scipy.stats.geom` to explore probabilities and compute expected values. Remember that scipy's definition counts failures before the first success, so we'll translate between the two interpretations.

:::{dropdown} Python Implementation

```{code-cell} ipython3
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Using scipy.stats.geom
p_pass = 0.6
geom_rv = stats.geom(p=p_pass)

# PMF: Probability that the first success occurs on trial k (k=1, 2, ...)
# Using scipy: geom_rv.pmf(k-1)
k_trial = 3 # Third attempt
print(f"P(First pass on attempt {k_trial}): {geom_rv.pmf(k_trial - 1):.4f}")
```

```{code-cell} ipython3
# CDF: Probability that the first success occurs on or before trial k
k_or_before = 2
print(f"P(First pass on or before attempt {k_or_before}): {geom_rv.cdf(k_or_before - 1):.4f}")
print(f"P(First pass takes more than {k_or_before} attempts): {1 - geom_rv.cdf(k_or_before - 1):.4f}")
print(f"P(First pass takes more than {k_or_before} attempts) (using sf): {geom_rv.sf(k_or_before - 1):.4f}")
```

```{code-cell} ipython3
# Mean and Variance (based on scipy's definition k=0, 1, 2...)
mean_scipy = geom_rv.mean()
var_scipy = geom_rv.var()
print(f"Mean number of failures before success (scipy): {mean_scipy:.2f}")
print(f"Variance of failures before success (scipy): {var_scipy:.2f}")
```

```{code-cell} ipython3
# Mean and Variance (based on our definition k=1, 2, 3...)
mean_trials = 1 / p_pass
var_trials = (1 - p_pass) / p_pass**2
print(f"Mean number of attempts until first pass: {mean_trials:.2f}")
print(f"Variance of number of attempts: {var_trials:.2f}")
```

```{code-cell} ipython3
# Generate random samples (number of failures before first success)
n_simulations = 1000
samples_failures = geom_rv.rvs(size=n_simulations)
# Convert to trial number (failures + 1)
samples_trials = samples_failures + 1
# print(f"\nSimulated number of attempts until first pass ({n_simulations} simulations): {samples_trials[:20]}...")
```

:::

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the PMF (using trial number k=1, 2, ...)
k_values_trials = np.arange(1, 11) # Plot first 10 trials
pmf_values = geom_rv.pmf(k_values_trials - 1) # Adjust k for scipy

plt.figure(figsize=(8, 4))
plt.bar(k_values_trials, pmf_values, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Geometric PMF (p={p_pass})")
plt.xlabel("Trial Number (k)")
plt.ylabel("Probability P(X=k)")
plt.xticks(k_values_trials)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_geometric_pmf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Geometric PMF](ch07_geometric_pmf.svg)

The PMF shows exponentially decreasing probabilities for the exam example with p = 0.6.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF (using trial number k=1, 2, ...)
cdf_values = geom_rv.cdf(k_values_trials - 1) # Adjust k for scipy

plt.figure(figsize=(8, 4))
plt.step(k_values_trials, cdf_values, where='mid', color='darkgreen', linewidth=2)
plt.title(f"Geometric CDF (p={p_pass})")
plt.xlabel("Trial Number (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.xticks(k_values_trials)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_geometric_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Geometric CDF](ch07_geometric_cdf.svg)

The CDF shows P(X ≤ k), increasing toward 1 as the trial number increases.

:::

:::

**Quick Check Questions**

1. You flip a coin until you get your first Heads. What distribution models this and what is the parameter?

2. For a Geometric distribution with p = 0.25, what is the expected value (mean)?

3. Which is more likely for a Geometric distribution with p = 0.5: success on the 1st trial or success on the 3rd trial?

```{admonition} Answers
:class: dropdown

1. **Geometric distribution with p = 0.5** - Counting trials until first success, each trial has p = 0.5 success probability.

2. **E[X] = 1/p = 1/0.25 = 4** - Expected number of trials until first success.

3. **1st trial is more likely** - Geometric PMF decreases exponentially, so P(X=1) > P(X=3).
```

+++

## 4. Negative Binomial Distribution

The Negative Binomial distribution models the number of independent Bernoulli trials needed to achieve a *fixed number* of successes ($r$). It generalizes the Geometric distribution (where $r=1$).

**Concrete Example**

You're rolling a die until you get 3 sixes. Each roll has p = 1/6 probability of rolling a six. How many rolls will it take to get your 3rd six?

We model this with a random variable $X$:
- $X$ = the trial number on which the 3rd six appears
- $X$ can take values 3, 4, 5, ... (minimum 3 rolls, could be more)

The probabilities are:
- $P(X = 3)$ = all three rolls are sixes = $(1/6)^3 \approx 0.0046$
- $P(X = 4)$ = 2 sixes in first 3 rolls, then a six on 4th roll
- And so on...

**The Negative Binomial PMF**

For trials with success probability $p$ and target $r$ successes:

$$ P(X=k) = \binom{k-1}{r-1} p^r (1-p)^{k-r} \quad \text{for } k = r, r+1, r+2, \dots $$

This means $r-1$ successes in the first $k-1$ trials, and the $k$-th trial is the $r$-th success.

**Key Characteristics**

- **Scenarios**: Coin flips until getting r Heads, products inspected to find r defects, interviews until making r hires
- **Parameters**:
    - $r$: target number of successes ($r \ge 1$)
    - $p$: probability of success on each trial ($0 < p \le 1$)
- **Random Variable**: $X \in \{r, r+1, r+2, ...\}$

**Mean:** $E[X] = \frac{r}{p}$

**Variance:** $Var(X) = \frac{r(1-p)}{p^2}$

:::{admonition} Note
:class: note

`scipy.stats.nbinom` counts the number of *failures* before the $r$-th success, not total trials. We'll use scipy's definition in code but state results in terms of total trials.
:::

**Visualizing the Distribution**

Let's visualize a Negative Binomial distribution with $r = 3$ and $p = 0.2$ (easier to see than our 1/6 example):

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Create Negative Binomial distribution for visualization (r=3, p=0.2)
r_viz = 3
p_viz = 0.2
nbinom_viz = stats.nbinom(n=r_viz, p=p_viz)

# Plotting the PMF
k_values_viz = np.arange(r_viz, 30)  # Total trials from r to 30
pmf_values_viz = nbinom_viz.pmf(k_values_viz - r_viz)  # Adjust for scipy

plt.figure(figsize=(8, 4))
plt.bar(k_values_viz, pmf_values_viz, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Negative Binomial PMF (r={r_viz}, p={p_viz})")
plt.xlabel("Total Number of Trials (k)")
plt.ylabel("Probability P(X=k)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_negative_binomial_pmf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Negative Binomial PMF](ch07_negative_binomial_pmf_generic.svg)

The PMF shows the distribution is centered around the expected value r/p = 3/0.2 = 15 trials.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values_viz = nbinom_viz.cdf(k_values_viz - r_viz)

plt.figure(figsize=(8, 4))
plt.step(k_values_viz, cdf_values_viz, where='mid', color='darkgreen', linewidth=2)
plt.title(f"Negative Binomial CDF (r={r_viz}, p={p_viz})")
plt.xlabel("Total Number of Trials (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_negative_binomial_cdf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Negative Binomial CDF](ch07_negative_binomial_cdf_generic.svg)

The CDF shows P(X ≤ k), the cumulative probability of achieving r successes within k trials.

:::{admonition} Example: Quality Control with r = 3, p = 0.05
:class: tip

A quality control inspector tests electronic components until finding 3 defective ones. The defect rate is p = 0.05.

We'll use `scipy.stats.nbinom` to calculate the probability of needing a certain number of trials and compute expected values, keeping in mind scipy's definition of counting failures.

:::{dropdown} Python Implementation

```{code-cell} ipython3
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Using scipy.stats.nbinom
r_defective = 3
p_defective = 0.05
nbinom_rv = stats.nbinom(n=r_defective, p=p_defective)

# PMF: Probability of needing k components tested to find r defective
k_components = 80
num_good = k_components - r_defective
if num_good >= 0:
    prob_k_components = nbinom_rv.pmf(num_good)
    print(f"P(Need exactly {k_components} components to find {r_defective} defective): {prob_k_components:.4f}")
else:
    print(f"Cannot find {r_defective} defective in fewer than {r_defective} components.")
```

```{code-cell} ipython3
# CDF: Probability of needing k or fewer components
k_or_fewer_components = 100
num_good_max = k_or_fewer_components - r_defective
if num_good_max >= 0:
    prob_k_or_fewer = nbinom_rv.cdf(num_good_max)
    print(f"P(Need {k_or_fewer_components} or fewer components to find {r_defective} defective): {prob_k_or_fewer:.4f}")
else:
    print(f"Cannot find {r_defective} defective in fewer than {r_defective} components.")
```

```{code-cell} ipython3
# Mean and Variance (scipy's definition: number of non-defective items)
mean_good_scipy = nbinom_rv.mean()
var_good_scipy = nbinom_rv.var()
print(f"Mean number of good components before {r_defective} defective (scipy): {mean_good_scipy:.2f}")
print(f"Variance of good components before {r_defective} defective (scipy): {var_good_scipy:.2f}")
```

```{code-cell} ipython3
# Mean and Variance (our definition: total components tested)
mean_components = r_defective / p_defective
var_components = r_defective * (1 - p_defective) / p_defective**2
print(f"Mean number of components to test for {r_defective} defective: {mean_components:.2f}")
print(f"Variance of number of components: {var_components:.2f}")
```

```{code-cell} ipython3
# Generate random samples (number of good components before r defective)
n_simulations = 1000
samples_good_nb = nbinom_rv.rvs(size=n_simulations)
# Convert to total components tested (good + r defective)
samples_components_nb = samples_good_nb + r_defective
# print(f"\nSimulated components tested to find {r_defective} defective ({n_simulations} sims): {samples_components_nb[:20]}...")
```

:::

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the PMF (using total components tested k = r, r+1, ...)
k_values_components = np.arange(r_defective, r_defective + 150) # Plot a range
pmf_values_nb = nbinom_rv.pmf(k_values_components - r_defective) # Adjust k for scipy

plt.figure(figsize=(8, 4))
plt.bar(k_values_components, pmf_values_nb, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Negative Binomial PMF (r={r_defective}, p={p_defective})")
plt.xlabel("Total Number of Components Tested (k)")
plt.ylabel("Probability P(X=k)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_negative_binomial_pmf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Negative Binomial PMF](ch07_negative_binomial_pmf.svg)

The PMF shows the distribution centered around r/p = 60 components with considerable variability.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF (using total components tested k = r, r+1, ...)
cdf_values_nb = nbinom_rv.cdf(k_values_components - r_defective) # Adjust k for scipy

plt.figure(figsize=(8, 4))
plt.step(k_values_components, cdf_values_nb, where='mid', color='darkgreen', linewidth=2)
plt.title(f"Negative Binomial CDF (r={r_defective}, p={p_defective})")
plt.xlabel("Total Number of Components Tested (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_negative_binomial_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Negative Binomial CDF](ch07_negative_binomial_cdf.svg)

The CDF shows P(X ≤ k), the cumulative probability of finding 3 defective items within k tests.

:::

:::

**Quick Check Questions**

1. You flip a fair coin until you get 5 Heads. What distribution models this and what are the parameters?

2. For a Negative Binomial distribution with r = 4 and p = 0.5, what is the expected value (mean)?

3. How is Negative Binomial related to Geometric distribution?

```{admonition} Answers
:class: dropdown

1. **Negative Binomial with r = 5, p = 0.5** - Counting trials until getting r successes, each trial has p = 0.5.

2. **E[X] = r/p = 4/0.5 = 8** - Expected number of trials to get 4 successes.

3. **Geometric is a special case where r = 1** - Negative Binomial with r=1 is identical to Geometric.
```

+++

## 5. Poisson Distribution

The Poisson distribution models the number of events occurring in a fixed interval of time or space when events happen independently at a constant average rate.

**Concrete Example**

You receive an average of 4 customer calls per hour. How many calls will you get in the next hour?

We model this with a random variable $X$:
- $X$ = the number of calls in one hour
- $X$ can take values 0, 1, 2, 3, ... (any non-negative integer)

The average rate is $\lambda = 4$ calls/hour.

**The Poisson PMF**

For events occurring at average rate $\lambda$:

$$ P(X=k) = \frac{e^{-\lambda} \lambda^k}{k!} \quad \text{for } k = 0, 1, 2, \dots $$

where $e \approx 2.71828$ is Euler's number.

Let's verify for our example (λ=4):
- $P(X=4) = \frac{e^{-4} \times 4^4}{4!} \approx 0.195$ ✓

**Key Characteristics**

- **Scenarios**: Emails per hour, customer arrivals per day, typos per page, emergency calls per shift, defects per unit area
- **Parameter**: $\lambda$, average number of events in the interval ($\lambda > 0$)
- **Random Variable**: $X \in \{0, 1, 2, ...\}$

**Mean:** $E[X] = \lambda$

**Variance:** $Var(X) = \lambda$

Note: Mean and variance are equal in a Poisson distribution.

**Visualizing the Distribution**

Let's visualize a Poisson distribution with $\lambda = 4$ (our call center example):

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Create Poisson distribution for visualization (λ=4)
lambda_viz = 4
poisson_viz = stats.poisson(mu=lambda_viz)

# Plotting the PMF
k_values_viz = np.arange(0, 15)
pmf_values_viz = poisson_viz.pmf(k_values_viz)

plt.figure(figsize=(8, 4))
plt.bar(k_values_viz, pmf_values_viz, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Poisson PMF (λ={lambda_viz})")
plt.xlabel("Number of Events (k)")
plt.ylabel("Probability P(X=k)")
plt.xticks(k_values_viz)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_poisson_pmf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Poisson PMF](ch07_poisson_pmf_generic.svg)

The PMF shows the distribution centered around λ = 4 with reasonable probability for nearby values.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values_viz = poisson_viz.cdf(k_values_viz)

plt.figure(figsize=(8, 4))
plt.step(k_values_viz, cdf_values_viz, where='mid', color='darkgreen', linewidth=2)
plt.title(f"Poisson CDF (λ={lambda_viz})")
plt.xlabel("Number of Events (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.xticks(k_values_viz)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_poisson_cdf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Poisson CDF](ch07_poisson_cdf_generic.svg)

The CDF shows P(X ≤ k), useful for questions like "What's the probability of 6 or fewer calls?"

:::{admonition} Example: Email Arrivals with λ = 5
:class: tip

Modeling the number of emails received per hour with an average rate of λ = 5 emails/hour.

Let's use `scipy.stats.poisson` to calculate the probability of observing different numbers of events and verify that the mean equals the variance.

:::{dropdown} Python Implementation

```{code-cell} ipython3
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Using scipy.stats.poisson
lambda_rate = 5
poisson_rv = stats.poisson(mu=lambda_rate)

# PMF: Probability of exactly k events
k_events = 3
print(f"P(X={k_events} emails in an hour | lambda={lambda_rate}): {poisson_rv.pmf(k_events):.4f}")
```

```{code-cell} ipython3
# CDF: Probability of k or fewer events
k_or_fewer_events = 6
print(f"P(X <= {k_or_fewer_events} emails in an hour): {poisson_rv.cdf(k_or_fewer_events):.4f}")
print(f"P(X > {k_or_fewer_events} emails in an hour): {1 - poisson_rv.cdf(k_or_fewer_events):.4f}")
print(f"P(X > {k_or_fewer_events} emails in an hour) (using sf): {poisson_rv.sf(k_or_fewer_events):.4f}")
```

```{code-cell} ipython3
# Mean and Variance
print(f"Mean (Expected number of emails): {poisson_rv.mean():.2f}")
print(f"Variance: {poisson_rv.var():.2f}")
```

```{code-cell} ipython3
# Generate random samples
n_simulations = 1000
samples = poisson_rv.rvs(size=n_simulations)
# print(f"\nSimulated number of emails per hour ({n_simulations} simulations): {samples[:20]}...")
```

:::

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the PMF
k_values = np.arange(0, 16) # Plot for k=0 to 15
pmf_values = poisson_rv.pmf(k_values)

plt.figure(figsize=(8, 4))
plt.bar(k_values, pmf_values, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Poisson PMF (λ={lambda_rate})")
plt.xlabel("Number of Events (k)")
plt.ylabel("Probability P(X=k)")
plt.xticks(k_values)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_poisson_pmf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Poisson PMF](ch07_poisson_pmf.svg)

The PMF shows the distribution centered around λ = 5 events.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values = poisson_rv.cdf(k_values)

plt.figure(figsize=(8, 4))
plt.step(k_values, cdf_values, where='mid', color='darkgreen', linewidth=2)
plt.title(f"Poisson CDF (λ={lambda_rate})")
plt.xlabel("Number of Events (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.xticks(k_values)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_poisson_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Poisson CDF](ch07_poisson_cdf.svg)

The CDF shows P(X ≤ k), useful for questions like "What's the probability of 6 or fewer emails?"

:::

:::

**Quick Check Questions**

1. A call center receives an average of 12 calls per hour. What distribution models the number of calls in one hour and what is the parameter?

2. For a Poisson distribution with λ = 7, what are the mean and variance?

3. True or False: In a Poisson distribution, the mean can be different from the variance.

```{admonition} Answers
:class: dropdown

1. **Poisson distribution with λ = 12** - Events occurring at constant average rate in fixed interval.

2. **Mean = 7, Variance = 7** - For Poisson, both equal λ.

3. **False** - A key property of Poisson is that mean = variance = λ.
```

+++

## 6. Hypergeometric Distribution

The Hypergeometric distribution models the number of successes in a sample drawn *without replacement* from a finite population. This is different from Binomial, which assumes sampling with replacement (or infinite population).

**Concrete Example**

You draw 5 cards from a standard deck of 52 cards. How many Aces will you get?

We model this with a random variable $X$:
- $X$ = the number of Aces in the 5-card hand
- Population: N = 52 cards total
- Successes in population: K = 4 Aces
- Sample size: n = 5 cards drawn
- $X$ can take values 0, 1, 2, 3, 4 (can't get more than 4 Aces!)

**The Hypergeometric PMF**

For sampling without replacement:

$$ P(X=k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}} $$

This is: (ways to choose k successes from K) × (ways to choose n-k failures from N-K) / (total ways to choose n items from N).

**Key Characteristics**

- **Scenarios**: Cards from a deck, defective items in small batch, tagged fish in sample, jury selection from finite pool
- **Parameters**:
    - $N$: total population size
    - $K$: total number of successes in population
    - $n$: sample size ($n \le N$)
- **Random Variable**: $X$, bounded by $\max(0, n-(N-K)) \le X \le \min(n, K)$

**Mean:** $E[X] = n \frac{K}{N}$

**Variance:** $Var(X) = n \frac{K}{N} \left(1 - \frac{K}{N}\right) \left(\frac{N-n}{N-1}\right)$

The term $\frac{N-n}{N-1}$ is the *finite population correction factor*. As $N \to \infty$, this approaches 1, and Hypergeometric → Binomial with $p = K/N$.

**Visualizing the Distribution**

Let's visualize a Hypergeometric distribution with N=52, K=4, n=5 (our card example):

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Create Hypergeometric distribution for visualization (N=52, K=4, n=5)
N_viz = 52
K_viz = 4
n_viz = 5
hypergeom_viz = stats.hypergeom(M=N_viz, n=K_viz, N=n_viz)

# Plotting the PMF
k_values_viz = np.arange(0, min(n_viz, K_viz) + 1)
pmf_values_viz = hypergeom_viz.pmf(k_values_viz)

plt.figure(figsize=(8, 4))
plt.bar(k_values_viz, pmf_values_viz, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Hypergeometric PMF (N={N_viz}, K={K_viz}, n={n_viz})")
plt.xlabel("Number of Successes in Sample (k)")
plt.ylabel("Probability P(X=k)")
plt.xticks(k_values_viz)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_hypergeometric_pmf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Hypergeometric PMF](ch07_hypergeometric_pmf_generic.svg)

The PMF shows most likely to get 0 Aces (about 0.66 probability), less likely to get 1 or 2.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values_viz = hypergeom_viz.cdf(k_values_viz)

plt.figure(figsize=(8, 4))
plt.step(k_values_viz, cdf_values_viz, where='mid', color='darkgreen', linewidth=2)
plt.title(f"Hypergeometric CDF (N={N_viz}, K={K_viz}, n={n_viz})")
plt.xlabel("Number of Successes in Sample (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.xticks(k_values_viz)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_hypergeometric_cdf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Hypergeometric CDF](ch07_hypergeometric_cdf_generic.svg)

The CDF shows P(X ≤ k), useful for questions like "What's the probability of getting at most 1 Ace?"

:::{admonition} Example: Lottery Tickets with N=100, K=20, n=10
:class: tip

Modeling the number of winning lottery tickets in a sample of 10 drawn from a box of 100 tickets where 20 are winners.

We'll use `scipy.stats.hypergeom` to calculate probabilities for sampling without replacement and see how the mean relates to the population proportion.

:::{dropdown} Python Implementation

```{code-cell} ipython3
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Using scipy.stats.hypergeom
N_population = 100
K_successes_pop = 20
n_sample = 10
hypergeom_rv = stats.hypergeom(M=N_population, n=K_successes_pop, N=n_sample)

# PMF: Probability of exactly k successes in the sample
k_successes_sample = 3
print(f"P(X={k_successes_sample} winning tickets in sample of {n_sample}): {hypergeom_rv.pmf(k_successes_sample):.4f}")
```

```{code-cell} ipython3
# CDF: Probability of k or fewer successes in the sample
k_or_fewer_sample = 2
print(f"P(X <= {k_or_fewer_sample} winning tickets in sample): {hypergeom_rv.cdf(k_or_fewer_sample):.4f}")
print(f"P(X > {k_or_fewer_sample} winning tickets in sample): {1 - hypergeom_rv.cdf(k_or_fewer_sample):.4f}")
print(f"P(X > {k_or_fewer_sample} winning tickets in sample) (using sf): {hypergeom_rv.sf(k_or_fewer_sample):.4f}")
```

```{code-cell} ipython3
# Mean and Variance
print(f"Mean (Expected number of winning tickets in sample): {hypergeom_rv.mean():.2f}")
print(f"Variance: {hypergeom_rv.var():.2f}")
print(f"Standard Deviation: {hypergeom_rv.std():.2f}")
# Theoretical mean: E[X] = n * (K/N) = 10 * (20/100) = 2.0
```

```{code-cell} ipython3
# Generate random samples
n_simulations = 1000
samples = hypergeom_rv.rvs(size=n_simulations)
# print(f"\nSimulated number of winning tickets ({n_simulations} simulations): {samples[:20]}...")
```

:::

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the PMF
# Determine possible k values: max(0, n-(N-K)) <= k <= min(n, K)
min_k = max(0, n_sample - (N_population - K_successes_pop))
max_k = min(n_sample, K_successes_pop)
k_values = np.arange(min_k, max_k + 1)
pmf_values = hypergeom_rv.pmf(k_values)

plt.figure(figsize=(8, 4))
plt.bar(k_values, pmf_values, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Hypergeometric PMF (N={N_population}, K={K_successes_pop}, n={n_sample})")
plt.xlabel("Number of Successes in Sample (k)")
plt.ylabel("Probability P(X=k)")
plt.xticks(k_values)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_hypergeometric_pmf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Hypergeometric PMF](ch07_hypergeometric_pmf.svg)

The PMF shows the probability distribution for the number of winning tickets in a sample of n = 10 tickets. With N = 100 total tickets and K = 20 winners (from our example), the expected value is n × (K/N) = 10 × 0.2 = 2 winning tickets.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values = hypergeom_rv.cdf(k_values)

plt.figure(figsize=(8, 4))
plt.step(k_values, cdf_values, where='mid', color='darkgreen', linewidth=2)
plt.title(f"Hypergeometric CDF (N={N_population}, K={K_successes_pop}, n={n_sample})")
plt.xlabel("Number of Successes in Sample (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.xticks(k_values)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_hypergeometric_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Hypergeometric CDF](ch07_hypergeometric_cdf.svg)

The CDF shows P(X ≤ k), the cumulative probability of getting k or fewer winning tickets.

:::

:::

**Quick Check Questions**

1. You draw 7 cards from a deck of 52. You want to know how many hearts you get. What distribution models this and what are the parameters?

2. For a Hypergeometric distribution with N=50, K=10, n=5, what is the expected value (mean)?

3. What's the key difference between Binomial and Hypergeometric distributions?

```{admonition} Answers
:class: dropdown

1. **Hypergeometric with N=52, K=13, n=7** - Sampling without replacement from finite population (13 hearts in 52 cards).

2. **E[X] = n(K/N) = 5 × (10/50) = 1** - Expected number of successes in sample.

3. **Hypergeometric samples WITHOUT replacement** (finite population), while Binomial samples WITH replacement (or assumes infinite population).
```

+++

## 7. Relationships Between Distributions

Understanding the connections between these distributions can deepen insight and provide useful approximations.

1.  **Bernoulli as a special case of Binomial**: A Binomial distribution with $n=1$ trial ($Binomial(1, p)$) is equivalent to a Bernoulli distribution ($Bernoulli(p)$).

2.  **Geometric as a special case of Negative Binomial**: A Negative Binomial distribution modeling the number of trials until the first success ($r=1$) ($NegativeBinomial(1, p)$) is equivalent to a Geometric distribution ($Geometric(p)$).

3.  **Binomial Approximation to Hypergeometric**: If the population size $N$ is much larger than the sample size $n$ (e.g., $N > 20n$), then drawing without replacement (Hypergeometric) is very similar to drawing with replacement. In this case, the Hypergeometric($N, K, n$) distribution can be well-approximated by the Binomial($n, p=K/N$) distribution. The finite population correction factor $\frac{N-n}{N-1}$ approaches 1.

4.  **Poisson Approximation to Binomial**: If the number of trials $n$ in a Binomial distribution is large, and the success probability $p$ is small, such that the mean $\lambda = np$ is moderate, then the Binomial($n, p$) distribution can be well-approximated by the Poisson($\lambda = np$) distribution. This is useful because the Poisson PMF is often easier to compute than the Binomial PMF when $n$ is large. A common rule of thumb is to use this approximation if $n \ge 20$ and $p \le 0.05$, or $n \ge 100$ and $np \le 10$.

**Example: Poisson approximation to Binomial**
Consider $Binomial(n=1000, p=0.005)$. Here $n$ is large, $p$ is small. The mean is $\lambda = np = 1000 \times 0.005 = 5$. We can approximate this with $Poisson(\lambda=5)$.

Let's compare the PMF values of both distributions to see how well the Poisson approximation works in practice.

:::{dropdown} Python Implementation

```{code-cell} ipython3
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Setup distributions
n_binom_approx = 1000
p_binom_approx = 0.005
lambda_approx = n_binom_approx * p_binom_approx

binom_rv_approx = stats.binom(n=n_binom_approx, p=p_binom_approx)
poisson_rv_approx = stats.poisson(mu=lambda_approx)

# Compare PMFs
k_vals_compare = np.arange(0, 15)
binom_pmf = binom_rv_approx.pmf(k_vals_compare)
poisson_pmf = poisson_rv_approx.pmf(k_vals_compare)

print(f"Comparing Binomial(n={n_binom_approx}, p={p_binom_approx}) and Poisson(lambda={lambda_approx:.1f})")
print("k\tBinomial P(X=k)\tPoisson P(X=k)\tDifference")
for k, bp, pp in zip(k_vals_compare, binom_pmf, poisson_pmf):
    print(f"{k}\t{bp:.6f}\t{pp:.6f}\t{abs(bp-pp):.6f}")
```

:::

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the comparison
plt.figure(figsize=(10, 5))
plt.bar(k_vals_compare - 0.2, binom_pmf, width=0.4, label=f'Binomial(n={n_binom_approx}, p={p_binom_approx})', align='center', color='skyblue', edgecolor='black', alpha=0.7)
plt.bar(k_vals_compare + 0.2, poisson_pmf, width=0.4, label=f'Poisson(lambda={lambda_approx:.1f})', align='center', color='lightcoral', edgecolor='black', alpha=0.7)
plt.title("Poisson Approximation to Binomial")
plt.xlabel("Number of Successes (k)")
plt.ylabel("Probability")
plt.xticks(k_vals_compare)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('ch07_poisson_binomial_approximation.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Poisson Approximation to Binomial](ch07_poisson_binomial_approximation.svg)

The chart compares the Binomial(100, 0.03) distribution (blue bars) with the Poisson(3.0) approximation (red bars). The distributions are nearly identical, demonstrating that when n is large and p is small, the Poisson provides an excellent and computationally simpler approximation to the Binomial.

+++

## Summary

In this chapter, we explored six fundamental discrete probability distributions:

* **Bernoulli**: Single trial, two outcomes (Success/Failure).
* **Binomial**: Fixed number of independent trials, counts successes.
* **Geometric**: Number of trials until the *first* success.
* **Negative Binomial**: Number of trials until a *fixed number* ($r$) of successes.
* **Poisson**: Number of events in a fixed interval of time/space, given an average rate.
* **Hypergeometric**: Number of successes in a sample drawn *without* replacement from a finite population.

We learned the scenarios each distribution models, their parameters, PMFs, means, and variances. Critically, we saw how to leverage `scipy.stats` functions (`pmf`, `cdf`, `rvs`, `mean`, `var`, `std`, `sf`) to perform calculations, generate simulations, and visualize these distributions. We also discussed important relationships, such as the Poisson approximation to the Binomial and the Binomial approximation to the Hypergeometric.

Mastering these distributions provides a powerful toolkit for modeling various random phenomena encountered in data analysis, science, engineering, and business. In the next chapters, we will transition to continuous random variables and their corresponding common distributions.

## Exercises

1. **Customer Arrivals:** The average number of customers arriving at a small cafe is 10 per hour. Assume arrivals follow a Poisson distribution.
    a. What is the probability that exactly 8 customers arrive in a given hour?
    b. What is the probability that 12 or fewer customers arrive in a given hour?
    c. What is the probability that more than 15 customers arrive in a given hour?
    d. Simulate 1000 hours of customer arrivals and plot a histogram of the results. Compare it to the theoretical PMF.

    ```{admonition} Answer
    :class: dropdown

    a) Using the Poisson distribution with $\lambda = 10$:

    ```{code-cell} ipython3
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    lambda_cafe = 10
    cafe_rv = stats.poisson(mu=lambda_cafe)
    prob_8 = cafe_rv.pmf(8)
    print(f"P(Exactly 8 customers) = {prob_8:.4f}")
    ```

    b) The probability of 12 or fewer customers:

    ```{code-cell} ipython3
    prob_12_or_fewer = cafe_rv.cdf(12)
    print(f"P(12 or fewer customers) = {prob_12_or_fewer:.4f}")
    ```

    c) The probability of more than 15 customers:

    ```{code-cell} ipython3
    prob_over_15 = cafe_rv.sf(15)
    print(f"P(More than 15 customers) = {prob_over_15:.4f}")
    ```

    d) Simulation and visualization:

    ```{code-cell} ipython3
    n_sim_hours = 1000
    sim_arrivals = cafe_rv.rvs(size=n_sim_hours)

    plt.figure(figsize=(10, 5))
    max_observed = np.max(sim_arrivals)
    bins = np.arange(0, max_observed + 2) - 0.5
    plt.hist(sim_arrivals, bins=bins, density=True, alpha=0.6, color='lightgreen', edgecolor='black', label='Simulated Arrivals')

    # Overlay theoretical PMF
    k_vals_cafe = np.arange(0, max_observed + 1)
    pmf_cafe = cafe_rv.pmf(k_vals_cafe)
    plt.plot(k_vals_cafe, pmf_cafe, 'ro-', linewidth=2, markersize=6, label='Theoretical PMF')

    plt.title(f'Simulated Customer Arrivals vs Poisson PMF (lambda={lambda_cafe})')
    plt.xlabel('Number of Customers per Hour')
    plt.ylabel('Probability / Density')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xlim(-0.5, max_observed + 1.5)
    plt.show()
    ```

    The histogram closely matches the theoretical PMF, confirming the Poisson model.
    ```

2. **Quality Control:** A batch contains 50 items, of which 5 are defective. You randomly sample 8 items without replacement.
    a. What distribution models the number of defective items in your sample? State the parameters.
    b. What is the probability that exactly 1 item in your sample is defective?
    c. What is the probability that at most 2 items in your sample are defective?
    d. What is the expected number of defective items in your sample?

    ```{admonition} Answer
    :class: dropdown

    a) This follows a Hypergeometric distribution since we're sampling without replacement from a finite population. The parameters are: $N=50$ (population size), $K=5$ (defective items in population), $n=8$ (sample size).

    ```{code-cell} ipython3
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    N_qc = 50
    K_qc = 5
    n_qc = 8
    qc_rv = stats.hypergeom(M=N_qc, n=K_qc, N=n_qc)
    print(f"Distribution: Hypergeometric(N={N_qc}, K={K_qc}, n={n_qc})")
    ```

    b) Probability of exactly 1 defective item:

    ```{code-cell} ipython3
    prob_1_defective = qc_rv.pmf(1)
    print(f"P(Exactly 1 defective in sample) = {prob_1_defective:.4f}")
    ```

    c) Probability of at most 2 defective items:

    ```{code-cell} ipython3
    prob_at_most_2 = qc_rv.cdf(2)
    print(f"P(At most 2 defectives in sample) = {prob_at_most_2:.4f}")
    ```

    d) Expected number of defective items:

    ```{code-cell} ipython3
    expected_defective = qc_rv.mean()
    print(f"Expected number of defectives in sample = {expected_defective:.4f}")
    # Theoretical: E[X] = n * (K/N) = 8 * (5/50) = 0.8
    ```
    ```

3. **Website Success:** A new website feature has a 3% chance of being used by a visitor ($p=0.03$). Assume visitors are independent.
    a. If 100 visitors come to the site, what is the probability that exactly 3 visitors use the feature? What distribution applies?
    b. What is the probability that 5 or fewer visitors use the feature out of 100?
    c. What is the expected number of users out of 100 visitors?
    d. A developer tests the feature repeatedly until the first user successfully uses it. What is the probability that the first success occurs on the 20th visitor? What distribution applies?
    e. What is the expected number of visitors needed to see the first success?
    f. How many visitors are expected until the 5th user is observed? What distribution applies?

    ```{admonition} Answer
    :class: dropdown

    a) This follows a Binomial distribution with $n=100$ trials and $p=0.03$:

    ```{code-cell} ipython3
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    p_ws = 0.03
    n_ws = 100
    ws_binom_rv = stats.binom(n=n_ws, p=p_ws)
    prob_3_users = ws_binom_rv.pmf(3)
    print(f"Distribution: Binomial(n={n_ws}, p={p_ws})")
    print(f"P(Exactly 3 users) = {prob_3_users:.4f}")
    ```

    b) Probability of 5 or fewer users:

    ```{code-cell} ipython3
    prob_5_or_fewer = ws_binom_rv.cdf(5)
    print(f"P(5 or fewer users) = {prob_5_or_fewer:.4f}")
    ```

    c) Expected number of users:

    ```{code-cell} ipython3
    expected_users = ws_binom_rv.mean()
    print(f"Expected number of users = {expected_users:.2f}")
    # Theoretical: E[X] = n*p = 100 * 0.03 = 3
    ```

    d) This follows a Geometric distribution. The probability that the first success occurs on trial 20:

    ```{code-cell} ipython3
    ws_geom_rv = stats.geom(p=p_ws)
    prob_first_on_20 = ws_geom_rv.pmf(19)  # scipy counts 19 failures before success
    print(f"Distribution: Geometric(p={p_ws})")
    print(f"P(First success on trial 20) = {prob_first_on_20:.4f}")
    ```

    e) Expected number of visitors until first success:

    ```{code-cell} ipython3
    expected_trials_geom = 1 / p_ws
    print(f"Expected visitors until first success = {expected_trials_geom:.2f}")
    # Theoretical: E[X] = 1/p = 1/0.03 ≈ 33.33
    ```

    f) This follows a Negative Binomial distribution with $r=5$ successes:

    ```{code-cell} ipython3
    r_ws = 5
    expected_trials_nbinom = r_ws / p_ws
    print(f"Distribution: Negative Binomial(r={r_ws}, p={p_ws})")
    print(f"Expected visitors until 5th success = {expected_trials_nbinom:.2f}")
    # Theoretical: E[X] = r/p = 5/0.03 ≈ 166.67
    ```
    ```
