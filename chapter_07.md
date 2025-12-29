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

The Bernoulli distribution is the simplest discrete distribution. It models a single trial with only two possible outcomes, often labeled "success" (usually encoded as 1) and "failure" (usually encoded as 0).

- **Scenario**: A single coin flip (Heads/Tails), a single product inspection (Defective/Not Defective), a single customer interaction (Purchase/No Purchase), medical test result (Positive/Negative), free throw attempt (Make/Miss).
- **Parameter**: $p$, the probability of success ($0 \le p \le 1$). The probability of failure is then $q = 1-p$.
- **Random Variable**: $X$ takes value 1 (success) with probability $p$, and 0 (failure) with probability $1-p$.

**PMF:**

$$ P(X=k) = \begin{cases} p & \text{if } k=1 \\ 1-p & \text{if } k=0 \\ 0 & \text{otherwise} \end{cases} $$

This can be written concisely as:

$$ P(X=k) = p^k (1-p)^{1-k} \quad \text{for } k \in \{0, 1\} $$

**Mean (Expected Value):** $E[X] = p$

**Variance:** $Var(X) = p(1-p)$

**Example:** Modeling the outcome of a single customer purchase where the probability of purchase ($p$) is 0.1.

Let's use `scipy.stats.bernoulli` to calculate probabilities, compute the mean and variance, and generate random samples.

:::{dropdown} Python Implementation

```{code-cell} ipython3
# Using scipy.stats.bernoulli
p_purchase = 0.1
bernoulli_rv = stats.bernoulli(p=p_purchase)

# PMF: Probability of success (k=1) and failure (k=0)
print(f"P(X=1) (Purchase): {bernoulli_rv.pmf(1):.2f}")
print(f"P(X=0) (No Purchase): {bernoulli_rv.pmf(0):.2f}")

# Mean and Variance
print(f"Mean (Expected Value): {bernoulli_rv.mean():.2f}")
print(f"Variance: {bernoulli_rv.var():.2f}")
```

```{code-cell} ipython3
# Generate random samples
n_samples = 10
samples = bernoulli_rv.rvs(size=n_samples)
print(f"{n_samples} simulated customer outcomes (1=Purchase, 0=No Purchase):")
print(samples)
```

:::

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the PMF
k_values = [0, 1]
pmf_values = bernoulli_rv.pmf(k_values)

plt.figure(figsize=(8, 4))
plt.bar(k_values, pmf_values, tick_label=["No Purchase (0)", "Purchase (1)"], color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Bernoulli PMF (p={p_purchase})")
plt.xlabel("Outcome")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_bernoulli_pmf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Bernoulli PMF](ch07_bernoulli_pmf.svg)

The PMF shows the probability of each outcome: 0.9 for "No Purchase" and 0.1 for "Purchase".

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values = bernoulli_rv.cdf(k_values)

plt.figure(figsize=(8, 4))
plt.step(k_values, cdf_values, where='post', color='darkgreen', linewidth=2)
plt.title(f"Bernoulli CDF (p={p_purchase})")
plt.xlabel("Outcome")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.ylim(0, 1.1)
plt.xticks([0, 1])
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_bernoulli_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Bernoulli CDF](ch07_bernoulli_cdf.svg)

The CDF shows cumulative probabilities: P(X ≤ 0) = 0.9 (the probability of getting outcome 0 or less), and P(X ≤ 1) = 1.0 (the probability of getting outcome 1 or less, which includes all possible outcomes).

+++

## 2. Binomial Distribution

The Binomial distribution models the number of successes in a *fixed number* of independent Bernoulli trials, where each trial has the same probability of success.

- **Scenario**: The number of heads in 10 coin flips, the number of defective items in a batch of 50, the number of successful free throws out of 20 attempts, the number of correct answers on a 25-question multiple choice test (with random guessing), the number of customers who make a purchase out of 100 website visitors.
- **Parameters**:
    - $n$: the number of independent trials.
    - $p$: the probability of success on each trial ($0 \le p \le 1$).
- **Random Variable**: $X$, the total number of successes in $n$ trials. $X$ can take values $k = 0, 1, 2, ..., n$.

**PMF:**
The probability of getting exactly $k$ successes in $n$ trials is given by:

$$ P(X=k) = \binom{n}{k} p^k (1-p)^{n-k} \quad \text{for } k = 0, 1, \dots, n $$

where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the binomial coefficient, representing the number of ways to choose $k$ successes from $n$ trials.

**Mean:** $E[X] = np$

**Variance:** $Var(X) = np(1-p)$

**Example:** Modeling the number of successful sales calls out of $n=20$, if the probability of success ($p$) for each call is 0.15.

We'll demonstrate how to use `scipy.stats.binom` to calculate PMF and CDF values, compute statistics, and generate random samples.

:::{dropdown} Python Implementation

```{code-cell} ipython3
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

The PMF shows the probability distribution for the number of successful calls out of 20 attempts. The distribution is centered around the expected value (mean = np = 20 × 0.15 = 3).

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values = binomial_rv.cdf(k_values)

plt.figure(figsize=(8, 4))
plt.step(k_values, cdf_values, where='post', color='darkgreen', linewidth=2)
plt.title(f"Binomial CDF (n={n_calls}, p={p_success_call})")
plt.xlabel("Number of Successes (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_binomial_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Binomial CDF](ch07_binomial_cdf.svg)

The CDF shows the cumulative probability P(X ≤ k) for each value of k. For example, it tells us the probability of getting k or fewer successful calls.

+++

## 3. Geometric Distribution

The Geometric distribution models the number of independent Bernoulli trials needed to get the *first* success.

- **Scenario**: The number of coin flips until the first Head appears, the number of job applications until the first interview offer, the number of attempts needed to pass a certification exam, the number of customers contacted before making the first sale, the number of at-bats until a baseball player gets their first hit.
- **Parameter**: $p$, the probability of success on each trial ($0 < p \le 1$).
- **Random Variable**: $X$, the number of trials required to achieve the first success. $X$ can take values $k = 1, 2, 3, ...$.

**PMF:**
The probability that the first success occurs on the $k$-th trial is:

$$ P(X=k) = (1-p)^{k-1} p \quad \text{for } k = 1, 2, 3, \dots $$

This means we have $k-1$ failures followed by one success.

**Mean:** $E[X] = \frac{1}{p}$

**Variance:** $Var(X) = \frac{1-p}{p^2}$

:::{admonition} Note
:class: note

`scipy.stats.geom` defines $k$ as the number of *failures before* the first success ($k=0, 1, 2, ...$). This shifts the distribution by 1 compared to the definition above where $k$ is the trial number ($k=1, 2, 3, ...$). We'll use the `scipy` definition ($k=0, 1, 2, ...$) in the code examples, but state results in terms of the trial number ($k+1$).
:::

**Example:** Modeling the number of attempts needed to pass a certification exam, where the probability of passing ($p$) on any given attempt is 0.6.

Let's use `scipy.stats.geom` to explore probabilities and compute expected values. Remember that scipy's definition counts failures before the first success, so we'll translate between the two interpretations.

:::{dropdown} Python Implementation

```{code-cell} ipython3
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
plt.title(f"Geometric PMF (p={p_pass}) - Trial number of first success")
plt.xlabel("Trial Number (k)")
plt.ylabel("Probability P(X=k)")
plt.xticks(k_values_trials)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_geometric_pmf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Geometric PMF](ch07_geometric_pmf.svg)

The PMF shows the probability of the first success occurring on each trial number. The probabilities decrease exponentially as the number of trials increases.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF (using trial number k=1, 2, ...)
cdf_values = geom_rv.cdf(k_values_trials - 1) # Adjust k for scipy

plt.figure(figsize=(8, 4))
plt.step(k_values_trials, cdf_values, where='post', color='darkgreen', linewidth=2)
plt.title(f"Geometric CDF (p={p_pass}) - Trial number of first success")
plt.xlabel("Trial Number (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.xticks(k_values_trials)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_geometric_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Geometric CDF](ch07_geometric_cdf.svg)

The CDF shows P(X ≤ k), the probability that the first success occurs on or before trial k. It increases toward 1 as k increases, since eventually success is nearly certain.

+++

## 4. Negative Binomial Distribution

The Negative Binomial distribution models the number of independent Bernoulli trials needed to achieve a *fixed number* of successes ($r$). It generalizes the Geometric distribution (where $r=1$).

- **Scenario**: The number of coin flips needed to get 5 Heads, the number of products to inspect to find 3 defective items, the number of patients tested until finding 10 with a specific condition, the number of job interviews conducted until making 3 hires.
- **Parameters**:
    - $r$: the target number of successes ($r \ge 1$).
    - $p$: the probability of success on each trial ($0 < p \le 1$).
- **Random Variable**: $X$, the total number of trials required to achieve $r$ successes. $X$ can take values $k = r, r+1, r+2, ...$.

**PMF:**
The probability that the $r$-th success occurs on the $k$-th trial is:

$$ P(X=k) = \binom{k-1}{r-1} p^r (1-p)^{k-r} \quad \text{for } k = r, r+1, r+2, \dots $$

This means we have $r-1$ successes in the first $k-1$ trials, and the $k$-th trial is the $r$-th success.

**Mean:** $E[X] = \frac{r}{p}$

**Variance:** $Var(X) = \frac{r(1-p)}{p^2}$

:::{admonition} Note
:class: note

Like `geom`, `scipy.stats.nbinom` defines the variable differently: it counts the number of *failures* ($k$) that occur before the $r$-th success. So, the total number of trials in our definition is $k + r$ in SciPy's terms. We'll use the `scipy` definition ($k=0, 1, 2, ...$ failures) in the code, stating results in terms of the total number of trials.
:::

**Example:** A quality control inspector tests electronic components until finding $r=3$ defective ones. If the probability that any component is defective ($p$) is 0.05, how many components should we expect to test?

We'll use `scipy.stats.nbinom` to calculate the probability of needing a certain number of trials and compute expected values, keeping in mind scipy's definition of counting failures.

:::{dropdown} Python Implementation

```{code-cell} ipython3
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
plt.title(f"Negative Binomial PMF (r={r_defective}, p={p_defective}) - Components tested")
plt.xlabel("Total Number of Components Tested (k)")
plt.ylabel("Probability P(X=k)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_negative_binomial_pmf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Negative Binomial PMF](ch07_negative_binomial_pmf.svg)

The PMF shows the probability distribution for the total number of components tested to find 3 defective items. The distribution shows the most likely values are around 10-15 components.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF (using total components tested k = r, r+1, ...)
cdf_values_nb = nbinom_rv.cdf(k_values_components - r_defective) # Adjust k for scipy

plt.figure(figsize=(8, 4))
plt.step(k_values_components, cdf_values_nb, where='post', color='darkgreen', linewidth=2)
plt.title(f"Negative Binomial CDF (r={r_defective}, p={p_defective}) - Components tested")
plt.xlabel("Total Number of Components Tested (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_negative_binomial_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Negative Binomial CDF](ch07_negative_binomial_cdf.svg)

The CDF shows P(X ≤ k), the cumulative probability that we'll have found 3 defective items after testing k or fewer components.

+++

## 5. Poisson Distribution

The Poisson distribution models the number of events occurring in a fixed interval of time or space, given the average rate of occurrence, assuming events happen independently and at a constant average rate.

- **Scenario**: Number of emails received per hour, number of customer arrivals at a store per day, number of typos per page of a book, number of mutations in a DNA strand of a certain length, number of emergency calls received at a fire station per shift, number of defects per square meter of fabric, number of meteor impacts per year in a region.
- **Parameter**: $\lambda$ (lambda), the average number of events in the interval ($\lambda > 0$).
- **Random Variable**: $X$, the number of events in the interval. $X$ can take values $k = 0, 1, 2, ...$.

**PMF:**

$$ P(X=k) = \frac{e^{-\lambda} \lambda^k}{k!} \quad \text{for } k = 0, 1, 2, \dots $$

where $e \approx 2.71828$ is Euler's number.

**Mean:** $E[X] = \lambda$

**Variance:** $Var(X) = \lambda$

Note: The mean and variance are equal in a Poisson distribution.

**Example:** Modeling the number of emails received per hour, if the average rate ($\lambda$) is 5 emails/hour.

Let's use `scipy.stats.poisson` to calculate the probability of observing different numbers of events and verify that the mean equals the variance.

:::{dropdown} Python Implementation

```{code-cell} ipython3
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
plt.title(f"Poisson PMF (lambda={lambda_rate})")
plt.xlabel("Number of Events (k)")
plt.ylabel("Probability P(X=k)")
plt.xticks(k_values)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_poisson_pmf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Poisson PMF](ch07_poisson_pmf.svg)

The PMF shows the probability distribution for the number of events (customer arrivals) in the time period. With λ = 4.5, the distribution is centered around 4-5 events.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values = poisson_rv.cdf(k_values)

plt.figure(figsize=(8, 4))
plt.step(k_values, cdf_values, where='post', color='darkgreen', linewidth=2)
plt.title(f"Poisson CDF (lambda={lambda_rate})")
plt.xlabel("Number of Events (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.xticks(k_values)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_poisson_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Poisson CDF](ch07_poisson_cdf.svg)

The CDF shows P(X ≤ k), the cumulative probability of observing k or fewer events. This is useful for questions like "What's the probability of 5 or fewer customer arrivals?"

+++

## 6. Hypergeometric Distribution

The Hypergeometric distribution models the number of successes in a sample drawn *without replacement* from a finite population containing a known number of successes. Contrast this with the Binomial, which assumes independence (sampling *with* replacement or from a very large population).

- **Scenario**: Number of winning lottery tickets in a handful drawn from a box, number of defective items in a sample taken from a small batch, number of Aces drawn in a 5-card poker hand from a standard deck, number of tagged fish caught in a sample when studying wildlife populations, number of Democrats on a jury randomly selected from a pool of registered voters.
- **Parameters**:
    - $N$: the total size of the population.
    - $K$: the total number of success items in the population.
    - $n$: the size of the sample drawn from the population ($n \le N$).
- **Random Variable**: $X$, the number of successes in the sample of size $n$. $X$ can take values $k$ such that $\max(0, n - (N-K)) \le k \le \min(n, K)$.

**PMF:**

$$ P(X=k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}} $$

This represents (ways to choose $k$ successes from $K$) * (ways to choose $n-k$ failures from $N-K$) / (total ways to choose $n$ items from $N$).

**Mean:** $E[X] = n \frac{K}{N}$

**Variance:** $Var(X) = n \frac{K}{N} \left(1 - \frac{K}{N}\right) \left(\frac{N-n}{N-1}\right)$

The term $\frac{N-n}{N-1}$ is the *finite population correction factor*. As $N \to \infty$, this factor approaches 1, and the Hypergeometric distribution approaches the Binomial distribution with $p = K/N$.

**Example:** Modeling the number of winning lottery tickets ($k$) in a sample of $n=10$ tickets drawn from a box containing $N=100$ tickets, where $K=20$ are winners.

We'll use `scipy.stats.hypergeom` to calculate probabilities for sampling without replacement and see how the mean relates to the population proportion.

:::{dropdown} Python Implementation

```{code-cell} ipython3
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

The PMF shows the probability distribution for the number of defective items in the sample of 10. Since we're sampling without replacement from a finite population, the probabilities depend on both the sample size and the population composition.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values = hypergeom_rv.cdf(k_values)

plt.figure(figsize=(8, 4))
plt.step(k_values, cdf_values, where='post', color='darkgreen', linewidth=2)
plt.title(f"Hypergeometric CDF (N={N_population}, K={K_successes_pop}, n={n_sample})")
plt.xlabel("Number of Successes in Sample (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.xticks(k_values)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_hypergeometric_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Hypergeometric CDF](ch07_hypergeometric_cdf.svg)

The CDF shows P(X ≤ k), the cumulative probability of finding k or fewer defective items in the sample. This helps answer questions like "What's the probability of finding at most 2 defective items?"

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
