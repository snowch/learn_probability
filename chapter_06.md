---
jupytext:
  formats: ipynb,md:myst
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

# Chapter 6: Discrete Random Variables

Welcome to Part 3 of our journey! We've built a solid foundation in basic probability, counting, conditional probability, and independence. Now, we introduce a central concept that bridges probability theory and data analysis: the **Random Variable**. Random variables allow us to quantitatively describe the outcomes of random phenomena. In this chapter, we'll focus specifically on **Discrete Random Variables**, which take on a finite or countably infinite number of values. We'll learn how to describe their behavior using Probability Mass Functions (PMFs) and Cumulative Distribution Functions (CDFs), and how to summarize them using measures like Expected Value (Mean) and Variance.

+++

## What is a Random Variable?

In many experiments, we're not interested in the specific outcome itself, but rather in some numerical property associated with that outcome.

**Definition:** A **Random Variable** is a variable whose value is a numerical outcome of a random phenomenon. More formally, it's a function that maps each outcome in the sample space $\Omega$ to a real number.

We typically denote random variables with uppercase letters (e.g., $X, Y, Z$) and their specific values with lowercase letters (e.g., $x, y, z$).

**Types of Random Variables:**
1.  **Discrete Random Variable:** A random variable that can only take on a finite or countably infinite number of distinct values. Often associated with counting processes.
2.  **Continuous Random Variable:** A random variable that can take on any value within a given range or interval. Often associated with measurement processes. (We'll cover these in Chapter 8).

**Example:** Consider rolling a fair six-sided die.
* The sample space is $\Omega = \{1, 2, 3, 4, 5, 6\}$.
* We can define a random variable $X$ to be the number shown on the die after the roll. $X$ maps each outcome (which is already a number in this case) to itself.
* $X$ is a **discrete random variable** because it can only take on the specific values $\{1, 2, 3, 4, 5, 6\}$.

**Another Example:** Consider flipping a coin twice.
* The sample space is $\Omega = \{HH, HT, TH, TT\}$.
* Let $Y$ be the random variable representing the *number of heads* obtained.
* $Y$ maps the outcomes to numbers: $Y(HH) = 2$, $Y(HT) = 1$, $Y(TH) = 1$, $Y(TT) = 0$.
* $Y$ is a **discrete random variable** because it can only take on the values $\{0, 1, 2\}$.

+++

## Probability Mass Function (PMF)

For a discrete random variable, we want to know the probability associated with each possible value it can take. This is captured by the Probability Mass Function (PMF).

**Definition:** The **Probability Mass Function (PMF)** of a discrete random variable $X$ is a function, denoted by $p_X(x)$ or simply $p(x)$, that gives the probability that $X$ is exactly equal to some value $x$.

$$
p_X(x) = P(X = x)
$$

A valid PMF must satisfy two conditions:
1.  $p_X(x) \ge 0$ for all possible values $x$. (Probabilities cannot be negative).
2.  $\sum_{x} p_X(x) = 1$, where the sum is taken over all possible values $x$ that $X$ can assume. (The total probability must be 1).

:::{admonition} Example: Fair Die PMF
:class: tip dropdown

For the fair die roll, let $X$ be the outcome. The possible values are $\{1, 2, 3, 4, 5, 6\}$. Since the die is fair, each outcome has a probability of $\frac{1}{6}$. The PMF is:

$$
p_X(x) =
\begin{cases}
1/6 & \text{if } x \in \{1, 2, 3, 4, 5, 6\} \\
0 & \text{otherwise}
\end{cases}
$$

So, $P(X=1) = 1/6$, $P(X=2) = 1/6$, ..., $P(X=6) = 1/6$.

The sum is $6 \times \frac{1}{6} = 1$.
:::

Let's represent and visualize this PMF in Python.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
```

:::{dropdown} Python Implementation
```{code-cell} ipython3
# Define the possible outcomes (values) and their probabilities for the die roll
die_values = np.arange(1, 7) # Possible values x: 1, 2, 3, 4, 5, 6
die_probs = np.array([1/6] * 6) # P(X=x) for each value

# Create a dictionary for easier lookup
die_pmf_dict = {val: prob for val, prob in zip(die_values, die_probs)}
print(f"PMF Dictionary: {die_pmf_dict}")
print(f"Sum of probabilities: {sum(die_pmf_dict.values())}")
```
:::

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Visualize the PMF
plt.figure(figsize=(8, 4))
plt.bar(die_values, die_probs, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel("Outcome (x)")
plt.ylabel("Probability P(X=x)")
plt.title("Probability Mass Function (PMF) of a Fair Die Roll")
plt.xticks(die_values)
plt.ylim(0, 0.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('ch06_pmf_die.svg', format='svg', bbox_inches='tight')
plt.show()
```

```{figure} ch06_pmf_die.svg
---
width: 80%
---
PMF of a fair die roll showing uniform probability of 1/6 for each outcome.
```

+++

## Cumulative Distribution Function (CDF)

Sometimes, we are interested not just in the probability of $X$ being *exactly* a certain value, but in the probability that $X$ is *less than or equal to* a certain value. This is captured by the Cumulative Distribution Function (CDF).

**Definition:** The **Cumulative Distribution Function (CDF)** of a random variable $X$ (discrete or continuous), denoted by $F_X(x)$ or simply $F(x)$, gives the probability that $X$ takes on a value less than or equal to $x$.

$$
F_X(x) = P(X \le x)
$$

For a discrete random variable $X$, the CDF is calculated by summing the PMF values for all outcomes less than or equal to $x$:

$$
F_X(x) = \sum_{k \le x} p_X(k)
$$

**Properties of a CDF:**
1.  $0 \le F_X(x) \le 1$ for all $x$.
2.  $F_X(x)$ is a non-decreasing function of $x$: if $a < b$, then $F_X(a) \le F_X(b)$.
3.  $\lim_{x \to -\infty} F_X(x) = 0$ and $\lim_{x \to +\infty} F_X(x) = 1$.
4.  For a discrete random variable, the CDF is a step function, increasing at the points where the PMF is positive.
5.  $P(X > x) = 1 - F_X(x)$.
6.  $P(a < X \le b) = F_X(b) - F_X(a)$ for $a < b$.
7.  $P(X=x) = F_X(x) - \lim_{y \to x^-} F_X(y)$ (the size of the jump at $x$).

**Example:** For the fair die roll $X$:
* $F_X(0) = P(X \le 0) = 0$
* $F_X(1) = P(X \le 1) = P(X=1) = 1/6$
* $F_X(2) = P(X \le 2) = P(X=1) + P(X=2) = 1/6 + 1/6 = 2/6$
* $F_X(3) = P(X \le 3) = P(X=1) + P(X=2) + P(X=3) = 3/6$
* ...
* $F_X(6) = P(X \le 6) = 6/6 = 1$
* $F_X(6.5) = P(X \le 6.5) = P(X \le 6) = 1$

Let's calculate and visualize the CDF.

:::{dropdown} Python Implementation
```{code-cell} ipython3
# Calculate the CDF values
die_cdf_values = np.cumsum(die_probs)
print(f"CDF Values: {die_cdf_values}")

# Create a function representation of the CDF
def die_cdf_func(x):
    if x < 1:
        return 0.0
    elif x >= 6:
        return 1.0
    else:
        # Find the largest integer <= x that is in our die_values
        idx = np.searchsorted(die_values, x, side='right') - 1
        return die_cdf_values[idx]

# Test the function
print(f"F(0.5) = {die_cdf_func(0.5)}")
print(f"F(3) = {die_cdf_func(3)}")
print(f"F(3.7) = {die_cdf_func(3.7)}")
print(f"F(6) = {die_cdf_func(6)}")
print(f"F(10) = {die_cdf_func(10)}")
```
:::

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Visualize the CDF
x_plot = np.linspace(-1, 8, 500) # Range for plotting
y_plot = [die_cdf_func(val) for val in x_plot]

plt.figure(figsize=(8, 4))
plt.plot(x_plot, y_plot, drawstyle='steps-post', linestyle='-', color='darkgreen')
# Add points at the jumps
plt.scatter(die_values, die_cdf_values, color='darkgreen', zorder=5)
plt.scatter(die_values, die_cdf_values - die_probs, facecolors='none', edgecolors='darkgreen', zorder=5)

plt.xlabel("Value (x)")
plt.ylabel("Cumulative Probability P(X ≤ x)")
plt.title("Cumulative Distribution Function (CDF) of a Fair Die Roll")
plt.xticks(np.arange(0, 8))
plt.yticks(np.linspace(0, 1, 7))
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.ylim(-0.05, 1.05)
plt.savefig('ch06_cdf_die.svg', format='svg', bbox_inches='tight')
plt.show()
```

```{figure} ch06_cdf_die.svg
---
width: 80%
---
CDF of a fair die roll showing the cumulative probability as a step function.
```

+++

## Expected Value (Mean)

The expected value, or mean, of a discrete random variable is a weighted average of its possible values, where the weights are the probabilities (PMF values). It represents the long-run average value we would expect if we observed the random variable many times.

**Definition:** The **Expected Value** (or **Mean**) of a discrete random variable $X$, denoted by $E[X]$ or $\mu_X$, is defined as:

$$
E[X] = \mu_X = \sum_{x} x \cdot p_X(x)
$$

where the sum is over all possible values $x$ that $X$ can take.

The expected value doesn't have to be one of the possible values of $X$.

:::{admonition} Example: Calculating Expected Value
:class: tip dropdown

For the fair die roll $X$:

$$
\begin{align*}
E[X] &= (1 \times \frac{1}{6}) + (2 \times \frac{1}{6}) + (3 \times \frac{1}{6}) + (4 \times \frac{1}{6}) + (5 \times \frac{1}{6}) + (6 \times \frac{1}{6}) \\
&= \frac{1+2+3+4+5+6}{6} \\
&= \frac{21}{6} \\
&= 3.5
\end{align*}
$$

Even though the die can never land on 3.5, the long-run average value of many rolls is expected to be 3.5.
:::

:::{dropdown} Python Implementation
```{code-cell} ipython3
# Calculate the expected value
expected_value = np.sum(die_values * die_probs)
# Alternatively using dot product:
# expected_value = np.dot(die_values, die_probs)

print(f"Theoretical Expected Value E[X]: {expected_value}")
```
:::

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Visualize the mean on the PMF plot
plt.figure(figsize=(8, 4))
plt.bar(die_values, die_probs, color='skyblue', edgecolor='black', alpha=0.7, label='PMF')
plt.axvline(expected_value, color='red', linestyle='--', linewidth=2, label=f'Expected Value E[X] = {expected_value:.1f}')
plt.xlabel("Outcome (x)")
plt.ylabel("Probability P(X=x)")
plt.title("PMF of a Fair Die Roll with Expected Value")
plt.xticks(die_values)
plt.ylim(0, 0.2)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('ch06_pmf_with_mean.svg', format='svg', bbox_inches='tight')
plt.show()
```

```{figure} ch06_pmf_with_mean.svg
---
width: 80%
---
PMF with expected value marked at 3.5, the theoretical long-run average.
```

+++

## Variance and Standard Deviation

While the expected value tells us the center of the distribution, the variance and standard deviation measure the *spread* or *dispersion* of the random variable's values around the mean.

**Definition:** The **Variance** of a random variable $X$, denoted by $Var(X)$ or $\sigma_X^2$, is the expected value of the squared difference between $X$ and its mean $E[X] = \mu_X$.

$$
Var(X) = \sigma_X^2 = E[(X - \mu_X)^2]
$$

For a discrete random variable, this is calculated as:

$$
Var(X) = \sum_{x} (x - \mu_X)^2 \cdot p_X(x)
$$

A computationally simpler formula for variance is often used:

$$
Var(X) = E[X^2] - (E[X])^2
$$

where $E[X^2] = \sum_{x} x^2 \cdot p_X(x)$.

**Definition:** The **Standard Deviation** of a random variable $X$, denoted by $SD(X)$ or $\sigma_X$, is the positive square root of the variance.

$$
SD(X) = \sigma_X = \sqrt{Var(X)}
$$

The standard deviation is often preferred because it has the same units as the random variable $X$.

:::{admonition} Example: Calculating Variance and Standard Deviation
:class: tip dropdown

For the fair die roll $X$, we know $\mu_X = 3.5$.

Let's calculate $E[X^2]$ first:

$$
\begin{align*}
E[X^2] &= (1^2 \times \frac{1}{6}) + (2^2 \times \frac{1}{6}) + (3^2 \times \frac{1}{6}) + (4^2 \times \frac{1}{6}) + (5^2 \times \frac{1}{6}) + (6^2 \times \frac{1}{6}) \\
&= \frac{1 + 4 + 9 + 16 + 25 + 36}{6} \\
&= \frac{91}{6} \approx 15.167
\end{align*}
$$

Now, calculate the variance:

$$
\begin{align*}
Var(X) &= E[X^2] - (E[X])^2 \\
&= \frac{91}{6} - (3.5)^2 \\
&= \frac{91}{6} - (7/2)^2 \\
&= \frac{91}{6} - \frac{49}{4} \\
&= \frac{182}{12} - \frac{147}{12} \\
&= \frac{35}{12} \approx 2.917
\end{align*}
$$

And the standard deviation:

$$
SD(X) = \sigma_X = \sqrt{\frac{35}{12}} \approx \sqrt{2.917} \approx 1.708
$$
:::

:::{dropdown} Python Implementation
```{code-cell} ipython3
# Calculate E[X^2]
e_x_squared = np.sum((die_values**2) * die_probs)
print(f"E[X^2]: {e_x_squared:.4f} (Exact: 91/6)")

# Calculate Variance using the computational formula
variance = e_x_squared - (expected_value**2)
print(f"Theoretical Variance Var(X): {variance:.4f} (Exact: 35/12)")

# Alternatively, calculate using the definition: E[(X - mu)^2]
variance_def = np.sum(((die_values - expected_value)**2) * die_probs)
print(f"Variance using definition: {variance_def:.4f}")

# Calculate Standard Deviation
std_dev = np.sqrt(variance)
print(f"Theoretical Standard Deviation SD(X): {std_dev:.4f} (Exact: sqrt(35/12))")
```
:::

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Visualize the standard deviation on the PMF plot
plt.figure(figsize=(10, 5))
plt.bar(die_values, die_probs, color='skyblue', edgecolor='black', alpha=0.7, label='PMF')
plt.axvline(expected_value, color='red', linestyle='--', linewidth=2, label=f'E[X] = {expected_value:.1f}')

# Add lines for +/- 1 standard deviation
plt.axvline(expected_value + std_dev, color='orange', linestyle=':', linewidth=2, label=f'E[X] ± σ ≈ {expected_value+std_dev:.2f}')
plt.axvline(expected_value - std_dev, color='orange', linestyle=':', linewidth=2, label=f'E[X] - σ ≈ {expected_value-std_dev:.2f}')

# Add lines for +/- 2 standard deviations
plt.axvline(expected_value + 2*std_dev, color='purple', linestyle=':', linewidth=2, label=f'E[X] ± 2σ ≈ {expected_value+2*std_dev:.2f}')
plt.axvline(expected_value - 2*std_dev, color='purple', linestyle=':', linewidth=2, label=f'E[X] - 2σ ≈ {expected_value-2*std_dev:.2f}')

plt.xlabel("Outcome (x)")
plt.ylabel("Probability P(X=x)")
plt.title("PMF, Mean, and Standard Deviation Bands for a Fair Die Roll")
plt.xticks(die_values)
plt.ylim(0, 0.2)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ch06_pmf_with_std.svg', format='svg', bbox_inches='tight')
plt.show()
```

```{figure} ch06_pmf_with_std.svg
---
width: 100%
---
PMF showing mean and standard deviation bands, illustrating the spread of the distribution.
```

+++

## Functions of a Random Variable

Often, we are interested in a quantity that is derived from a random variable. If $X$ is a random variable and $g$ is a function, then $Y = g(X)$ is also a random variable.

If $X$ is discrete with PMF $p_X(x)$, we can find the PMF of $Y = g(X)$, denoted $p_Y(y)$, by summing the probabilities of all $x$ values such that $g(x) = y$:

$$
\begin{align*}
p_Y(y) &= P(Y=y) \\
&= P(g(X)=y) \\
&= \sum_{x: g(x)=y} p_X(x)
\end{align*}
$$

```{admonition} Reading the notation
:class: note

The notation $\sum_{x: g(x)=y}$ is read as "sum over all values of $x$ such that $g(x) = y$". The colon (:) means "such that" or "where". This is a concise way to write a conditional sum - we only include terms where the condition $g(x)=y$ is true.

**Example:** Consider $X$ as a fair die roll and $Y = X^2$. To find $p_Y(4) = P(Y=4)$:
$$p_Y(4) = \sum_{x: x^2=4} p_X(x)$$

Only $x=2$ satisfies $x^2=4$, so:
$$p_Y(4) = p_X(2) = \frac{1}{6}$$

The other values ($x \in \{1, 3, 4, 5, 6\}$) do NOT satisfy the condition $x^2=4$, so they are not included in the sum.
```

### Expected Value of a Function of a Random Variable (LOTUS)

A very useful result, sometimes called the Law of the Unconscious Statistician (LOTUS), allows us to calculate the expected value of $Y=g(X)$ without explicitly finding the PMF of $Y$.

**Definition:** For a discrete random variable $X$ with PMF $p_X(x)$ and a function $g$, the expected value of $Y = g(X)$ is:

$$
E[Y] = E[g(X)] = \sum_{x} g(x) \cdot p_X(x)
$$

Notice this is similar to the definition of $E[X]$, but we replace $x$ with $g(x)$. This is how we calculated $E[X^2]$ earlier, where $g(x) = x^2$.

:::{admonition} Example: PMF and Expected Value of Y = X²
:class: tip dropdown

Let $X$ be the outcome of a fair die roll. Let $Y = X^2$. What are the PMF and expected value of $Y$?

* The possible values of $X$ are $\{1, 2, 3, 4, 5, 6\}$, each with probability $1/6$.
* The possible values of $Y = X^2$ are $\{1^2, 2^2, 3^2, 4^2, 5^2, 6^2\} = \{1, 4, 9, 16, 25, 36\}$.
* Since each $x$ value maps to a unique $y=x^2$ value, the probability of each $y$ is the same as the probability of the corresponding $x$.
* The PMF of $Y$ is:
    $p_Y(y) = 1/6$ for $y \in \{1, 4, 9, 16, 25, 36\}$, and $0$ otherwise.

**Calculating E[Y] using the PMF of Y:**

$$
\begin{align*}
E[Y] &= (1 \times \frac{1}{6}) + (4 \times \frac{1}{6}) + (9 \times \frac{1}{6}) + (16 \times \frac{1}{6}) + (25 \times \frac{1}{6}) + (36 \times \frac{1}{6}) \\
&= \frac{1+4+9+16+25+36}{6} \\
&= \frac{91}{6}
\end{align*}
$$

**Alternatively, using LOTUS:**

$$
\begin{align*}
E[Y] = E[X^2] &= \sum_{x=1}^{6} x^2 \cdot p_X(x) \\
&= \sum_{x=1}^{6} x^2 \cdot \frac{1}{6} \\
&= \frac{1^2+2^2+3^2+4^2+5^2+6^2}{6} \\
&= \frac{91}{6}
\end{align*}
$$

This confirms our earlier calculation of $E[X^2]$.
:::

:::{dropdown} Python Implementation
```{code-cell} ipython3
# Define the function g(x) = x^2
def g(x):
  return x**2

# Possible values for X and Y
x_values = die_values
y_values = g(x_values)

# PMF for Y (since g(x) is one-to-one for x in {1..6}, probs are the same)
y_probs = die_probs

# PMF dictionary for Y
y_pmf_dict = {val_y: prob for val_y, prob in zip(y_values, y_probs)}
print(f"PMF Dictionary for Y=X^2: {y_pmf_dict}")

# Calculate E[Y] using the PMF of Y
expected_value_y = np.sum(y_values * y_probs)
print(f"E[Y] calculated using PMF of Y: {expected_value_y:.4f} (Exact: 91/6)")

# Calculate E[Y] = E[g(X)] using LOTUS
expected_value_y_lotus = np.sum(g(x_values) * die_probs)
print(f"E[Y] calculated using LOTUS E[g(X)]: {expected_value_y_lotus:.4f} (Exact: 91/6)")
```
:::

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Visualize the PMF of Y = X^2
plt.figure(figsize=(8, 4))
plt.bar(y_values, y_probs, color='lightcoral', edgecolor='black', alpha=0.7)
plt.xlabel("Outcome (y = x^2)")
plt.ylabel("Probability P(Y=y)")
plt.title("Probability Mass Function (PMF) of Y = X^2 (Squared Die Roll)")
plt.xticks(y_values)
plt.ylim(0, 0.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('ch06_pmf_y_squared.svg', format='svg', bbox_inches='tight')
plt.show()
```

```{figure} ch06_pmf_y_squared.svg
---
width: 80%
---
PMF of $Y = X^2$ showing the transformed distribution.
```

+++

## Hands-on: Simulation and Comparison

The Law of Large Numbers (which we'll study later) tells us that if we simulate a random variable many times, the average of the outcomes (the *sample mean*) should get close to the theoretical expected value $E[X]$. Similarly, the variance of the outcomes (the *sample variance*) should approach $Var(X)$. Let's verify this for our die roll example.

We will:
1.  Simulate a large number of fair die rolls using `numpy.random.randint`.
2.  Calculate the sample mean and sample variance of the simulated outcomes.
3.  Compare these empirical results to the theoretical values ($E[X]=3.5$, $Var(X) \approx 2.917$).
4.  Visualize the distribution of the simulated outcomes (empirical PMF) and compare it to the theoretical PMF.
5.  Visualize the empirical CDF and compare it to the theoretical CDF.

```{code-cell} ipython3
# Number of simulations
num_simulations = 10000

# Simulate die rolls
simulated_rolls = np.random.randint(1, 7, size=num_simulations)

# Calculate empirical mean and variance
sample_mean = np.mean(simulated_rolls)
sample_variance = np.var(simulated_rolls, ddof=1)  # ddof=1 for unbiased estimator
sample_std_dev = np.std(simulated_rolls, ddof=1)

# Compare empirical vs theoretical
print(f"--- Comparison after {num_simulations} simulations ---")
print(f"Theoretical E[X]: {expected_value:.4f}")
print(f"Sample Mean:      {sample_mean:.4f}")
print(f"Difference (Mean): {abs(sample_mean - expected_value):.4f}\n")

print(f"Theoretical Var(X): {variance:.4f}")
print(f"Sample Variance:    {sample_variance:.4f}")
print(f"Difference (Var):   {abs(sample_variance - variance):.4f}\n")

print(f"Theoretical SD(X): {std_dev:.4f}")
print(f"Sample Std Dev:     {sample_std_dev:.4f}")
print(f"Difference (SD):    {abs(sample_std_dev - std_dev):.4f}")
```

+++

### Visualizing Empirical vs Theoretical Distributions

Now let's plot the frequencies of our simulated results and compare them to the theoretical probabilities (PMF), and do the same for the cumulative distributions (CDF).

```{code-cell} ipython3
# Calculate empirical PMF and CDF
unique_outcomes, counts = np.unique(simulated_rolls, return_counts=True)
empirical_pmf = counts / num_simulations
empirical_cdf = np.cumsum(empirical_pmf)
```

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Create comparison plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Empirical PMF vs Theoretical PMF
ax1.bar(unique_outcomes, empirical_pmf, color='lightgreen', alpha=0.7, label=f'Empirical PMF (N={num_simulations})')
ax1.plot(die_values, die_probs, 'ro--', markersize=8, label='Theoretical PMF (1/6)')
ax1.set_xlabel("Outcome")
ax1.set_ylabel("Probability / Relative Frequency")
ax1.set_title("Empirical vs Theoretical PMF")
ax1.set_xticks(die_values)
ax1.set_ylim(0, max(np.max(empirical_pmf), np.max(die_probs)) * 1.1)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Empirical CDF vs Theoretical CDF
ax2.step(unique_outcomes, empirical_cdf, where='post', color='orange', linewidth=2, label=f'Empirical CDF (N={num_simulations})')
x_plot_cdf = np.linspace(-1, 8, 500)
y_plot_cdf = [die_cdf_func(val) for val in x_plot_cdf]
ax2.plot(x_plot_cdf, y_plot_cdf, 'b--', linewidth=2, label='Theoretical CDF')
ax2.scatter(die_values, die_cdf_values, color='blue', zorder=5, s=50)
ax2.set_xlabel("Outcome")
ax2.set_ylabel("Cumulative Probability")
ax2.set_title("Empirical vs Theoretical CDF")
ax2.set_xticks(np.arange(0, 8))
ax2.set_yticks(np.linspace(0, 1, 7))
ax2.set_ylim(-0.05, 1.05)
ax2.legend()
ax2.grid(True, which='both', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('ch06_empirical_vs_theoretical.svg', format='svg', bbox_inches='tight')
plt.show()
```

```{figure} ch06_empirical_vs_theoretical.svg
---
width: 100%
figclass: full-width
---
Comparison of empirical (simulated) and theoretical distributions, demonstrating convergence.
```

As you can see from the simulation results and the plots, the empirical values (sample mean, sample variance, empirical PMF/CDF) obtained from a large number of simulations closely approximate the theoretical values we derived. This demonstrates the connection between probability theory and real-world observations or simulations.

+++

## Summary

In this chapter, we introduced the fundamental concept of a discrete random variable.

* A **Random Variable** assigns a numerical value to each outcome of a random experiment.
* A **Discrete Random Variable** takes on a finite or countably infinite number of values.
* The **Probability Mass Function (PMF)**, $p_X(x) = P(X=x)$, gives the probability for each possible value $x$.
* The **Cumulative Distribution Function (CDF)**, $F_X(x) = P(X \le x)$, gives the cumulative probability up to a value $x$.
* The **Expected Value (Mean)**, $E[X] = \sum x \cdot p_X(x)$, represents the long-run average value.
* The **Variance**, $Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$, measures the spread around the mean.
* The **Standard Deviation**, $SD(X) = \sqrt{Var(X)}$, also measures spread but in the original units.
* We can analyze **Functions of Random Variables**, $Y = g(X)$, and find their PMFs by summing probabilities: $p_Y(y) = \sum_{x: g(x)=y} p_X(x)$.
* **LOTUS (Law of the Unconscious Statistician)** allows us to calculate expected values of functions directly: $E[g(X)] = \sum g(x) p_X(x)$.
* Simulations using Python (`numpy`) allow us to generate empirical data that converges to theoretical probability distributions and their parameters as the number of simulations increases.

In the next chapter, we will explore several important families of discrete distributions that model common real-world scenarios.

+++

---

## Exercises

1.  **Biased Coin:** Consider a biased coin where the probability of getting Heads (H) is $P(H) = 0.7$. Let $X$ be the random variable representing the number of heads in a single flip (so $X=1$ for Heads, $X=0$ for Tails).
    a.  What is the PMF of $X$?
    b.  What is the CDF of $X$? Plot it.
    c.  Calculate the expected value $E[X]$.
    d.  Calculate the variance $Var(X)$ and standard deviation $SD(X)$.

    ```{admonition} Answer
    :class: dropdown

    a) PMF: $P(X=0) = 0.3$, $P(X=1) = 0.7$

    b) CDF: $F(x) = 0$ for $x < 0$; $F(x) = 0.3$ for $0 \le x < 1$; $F(x) = 1$ for $x \ge 1$

    c) $E[X] = 0 \times 0.3 + 1 \times 0.7 = 0.7$

    d) $E[X^2] = 0^2 \times 0.3 + 1^2 \times 0.7 = 0.7$

    $Var(X) = E[X^2] - (E[X])^2 = 0.7 - 0.49 = 0.21$

    $SD(X) = \sqrt{0.21} \approx 0.458$
    ```

2.  **Two Dice Sum:** Let $X$ be the random variable representing the sum of the outcomes when two fair six-sided dice are rolled.
    a.  What are the possible values for $X$?
    b.  Determine the PMF of $X$. (Hint: There are 36 equally likely outcomes for the pair of dice.)
    c.  Calculate $E[X]$. Is there an easier way than using the PMF directly? (Hint: Linearity of Expectation)
    d.  Calculate $Var(X)$. (Hint: Variance of sums of independent variables)
    e.  Find $P(X > 7)$.
    f.  Find $P(X \text{ is even})$.

    ```{admonition} Answer
    :class: dropdown

    a) Possible values: $\{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\}$

    b) PMF: Count outcomes for each sum. For example, $P(X=7) = 6/36 = 1/6$ (six ways: 1+6, 2+5, 3+4, 4+3, 5+2, 6+1)

    c) Using linearity: $E[X] = E[D_1] + E[D_2] = 3.5 + 3.5 = 7$

    d) For independent dice: $Var(X) = Var(D_1) + Var(D_2) = 35/12 + 35/12 = 70/12 \approx 5.833$

    e) $P(X > 7) = P(X \in \{8,9,10,11,12\}) = (5+4+3+2+1)/36 = 15/36 = 5/12$

    f) Count even sums: $P(X \text{ even}) = 18/36 = 1/2$
    ```

3.  **Game Value:** You pay £2 to play a game. You roll a fair six-sided die. If you roll a 6, you win £5 (getting your £2 back plus £3 profit). If you roll a 4 or 5, you win £2 (getting your £2 back). If you roll a 1, 2, or 3, you win nothing (losing your £2). Let $W$ be the random variable representing your *net* winnings (profit/loss) from playing the game once.
    a.  What are the possible values for $W$?
    b.  Determine the PMF of $W$.
    c.  Calculate the expected net winnings $E[W]$. Is this a fair game? (A fair game has $E[W]=0$).
    d.  Calculate the variance $Var(W)$.

    ```{admonition} Answer
    :class: dropdown

    a) Possible values: $\{-2, 0, 3\}$ (net: lose £2, break even, or profit £3)

    b) PMF:
    - $P(W=-2) = 3/6 = 1/2$ (roll 1, 2, or 3)
    - $P(W=0) = 2/6 = 1/3$ (roll 4 or 5)
    - $P(W=3) = 1/6$ (roll 6)

    c) $E[W] = (-2)(1/2) + (0)(1/3) + (3)(1/6) = -1 + 0 + 0.5 = -0.5$

    Not fair; expected loss of £0.50 per game

    d) $E[W^2] = 4(1/2) + 0(1/3) + 9(1/6) = 2 + 0 + 1.5 = 3.5$

    $Var(W) = 3.5 - (-0.5)^2 = 3.5 - 0.25 = 3.25$
    ```

4.  **Simulation Comparison:** Simulate rolling two fair dice and calculating their sum $10,000$ times.
    a.  Calculate the sample mean and sample variance of the simulated sums.
    b.  Compare these to the theoretical values you calculated in Exercise 2.
    c.  Plot the empirical PMF of the simulated sums and compare it to the theoretical PMF from Exercise 2.

    ```{admonition} Hint
    :class: tip

    Use `np.random.randint(1, 7, size=(10000, 2))` to simulate two dice 10000 times, then sum along axis 1.
    ```

---

+++

*(Solutions/Hints Appendix)*

:::{dropdown} Example Code for Exercise 1
```{code-cell} ipython3
# Exercise 1: Biased Coin
p_heads = 0.7
p_tails = 1 - p_heads

# a) PMF
x_values_coin = np.array([0, 1]) # 0 for Tails, 1 for Heads
pmf_coin = np.array([p_tails, p_heads])
pmf_coin_dict = {val: prob for val, prob in zip(x_values_coin, pmf_coin)}
print(f"Ex 1(a) - PMF: {pmf_coin_dict}")

# b) CDF
cdf_coin_values = np.cumsum(pmf_coin)
def coin_cdf_func(x):
    if x < 0: return 0.0
    if x >= 1: return 1.0
    return cdf_coin_values[0] # P(X<=0)

# c) Expected Value
ex_coin = np.sum(x_values_coin * pmf_coin)
print(f"Ex 1(c) - E[X]: {ex_coin}")

# d) Variance
ex2_coin = np.sum((x_values_coin**2) * pmf_coin)
var_coin = ex2_coin - ex_coin**2
sd_coin = np.sqrt(var_coin)
print(f"Ex 1(d) - Var(X): {var_coin:.4f}")
print(f"Ex 1(d) - SD(X): {sd_coin:.4f}")
```
:::
