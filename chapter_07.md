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

These "common" distributions serve as powerful models for a wide variety of real-world processes. Understanding their properties and when to apply them is crucial for probabilistic modeling. In this chapter, we will explore nine fundamental discrete distributions: Bernoulli, Binomial, Geometric, Negative Binomial, Poisson, Hypergeometric, Discrete Uniform, Categorical, and Multinomial.

We'll examine the scenarios each distribution models, their key characteristics (PMF, mean, variance), and how to work with them efficiently using Python's `scipy.stats` library. This library provides tools to calculate probabilities (PMF, CDF), generate random samples, and more, significantly simplifying our practical work.

:::{admonition} Focus on Understanding, Not Memorization
:class: tip

As you work through this chapter, remember: **understanding the underlying probabilistic structure is more important than memorizing formulas**.

For each distribution, focus on:
- **When to use it**: What real-world scenario does it model?
- **Why it works**: What's the intuition behind the formula?
- **How distributions relate**: How does it connect to other distributions you know?

Don't worry about memorizing every PMF formula—you can always look them up or use `scipy.stats`. Instead, build intuition about when and why to use each distribution. This deeper understanding will serve you far better than rote memorization!
:::

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

:::{admonition} Why Start with Something So Simple?
:class: note

The Bernoulli distribution might seem almost trivially simple—it's just a single trial with two outcomes! However, it's incredibly important because:

- **It's the fundamental building block**: Several key distributions in this chapter (Binomial, Geometric, Negative Binomial) build directly on repeated Bernoulli trials, while Categorical generalizes the Bernoulli distribution to more than two outcomes
- **Single binary events are everywhere**: Many real-world scenarios involve a single yes/no, success/failure, or on/off decision
- **It establishes key patterns**: The $p$ and $(1-p)$ structure appears throughout probability theory
- **Mathematical note**: The Bernoulli distribution is technically just the Binomial distribution with $n=1$, but we give it its own name because of its foundational importance

Think of it like learning to add before learning to multiply—simple, but essential!
:::

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

Expanding this for both cases to make it crystal clear:

**When k = 1 (success):**

$$
\begin{align}
P(X=1) &= p^1 (1-p)^{1-1} \\
&= p^1 (1-p)^0 \\
&= p \times 1 \\
&= p
\end{align}
$$

**When k = 0 (failure):**

$$
\begin{align}
P(X=0) &= p^0 (1-p)^{1-0} \\
&= p^0 (1-p)^1 \\
&= 1 \times (1-p) \\
&= 1-p
\end{align}
$$

Let's verify this works for our example where $p = 0.3$:
- When $k = 1$: $P(X=1) = (0.3)^1 (0.7)^0 = 0.3 \times 1 = 0.3$ ✓
- When $k = 0$: $P(X=0) = (0.3)^0 (0.7)^1 = 1 \times 0.7 = 0.7$ ✓

**Key Characteristics**

- **Scenarios**: Coin flip (Heads/Tails), product inspection (Defective/Not Defective), medical test (Positive/Negative), free throw (Make/Miss)
- **Parameter**: $p$, the probability of success ($0 \le p \le 1$)
- **Random Variable**: $X \in \{0, 1\}$

**Mean:** $E[X] = p$

**Variance:** $Var(X) = p(1-p)$

**Standard Deviation:** $SD(X) = \sqrt{p(1-p)}$

**Visualizing the Distribution**

Let's visualize a Bernoulli distribution with $p = 0.3$ (our medical test example from above):

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Create Bernoulli distribution for visualization (p=0.3)
p_viz = 0.3
bernoulli_viz = stats.bernoulli(p=p_viz)

# Calculate mean and std
mean_viz = bernoulli_viz.mean()
std_viz = bernoulli_viz.std()

# Plotting the PMF
k_values_viz = [0, 1]
pmf_values_viz = bernoulli_viz.pmf(k_values_viz)

plt.figure(figsize=(10, 5))
plt.bar(k_values_viz, pmf_values_viz, tick_label=["Failure (0)", "Success (1)"], color='skyblue', edgecolor='black', alpha=0.7)

# Add mean line
plt.axvline(mean_viz, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_viz:.2f}')

# Add mean ± 1 std region
plt.axvspan(mean_viz - std_viz, mean_viz + std_viz, alpha=0.2, color='orange',
            label=f'Mean ± 1 SD = [{mean_viz - std_viz:.2f}, {mean_viz + std_viz:.2f}]')

plt.title(f"Bernoulli PMF (p={p_viz})")
plt.xlabel("Outcome")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.legend(loc='upper right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_bernoulli_pmf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Bernoulli PMF](ch07_bernoulli_pmf_generic.svg)

The PMF shows two bars: P(X=0) = 0.7 for a negative test and P(X=1) = 0.3 for a positive test. The red dashed line marks the mean ($p = 0.3$), and the orange shaded region shows mean ± 1 standard deviation.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
k_values_viz = [0, 1]
cdf_values_viz = bernoulli_viz.cdf(k_values_viz)

plt.figure(figsize=(10, 5))
# Add points to show the full step function including the start at 0
plt.step([-0.5] + k_values_viz, [0] + list(cdf_values_viz), where='post', color='darkgreen', linewidth=2)

# Add mean line
plt.axvline(mean_viz, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_viz:.2f}')

plt.title(f"Bernoulli CDF (p={p_viz})")
plt.xlabel("Outcome")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.ylim(0, 1.1)
plt.xlim(-0.5, 1.5)
plt.xticks([0, 1])
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_bernoulli_cdf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Bernoulli CDF](ch07_bernoulli_cdf_generic.svg)

The CDF shows the step function: starts at 0 for x < 0, jumps to 0.7 at x=0 (the value when outcome is 0), stays flat at 0.7 until x=1, then jumps to 1.0 at x=1 (the value when including both outcomes 0 and 1). The red dashed line marks the mean.

Note: Here, P(X ≤ 0) = P(X = 0) = 0.7 because X can't take negative values; in general, "X ≤ 0" means "at or below 0", not "exactly 0".

**Reading the PMF**

- **What it shows:** The height of each bar represents the probability of that exact outcome
- **How to read:** Look at the bar height to find P(X = k) for any specific value k
- **Practical use:** Answer questions like "What's the probability of success?" or "What's the probability of exactly 1 positive test?"
- **Key property:** All bar heights must sum to 1.0 (total probability)
- **Visualization aids:** The red dashed line marks the mean (expected value), and the orange shaded region shows mean ± 1 standard deviation (where ~68% of values typically fall)

**Reading the CDF**

- **What it shows:** The cumulative probability P(X ≤ k) up to and including each value k
- **How to read:** The height at position k tells you the probability of getting k or fewer successes
- **Why step functions?** For discrete distributions, probability accumulates in jumps at each possible value. Between possible values, the CDF stays constant (no additional probability)
- **Key identity:** The jump at k equals P(X = k) — the size of each step up is the PMF value
- **Practical uses:**
  - Find P(X ≤ k) directly by reading the height at k
  - Find P(X > k) by calculating 1 - P(X ≤ k)
  - Find P(a < X ≤ b) by calculating P(X ≤ b) - P(X ≤ a)
- **Key property:** The CDF is right-continuous, always increases (or stays flat), and approaches 1.0
- **Visualization aids:** The red dashed line marks the mean (expected value) as a reference point

**Note on CDF visualization:** The charts use `where='post'` in the step plot to create proper right-continuous step functions. This means the CDF jumps up at each value and includes that value in the cumulative probability.

::::{admonition} Example: Medical Diagnostic Test with p = 0.1
:class: tip

Modeling the outcome of a single medical diagnostic test where the probability of a positive result is 0.1.

Let's use [`scipy.stats.bernoulli`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html) to calculate probabilities, compute the mean and variance, and generate random samples.

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

:::{admonition} Working with Frozen Random Variables in scipy.stats
:class: note

When we write `bernoulli_rv = stats.bernoulli(p=p_positive)`, we're creating a **frozen random variable** — a distribution object with parameters locked in.

**Think of it like partial immutability:** Similar to how Python's immutable objects (strings, tuples) can't be changed after creation, a frozen RV's distribution parameters are fixed and can't be modified. The difference is that frozen RVs only "freeze" the parameters (like p=0.1), not the entire object.

**Two ways to use scipy.stats:**

1. **Non-frozen** (pass parameters every time):
   ```python
   stats.bernoulli.pmf(1, p=0.1)
   stats.bernoulli.cdf(0, p=0.1)
   stats.bernoulli.mean(p=0.1)
   ```

2. **Frozen** (set parameters once, reuse):
   ```python
   rv = stats.bernoulli(p=0.1)  # Create frozen RV
   rv.pmf(1)                     # Use it multiple times
   rv.cdf(0)
   rv.mean()
   ```

**Benefits of frozen RVs:**
- Cleaner, more readable code
- More efficient (parameters validated once)
- Easier to pass distributions to functions
- Matches the pattern in scipy documentation

Throughout this chapter, we use frozen RVs for all examples. This is the recommended approach when working with the same distribution parameters multiple times.
:::

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
# Add points to show the full step function including the start at 0
plt.step([-0.5] + k_values, [0] + list(cdf_values), where='post', color='darkgreen', linewidth=2)
plt.title(f"Bernoulli CDF (p={p_positive})")
plt.xlabel("Outcome")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.ylim(0, 1.1)
plt.xlim(-0.5, 1.5)
plt.xticks([0, 1])
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_bernoulli_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Bernoulli CDF](ch07_bernoulli_cdf.svg)

The CDF shows the step function: starts at 0 for x < 0, jumps to 0.9 at x=0, stays flat at 0.9 until x=1, then jumps to 1.0 at x=1.

::::

**Quick Check Questions**

1. A quality control inspector checks a single product. It's either defective or not defective. Is this scenario well-modeled by a Bernoulli distribution? Why or why not?

```{admonition} Answer
:class: dropdown

**Yes** - This scenario perfectly fits the Bernoulli distribution requirements:
- **Single trial**: Checking one product
- **Two possible outcomes**: Defective (success/1) or not defective (failure/0)
- **Fixed probability**: The defect rate is constant for each product

If the defect rate is 5%, we'd use Bernoulli(p=0.05).
```

2. For a Bernoulli distribution with p = 0.3, what is P(X = 0)?

```{admonition} Answer
:class: dropdown

**P(X = 0) = 1 - p = 0.7** - The probability of failure is 1 - p.
```

3. A basketball player has a 75% free throw success rate. If we model a single free throw as a Bernoulli trial, what are the mean and variance?

```{admonition} Answer
:class: dropdown

**Mean = 0.75, Variance = 0.75 × 0.25 = 0.1875**

Using the formulas E[X] = p and Var(X) = p(1-p):
- E[X] = 0.75
- Var(X) = 0.75 × (1 - 0.75) = 0.75 × 0.25 = 0.1875
```

4. You roll a six-sided die once. Is this well-modeled by a Bernoulli distribution?

```{admonition} Answer
:class: dropdown

**No** - A Bernoulli distribution requires exactly **two possible outcomes**. A die roll has 6 outcomes (1, 2, 3, 4, 5, 6), so Bernoulli doesn't apply directly.

**However**, you *could* use Bernoulli if you redefined the experiment with a binary outcome:
- "Does the die show a 6?" (Yes/No) → Bernoulli with p = 1/6
- "Is the result even?" (Yes/No) → Bernoulli with p = 1/2

The key: Bernoulli requires exactly two outcomes.
```

5. True or False: A Bernoulli random variable can only take on the values 0 and 1.

```{admonition} Answer
:class: dropdown

**True** - By definition, a Bernoulli random variable X ∈ {0, 1}, where:
- X = 1 represents "success" with probability p
- X = 0 represents "failure" with probability 1-p

These are the only two possible outcomes.
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

:::{admonition} Understanding the Binomial Formula
:class: note

The Binomial PMF formula combines probability and counting:

$$P(X=k) = \binom{n}{k} \cdot p^k \cdot (1-p)^{n-k}$$

**Breaking down the formula:**
- **$p^k$**: Probability of $k$ successes — each success is an independent Bernoulli trial with probability $p$
- **$(1-p)^{n-k}$**: Probability of $(n-k)$ failures — each failure is an independent Bernoulli trial with probability $1-p$
- **$\binom{n}{k}$**: Number of ways to arrange $k$ successes among $n$ trial positions

**The probabilistic view (Bernoulli trials):**

Any specific sequence of $k$ successes and $(n-k)$ failures has probability $p^k(1-p)^{n-k}$ (by independence of trials). For example, the sequence "success, success, failure" has probability $p \cdot p \cdot (1-p) = p^2(1-p)$.

This shows why Binomial "counts successes in repeated Bernoulli trials": it's built from the ground up using the Bernoulli probability $p$ for each trial.

**The combinatorial view (counting techniques):**

The binomial coefficient $\binom{n}{k}$ is a **combination** that counts the number of ways to choose $k$ items from $n$ items when order doesn't matter (see [Chapter 3: Combinations](chapter_03.md#combinations-when-order-doesnt-matter)).

In our context, it counts **how many different sequences** of $n$ trials yield exactly $k$ successes. For example, with $n=3$ trials and $k=2$ successes: $\binom{3}{2} = 3$ represents the three sequences SSF, SFS, and FSS (where S=success, F=failure).

**Why we multiply:** Each of the $\binom{n}{k}$ sequences has the same probability $p^k(1-p)^{n-k}$. To get the total probability of exactly $k$ successes (in any order), we multiply the number of sequences by the probability of each sequence.

**Visual example:** Here's how it works for $n=3$ trials, $k=2$ successes, with $p=0.6$:

```{code-cell} ipython3
:tags: [remove-input, remove-output]

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# --- Parameters ---
n, k, p = 3, 2, 0.6
q = 1 - p
prob_each = (p**k) * (q**(n-k))
total = 3 * prob_each

fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# ---------------- helpers ----------------
def rounded_box(ax, xy, w, h, fc, ec, lw=2, pad=0.012, r=0.02, z=3):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size={r}",
        transform=ax.transAxes,
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z
    )
    ax.add_patch(patch)
    return patch

def draw_sequence(ax, cx, cy, label):
    w, h = 0.14, 0.075
    rounded_box(ax, (cx - w/2, cy - h/2), w, h,
                fc="lightblue", ec="steelblue", lw=2.2, r=0.02)

    ax.text(cx, cy, label, transform=ax.transAxes,
            ha="center", va="center",
            fontsize=24, weight="bold", family="monospace", zorder=4)

    ax.text(cx, cy - 0.075, rf"${p:.1f}\times{p:.1f}\times{q:.1f}$",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=18, zorder=4)

    ax.text(cx, cy - 0.115, rf"$= {prob_each:.3f}$",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=18, weight="bold", zorder=4)

    # return (left, bottom, right, top) for arrow anchoring
    return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)

# ---------------- layout ----------------
ax.text(0.5, 0.955, rf"Binomial Formula Breakdown: $n={n},\,k={k},\,p={p}$",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=24, weight="bold", zorder=4)

# Top row: sequences
seq_y = 0.78
seq_boxes = {}
for lbl, x in [("SSF", 0.24), ("SFS", 0.50), ("FSS", 0.76)]:
    seq_boxes[lbl] = draw_sequence(ax, x, seq_y, lbl)

# Count box
count_w, count_h = 0.30, 0.08
count_xy = (0.5 - count_w/2, 0.56 - count_h/2)
rounded_box(ax, count_xy, count_w, count_h,
            fc="lightyellow", ec="orange", lw=2.2, r=0.02)

ax.text(0.5, 0.56, r"$\binom{3}{2}=3$ sequences",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=20, weight="bold", zorder=4)

# Formula block
ax.text(0.5, 0.44, "Formula:", transform=ax.transAxes,
        ha="center", va="center", fontsize=20, weight="bold", zorder=4)

ax.text(0.5, 0.385, r"$P(X=2)=\binom{3}{2}\cdot p^2\cdot (1-p)^1$",
        transform=ax.transAxes, ha="center", va="center", fontsize=18, zorder=4)

ax.text(0.5, 0.33, r"$=3\times 0.36\times 0.4$",
        transform=ax.transAxes, ha="center", va="center", fontsize=18, zorder=4)

# Result box (moved DOWN a bit)
res_w, res_h = 0.18, 0.075
res_center_y = 0.235  # was 0.26
res_xy = (0.5 - res_w/2, res_center_y - res_h/2)
rounded_box(ax, res_xy, res_w, res_h,
            fc="lightgreen", ec="green", lw=2.2, r=0.02)

ax.text(0.5, res_center_y, rf"$= {total:.3f}$",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=20, weight="bold", zorder=4)

# ---- Callouts: arrows hit box edges and avoid overlaps ----
# Orange (label moved down a bit)
count_tip = (count_xy[0], count_xy[1] + count_h*0.55)  # left edge of yellow box
ax.annotate("Count sequences",
            xy=count_tip, xycoords=ax.transAxes,
            xytext=(0.08, 0.595), textcoords=ax.transAxes,  # moved down
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.10",
                            lw=2.5, color="orange",
                            shrinkA=6, shrinkB=8),
            fontsize=16, color="orange", weight="bold",
            ha="left", va="center", zorder=5)

# Blue -> right edge of FSS box
x0, y0, x1, y1 = seq_boxes["FSS"]
fss_tip = (x1, (y0 + y1) / 2)
ax.annotate("Each sequence\nhas same\nprobability",
            xy=fss_tip, xycoords=ax.transAxes,
            xytext=(0.88, 0.84), textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=0.18",
                            lw=2.5, color="steelblue",
                            shrinkA=6, shrinkB=10),
            fontsize=16, color="steelblue", weight="bold",
            ha="left", va="center", zorder=5)

# Green -> right edge of result box
res_tip = (res_xy[0] + res_w, res_xy[1] + res_h*0.55)
ax.annotate("Multiply!",
            xy=res_tip, xycoords=ax.transAxes,
            xytext=(0.78, 0.19), textcoords=ax.transAxes,  # nudged down
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=0.10",
                            lw=2.5, color="green",
                            shrinkA=6, shrinkB=10),
            fontsize=16, color="green", weight="bold",
            ha="left", va="center", zorder=5)

# Bottom explanation
why = (
    f"Why it works: Each sequence occurs with probability {total:.3f}/3 = {prob_each:.3f}, "
    f"and there are 3 ways to get exactly 2 successes, so total = 3 × {prob_each:.3f} = {total:.3f}."
)
ax.text(0.5, 0.10, why,
        transform=ax.transAxes, ha="center", va="center",
        fontsize=14, style="italic", wrap=True, zorder=4)

plt.savefig('ch07_binomial_formula_breakdown.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Binomial Formula Breakdown](ch07_binomial_formula_breakdown.svg)

The diagram shows how the formula components work together: we count the sequences (3), calculate the probability of each sequence (0.144), and multiply to get the total probability of exactly 2 successes (0.432).
:::

Let's verify this works for our coin flip example (n=10, p=0.5):

$$
\begin{align}
P(X=5) &= \binom{10}{5} p^5 (1-p)^{10-5} \\
&= \binom{10}{5} (0.5)^5 (1-0.5)^5 \\
&= \binom{10}{5} (0.5)^5 (0.5)^5 \\
&= 252 \times 0.03125 \times 0.03125 \\
&\approx 0.246 \quad \checkmark
\end{align}
$$

**Key Characteristics**

- **Scenarios**: Number of heads in coin flips, defective items in a batch, successful free throws, correct guesses on a test, customers who purchase
- **Parameters**:
    - $n$: number of independent trials
    - $p$: probability of success on each trial ($0 \le p \le 1$)
- **Random Variable**: $X \in \{0, 1, 2, ..., n\}$

**Mean:** $E[X] = np$

**Variance:** $Var(X) = np(1-p)$

**Standard Deviation:** $SD(X) = \sqrt{np(1-p)}$

**Visualizing the Distribution**

Let's visualize a Binomial distribution with $n = 10$ and $p = 0.5$ (our coin flip example):

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Create Binomial distribution for visualization (n=10, p=0.5)
n_viz = 10
p_viz = 0.5
binomial_viz = stats.binom(n=n_viz, p=p_viz)

# Calculate mean and std
mean_viz = binomial_viz.mean()
std_viz = binomial_viz.std()

# Plotting the PMF
k_values_viz = np.arange(0, n_viz + 1)
pmf_values_viz = binomial_viz.pmf(k_values_viz)

plt.figure(figsize=(10, 5))
plt.bar(k_values_viz, pmf_values_viz, color='skyblue', edgecolor='black', alpha=0.7)

# Add mean line
plt.axvline(mean_viz, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_viz:.1f}')

# Add mean ± 1 std region
plt.axvspan(mean_viz - std_viz, mean_viz + std_viz, alpha=0.2, color='orange',
            label=f'Mean ± 1 SD = [{mean_viz - std_viz:.1f}, {mean_viz + std_viz:.1f}]')

plt.title(f"Binomial PMF (n={n_viz}, p={p_viz})")
plt.xlabel("Number of Successes (k)")
plt.ylabel("Probability")
plt.legend(loc='upper right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_binomial_pmf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Binomial PMF](ch07_binomial_pmf_generic.svg)

The PMF shows the probability distribution for the number of heads in 10 coin flips. The distribution is symmetric around the mean ($np = 5$) since $p = 0.5$. The shaded region shows mean ± 1 standard deviation ($\sqrt{np(1-p)} = \sqrt{2.5} \approx 1.58$).

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values_viz = binomial_viz.cdf(k_values_viz)

plt.figure(figsize=(10, 5))
plt.step(k_values_viz, cdf_values_viz, where='post', color='darkgreen', linewidth=2)

# Add mean line
plt.axvline(mean_viz, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_viz:.1f}')

plt.title(f"Binomial CDF (n={n_viz}, p={p_viz})")
plt.xlabel("Number of Successes (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_binomial_cdf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Binomial CDF](ch07_binomial_cdf_generic.svg)

The CDF shows P(X ≤ k), the cumulative probability of getting k or fewer heads. The red dashed line marks the mean.

:::{admonition} Example: Sales Calls with n = 20, p = 0.15
:class: tip

Modeling the number of successful sales calls out of 20, where each call has a 0.15 probability of success.

We'll demonstrate how to use [`scipy.stats.binom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html) to calculate probabilities, compute statistics, and generate random samples.

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

**Note on Survival Function (`sf`):** The `sf()` method computes the **survival function**, which is P(X > k) = 1 - P(X ≤ k). While mathematically equivalent to `1 - cdf(k)`, using `sf(k)` directly is preferable because it provides better numerical accuracy when dealing with very small or very large probabilities, and makes the code's intent clearer.

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
plt.step(k_values, cdf_values, where='post', color='darkgreen', linewidth=2)
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

**Quick Check Questions**

1. You roll a die 12 times and count how many times you get a 6. Which distribution models this and what are the parameters?

```{admonition} Answer
:class: dropdown

**Binomial distribution with n = 12, p = 1/6** - Fixed number of trials (12 rolls), each with the same success probability (1/6 for rolling a 6).
```

2. For a Binomial distribution with n = 8 and p = 0.25, what is the expected value (mean)?

```{admonition} Answer
:class: dropdown

**E[X] = np = 8 × 0.25 = 2** - The expected number of successes in 8 trials is 2.
```

3. You're quality testing a batch of 100 products by examining each one. 5% are typically defective. Is this scenario best modeled by Binomial or Hypergeometric distribution?

```{admonition} Answer
:class: dropdown

**Binomial distribution** - Although you're sampling from a finite population, if the batch is large relative to your sample (or you replace items after testing), Binomial is appropriate. Each test is independent with constant p = 0.05.

If you were sampling a significant fraction of the batch *without replacement*, then Hypergeometric would be more appropriate.
```

4. For a Binomial(n=20, p=0.3) distribution, what is the variance?

```{admonition} Answer
:class: dropdown

**Var(X) = np(1-p) = 20 × 0.3 × 0.7 = 4.2**

Using the variance formula for Binomial distributions.
```

5. True or False: In a Binomial distribution, each trial must have the same probability of success.

```{admonition} Answer
:class: dropdown

**True** - The Binomial distribution requires:
1. Fixed number of independent trials (n)
2. Each trial has only two outcomes (success/failure)
3. **Constant success probability (p) across all trials**
4. Trials are independent

If the success probability changes from trial to trial, Binomial doesn't apply.
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

:::{admonition} Why This Formula Works
:class: note

The formula $(1-p)^{k-1} p$ has an intuitive structure:

- **$(1-p)^{k-1}$**: Probability of $k-1$ consecutive failures
- **$p$**: Probability of success on the $k$-th trial
- **Multiply them**: Since trials are independent, we multiply the probabilities

**Example:** For $P(X=3)$ with $p=0.4$:
- First two trials must fail: $(0.6) \times (0.6) = 0.36$
- Third trial must succeed: $0.4$
- Combined: $0.36 \times 0.4 = 0.144$

This is why the formula captures "trials until first success" - it requires all previous trials to fail and the final trial to succeed.
:::

**Key Characteristics**

- **Scenarios**: Coin flips until first Head, job applications until first offer, attempts to pass an exam, at-bats until first hit
- **Parameter**: $p$, probability of success on each trial ($0 < p \le 1$)
- **Random Variable**: $X \in \{1, 2, 3, ...\}$

**Mean:** $E[X] = \frac{1}{p}$

**Variance:** $Var(X) = \frac{1-p}{p^2}$

**Standard Deviation:** $SD(X) = \frac{\sqrt{1-p}}{p}$

**Relationship to Other Distributions:** The Geometric distribution is built from independent **Bernoulli trials** and is a special case of the **Negative Binomial distribution** with $r=1$ (waiting for just one success instead of $r$ successes).

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

# Calculate mean and std (adjusted for trial number definition)
mean_viz = 1 / p_viz
std_viz = np.sqrt((1 - p_viz) / p_viz**2)

# Plotting the PMF
k_values_viz = np.arange(1, 11)
pmf_values_viz = geom_viz.pmf(k_values_viz - 1)

plt.figure(figsize=(10, 5))
plt.bar(k_values_viz, pmf_values_viz, color='skyblue', edgecolor='black', alpha=0.7)

# Add mean line
plt.axvline(mean_viz, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_viz:.2f}')

# Add mean ± 1 std region
plt.axvspan(mean_viz - std_viz, mean_viz + std_viz, alpha=0.2, color='orange',
            label=f'Mean ± 1 SD = [{mean_viz - std_viz:.1f}, {mean_viz + std_viz:.1f}]')

plt.title(f"Geometric PMF (p={p_viz})")
plt.xlabel("Trial Number (k)")
plt.ylabel("Probability P(X=k)")
plt.xticks(k_values_viz)
plt.legend(loc='upper right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_geometric_pmf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Geometric PMF](ch07_geometric_pmf_generic.svg)

The PMF shows exponentially decreasing probabilities - you're most likely to succeed on the first few trials. The shaded region shows mean ± 1 standard deviation.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values_viz = geom_viz.cdf(k_values_viz - 1)

plt.figure(figsize=(10, 5))
plt.step(k_values_viz, cdf_values_viz, where='post', color='darkgreen', linewidth=2)

# Add mean line
plt.axvline(mean_viz, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_viz:.2f}')

plt.title(f"Geometric CDF (p={p_viz})")
plt.xlabel("Trial Number (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.xticks(k_values_viz)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_geometric_cdf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Geometric CDF](ch07_geometric_cdf_generic.svg)

The CDF shows P(X ≤ k), approaching 1 as k increases (eventually you'll succeed). The red dashed line marks the mean.

:::{admonition} Example: Certification Exam with p = 0.6
:class: tip

Modeling the number of attempts needed to pass a certification exam where the pass probability is 0.6.

Let's use [`scipy.stats.geom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geom.html) to explore probabilities and compute expected values. Remember that scipy's definition counts failures before the first success, so we'll translate between the two interpretations.

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
plt.step(k_values_trials, cdf_values, where='post', color='darkgreen', linewidth=2)
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

**Quick Check Questions**

1. You flip a coin until you get your first Heads. What distribution models this and what is the parameter?

```{admonition} Answer
:class: dropdown

**Geometric distribution with p = 0.5** - Counting trials until first success, each trial has p = 0.5 success probability.
```

2. For a Geometric distribution with p = 0.25, what is the expected value (mean)?

```{admonition} Answer
:class: dropdown

**E[X] = 1/p = 1/0.25 = 4** - Expected number of trials until first success.
```

3. You're calling customer service and have a 20% chance each attempt of getting through. Should you model this with Geometric or Binomial?

```{admonition} Answer
:class: dropdown

**Geometric distribution** - You're waiting for the *first* success (getting through), not counting successes in a fixed number of tries. Geometric models "how many attempts until success" with p = 0.20.

Binomial would apply if you made a fixed number of calls and counted how many got through.
```

4. Which is more likely for a Geometric distribution with p = 0.5: success on the 1st trial or success on the 3rd trial?

```{admonition} Answer
:class: dropdown

**1st trial is more likely** - The Geometric PMF decreases exponentially with k, so P(X=1) > P(X=3).

Specifically: P(X=1) = 0.5, while P(X=3) = (0.5)³ = 0.125
```

5. For a Geometric distribution, why does the variance equal (1-p)/p²?

```{admonition} Answer
:class: dropdown

The variance formula Var(X) = (1-p)/p² reflects the increasing uncertainty as p decreases:

- When p is high (easy to succeed): variance is low (more predictable)
- When p is low (hard to succeed): variance is high (could take many tries or get lucky early)

For example:
- p = 0.5: Var(X) = 0.5/0.25 = 2
- p = 0.1: Var(X) = 0.9/0.01 = 90 (much more variable!)
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

:::{admonition} Why This Formula Works
:class: note

The formula breaks down into three parts:

- **$\binom{k-1}{r-1}$**: Choose which $r-1$ of the first $k-1$ trials are successes (the $k$-th trial must be a success, so we only choose positions for $r-1$ successes)
- **$p^r$**: Probability of $r$ successes
- **$(1-p)^{k-r}$**: Probability of $k-r$ failures

**Example:** For $r=3$ successes in $k=5$ trials with $p=0.4$:
- Need exactly 2 successes in first 4 trials: $\binom{4}{2} = 6$ ways (e.g., SSFF, SFSF, SFFS, FSSF, FSFS, FFSS)
- Each arrangement has probability $(0.4)^2 (0.6)^2$ for the first 4 trials
- 5th trial must succeed: $0.4$
- Combined: $6 \times (0.4)^3 \times (0.6)^2$

The binomial coefficient ensures we count all possible arrangements where the $r$-th success occurs exactly on trial $k$.
:::

**Key Characteristics**

- **Scenarios**: Coin flips until getting r Heads, products inspected to find r defects, interviews until making r hires
- **Parameters**:
    - $r$: target number of successes ($r \ge 1$)
    - $p$: probability of success on each trial ($0 < p \le 1$)
- **Random Variable**: $X \in \{r, r+1, r+2, ...\}$

**Mean:** $E[X] = \frac{r}{p}$

**Variance:** $Var(X) = \frac{r(1-p)}{p^2}$

**Standard Deviation:** $SD(X) = \frac{\sqrt{r(1-p)}}{p}$

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

# Calculate mean and std
mean_viz = r_viz / p_viz
std_viz = np.sqrt(r_viz * (1 - p_viz)) / p_viz

# Plotting the PMF
k_values_viz = np.arange(r_viz, 30)  # Total trials from r to 30
pmf_values_viz = nbinom_viz.pmf(k_values_viz - r_viz)  # Adjust for scipy

plt.figure(figsize=(10, 5))
plt.bar(k_values_viz, pmf_values_viz, color='skyblue', edgecolor='black', alpha=0.7)

# Add mean line
plt.axvline(mean_viz, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_viz:.1f}')

# Add mean ± 1 std region
plt.axvspan(mean_viz - std_viz, mean_viz + std_viz, alpha=0.2, color='orange',
            label=f'Mean ± 1 SD = [{mean_viz - std_viz:.1f}, {mean_viz + std_viz:.1f}]')

plt.title(f"Negative Binomial PMF (r={r_viz}, p={p_viz})")
plt.xlabel("Total Number of Trials (k)")
plt.ylabel("Probability P(X=k)")
plt.legend(loc='upper right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_negative_binomial_pmf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Negative Binomial PMF](ch07_negative_binomial_pmf_generic.svg)

The PMF shows the distribution is centered around the expected value r/p = 3/0.2 = 15 trials. The shaded region shows mean ± 1 standard deviation.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values_viz = nbinom_viz.cdf(k_values_viz - r_viz)

plt.figure(figsize=(10, 5))
plt.step(k_values_viz, cdf_values_viz, where='post', color='darkgreen', linewidth=2)

# Add mean line
plt.axvline(mean_viz, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_viz:.1f}')

plt.title(f"Negative Binomial CDF (r={r_viz}, p={p_viz})")
plt.xlabel("Total Number of Trials (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_negative_binomial_cdf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Negative Binomial CDF](ch07_negative_binomial_cdf_generic.svg)

The CDF shows P(X ≤ k), the cumulative probability of achieving r successes within k trials. The red dashed line marks the mean.

:::{admonition} Example: Quality Control with r = 3, p = 0.05
:class: tip

A quality control inspector tests electronic components until finding 3 defective ones. The defect rate is p = 0.05.

We'll use [`scipy.stats.nbinom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html) to calculate the probability of needing a certain number of trials and compute expected values, keeping in mind scipy's definition of counting failures.

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
plt.step(k_values_components, cdf_values_nb, where='post', color='darkgreen', linewidth=2)
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

**Quick Check Questions**

1. You flip a fair coin until you get 5 Heads. What distribution models this and what are the parameters?

```{admonition} Answer
:class: dropdown

**Negative Binomial with r = 5, p = 0.5** - Counting trials until getting r successes, each trial has p = 0.5.
```

2. For a Negative Binomial distribution with r = 4 and p = 0.5, what is the expected value (mean)?

```{admonition} Answer
:class: dropdown

**E[X] = r/p = 4/0.5 = 8** - Expected number of trials to get 4 successes.
```

3. A basketball player practices free throws until making 10 successful shots. Each shot has a 70% success rate. Which distribution and why?

```{admonition} Answer
:class: dropdown

**Negative Binomial with r = 10, p = 0.7** - We're waiting for a fixed number of successes (r = 10), not just the first success. Each trial (shot) is independent with constant probability p = 0.7.

This is NOT Geometric because we need 10 successes, not just 1.
```

4. How is Negative Binomial related to Geometric distribution?

```{admonition} Answer
:class: dropdown

**Geometric is a special case where r = 1** - Negative Binomial with r=1 is identical to Geometric.

- Geometric: waiting for 1st success
- Negative Binomial: waiting for r-th success (r ≥ 1)
```

5. For Negative Binomial, why is the variance r(1-p)/p²?

```{admonition} Answer
:class: dropdown

The variance r(1-p)/p² grows with both r and uncertainty:

- **Increases with r**: Waiting for more successes means more trials and more variability
- **Increases as p decreases**: Lower success probability means higher uncertainty in when you'll reach r successes

For example:
- r=1, p=0.5: Var = 1×0.5/0.25 = 2
- r=5, p=0.5: Var = 5×0.5/0.25 = 10 (more variable with more successes needed)
- r=5, p=0.2: Var = 5×0.8/0.04 = 100 (much more variable with low p!)
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

:::{admonition} Why This Formula Works
:class: note

The Poisson formula $\frac{e^{-\lambda} \lambda^k}{k!}$ emerges from the mathematics of rare events:

- **$\lambda^k / k!$**: Represents the "raw" likelihood of $k$ events based on the rate $\lambda$
- **$e^{-\lambda}$**: A normalization factor that ensures all probabilities sum to 1

**Intuition:** The Poisson distribution arises as the limit of the Binomial distribution when:
- You divide a time interval into many tiny sub-intervals ($n$ very large)
- The probability of an event in each sub-interval is very small ($p$ very small)
- The average rate $\lambda = np$ stays constant

For example, "4 calls per hour" could be modeled as 3600 one-second intervals where each second has probability $p = 4/3600$ of receiving a call.

**Why mean = variance = λ?** This unique property reflects the "memoryless" nature of the Poisson process - events occur randomly and independently at a constant average rate.
:::

**Key Characteristics**

- **Scenarios**: Emails per hour, customer arrivals per day, typos per page, emergency calls per shift, defects per unit area
- **Parameter**: $\lambda$, average number of events in the interval ($\lambda > 0$)
- **Random Variable**: $X \in \{0, 1, 2, ...\}$

**Mean:** $E[X] = \lambda$

**Variance:** $Var(X) = \lambda$

**Standard Deviation:** $SD(X) = \sqrt{\lambda}$

Note: Mean and variance are equal in a Poisson distribution, so the standard deviation is simply the square root of λ.

**Relationship to Other Distributions:** The Poisson distribution is an approximation to the **Binomial distribution** when $n$ is large, $p$ is small, and $\lambda = np$ is moderate. Rule of thumb: use Poisson approximation when $n \ge 20$ and $p \le 0.05$.

**Visualizing the Distribution**

Let's visualize a Poisson distribution with $\lambda = 4$ (our call center example):

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Create Poisson distribution for visualization (λ=4)
lambda_viz = 4
poisson_viz = stats.poisson(mu=lambda_viz)

# Calculate mean and std
mean_viz = poisson_viz.mean()
std_viz = poisson_viz.std()

# Plotting the PMF
k_values_viz = np.arange(0, 15)
pmf_values_viz = poisson_viz.pmf(k_values_viz)

plt.figure(figsize=(10, 5))
plt.bar(k_values_viz, pmf_values_viz, color='skyblue', edgecolor='black', alpha=0.7)

# Add mean line
plt.axvline(mean_viz, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_viz:.1f}')

# Add mean ± 1 std region
plt.axvspan(mean_viz - std_viz, mean_viz + std_viz, alpha=0.2, color='orange',
            label=f'Mean ± 1 SD = [{mean_viz - std_viz:.1f}, {mean_viz + std_viz:.1f}]')

plt.title(f"Poisson PMF (λ={lambda_viz})")
plt.xlabel("Number of Events (k)")
plt.ylabel("Probability P(X=k)")
plt.xticks(k_values_viz)
plt.legend(loc='upper right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_poisson_pmf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Poisson PMF](ch07_poisson_pmf_generic.svg)

The PMF shows the distribution centered around λ = 4 with reasonable probability for nearby values. The shaded region shows mean ± 1 standard deviation ($\sqrt{4} = 2$).

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values_viz = poisson_viz.cdf(k_values_viz)

plt.figure(figsize=(10, 5))
plt.step(k_values_viz, cdf_values_viz, where='post', color='darkgreen', linewidth=2)

# Add mean line
plt.axvline(mean_viz, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_viz:.1f}')

plt.title(f"Poisson CDF (λ={lambda_viz})")
plt.xlabel("Number of Events (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.xticks(k_values_viz)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_poisson_cdf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Poisson CDF](ch07_poisson_cdf_generic.svg)

The CDF shows P(X ≤ k), useful for questions like "What's the probability of 6 or fewer calls?" The red dashed line marks the mean.

:::{admonition} Example: Email Arrivals with λ = 5
:class: tip

Modeling the number of emails received per hour with an average rate of λ = 5 emails/hour.

Let's use [`scipy.stats.poisson`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html) to calculate the probability of observing different numbers of events and verify that the mean equals the variance.

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
plt.step(k_values, cdf_values, where='post', color='darkgreen', linewidth=2)
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

**Quick Check Questions**

1. A call center receives an average of 12 calls per hour. What distribution models the number of calls in one hour and what is the parameter?

```{admonition} Answer
:class: dropdown

**Poisson distribution with λ = 12** - Events occurring at a constant average rate in a fixed interval.
```

2. For a Poisson distribution with λ = 7, what are the mean and variance?

```{admonition} Answer
:class: dropdown

**Mean = 7, Variance = 7** - For Poisson, both equal λ. This is a unique property of the Poisson distribution.
```

3. You count the number of typos on a random page of a book. The average is 2 typos per page. Which distribution?

```{admonition} Answer
:class: dropdown

**Poisson with λ = 2** - Counting discrete events (typos) occurring in a fixed space (one page) at a constant average rate.

This fits Poisson's requirements:
- Events happen independently
- Constant average rate
- Counting occurrences in fixed interval/space
```

4. True or False: In a Poisson distribution, the mean can be different from the variance.

```{admonition} Answer
:class: dropdown

**False** - A key property of Poisson is that mean = variance = λ.

This property can help you identify when Poisson might not be the best fit. If your data has variance much larger or smaller than the mean, consider other distributions (e.g., Negative Binomial for overdispersion).
```

5. When can Poisson approximate Binomial?

```{admonition} Answer
:class: dropdown

**When n is large, p is small, and np is moderate** - Specifically:
- n ≥ 20 and p ≤ 0.05, or
- n ≥ 100 and np ≤ 10

Then Binomial(n, p) ≈ Poisson(λ = np)

Example: Binomial(n=1000, p=0.003) ≈ Poisson(λ=3)

This works because rare events in many trials behave like events occurring at a constant rate.
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

:::{admonition} Why This Formula Works
:class: note

The Hypergeometric formula uses counting principles:

- **$\binom{N}{n}$** (denominator): Total ways to choose $n$ items from $N$ - this is all possible samples
- **$\binom{K}{k}$** (numerator): Ways to choose $k$ successes from the $K$ successes available
- **$\binom{N-K}{n-k}$** (numerator): Ways to choose $n-k$ failures from the $N-K$ failures available

**Example:** Drawing 5 cards hoping for 2 Aces (N=52, K=4, n=5, k=2):
- Ways to choose 2 Aces from 4: $\binom{4}{2} = 6$
- Ways to choose 3 non-Aces from 48: $\binom{48}{3} = 17,296$
- Ways to choose any 5 cards: $\binom{52}{5} = 2,598,960$
- Probability: $\frac{6 \times 17,296}{2,598,960} \approx 0.040$

The formula is essentially: **(favorable outcomes) / (total possible outcomes)** from basic probability, using combinations to count!
:::

**Key Characteristics**

- **Scenarios**: Cards from a deck, defective items in small batch, tagged fish in sample, jury selection from finite pool
- **Parameters**:
    - $N$: total population size
    - $K$: total number of successes in population
    - $n$: sample size ($n \le N$)
- **Random Variable**: $X$, bounded by $\max(0, n-(N-K)) \le X \le \min(n, K)$

**Mean:** $E[X] = n \frac{K}{N}$

**Variance:** $Var(X) = n \frac{K}{N} \left(1 - \frac{K}{N}\right) \left(\frac{N-n}{N-1}\right)$

**Standard Deviation:** $SD(X) = \sqrt{n \frac{K}{N} \left(1 - \frac{K}{N}\right) \left(\frac{N-n}{N-1}\right)}$

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

# Calculate mean and std
mean_viz = hypergeom_viz.mean()
std_viz = hypergeom_viz.std()

# Plotting the PMF
k_values_viz = np.arange(0, min(n_viz, K_viz) + 1)
pmf_values_viz = hypergeom_viz.pmf(k_values_viz)

plt.figure(figsize=(10, 5))
plt.bar(k_values_viz, pmf_values_viz, color='skyblue', edgecolor='black', alpha=0.7)

# Add mean line
plt.axvline(mean_viz, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_viz:.2f}')

# Add mean ± 1 std region
plt.axvspan(mean_viz - std_viz, mean_viz + std_viz, alpha=0.2, color='orange',
            label=f'Mean ± 1 SD = [{mean_viz - std_viz:.2f}, {mean_viz + std_viz:.2f}]')

plt.title(f"Hypergeometric PMF (N={N_viz}, K={K_viz}, n={n_viz})")
plt.xlabel("Number of Successes in Sample (k)")
plt.ylabel("Probability P(X=k)")
plt.xticks(k_values_viz)
plt.legend(loc='upper right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_hypergeometric_pmf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Hypergeometric PMF](ch07_hypergeometric_pmf_generic.svg)

The PMF shows most likely to get 0 Aces (about 0.66 probability), less likely to get 1 or 2. The red dashed line marks the mean, and the orange shaded region shows mean ± 1 standard deviation.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values_viz = hypergeom_viz.cdf(k_values_viz)

plt.figure(figsize=(10, 5))
plt.step(k_values_viz, cdf_values_viz, where='post', color='darkgreen', linewidth=2)

# Add mean line
plt.axvline(mean_viz, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_viz:.2f}')

plt.title(f"Hypergeometric CDF (N={N_viz}, K={K_viz}, n={n_viz})")
plt.xlabel("Number of Successes in Sample (k)")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.xticks(k_values_viz)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_hypergeometric_cdf_generic.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Hypergeometric CDF](ch07_hypergeometric_cdf_generic.svg)

The CDF shows P(X ≤ k), useful for questions like "What's the probability of getting at most 1 Ace?" The red dashed line marks the mean.

:::{admonition} Example: Lottery Tickets with N=100, K=20, n=10
:class: tip

Modeling the number of winning lottery tickets in a sample of 10 drawn from a box of 100 tickets where 20 are winners.

We'll use [`scipy.stats.hypergeom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html) to calculate probabilities for sampling without replacement and see how the mean relates to the population proportion.

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

The CDF shows P(X ≤ k), the cumulative probability of getting k or fewer winning tickets.

:::

**Quick Check Questions**

1. You draw 7 cards from a deck of 52. You want to know how many hearts you get. What distribution models this and what are the parameters?

```{admonition} Answer
:class: dropdown

**Hypergeometric with N=52, K=13, n=7** - Sampling without replacement from a finite population (13 hearts in 52 cards).
```

2. For a Hypergeometric distribution with N=50, K=10, n=5, what is the expected value (mean)?

```{admonition} Answer
:class: dropdown

**E[X] = n(K/N) = 5 × (10/50) = 1** - Expected number of successes in the sample.
```

3. A quality inspector randomly selects 10 products from a batch of 100 (where 15 are defective) without replacement. Which distribution?

```{admonition} Answer
:class: dropdown

**Hypergeometric with N=100, K=15, n=10** - Sampling without replacement from a finite population.

This is NOT Binomial because:
- We're sampling without replacement
- The sample size (10) is significant relative to population (100)
- Each draw changes the probability for subsequent draws
```

4. What's the key difference between Binomial and Hypergeometric distributions?

```{admonition} Answer
:class: dropdown

**Hypergeometric samples WITHOUT replacement** (finite population), while Binomial samples WITH replacement (or assumes infinite population).

Key implications:
- Hypergeometric: Trials are NOT independent (each draw affects the next)
- Binomial: Trials ARE independent (constant probability p)

Rule of thumb: If N > 20n, Hypergeometric ≈ Binomial because the sample is small relative to the population.
```

5. When can Hypergeometric be approximated by Binomial?

```{admonition} Answer
:class: dropdown

**When the population is much larger than the sample** - Specifically, when N > 20n.

In this case, Hypergeometric(N, K, n) ≈ Binomial(n, p=K/N)

Example: Drawing 10 cards from a population of 1000 cards. The small sample barely affects the population proportions, so with/without replacement gives nearly identical results.
```

+++

## 7. Discrete Uniform Distribution

The Discrete Uniform distribution models selecting one outcome from a finite set where all outcomes are equally likely.

**Concrete Example**

Suppose you roll a fair six-sided die. Each face (1, 2, 3, 4, 5, 6) has an equal probability of appearing.

We model this with a random variable $X$:
- $X$ = the number showing on the die
- $X$ can take values 1, 2, 3, 4, 5, 6

The probabilities are:
- $P(X = 1) = P(X = 2) = \cdots = P(X = 6) = \frac{1}{6}$

**The Discrete Uniform PMF**

For a Discrete Uniform distribution on the integers from $a$ to $b$ (inclusive):

$$ P(X=k) = \begin{cases} \frac{1}{b-a+1} & \text{if } k \in \{a, a+1, \ldots, b\} \\ 0 & \text{otherwise} \end{cases} $$

For our die example with $a = 1$ and $b = 6$:
- $P(X=k) = \frac{1}{6-1+1} = \frac{1}{6}$ for $k \in \{1, 2, 3, 4, 5, 6\}$

:::{admonition} Why This Formula Works
:class: note

The Discrete Uniform distribution is the simplest probability distribution:

- **Total outcomes**: $b - a + 1$ (the "+1" counts both endpoints)
- **Each outcome equally likely**: Probability = $\frac{1}{\text{total outcomes}}$

**Example:** For values 5 through 15:
- Total values: $15 - 5 + 1 = 11$ values
- Each has probability: $\frac{1}{11} \approx 0.091$

This directly implements the classical definition of probability: **(favorable outcomes) / (total equally likely outcomes)**.
:::

**Key Characteristics**

- **Scenarios**: Fair die roll, random selection from a list, lottery number selection, random password digit
- **Parameters**:
    - $a$: minimum value (integer)
    - $b$: maximum value (integer, $b \ge a$)
- **Random Variable**: $X \in \{a, a+1, \ldots, b\}$

**Mean:** $E[X] = \frac{a+b}{2}$

**Variance:** $Var(X) = \frac{(b-a+1)^2 - 1}{12}$

**Standard Deviation:** $SD(X) = \sqrt{\frac{(b-a+1)^2 - 1}{12}}$

**Relationship to Other Distributions:** The Discrete Uniform distribution is a special case of the **Categorical distribution** where all $k$ categories have equal probability $p_i = 1/k$. If outcomes aren't equally likely, use Categorical instead.

**Visualizing the Distribution**

Let's visualize a Discrete Uniform distribution for a fair die ($a = 1$, $b = 6$):

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Create Discrete Uniform distribution for visualization (fair die)
a_viz = 1
b_viz = 6
from scipy.stats import randint
# scipy.stats.randint uses [low, high) so we add 1 to b
uniform_viz = randint(low=a_viz, high=b_viz+1)

# Calculate mean and std
mean_viz = uniform_viz.mean()
std_viz = uniform_viz.std()

# Plotting the PMF
k_values_viz = np.arange(a_viz, b_viz+1)
pmf_values_viz = uniform_viz.pmf(k_values_viz)

plt.figure(figsize=(10, 5))
plt.bar(k_values_viz, pmf_values_viz, color='skyblue', edgecolor='black', alpha=0.7)

# Add mean line
plt.axvline(mean_viz, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_viz:.2f}')

# Add mean ± 1 std region
plt.axvspan(mean_viz - std_viz, mean_viz + std_viz, alpha=0.2, color='orange',
            label=f'Mean ± 1 SD = [{mean_viz - std_viz:.2f}, {mean_viz + std_viz:.2f}]')

plt.title(f"Discrete Uniform PMF (a={a_viz}, b={b_viz})")
plt.xlabel("Outcome")
plt.ylabel("Probability")
plt.ylim(0, 0.25)
plt.xticks(k_values_viz)
plt.legend(loc='upper right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_discrete_uniform_pmf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Discrete Uniform PMF](ch07_discrete_uniform_pmf.svg)

The PMF shows six equal bars, each with probability 1/6, representing the fair die. The shaded region shows mean ± 1 standard deviation.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values_viz = uniform_viz.cdf(k_values_viz)

plt.figure(figsize=(10, 5))
plt.step(k_values_viz, cdf_values_viz, where='post', color='darkgreen', linewidth=2)

# Add mean line
plt.axvline(mean_viz, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_viz:.2f}')

plt.title(f"Discrete Uniform CDF (a={a_viz}, b={b_viz})")
plt.xlabel("Outcome")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.ylim(0, 1.1)
plt.xticks(k_values_viz)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_discrete_uniform_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Discrete Uniform CDF](ch07_discrete_uniform_cdf.svg)

The CDF increases in equal steps of 1/6 at each value, reaching 1.0 at the maximum value. The red dashed line marks the mean.

:::{admonition} Example: Random Selection from 1 to 20
:class: tip

Modeling a random integer selection from 1 to 20, where each number is equally likely to be chosen.

Let's use [`scipy.stats.randint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html) to calculate probabilities and generate samples.

```{code-cell} ipython3
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Using scipy.stats.randint (note: uses [low, high) interval)
a_sel = 1
b_sel = 20
uniform_rv = stats.randint(low=a_sel, high=b_sel+1)

# PMF: Probability of any specific value
k_val = 7
print(f"P(X={k_val}): {uniform_rv.pmf(k_val):.4f}")
print(f"This equals 1/{b_sel-a_sel+1} = {1/(b_sel-a_sel+1):.4f}")
```

```{code-cell} ipython3
# CDF: Probability of k or fewer
k_threshold = 10
print(f"P(X <= {k_threshold}): {uniform_rv.cdf(k_threshold):.4f}")
print(f"P(X > {k_threshold}): {uniform_rv.sf(k_threshold):.4f}")
```

```{code-cell} ipython3
# Mean and Variance
print(f"Mean (Expected value): {uniform_rv.mean():.2f}")
print(f"Theoretical mean (a+b)/2: {(a_sel+b_sel)/2:.2f}")
print(f"Variance: {uniform_rv.var():.2f}")
```

```{code-cell} ipython3
# Generate random samples
n_samples = 10
samples = uniform_rv.rvs(size=n_samples)
print(f"{n_samples} random selections from 1 to {b_sel}:")
print(samples)
```

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the PMF
k_values = np.arange(a_sel, b_sel+1)
pmf_values = uniform_rv.pmf(k_values)

plt.figure(figsize=(10, 4))
plt.bar(k_values, pmf_values, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Discrete Uniform PMF (a={a_sel}, b={b_sel})")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_discrete_uniform_pmf_example.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Discrete Uniform PMF](ch07_discrete_uniform_pmf_example.svg)

All 20 values have equal probability of 0.05 (1/20).

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values = uniform_rv.cdf(k_values)

plt.figure(figsize=(10, 4))
plt.step(k_values, cdf_values, where='post', color='darkgreen', linewidth=2)
plt.title(f"Discrete Uniform CDF (a={a_sel}, b={b_sel})")
plt.xlabel("Value")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_discrete_uniform_cdf_example.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Discrete Uniform CDF](ch07_discrete_uniform_cdf_example.svg)

The CDF increases linearly in equal steps, showing the uniform nature of the distribution.

:::

**Quick Check Questions**

1. You randomly select a card from a standard deck (52 cards). If X represents the card number (1-13, where 1=Ace, 11=Jack, 12=Queen, 13=King), what distribution models this and what are the parameters?

```{admonition} Answer
:class: dropdown

**Discrete Uniform distribution with a = 1, b = 13** - Each card number is equally likely (4 of each in the deck).
```

2. For a Discrete Uniform distribution with a = 5 and b = 15, what is the probability of getting exactly 10?

```{admonition} Answer
:class: dropdown

**P(X = 10) = 1/(15-5+1) = 1/11 ≈ 0.091** - All values in the range are equally likely.
```

3. What is the mean of a Discrete Uniform distribution on the integers from 1 to 100?

```{admonition} Answer
:class: dropdown

**Mean = (1+100)/2 = 50.5** - The mean is the midpoint of the range.
```

4. You're modeling the outcome of rolling a fair six-sided die. Should you use Discrete Uniform or Categorical distribution?

```{admonition} Answer
:class: dropdown

**Discrete Uniform distribution** - Since it's a *fair* die, all outcomes (1-6) are equally likely with probability 1/6 each.

Use Discrete Uniform(a=1, b=6).

**Note:** If the die were *loaded* (unequal probabilities), you'd use the Categorical distribution instead.
```

5. For a Discrete Uniform distribution on integers from a to b, why is the variance equal to $\frac{(b-a)(b-a+2)}{12}$?

```{admonition} Answer
:class: dropdown

The variance formula reflects how spread out the values are:

- **Larger range (b-a)**: Higher variance - values are more spread out
- **Formula intuition**: The variance grows with the *square* of the range, similar to continuous uniform distributions

**Example:**
- Discrete Uniform(1, 6): Variance = (6-1)(6-1+2)/12 = 5×7/12 ≈ 2.92
- Discrete Uniform(1, 100): Variance = 99×101/12 ≈ 833.25

Much larger range → much larger variance.
```

+++

## 8. Categorical Distribution

The Categorical distribution models a single trial with multiple possible outcomes (more than 2), where each outcome has its own probability. It's the generalization of the Bernoulli distribution to more than two categories.

**Concrete Example**

Suppose you're rolling a loaded six-sided die where the faces have different probabilities:
- Face 1: probability 0.1
- Face 2: probability 0.15
- Face 3: probability 0.20
- Face 4: probability 0.25
- Face 5: probability 0.20
- Face 6: probability 0.10

We model this with a random variable $X$:
- $X$ = the face that appears
- $X$ can take values in $\{1, 2, 3, 4, 5, 6\}$
- Each value has its own probability: $P(X=1)=0.1, P(X=2)=0.15,$ etc.

**The Categorical PMF**

For a Categorical distribution with $k$ possible outcomes and probabilities $p_1, p_2, \ldots, p_k$ where $\sum_{i=1}^k p_i = 1$:

$$ P(X=i) = p_i \quad \text{for } i = 1, 2, \ldots, k $$

For our loaded die example:
- $P(X=1) = 0.1,\, P(X=2) = 0.15,\, P(X=3) = 0.20$
- $P(X=4) = 0.25,\, P(X=5) = 0.20,\, P(X=6) = 0.10$
- Sum: $0.1 + 0.15 + 0.20 + 0.25 + 0.20 + 0.10 = 1.0$ ✓

:::{admonition} Why This Formula Works
:class: note

The Categorical PMF is straightforward - each outcome has its own assigned probability:

- **Single trial**: Only one outcome occurs
- **Each outcome $i$ has probability $p_i$**: Directly specified
- **Constraint**: All probabilities must sum to 1 (ensuring exactly one outcome occurs)

This is the most general discrete distribution for a single trial - every outcome can have a different probability. It generalizes simpler distributions:
- If $k=2$: Reduces to **Bernoulli**
- If all $p_i = 1/k$: Reduces to **Discrete Uniform**
:::

**Key Characteristics**

- **Scenarios**: Loaded die, customer choosing from menu categories, survey response (multiple choice), weather outcome (sunny/cloudy/rainy/snowy)
- **Parameters**:
    - $k$: number of categories
    - $p_1, p_2, \ldots, p_k$: probabilities for each category (must sum to 1)
- **Random Variable**: $X \in \{1, 2, \ldots, k\}$

**Mean:** $E[X] = \sum_{i=1}^k i \cdot p_i$ (weighted average of outcomes)

**Variance:** $Var(X) = \sum_{i=1}^k i^2 \cdot p_i - \left(\sum_{i=1}^k i \cdot p_i\right)^2$

**Relationship to Other Distributions:** Categorical generalizes **Bernoulli** (when $k=2$) and is a special case of **Discrete Uniform** (when all $p_i$ are equal). For multiple trials, use the **Multinomial distribution** instead.

**Visualizing the Distribution**

Let's visualize our loaded die Categorical distribution:

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Create Categorical distribution for visualization (loaded die)
probs_viz = np.array([0.1, 0.15, 0.20, 0.25, 0.20, 0.10])
from scipy.stats import rv_discrete
values_viz = np.arange(1, 7)
categorical_viz = rv_discrete(values=(values_viz, probs_viz))

# Plotting the PMF
plt.figure(figsize=(8, 4))
plt.bar(values_viz, probs_viz, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Categorical PMF (Loaded Die)")
plt.xlabel("Outcome")
plt.ylabel("Probability")
plt.ylim(0, 0.3)
plt.xticks(values_viz)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_categorical_pmf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Categorical PMF](ch07_categorical_pmf.svg)

The PMF shows the different probabilities for each face of the loaded die.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_values_viz = categorical_viz.cdf(values_viz)

plt.figure(figsize=(8, 4))
plt.step(values_viz, cdf_values_viz, where='post', color='darkgreen', linewidth=2)
plt.title("Categorical CDF (Loaded Die)")
plt.xlabel("Outcome")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.ylim(0, 1.1)
plt.xticks(values_viz)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_categorical_cdf.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Categorical CDF](ch07_categorical_cdf.svg)

The CDF increases by different amounts at each value, reflecting the varying probabilities.

:::{admonition} Example: Customer Product Choice
:class: tip

A coffee shop tracks customer drink preferences: 40% choose coffee, 30% choose tea, 20% choose juice, and 10% choose water.

Let's model this using a Categorical distribution with [`scipy.stats.rv_discrete`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html).

```{code-cell} ipython3
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Define the categorical distribution
choices = np.array([1, 2, 3, 4])  # 1=Coffee, 2=Tea, 3=Juice, 4=Water
probs = np.array([0.40, 0.30, 0.20, 0.10])
categorical_rv = stats.rv_discrete(values=(choices, probs))

# PMF: Probability of each choice
labels = ['Coffee', 'Tea', 'Juice', 'Water']
for i, (choice, label) in enumerate(zip(choices, labels)):
    print(f"P(X={choice}) [{label}]: {probs[i]:.2f}")
```

```{code-cell} ipython3
# CDF: Probability of choice i or lower
print(f"P(X <= 2) [Coffee or Tea]: {categorical_rv.cdf(2):.2f}")
print(f"P(X > 2) [Juice or Water]: {1 - categorical_rv.cdf(2):.2f}")
```

```{code-cell} ipython3
# Mean and Variance
print(f"Mean (Expected value): {categorical_rv.mean():.2f}")
print(f"Variance: {categorical_rv.var():.2f}")
```

```{code-cell} ipython3
# Generate random samples
n_customers = 100
samples = categorical_rv.rvs(size=n_customers)
print(f"\nSimulated choices for {n_customers} customers:")
for i, label in enumerate(labels, 1):
    count = np.sum(samples == i)
    print(f"{label}: {count} ({count/n_customers:.1%})")
```

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the PMF
plt.figure(figsize=(8, 4))
plt.bar(choices, probs, tick_label=labels, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Categorical PMF (Customer Drink Choice)")
plt.xlabel("Choice")
plt.ylabel("Probability")
plt.ylim(0, 0.5)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_categorical_pmf_example.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Categorical PMF](ch07_categorical_pmf_example.svg)

Coffee is the most popular choice, followed by tea, juice, and water.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plotting the CDF
cdf_vals = categorical_rv.cdf(choices)

plt.figure(figsize=(8, 4))
plt.step(choices, cdf_vals, where='post', color='darkgreen', linewidth=2)
plt.xticks(choices, labels)
plt.title("Categorical CDF (Customer Drink Choice)")
plt.xlabel("Choice")
plt.ylabel("Cumulative Probability P(X <= k)")
plt.ylim(0, 1.1)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.savefig('ch07_categorical_cdf_example.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Categorical CDF](ch07_categorical_cdf_example.svg)

The CDF shows cumulative probabilities across the ordered choices.

:::

**Quick Check Questions**

1. A traffic light can be red (50%), yellow (10%), or green (40%). What distribution models the color when you arrive at an intersection?

```{admonition} Answer
:class: dropdown

**Categorical distribution with k=3 categories and probabilities p₁=0.5, p₂=0.1, p₃=0.4** - Single trial with three possible outcomes.
```

2. For a Categorical distribution with 4 equally likely outcomes, what is P(X = 2)?

```{admonition} Answer
:class: dropdown

**P(X = 2) = 0.25** - For equally likely outcomes, each has probability 1/4.
```

3. How is the Categorical distribution related to the Bernoulli distribution?

```{admonition} Answer
:class: dropdown

**Bernoulli is a special case of Categorical with k=2** - When there are only two categories, Categorical reduces to Bernoulli.

Categorical generalizes Bernoulli from 2 outcomes to k outcomes.
```

4. You're observing a single customer's choice from a menu with 5 items having probabilities [0.3, 0.25, 0.2, 0.15, 0.1]. Should you use Categorical or Multinomial distribution?

```{admonition} Answer
:class: dropdown

**Categorical distribution** - You're observing a *single trial* (one customer making one choice).

**Key distinction:**
- **Categorical**: Single trial, multiple outcomes (this scenario)
- **Multinomial**: Multiple trials, counting how many times each outcome occurs

If you observed 100 customers and counted how many chose each item, *that* would be Multinomial.
```

5. When can you model a Categorical distribution as a Discrete Uniform distribution?

```{admonition} Answer
:class: dropdown

**When all k categories have equal probability** - If p₁ = p₂ = ... = pₖ = 1/k.

**Example:**
- Rolling a fair die (6 equally likely outcomes): Can use either Categorical(p=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]) or Discrete Uniform(a=1, b=6)
- Rolling a loaded die (unequal probabilities): Must use Categorical

Discrete Uniform is just a special case of Categorical where all probabilities are equal.
```

+++

## 9. Multinomial Distribution

The Multinomial distribution models performing a fixed number of independent trials where each trial has multiple possible outcomes (more than 2), and we count how many times each outcome occurs. It's the generalization of the Binomial distribution to more than two categories.

**Concrete Example**

Suppose you roll a fair six-sided die 20 times. We want to know how many times each face (1, 2, 3, 4, 5, 6) appears.

We model this with a random vector $\mathbf{X} = (X_1, X_2, X_3, X_4, X_5, X_6)$ where:
- $X_1$ = number of times face 1 appears
- $X_2$ = number of times face 2 appears
- ... and so on
- Constraint: $X_1 + X_2 + X_3 + X_4 + X_5 + X_6 = 20$

The probabilities for a fair die are:
- $p_1 = p_2 = p_3 = p_4 = p_5 = p_6 = \frac{1}{6}$

**The Multinomial PMF**

For $n$ independent trials with $k$ possible outcomes and probabilities $p_1, p_2, \ldots, p_k$ where $\sum_{i=1}^k p_i = 1$:

$$ P(X_1=x_1, X_2=x_2, \ldots, X_k=x_k) = \frac{n!}{x_1! x_2! \cdots x_k!} \, p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k} $$

where $x_1 + x_2 + \cdots + x_k = n$.

The term $\frac{n!}{x_1! x_2! \cdots x_k!}$ is the multinomial coefficient (see [Chapter 3: Permutations of Identical Objects](chapter_03.md#permutations-of-identical-objects)).

For our die example, the probability of getting exactly (3, 4, 2, 5, 4, 2) of each face:

$$
\begin{align}
P(X_1=3, X_2=4, X_3=2, X_4=5, X_5=4, X_6=2) &= \frac{20!}{3! \, 4! \, 2! \, 5! \, 4! \, 2!} \left(\frac{1}{6}\right)^{20} \\
&= 1.34 \times 10^{13} \times 3.39 \times 10^{-16} \\
&\approx 0.00454
\end{align}
$$

:::{admonition} Why This Formula Works
:class: note

The Multinomial formula extends the Binomial idea to multiple categories:

- **$\frac{n!}{x_1! x_2! \cdots x_k!}$**: The multinomial coefficient counts how many different sequences of $n$ trials produce exactly $x_1$ occurrences of category 1, $x_2$ of category 2, etc.
- **$p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k}$**: Probability of any specific sequence with those counts

**Example:** With 3 trials and outcomes (A, A, B):
- There are $\frac{3!}{2! \, 1!} = 3$ arrangements: AAB, ABA, BAA
- Each has probability $p_A^2 p_B^1$
- Combined: $3 \times p_A^2 p_B$

The multinomial coefficient is like the binomial coefficient, but for distributing $n$ items among $k$ categories instead of just 2.
:::

**Key Characteristics**

- **Scenarios**: Rolling a die n times (counting each face), survey with multiple choice options, customer purchases across product categories, DNA base frequencies in a sequence
- **Parameters**:
    - $n$: number of trials
    - $k$: number of categories
    - $p_1, p_2, \ldots, p_k$: probabilities for each category (must sum to 1)
- **Random Variables**: $X_1, X_2, \ldots, X_k$ where $X_i$ = count for category $i$, and $\sum_{i=1}^k X_i = n$

**Mean for each category:** $E[X_i] = n p_i$

**Variance for each category:** $Var(X_i) = n p_i (1-p_i)$

**Relationship to Other Distributions:** Multinomial generalizes **Binomial** (when $k=2$) and **Categorical** (single trial becomes multiple trials). Each individual category count $X_i$ follows a **Binomial** distribution with parameters $(n, p_i)$.

**Visualizing the Distribution**

Multinomial distributions are challenging to visualize since they involve multiple variables. Let's look at a simple case with $k=3$ categories:

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Simulate multinomial: rolling a 3-sided die 15 times
n_trials = 15
probs_3 = np.array([1/3, 1/3, 1/3])

# Generate many samples
n_sims = 10000
samples = np.random.multinomial(n_trials, probs_3, size=n_sims)

# Plot distribution of outcomes for Category 1 (marginal distribution)
category_1_counts = samples[:, 0]

plt.figure(figsize=(8, 4))
plt.hist(category_1_counts, bins=np.arange(0, n_trials+2)-0.5, density=True,
         color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Marginal Distribution of Category 1\n(Multinomial with n={n_trials}, k=3, all p=1/3)")
plt.xlabel("Count for Category 1")
plt.ylabel("Probability")
plt.xticks(range(0, n_trials+1))
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('ch07_multinomial_marginal.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Multinomial Marginal](ch07_multinomial_marginal.svg)

The marginal distribution of any single category in a Multinomial distribution is actually a Binomial distribution! Here, Category 1 follows Binomial(n=15, p=1/3).

:::{admonition} Example: Customer Product Purchases
:class: tip

A store tracks purchases across 4 product categories: Electronics (30%), Clothing (25%), Home Goods (25%), Food (20%). We observe 50 customers and count how many purchase from each category.

Let's use [`numpy.random.multinomial`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html) to work with this distribution.

```{code-cell} ipython3
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Define parameters
n_customers = 50
categories = ['Electronics', 'Clothing', 'Home Goods', 'Food']
probs = np.array([0.30, 0.25, 0.25, 0.20])

# Expected counts
expected_counts = n_customers * probs
print("Expected purchases per category:")
for cat, exp in zip(categories, expected_counts):
    print(f"  {cat}: {exp:.1f}")
```

```{code-cell} ipython3
# Generate one sample (one set of 50 customers)
one_sample = np.random.multinomial(n_customers, probs)
print(f"\nOne simulation of {n_customers} customers:")
for cat, count in zip(categories, one_sample):
    print(f"  {cat}: {count}")
print(f"Total: {np.sum(one_sample)}")
```

```{code-cell} ipython3
# Generate many samples to see the distribution
n_sims = 10000
samples = np.random.multinomial(n_customers, probs, size=n_sims)

# Compute mean and std for each category
for i, cat in enumerate(categories):
    counts = samples[:, i]
    print(f"{cat}:")
    print(f"  Mean: {np.mean(counts):.2f} (theoretical: {n_customers * probs[i]:.2f})")
    print(f"  Std: {np.std(counts):.2f} (theoretical: {np.sqrt(n_customers * probs[i] * (1-probs[i])):.2f})")
```

```{code-cell} ipython3
:tags: [remove-input, remove-output]

# Plot distributions for each category
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, (cat, ax) in enumerate(zip(categories, axes)):
    counts = samples[:, i]
    ax.hist(counts, bins=np.arange(0, n_customers+2)-0.5, density=True,
            color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(expected_counts[i], color='red', linestyle='--', linewidth=2, label=f'Expected: {expected_counts[i]:.1f}')
    ax.set_title(f"{cat} (p={probs[i]})")
    ax.set_xlabel("Number of Purchases")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('ch07_multinomial_example.svg', format='svg', bbox_inches='tight')
plt.show()
```

![Multinomial Example](ch07_multinomial_example.svg)

Each category's marginal distribution is Binomial with parameters (n=50, p=category probability).

:::

**Quick Check Questions**

1. You flip a fair coin 30 times and count heads and tails. What distribution models the counts?

```{admonition} Answer
:class: dropdown

**Multinomial distribution with n=30, k=2, and p₁=p₂=0.5** - Or equivalently, Binomial(n=30, p=0.5) for the number of heads, since there are only 2 categories.

When k=2, Multinomial is the same as Binomial.
```

2. For a Multinomial distribution with n=100 trials and k=4 equally likely categories, what is the expected count for any one category?

```{admonition} Answer
:class: dropdown

**E[X_i] = n × p_i = 100 × 0.25 = 25** - Each category is expected to appear 25 times.

Since all 4 categories are equally likely, p_i = 1/4 = 0.25 for each.
```

3. How is the Multinomial distribution related to the Binomial distribution?

```{admonition} Answer
:class: dropdown

**Binomial is a special case of Multinomial with k=2** - When there are only two categories, Multinomial reduces to Binomial.

Multinomial generalizes Binomial from 2 outcomes to k outcomes across multiple trials.
```

4. You roll a die 100 times and count how many times each face (1-6) appears. Should you use Categorical or Multinomial distribution?

```{admonition} Answer
:class: dropdown

**Multinomial distribution** - You're performing *multiple trials* (100 rolls) and counting how many times each outcome occurs.

**Key distinction:**
- **Categorical**: Single trial, multiple possible outcomes (one roll)
- **Multinomial**: Multiple trials, counting occurrences of each outcome (100 rolls)

Use Multinomial(n=100, k=6, p=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]) for a fair die.
```

5. In a Multinomial distribution, what is the relationship between the individual category counts X₁, X₂, ..., Xₖ?

```{admonition} Answer
:class: dropdown

**They must sum to n** - The constraint is: X₁ + X₂ + ... + Xₖ = n

This is because every trial must result in exactly one category, so the total count across all categories equals the number of trials.

**Important implication:** The counts are *not independent* - if you know k-1 of the counts, you can determine the last one.

**Example:** If n=100 and you know X₁=30, X₂=25, X₃=20 in a k=4 category case, then X₄ must equal 25 (since 30+25+20+25=100).
```

+++

## 10. Relationships Between Distributions

Understanding the connections between these distributions can deepen insight and provide useful approximations.

1.  **Bernoulli as a special case of Binomial**: A Binomial distribution with $n=1$ trial ($Binomial(1, p)$) is equivalent to a Bernoulli distribution ($Bernoulli(p)$).

2.  **Geometric as a special case of Negative Binomial**: A Negative Binomial distribution modeling the number of trials until the first success ($r=1$) ($NegativeBinomial(1, p)$) is equivalent to a Geometric distribution ($Geometric(p)$).

3.  **Binomial Approximation to Hypergeometric**: If the population size $N$ is much larger than the sample size $n$ (e.g., $N > 20n$), then drawing without replacement (Hypergeometric) is very similar to drawing with replacement. In this case, the Hypergeometric($N, K, n$) distribution can be well-approximated by the Binomial($n, p=K/N$) distribution. The finite population correction factor $\frac{N-n}{N-1}$ approaches 1.

4.  **Poisson Approximation to Binomial**: If the number of trials $n$ in a Binomial distribution is large, and the success probability $p$ is small, such that the mean $\lambda = np$ is moderate, then the Binomial($n, p$) distribution can be well-approximated by the Poisson($\lambda = np$) distribution. This is useful because the Poisson PMF is often easier to compute than the Binomial PMF when $n$ is large. A common rule of thumb is to use this approximation if $n \ge 20$ and $p \le 0.05$, or $n \ge 100$ and $np \le 10$.

**Example: Poisson approximation to Binomial**
Consider $Binomial(n=1000, p=0.005)$. Here $n$ is large, $p$ is small. The mean is $\lambda = np = 1000 \times 0.005 = 5$. We can approximate this with $Poisson(\lambda=5)$.

Let's compare the PMF values of both distributions to see how well the Poisson approximation works in practice.

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

5.  **Categorical as generalization of Bernoulli**: A Categorical distribution with $k=2$ categories ($Categorical(p_1, p_2)$ where $p_1 + p_2 = 1$) is equivalent to a Bernoulli distribution ($Bernoulli(p_1)$). Categorical extends Bernoulli to handle more than two outcomes in a single trial.

6.  **Multinomial as generalization of Binomial**: A Multinomial distribution with $k=2$ categories ($Multinomial(n, p_1, p_2)$ where $p_1 + p_2 = 1$) is equivalent to a Binomial distribution ($Binomial(n, p_1)$). Multinomial extends Binomial to count outcomes across more than two categories.

7.  **Discrete Uniform as special case of Categorical**: A Categorical distribution where all $k$ probabilities are equal ($p_1 = p_2 = \cdots = p_k = \frac{1}{k}$) is a Discrete Uniform distribution on $k$ values. This represents maximum uncertainty about a single trial's outcome.

8.  **Marginal distributions of Multinomial are Binomial**: If $(X_1, X_2, \ldots, X_k) \sim Multinomial(n, p_1, p_2, \ldots, p_k)$, then each individual count $X_i$ follows a Binomial distribution: $X_i \sim Binomial(n, p_i)$. This makes sense because we're just counting successes (category $i$) vs. failures (all other categories) across $n$ trials.

+++

## Summary

In this chapter, we explored nine fundamental discrete probability distributions:

* **Bernoulli**: Single trial, two outcomes (Success/Failure).
* **Binomial**: Fixed number of independent trials, counts successes.
* **Geometric**: Number of trials until the *first* success.
* **Negative Binomial**: Number of trials until a *fixed number* ($r$) of successes.
* **Poisson**: Number of events in a fixed interval of time/space, given an average rate.
* **Hypergeometric**: Number of successes in a sample drawn *without* replacement from a finite population.
* **Discrete Uniform**: Single trial where all outcomes are equally likely.
* **Categorical**: Single trial with multiple possible outcomes, each with its own probability.
* **Multinomial**: Fixed number of trials with multiple possible outcomes, counting occurrences of each outcome.

We learned the scenarios each distribution models, their parameters, PMFs, means, and variances. Critically, we saw how to leverage `scipy.stats` functions (`pmf`, `cdf`, `rvs`, `mean`, `var`, `std`, `sf`) to perform calculations, generate simulations, and visualize these distributions. We also discussed important relationships, such as:
- Bernoulli ↔ Binomial ↔ Categorical ↔ Multinomial (generalizations)
- Discrete Uniform as a special case of Categorical
- Poisson approximation to Binomial
- Binomial approximation to Hypergeometric

Mastering these distributions provides a powerful toolkit for modeling various random phenomena encountered in data analysis, science, engineering, and business. In the next chapters, we will transition to continuous random variables and their corresponding common distributions.

**Decision Tree: Choosing the Right Distribution**

Use this decision tree to help identify which distribution fits your scenario:

```{mermaid}
graph TD
    Start{What are you<br/>modeling?}

    Start -->|Single trial| Single{How many<br/>outcomes?}
    Start -->|Multiple trials| Multi{Fixed or<br/>variable trials?}
    Start -->|Events in interval| IntervalQ{Constant<br/>average rate?}

    Single -->|Only 2| Bernoulli[Bernoulli]
    Single -->|More than 2| MultiOut{All outcomes<br/>equally likely?}

    MultiOut -->|Yes| Uniform[Discrete Uniform]
    MultiOut -->|No| Categorical[Categorical]

    Multi -->|Fixed number n| Fixed{How many<br/>outcomes per trial?}
    Multi -->|Variable waiting| Waiting{Waiting for<br/>which success?}

    Fixed -->|Only 2| TwoOut{Sampling with or<br/>without replacement?}
    Fixed -->|More than 2| Multinomial[Multinomial]

    TwoOut -->|With replacement<br/>or infinite pop| Binomial[Binomial]
    TwoOut -->|Without replacement<br/>finite pop| Hypergeometric[Hypergeometric]

    Waiting -->|First success| Geometric[Geometric]
    Waiting -->|r-th success r>1| NegBinom[Negative Binomial]

    IntervalQ -->|Yes| PoissonDist[Poisson]
    IntervalQ -->|Need other| Other[See Exploring<br/>Additional Distributions]

    style Bernoulli fill:#e1f5ff
    style Binomial fill:#e1f5ff
    style Geometric fill:#e1f5ff
    style NegBinom fill:#e1f5ff
    style PoissonDist fill:#e1f5ff
    style Hypergeometric fill:#e1f5ff
    style Uniform fill:#ffe1f5
    style Categorical fill:#ffe1f5
    style Multinomial fill:#ffe1f5
```

**Key Questions to Ask:**

1. **How many trials?** Single → Bernoulli/Categorical/Discrete Uniform. Fixed number → Binomial/Multinomial/Hypergeometric. Variable → Geometric/Negative Binomial.

2. **How many outcomes per trial?** Two → Bernoulli/Binomial/Geometric/Negative Binomial. More than two → Categorical/Multinomial/Discrete Uniform.

3. **With or without replacement?** With replacement (or infinite population) → Binomial. Without replacement (finite population) → Hypergeometric.

4. **What are you counting?** Successes in fixed trials → Binomial/Multinomial. Trials until success → Geometric/Negative Binomial. Events in interval → Poisson.

5. **Are probabilities equal?** Yes → Discrete Uniform. No → Categorical.

**Example Applications:**

- "Flip a coin once" → Bernoulli (single trial, 2 outcomes)
- "Flip a coin 10 times, count heads" → Binomial (fixed trials, 2 outcomes, with replacement)
- "Roll a die until you get a 6" → Geometric (variable trials, waiting for first success)
- "Draw 5 cards from a deck, count hearts" → Hypergeometric (fixed trials, 2 outcomes, without replacement)
- "Count customers arriving per hour" → Poisson (events in interval)
- "Roll a die once" → Discrete Uniform (single trial, 6 equally likely outcomes)
- "Traffic light color when you arrive" → Categorical (single trial, 3 outcomes with different probabilities)
- "Roll a die 20 times, count each face" → Multinomial (fixed trials, 6 outcomes)

## Exploring Additional Distributions

While this chapter covers nine fundamental discrete distributions, many other distributions exist for specialized scenarios. Here's how to learn about distributions beyond this chapter:

**How to Approach Learning a New Distribution:**

When you encounter a new distribution, follow these steps:

1. **Understand the Scenario**: What real-world process does it model? What makes it different from distributions you already know?

2. **Identify the Parameters**: What values define the distribution? (like $n$ and $p$ for Binomial, $\lambda$ for Poisson)

3. **Study the PMF (or PDF for continuous)**: How are probabilities calculated? What's the formula?
   - PMF = Probability Mass Function (discrete distributions, like those in this chapter)
   - PDF = Probability Density Function (continuous distributions, covered in Chapters 8-9)

4. **Learn Key Properties**: What are the mean and variance? Are there special characteristics?

5. **Explore Relationships**: How does it relate to distributions you already know? Is it a special case or generalization of something familiar?

6. **See Examples**: Find concrete examples and visualizations to build intuition.

7. **Practice with Code**: Use `scipy.stats` or similar libraries to work with the distribution hands-on.

**Key Resources for Learning About Other Distributions:**

1. **Wikipedia** - Each distribution has a comprehensive article with a standardized format:
   - Definition and scenario
   - Parameters and support (possible values)
   - PMF formula (discrete) or PDF formula (continuous)
   - Mean, variance, and other properties
   - Relationships to other distributions
   - Examples and applications
   - Search for: "[Distribution name] distribution" (e.g., "Beta-Binomial distribution")

2. **SciPy Documentation** - Python's `scipy.stats` module includes 100+ distributions:
   - Complete reference: https://docs.scipy.org/doc/scipy/reference/stats.html
   - Each distribution has: PMF (discrete) or PDF (continuous), CDF, mean, variance, random sampling
   - Includes code examples showing how to use each distribution
   - Discrete distributions: `bernoulli`, `binom`, `geom`, `hypergeom`, `poisson`, `nbinom`, `randint`, and many more

3. **Interactive Distribution Explorers**:
   - Search for "distribution explorer" or "probability distribution visualizer"
   - These tools let you adjust parameters and see how distributions change
   - Helps build intuition about distribution behavior

4. **Classic Textbooks**:
   - *Introduction to Probability* by Bertsekas & Tsitsiklis
   - *A First Course in Probability* by Sheldon Ross
   - *Probability and Statistics* by DeGroot & Schervish
   - These provide rigorous treatment with proofs and derivations

5. **Online Resources**:
   - **NIST Engineering Statistics Handbook**: Comprehensive reference for common distributions
   - **Wolfram MathWorld**: Mathematical encyclopedia with detailed distribution information
   - **Stack Exchange (Cross Validated)**: Q&A site for statistics questions

**Examples of Other Discrete Distributions:**

Here are some distributions you might encounter that we didn't cover in detail:

- **Beta-Binomial**: Like Binomial, but the success probability $p$ itself is random (varies from trial to trial)
- **Logarithmic Distribution**: Used in ecology and information theory
- **Zipf Distribution**: Models frequency of words, website visits (follows power law)
- **Zero-Inflated Poisson**: Poisson with extra zeros, common in count data
- **Conway-Maxwell-Poisson**: Generalization of Poisson with extra dispersion parameter
- **Benford's Law**: Distribution of leading digits in real-world datasets

**Finding the Right Distribution:**

If you have data or a scenario and need to find which distribution fits:

1. **Identify the process**: Single trial? Fixed trials? Waiting time? Events in interval?

2. **Check the support**: What values can the random variable take? (e.g., 0/1, non-negative integers, finite range)

3. **Consider the parameters**: What aspects of the process can vary? (success probability, rate, sample size, etc.)

4. **Use the decision tree** (see below) to narrow down candidates

5. **Test candidate distributions** using visualizations and goodness-of-fit tests

6. **Consult domain literature**: See what distributions are commonly used in your field

:::{admonition} Key Takeaway
:class: tip

Understanding the underlying probabilistic structure is more important than memorizing formulas. Focus on building intuition about when and why to use each distribution!
:::

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
