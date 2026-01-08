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
  - file: notebooks/chapter_10.ipynb
---

# Chapter 10: Mastering scipy.stats in Practice

In Chapters 6-9, we explored random variables and common probability distributions, building intuition for when to use each distribution and why. We used basic `scipy.stats` methods like `.pmf()`, `.cdf()`, `.mean()`, and `.var()` to perform calculations. But `scipy.stats` offers a much richer toolkit for working with distributions.

This chapter serves as a **practical capstone** for Part 3, teaching you how to master the full `scipy.stats` API so you can work confidently with any probability distribution. Our goal is ambitious but achievable:

> **After this chapter, the [scipy.stats documentation](https://docs.scipy.org/doc/scipy/reference/stats.html) will be all you need to work with any distribution—both those covered in this book and the 80+ others available in scipy.**

:::{admonition} Learning Objectives
:class: tip

By the end of this chapter, you will be able to:
- Use the complete `scipy.stats` interface for any distribution
- Translate real-world questions into distribution queries
- Find quantiles and interpret percentiles
- Compare distributions side-by-side
- Validate understanding through simulation
- Navigate scipy.stats documentation independently
:::

```{code-cell} ipython3
:tags: [remove-output]

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Configure plots
plt.style.use('seaborn-v0_8-whitegrid')
```

## 1. The Unified scipy.stats Interface

One of scipy.stats' greatest strengths is its **consistent API**. Whether you're working with a Bernoulli, Poisson, Normal, or Gamma distribution, the methods work the same way.

### The Pattern: Frozen Distributions

The recommended approach is to create a **frozen distribution object** with fixed parameters, then query it:

```{code-cell} ipython3
# Create frozen distribution objects
binomial_dist = stats.binom(n=20, p=0.3)     # Discrete
poisson_dist = stats.poisson(mu=4)            # Discrete
normal_dist = stats.norm(loc=100, scale=15)   # Continuous
exponential_dist = stats.expon(scale=2)       # Continuous

print("Created 4 frozen distributions")
print(f"Binomial(n=20, p=0.3), Poisson(μ=4), Normal(μ=100, σ=15), Exponential(scale=2)")
```

:::{admonition} Why "Frozen" Distributions?
:class: note

The term "frozen" means the parameters are fixed when you create the object. This makes code cleaner:

```python
# Frozen (recommended - cleaner code)
dist = stats.poisson(mu=4)
dist.pmf(3)
dist.mean()

# Unfrozen (also works, but repetitive)
stats.poisson.pmf(3, mu=4)
stats.poisson.mean(mu=4)
```
:::

### Complete API Reference

Here's the full toolkit available for any `scipy.stats` distribution:

| Method | Purpose | Works On | Returns | Example |
|--------|---------|----------|---------|---------|
| **Probabilities** | | | | |
| `.pmf(k)` | P(X = k) | Discrete | Probability | `poisson_dist.pmf(3)` |
| `.pdf(x)` | Density at x | Continuous | Density | `normal_dist.pdf(110)` |
| `.cdf(x)` | P(X ≤ x) | Both | Cumulative prob | `binomial_dist.cdf(8)` |
| `.sf(x)` | P(X > x) = 1 - CDF | Both | Survival prob | `exponential_dist.sf(3)` |
| `.logpmf(k)` | log(P(X = k)) | Discrete | Log probability | `poisson_dist.logpmf(10)` |
| `.logpdf(x)` | log(density) | Continuous | Log density | `normal_dist.logpdf(110)` |
| `.logcdf(x)` | log(P(X ≤ x)) | Both | Log cumulative | `binomial_dist.logcdf(8)` |
| `.logsf(x)` | log(P(X > x)) | Both | Log survival | `exponential_dist.logsf(3)` |
| **Quantiles (Inverse CDF)** | | | | |
| `.ppf(q)` | Percent point function | Both | Value at quantile q | `normal_dist.ppf(0.95)` |
| `.isf(q)` | Inverse survival function | Both | Value where P(X>x)=q | `exponential_dist.isf(0.1)` |
| **Properties** | | | | |
| `.mean()` | E[X] | Both | Mean | `poisson_dist.mean()` |
| `.median()` | 50th percentile | Both | Median | `binomial_dist.median()` |
| `.var()` | Var(X) | Both | Variance | `normal_dist.var()` |
| `.std()` | σ | Both | Standard deviation | `binomial_dist.std()` |
| `.stats(moments)` | Multiple moments | Both | Tuple | `poisson_dist.stats(moments='mvsk')` |
| **Simulation** | | | | |
| `.rvs(size)` | Random samples | Both | Array of samples | `normal_dist.rvs(1000)` |
| **Intervals** | | | | |
| `.interval(alpha)` | Confidence interval | Both | (lower, upper) | `normal_dist.interval(0.95)` |

### Example: Exploring Poisson(μ=4) with the Full API

```{code-cell} ipython3
dist = stats.poisson(mu=4)

print("="*60)
print("EXPLORING POISSON(μ=4) WITH THE FULL scipy.stats API")
print("="*60)

# Properties
print("\n1. PROPERTIES:")
print(f"   Mean:     {dist.mean():.4f}")
print(f"   Median:   {dist.median():.4f}")
print(f"   Variance: {dist.var():.4f}")
print(f"   Std Dev:  {dist.std():.4f}")

# Get all moments at once
m, v, s, k = dist.stats(moments='mvsk')
print(f"\n   Using .stats(moments='mvsk'):")
print(f"   Skewness (s): {s:.4f} (positive = right tail)")
print(f"   Kurtosis (k): {k:.4f} (positive = heavier tails)")

# Probabilities
print("\n2. PROBABILITIES:")
print(f"   P(X = 4):     {dist.pmf(4):.4f}")
print(f"   P(X ≤ 6):     {dist.cdf(6):.4f}")
print(f"   P(X > 6):     {dist.sf(6):.4f}")
print(f"   Check: cdf + sf = {dist.cdf(6) + dist.sf(6):.4f}")

# Quantiles
print("\n3. QUANTILES (Inverse CDF):")
print(f"   50th percentile (median): {dist.ppf(0.50):.0f}")
print(f"   75th percentile:          {dist.ppf(0.75):.0f}")
print(f"   90th percentile:          {dist.ppf(0.90):.0f}")
print(f"   95th percentile:          {dist.ppf(0.95):.0f}")

# Confidence intervals
lower, upper = dist.interval(0.90)
print("\n4. CONFIDENCE INTERVALS:")
print(f"   90% interval: [{lower:.0f}, {upper:.0f}]")
print(f"   Meaning: P({lower:.0f} ≤ X ≤ {upper:.0f}) ≈ 0.90")

# Simulation
samples = dist.rvs(size=10000, random_state=42)
print("\n5. SIMULATION:")
print(f"   Generated 10,000 samples")
print(f"   Sample mean: {samples.mean():.4f} vs theoretical {dist.mean():.4f}")
print("="*60)
```

## 2. Understanding Quantiles and the PPF

The **percent point function** (`.ppf()`) is one of the most useful but initially confusing methods.

### What is PPF?

The PPF is the **inverse of the CDF**:

$$\text{ppf}(q) = \text{CDF}^{-1}(q) = \text{smallest } x \text{ where } P(X \le x) \ge q$$

**In plain English:** "What value puts me at the q-th quantile?"

```{code-cell} ipython3
# Example: Poisson(μ=5)
dist = stats.poisson(mu=5)

# Forward: value → probability
k_value = 7
prob = dist.cdf(k_value)
print(f"CDF: Given k={k_value}, probability P(X ≤ {k_value}) = {prob:.4f}")

# Inverse: probability → value
q = 0.867
k_inverse = dist.ppf(q)
print(f"PPF: Given probability q={q:.4f}, value k = {k_inverse:.0f}")
print(f"\nThey are inverses! CDF({k_value}) ≈ {prob:.4f}, PPF({prob:.4f}) = {k_value}")
```

:::{admonition} Discrete Distributions and PPF
:class: warning

For discrete distributions, `.ppf(q)` returns the **smallest integer k where CDF(k) ≥ q**.

This can create "jumps":
```python
dist = stats.poisson(mu=4)
dist.ppf(0.60)  # Returns 4
dist.ppf(0.70)  # Also returns 4
dist.ppf(0.78)  # Also returns 4
dist.ppf(0.79)  # Returns 5 (jump!)
```

This is correct behavior - it reflects the discrete nature.
:::

### Practical Applications of PPF

**Use Case 1: Setting Thresholds**

```{code-cell} ipython3
# Customer service: calls per hour ~ Poisson(μ=15)
calls_dist = stats.poisson(mu=15)

threshold_90 = calls_dist.ppf(0.90)
threshold_95 = calls_dist.ppf(0.95)

print("Customer Calls per Hour ~ Poisson(μ=15)")
print(f"\nStaffing for 90% of hours: {threshold_90:.0f} calls")
print(f"  Verification: P(X ≤ {threshold_90:.0f}) = {calls_dist.cdf(threshold_90):.4f}")
print(f"\nStaffing for 95% of hours: {threshold_95:.0f} calls")
print(f"  Verification: P(X ≤ {threshold_95:.0f}) = {calls_dist.cdf(threshold_95):.4f}")
```

**Use Case 2: Risk Analysis**

```{code-cell} ipython3
# Defects per batch ~ Poisson(μ=2.5)
defect_dist = stats.poisson(mu=2.5)

worst_case_99 = defect_dist.ppf(0.99)
worst_case_999 = defect_dist.ppf(0.999)

print(f"Defects per Batch ~ Poisson(μ=2.5)")
print(f"\nRisk Analysis:")
print(f"  99th percentile (1 in 100):    {worst_case_99:.0f} defects")
print(f"  99.9th percentile (1 in 1000): {worst_case_999:.0f} defects")
print(f"\nPlan for {worst_case_99:.0f} defects to handle 99% of batches")
```

## 3. Comparing Distributions

One powerful application is comparing distributions side-by-side.

### Example: When Does Poisson Approximate Binomial?

```{code-cell} ipython3
# Compare Binomial and Poisson approximation
scenarios = [
    (20, 0.05, "Good"),
    (100, 0.03, "Excellent"),
    (20, 0.5, "Poor"),
]

print("="*70)
print("COMPARING BINOMIAL AND POISSON APPROXIMATION")
print("="*70)

for n, p, quality in scenarios:
    lam = n * p
    binom_dist = stats.binom(n=n, p=p)
    poisson_dist = stats.poisson(mu=lam)

    print(f"\nn={n}, p={p}, λ={lam} ({quality} approximation expected):")
    print(f"  Binomial mean={binom_dist.mean():.4f}, var={binom_dist.var():.4f}")
    print(f"  Poisson  mean={poisson_dist.mean():.4f}, var={poisson_dist.var():.4f}")

    # Compare probabilities at mode
    mode_k = int(lam)
    binom_prob = binom_dist.pmf(mode_k)
    poisson_prob = poisson_dist.pmf(mode_k)
    print(f"  P(X={mode_k}): Binomial={binom_prob:.6f}, Poisson={poisson_prob:.6f}")
    print(f"  Difference: {abs(binom_prob - poisson_prob):.6f}")
```

**Rule validated:** Poisson approximates Binomial well when n ≥ 20 and p ≤ 0.05.

## 4. Simulation and Validation

The `.rvs()` method generates samples for validation.

```{code-cell} ipython3
# Example: Binomial(n=50, p=0.3)
true_dist = stats.binom(n=50, p=0.3)
np.random.seed(42)
samples = true_dist.rvs(size=10000)

print("="*70)
print("SIMULATION VALIDATION: Binomial(n=50, p=0.3)")
print("="*70)

print("\nTHEORETICAL vs EMPIRICAL:")
print(f"  Mean:     {true_dist.mean():.4f} vs {samples.mean():.4f}")
print(f"  Variance: {true_dist.var():.4f} vs {samples.var():.4f}")
print(f"  Std Dev:  {true_dist.std():.4f} vs {samples.std():.4f}")

print("\nQUANTILE COMPARISON:")
for q in [0.25, 0.50, 0.75, 0.90]:
    theoretical = true_dist.ppf(q)
    empirical = np.percentile(samples, q*100)
    print(f"  {q:.2f}: {theoretical:5.1f} vs {empirical:5.1f}")

print("="*70)
```

With 10,000 samples, empirical closely matches theoretical!

## 5. Reading the scipy.stats Documentation

:::{admonition} Documentation Structure
:class: tip

Every scipy.stats distribution page follows the same structure:

1. **Function signature** - Shows parameters
2. **Parameters section** - Describes each parameter
3. **Notes** - Mathematical definition and properties
4. **Methods** - Complete list of available methods
5. **Examples** - Copy-pasteable code

**Example:** [scipy.stats.poisson documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html)
:::

### Exercise: Learn the Geometric Distribution from Docs

Using only the [scipy.stats.geom docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geom.html), answer:
1. What parameter does it take?
2. What does it model?
3. What's the mean?
4. What's P(X = 5) if p = 0.2?
5. What's the 75th percentile?

```{code-cell} ipython3
# Solution using scipy.stats documentation
geom_dist = stats.geom(p=0.2)

print("="*70)
print("LEARNING GEOMETRIC DISTRIBUTION FROM SCIPY DOCS")
print("="*70)

print("\n1. Parameter: p = 0.2 (probability of success)")
print("2. Models: Number of trials until first success")
print(f"3. Mean (from docs: 1/p): {geom_dist.mean():.4f} = {1/0.2:.4f} ✓")
print(f"4. P(X = 5): {geom_dist.pmf(5):.6f}")

p75 = geom_dist.ppf(0.75)
print(f"5. 75th percentile: {p75:.0f} trials")

# Validate with simulation
samples = geom_dist.rvs(size=10000, random_state=42)
print(f"\nValidation (10,000 samples):")
print(f"  Empirical mean: {samples.mean():.4f} vs theory {geom_dist.mean():.4f}")
print("="*70)
```

**You just learned a distribution independently!**

## 6. Practical Workflows

### Workflow 1: Quality Control Decision

**Scenario:** Factory produces batches of 100 items. 2% are defective. Reject batch if > 5 defective. What's rejection probability?

```{code-cell} ipython3
print("="*70)
print("QUALITY CONTROL WORKFLOW")
print("="*70)

# Step 1: Model selection
defect_dist = stats.binom(n=100, p=0.02)
print("\nModel: Binomial(n=100, p=0.02)")
print(f"Expected defectives: {defect_dist.mean():.2f}")

# Step 2: Answer question
prob_reject = defect_dist.sf(5)  # P(X > 5)
print(f"\nP(X > 5) = {prob_reject:.6f}")
print(f"→ {prob_reject*100:.3f}% of batches will be rejected")

# Step 3: Sensitivity analysis
print("\nWhat if threshold was 3?")
prob_reject_3 = defect_dist.sf(3)
print(f"  Rejection rate: {prob_reject_3*100:.3f}%")

print("="*70)
```

### Workflow 2: Inventory Planning

**Scenario:** Daily demand ~ Poisson(μ=7). Stock enough for 95% of days. How much?

```{code-cell} ipython3
demand_dist = stats.poisson(mu=7)

stock_95 = demand_dist.ppf(0.95)
stock_99 = demand_dist.ppf(0.99)

print("="*70)
print("INVENTORY PLANNING")
print("="*70)

print(f"\nDaily Demand ~ Poisson(μ=7)")
print(f"\nFor 95% service: Stock {stock_95:.0f} units")
print(f"  Verification: P(Demand ≤ {stock_95:.0f}) = {demand_dist.cdf(stock_95):.4f}")

print(f"\nFor 99% service: Stock {stock_99:.0f} units")
print(f"  Verification: P(Demand ≤ {stock_99:.0f}) = {demand_dist.cdf(stock_99):.4f}")

print("="*70)
```

## 7. Advanced Topics (Preview)

### Log Probabilities for Numerical Stability

When working with very small probabilities, use log methods:

```{code-cell} ipython3
rare_dist = stats.poisson(mu=2)
k_large = 20

# Regular probability (may underflow)
regular_prob = rare_dist.pmf(k_large)

# Log probability (numerically stable)
log_prob = rare_dist.logpmf(k_large)

print("="*60)
print("NUMERICAL STABILITY WITH LOG PROBABILITIES")
print("="*60)

print(f"\nP(X = {k_large}) for Poisson(μ=2):")
print(f"  Regular .pmf({k_large}):   {regular_prob}")
print(f"  Log .logpmf({k_large}): {log_prob:.4f}")
print(f"  Recover: exp(log_prob) = {np.exp(log_prob)}")

print("\nWhen to use log methods:")
print("  - Very small probabilities (< 1e-10)")
print("  - Products of many probabilities")
print("  - Maximum likelihood estimation")

print("="*60)
```

### Parameter Estimation (Brief Preview)

```{code-cell} ipython3
# Observed data from unknown Poisson process
observed_data = np.array([3, 5, 4, 6, 3, 5, 4, 7, 2, 5, 4, 6, 5, 3, 4])

# For Poisson, MLE is simply the sample mean
mu_hat = observed_data.mean()

fitted_dist = stats.poisson(mu=mu_hat)

print("="*70)
print("PARAMETER ESTIMATION (PREVIEW)")
print("="*70)

print(f"\nObserved data: {observed_data}")
print(f"Estimated μ: {mu_hat:.4f}")
print(f"\nFitted distribution: Poisson(μ={mu_hat:.4f})")
print(f"  Mean: {fitted_dist.mean():.4f}")
print(f"  Variance: {fitted_dist.var():.4f}")

print("\nNote: Formal parameter estimation is covered in statistics courses.")
print("scipy.stats supports this through methods like .fit()")
print("="*70)
```

## 8. Summary and Next Steps

### What You've Learned

**Core Skills:**
- ✅ The unified scipy.stats API pattern (works for all 80+ distributions)
- ✅ Calculating probabilities with `.pmf()`, `.pdf()`, `.cdf()`, `.sf()`
- ✅ Finding quantiles and percentiles with `.ppf()`
- ✅ Querying properties with `.mean()`, `.median()`, `.var()`, `.std()`, `.stats()`
- ✅ Generating samples with `.rvs()` for simulation
- ✅ Comparing distributions visually and numerically

**Practical Workflows:**
- ✅ Translating real problems into distribution questions
- ✅ Setting thresholds based on confidence levels
- ✅ Risk analysis with quantiles
- ✅ Navigating scipy.stats documentation independently

### The scipy.stats Documentation is Now Your Resource

You can now:
1. Pick any distribution from the [scipy.stats list](https://docs.scipy.org/doc/scipy/reference/stats.html)
2. Read its documentation page
3. Understand the parameters, methods, and examples
4. Apply it to your problems confidently

:::{admonition} The Power of the Unified API
:class: important

The scipy.stats interface is **consistent across all distributions**. This pattern works for ANY distribution:

```python
dist = stats.DISTRIBUTION_NAME(params)

# Query properties
dist.mean(), dist.median(), dist.var(), dist.std()

# Calculate probabilities
dist.pmf(k) or dist.pdf(x)  # Point
dist.cdf(x)                  # Cumulative
dist.sf(x)                   # Survival

# Find quantiles
dist.ppf(q)                  # Inverse CDF
dist.interval(alpha)         # Confidence interval

# Generate samples
dist.rvs(size=n)             # Random variates
```

Learn this pattern → Work with ANY distribution!
:::

### Practice Exercises

1. **Distribution Comparison:** Compare Binomial(n=20, p=0.3) with Normal(μ=6, σ=2.05). How close is the normal approximation?

2. **Risk Analysis:** Website visitors ~ Poisson(μ=500). Server handles 650. What's crash probability? What capacity for <1% crash risk?

3. **Learn a New Distribution:** Pick Negative Binomial. Model: "Roll die until three 6's. Probability of exactly 20 rolls?"

4. **Simulation:** Generate 1000 samples from Exponential(λ=0.5). Compare sample mean to theoretical. Try 10,000 samples.

5. **Documentation Explorer:** Find docs for `scipy.stats.describe()`. Use it to analyze data and interpret all statistics.

---

**You're now scipy.stats-literate!** The documentation is your comprehensive reference for all future probability work. In the next chapter, we explore how multiple random variables interact (joint distributions, covariance, correlation).
