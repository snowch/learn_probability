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
  - file: notebooks/chapter_21.ipynb
---

# Chapter 21: Probability with SageMath

+++

Welcome to Chapter 21! In the previous chapter, we explored SymPy for symbolic probability computation. Now we'll discover **SageMath** (often called just "Sage"), a comprehensive free open-source mathematics software system that combines the power of many specialized libraries including NumPy, SciPy, SymPy, matplotlib, and dozens more.

SageMath is built on Python but extends it with a powerful mathematical syntax and extensive pre-built functionality for:
- Symbolic and numerical computation
- Probability and statistics
- Linear algebra and calculus
- Number theory and combinatorics
- Graph theory and cryptography
- And much more

While this entire book could have been written using SageMath, we've focused on standard Python libraries (NumPy, SciPy, SymPy) because they're more commonly used in data science and machine learning workflows. However, SageMath offers unique advantages for mathematical work and is worth knowing about.

+++

## Learning Objectives

* Understand what SageMath is and how it differs from standard Python
* Set up and access SageMath (locally or via CoCalc)
* Use SageMath for combinatorics and exact probability calculations
* Work with probability distributions in SageMath
* Leverage SageMath's symbolic capabilities for probability theory
* Compare SageMath with NumPy/SciPy/SymPy approaches
* Decide when to use SageMath vs standard Python libraries

+++

## What is SageMath?

### SageMath vs Python + Libraries

**Standard Python Approach** (what we've used in this book):
```python
import numpy as np
import scipy.stats as stats
import sympy as sp
import matplotlib.pyplot as plt
```

**SageMath Approach**:
- All major mathematical libraries are pre-integrated
- Enhanced syntax for mathematical operations
- Built-in support for exact arithmetic
- Extensive mathematical functions without imports
- Interactive environment (Sage REPL or Jupyter)

**Key Philosophy**: SageMath aims to be a free open-source alternative to Mathematica, Maple, and MATLAB.

+++

## Installation and Access

### Option 1: CoCalc (Recommended for Beginners)

**CoCalc** (https://cocalc.com) is a free online platform that provides SageMath in your browser:
- No installation required
- Includes Jupyter notebooks with SageMath kernel
- Free tier available
- Collaborative features
- Perfect for learning and experimentation

### Option 2: Local Installation

**Install SageMath locally**:

**On Ubuntu/Debian:**
```bash
sudo apt-get install sagemath
```

**On macOS (using Homebrew):**
```bash
brew install --cask sagemath
```

**On Windows:**
- Download from https://www.sagemath.org/
- Use WSL (Windows Subsystem for Linux) + Linux installation
- Or use Docker

**Via Conda (unofficial):**
```bash
conda install -c conda-forge sage
```

### Option 3: Docker

```bash
docker pull sagemath/sagemath
docker run -p 8888:8888 sagemath/sagemath:latest sage-jupyter
```

### Option 4: SageMathCell

For quick one-off calculations: https://sagecell.sagemath.org/

+++

:::{admonition} Note About This Chapter
:class: warning

The code examples in this chapter require **SageMath** to run. They will not work in a standard Python/Jupyter environment.

To run these examples:
- Use CoCalc (https://cocalc.com) with a SageMath kernel
- Install SageMath locally and use a Sage notebook
- Use SageMathCell (https://sagecell.sagemath.org/) for individual examples

The examples are presented to show SageMath's capabilities and syntax.
:::

+++

## SageMath Basics for Probability

### Exact Arithmetic by Default

Unlike Python where `1/3` gives `0.333...`, SageMath provides exact arithmetic by default:

```python
# In SageMath (this would give exact result)
sage: 1/3
1/3

sage: 1/3 + 1/3 + 1/3
1

sage: sqrt(2)
sqrt(2)

sage: sqrt(2).n()  # .n() for numerical approximation
1.41421356237310

sage: pi
pi

sage: pi.n(digits=50)
3.1415926535897932384626433832795028841971693993751
```

This exact arithmetic is similar to SymPy but built into the core language.

+++

### Combinatorics in SageMath

SageMath has extensive built-in combinatorics support:

```python
# Factorials
sage: factorial(10)
3628800

sage: factorial(100)  # Handles large numbers easily
93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000

# Binomial coefficients
sage: binomial(10, 3)
120

sage: binomial(52, 5)  # Poker hands
2598960

# Permutations
sage: factorial(8) / factorial(8-3)  # P(8,3)
336

# SageMath also has Permutations class for working with permutation objects
sage: Permutations(3).list()
[[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]

sage: Permutations(3).cardinality()
6

# Combinations as mathematical objects
sage: Combinations(5, 2).list()
[[1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]

sage: Combinations(5, 2).cardinality()
10
```

+++

### Symbolic Variables and Expressions

```python
# Declare symbolic variables
sage: var('n k p')
(n, k, p)

# Binomial probability formula
sage: prob = binomial(n, k) * p^k * (1-p)^(n-k)
sage: prob
(p - 1)^(-k + n)*p^k*binomial(n, k)

# Substitute values
sage: prob.subs(n=10, k=3, p=1/2)
15/128

# Simplify
sage: expr = (n * (n-1)) / n
sage: expr.simplify()
n - 1

# Expand
sage: expand((p + (1-p))^5)
1
```

+++

## Probability Distributions in SageMath

SageMath provides several ways to work with probability distributions.

### Using SciPy Through SageMath

SageMath includes SciPy, so you can use it exactly as we have throughout this book:

```python
sage: import scipy.stats as stats
sage: stats.binom.pmf(3, 10, 0.5)
0.1171875
```

### Symbolic Probability with SageMath

```python
# Exact probability calculations
sage: var('n k p')
sage: binomial_prob = binomial(n,k) * p^k * (1-p)^(n-k)

# Calculate for specific values
sage: binomial_prob(n=5, k=2, p=1/2)
5/16

# Verify sum to 1
sage: n_val = 5
sage: sum(binomial_prob(n=n_val, k=k_val, p=1/2) for k_val in range(n_val+1))
1
```

### Discrete Distributions

```python
# Creating a discrete probability distribution
sage: def coin_flip():
....:     return 'H' if random() < 0.5 else 'T'

sage: # Simulate flips
sage: [coin_flip() for _ in range(10)]
['T', 'H', 'H', 'T', 'H', 'T', 'T', 'H', 'H', 'T']

# Using GeneralDiscreteDistribution
sage: outcomes = [1, 2, 3, 4, 5, 6]
sage: probabilities = [1/6] * 6
sage: die_dist = GeneralDiscreteDistribution(probabilities)

sage: # Sample from distribution
sage: [outcomes[die_dist.get_random_element()] for _ in range(10)]
[4, 1, 6, 2, 3, 5, 1, 6, 4, 2]
```

+++

## Advanced Combinatorics Examples

### Example 1: Poker Hand Probabilities

```python
sage: # Total 5-card poker hands
sage: total_hands = binomial(52, 5)
sage: total_hands
2598960

sage: # Royal flush: A,K,Q,J,10 of same suit
sage: royal_flush = 4  # One per suit
sage: prob_royal_flush = royal_flush / total_hands
sage: prob_royal_flush
1/649740

sage: # Full house: 3 of one rank, 2 of another
sage: full_house = binomial(13,1) * binomial(4,3) * binomial(12,1) * binomial(4,2)
sage: full_house
3744
sage: prob_full_house = full_house / total_hands
sage: prob_full_house
6/4165

sage: # Convert to decimal
sage: prob_full_house.n()
0.00144057623049925
```

+++

### Example 2: Birthday Problem

```python
sage: def birthday_probability(n):
....:     """Exact probability that at least 2 people share a birthday"""
....:     prob_all_different = 1
....:     for i in range(n):
....:         prob_all_different *= (365 - i) / 365
....:     return 1 - prob_all_different

sage: # Test various group sizes
sage: for n in [10, 20, 23, 30, 50]:
....:     print(f"n={n:2d}: {birthday_probability(n).n():.6f}")
n=10: 0.116948
n=20: 0.411438
n=23: 0.507297
n=30: 0.706316
n=50: 0.970374

sage: # Find minimum n where probability > 0.5
sage: n = 1
sage: while birthday_probability(n) < 1/2:
....:     n += 1
sage: print(f"Minimum group size for >50%: {n}")
Minimum group size for >50%: 23
```

+++

### Example 3: Multinomial Coefficients

```python
sage: # Multinomial coefficient: n! / (k1! * k2! * ... * km!)
sage: # Example: Arrangements of MISSISSIPPI (11 letters)
sage: # M:1, I:4, S:4, P:2

sage: n = 11
sage: multinomial([1, 4, 4, 2])
34650

sage: # Verify with factorial formula
sage: factorial(11) / (factorial(1) * factorial(4) * factorial(4) * factorial(2))
34650

sage: # Probability that a random arrangement spells MISSISSIPPI
sage: 1 / multinomial([1, 4, 4, 2])
1/34650
```

+++

## Symbolic Probability Theory

### Deriving Expected Value Formulas

```python
sage: # Expected value of binomial distribution
sage: var('n p k')
sage: # E(X) = sum(k * P(X=k)) for k=0 to n

sage: # Using symbolic sum (this may be slow for symbolic n)
sage: # We can verify for small concrete values
sage: n_val = 5
sage: p_val = var('p')
sage: expected = sum(k * binomial(n_val, k) * p_val^k * (1-p_val)^(n_val-k)
....:                for k in range(n_val + 1))
sage: expected.simplify()
5*p

sage: # This confirms E(X) = np for Binomial(n,p)
```

+++

### Moment Generating Functions

```python
sage: # MGF of Binomial distribution: M(t) = (pe^t + (1-p))^n
sage: var('t n p')
sage: mgf = (p * exp(t) + (1-p))^n

sage: # First derivative at t=0 gives E(X)
sage: mgf_prime = diff(mgf, t)
sage: expected_value = mgf_prime.subs(t=0).simplify()
sage: expected_value
n*p

sage: # Second derivative for variance
sage: mgf_double_prime = diff(mgf, t, 2)
sage: second_moment = mgf_double_prime.subs(t=0).simplify()
sage: variance = (second_moment - expected_value^2).simplify()
sage: variance
n*p*(p - 1)
sage: # Which simplifies to n*p*(1-p)
```

+++

## Numerical Integration and Probability

SageMath can handle both symbolic and numerical integration:

```python
sage: # Numerical integration for continuous distributions
sage: var('x')

sage: # Standard normal PDF
sage: normal_pdf = 1/sqrt(2*pi) * exp(-x^2/2)

sage: # P(Z < 1) for standard normal
sage: prob = integral(normal_pdf, x, -oo, 1)
sage: prob.n()
0.841344746068543

sage: # P(0 < Z < 1)
sage: prob_range = integral(normal_pdf, x, 0, 1)
sage: prob_range.n()
0.341344746068543

sage: # Verify using scipy for comparison
sage: import scipy.stats as stats
sage: stats.norm.cdf(1)  # Should match first calculation
0.8413447460685429
```

+++

## Plotting Distributions

SageMath has built-in plotting that's similar to matplotlib but with some enhancements:

```python
sage: # Plot binomial PMF
sage: n, p = 10, 0.5
sage: points = [(k, binomial(n,k) * p^k * (1-p)^(n-k)) for k in range(n+1)]
sage: bar_chart(points, color='blue', width=0.5)

sage: # Plot normal PDF
sage: var('x')
sage: mu, sigma = 0, 1
sage: pdf = 1/(sigma*sqrt(2*pi)) * exp(-(x-mu)^2/(2*sigma^2))
sage: plot(pdf, (x, -4, 4), color='red', thickness=2,
....:      axes_labels=['x', 'PDF'], title='Standard Normal Distribution')

sage: # Multiple distributions on same plot
sage: p1 = plot(pdf, (x, -4, 4), color='blue', legend_label='N(0,1)')
sage: pdf2 = 1/(2*sqrt(2*pi)) * exp(-(x-1)^2/(2*4))  # N(1,2)
sage: p2 = plot(pdf2, (x, -4, 6), color='red', legend_label='N(1,2)')
sage: (p1 + p2).show()
```

+++

## Simulation and Monte Carlo

```python
sage: # Monte Carlo estimation of π
sage: def estimate_pi(n):
....:     inside = 0
....:     for _ in range(n):
....:         x, y = random(), random()
....:         if x^2 + y^2 <= 1:
....:             inside += 1
....:     return 4 * inside / n

sage: estimate_pi(10000)
3.1416  # Will vary

sage: # Simulate dice rolls
sage: rolls = [randint(1,6) for _ in range(1000)]
sage: # Count frequencies
sage: {i: rolls.count(i)/1000 for i in range(1,7)}
{1: 0.163, 2: 0.171, 3: 0.166, 4: 0.168, 5: 0.162, 6: 0.170}  # Approximately uniform
```

+++

## Comparison: SageMath vs NumPy/SciPy/SymPy

### When to Use SageMath

**✅ Use SageMath when:**
- You're doing primarily mathematical/theoretical work
- You want everything pre-integrated (no import hunting)
- You need powerful symbolic computation
- You're teaching mathematics or probability theory
- You want an interactive mathematical environment
- You're working on pure math research
- You prefer Mathematica-like syntax

**❌ Don't use SageMath when:**
- Building production data science/ML systems
- Working in standard Python environments (deployment, CI/CD)
- Need to integrate with industry-standard Python data stack
- Collaborating with data scientists using NumPy/pandas/scikit-learn
- Performance is critical (NumPy/SciPy are often faster)

### Syntax Comparison

```python
# Exact probability: P(X=3) for Binomial(10, 0.5)

# NumPy/SciPy (numerical)
from scipy.stats import binom
prob = binom.pmf(3, 10, 0.5)  # 0.1171875

# SymPy (symbolic)
from sympy.stats import Binomial, P
from sympy import Rational
X = Binomial('X', 10, Rational(1,2))
prob = P(X == 3)  # 15/128

# SageMath (natural mathematical syntax)
binomial(10, 3) * (1/2)^10  # 15/128
```

+++

## Practical Example: Complete Analysis

Let's solve a problem using SageMath's full capabilities:

**Problem**: A fair coin is flipped 100 times. What's the probability of getting between 45 and 55 heads (inclusive)?

```python
sage: # Method 1: Exact calculation
sage: n = 100
sage: p = 1/2
sage: prob_exact = sum(binomial(n,k) * p^n for k in range(45, 56))
sage: prob_exact.n()
0.728747317564360

sage: # Method 2: Using scipy for comparison
sage: import scipy.stats as stats
sage: prob_scipy = stats.binom.cdf(55, n, 0.5) - stats.binom.cdf(44, n, 0.5)
sage: prob_scipy
0.7287473175643597

sage: # Method 3: Normal approximation
sage: # Binomial(100, 0.5) ≈ Normal(50, 25)
sage: mu = n * p
sage: sigma = sqrt(n * p * (1-p))
sage: sigma.n()
5.00000000000000

sage: # Using continuity correction: P(44.5 < X < 55.5)
sage: from scipy.stats import norm
sage: prob_normal = norm.cdf(55.5, mu, sigma) - norm.cdf(44.5, mu, sigma)
sage: prob_normal
0.7287181077536644

sage: print(f"Exact: {prob_exact.n():.10f}")
sage: print(f"SciPy: {prob_scipy:.10f}")
sage: print(f"Normal approx: {prob_normal:.10f}")
```

+++

## Unique SageMath Features for Probability

### 1. Built-in Graph Theory (for Markov Chains)

```python
sage: # Create a transition matrix for a Markov chain
sage: P = matrix(QQ, [[1/2, 1/2, 0],
....:                  [1/4, 1/2, 1/4],
....:                  [0, 1/2, 1/2]])

sage: # Visualize as directed graph
sage: G = DiGraph(P, format='weighted_adjacency_matrix')
sage: G.plot(edge_labels=True)

sage: # Find stationary distribution
sage: # Solve π P = π
sage: eigenspaces = P.transpose().eigenspaces_right()
sage: # Extract eigenvector for eigenvalue 1
```

### 2. Automatic Simplification

```python
sage: # SageMath often simplifies automatically
sage: var('n')
sage: expr = binomial(n, 0) + binomial(n, n)
sage: expr
2  # Automatically simplified!

sage: expr2 = factorial(n) / (factorial(n-1) * n)
sage: expr2.simplify()
1
```

### 3. LaTeX Output

```python
sage: var('n k p')
sage: formula = binomial(n,k) * p^k * (1-p)^(n-k)
sage: latex(formula)
\left(p - 1\right)^{-k + n} p^{k} \binom{n}{k}

# This is great for generating formulas for papers and presentations!
```

+++

## Integration with Jupyter

SageMath works excellently with Jupyter notebooks:

1. **CoCalc**: Built-in Jupyter with SageMath kernel
2. **Local Jupyter**: After installing SageMath, use `sage -n jupyter`
3. **SageMath kernel**: Select "SageMath" kernel in Jupyter

**Advantages**:
- Mix SageMath code with markdown explanations
- Export to PDF, HTML
- Share notebooks easily
- Use LaTeX for mathematical notation

+++

## Limitations and Considerations

### Installation Complexity
- Not a simple `pip install sage`
- Larger download than individual libraries
- Platform-specific installation issues

### Ecosystem Compatibility
- Not as integrated with modern ML/data science tools
- Can't easily mix SageMath and pandas/sklearn in production
- Deployment is complex

### Performance
- NumPy/SciPy can be faster for numerical operations
- SageMath includes overhead from multiple systems

### Community Size
- Smaller community than NumPy/SciPy/pandas
- Fewer Stack Overflow answers
- Less industry adoption

+++

## Summary

**SageMath** is a powerful, comprehensive mathematical software system built on Python that offers:

**Strengths:**
- ✅ All-in-one mathematical environment
- ✅ Exact arithmetic by default
- ✅ Extensive symbolic capabilities
- ✅ Rich combinatorics and number theory support
- ✅ Excellent for teaching and research
- ✅ Free and open-source alternative to Mathematica/Maple

**Trade-offs:**
- ❌ More complex installation
- ❌ Less integrated with industry data science stack
- ❌ Smaller ecosystem than NumPy/SciPy
- ❌ Can be slower for pure numerical work

**Recommendation:**
- **For this book's audience** (data science/ML practitioners): Stick with NumPy/SciPy/SymPy
- **For mathematics students/researchers**: SageMath is excellent
- **For teaching probability theory**: SageMath offers great pedagogical benefits
- **For production systems**: Use standard Python stack

**Best of both worlds**: Learn the concepts with SageMath, implement production code with NumPy/SciPy!

+++

## Exercises

1. **Installation**: Set up SageMath using your preferred method (CoCalc, local, or Docker) and verify it works.

2. **Exact Probabilities**: Calculate the exact probability of getting exactly 5 heads in 12 coin flips. Express as both a fraction and decimal.

3. **Poker Probability**: Calculate the exact probability of getting "four of a kind" in 5-card poker using SageMath combinatorics.

4. **Symbolic Variance**: Derive the variance formula for a geometric distribution using symbolic computation in SageMath.

5. **Birthday Problem Extension**: Find the minimum number of people needed for a >90% chance of a shared birthday.

6. **Markov Chain**: Create a simple 3-state Markov chain transition matrix and find its stationary distribution using SageMath's linear algebra capabilities.

7. **Comparison**: Solve the same binomial probability problem using SciPy (numerical), SymPy (symbolic), and SageMath. Compare the results and execution times.

8. **Monte Carlo**: Implement a Monte Carlo simulation in SageMath to estimate the probability that the sum of two dice is greater than 8.

+++

## Further Reading

- **SageMath Official Website**: https://www.sagemath.org/
- **SageMath Documentation**: https://doc.sagemath.org/
- **SageMath Tutorial**: https://doc.sagemath.org/html/en/tutorial/
- **CoCalc**: https://cocalc.com/
- **SageMath for Combinatorics**: https://doc.sagemath.org/html/en/reference/combinat/
- **Computational Mathematics with SageMath** (book): Free online at http://sagebook.gforge.inria.fr/

+++

## Conclusion

You've now completed the journey through probability in practice with Python! From basic probability concepts to Monte Carlo methods, Markov chains, and now symbolic computation with SymPy and SageMath, you have a comprehensive toolkit for tackling probability problems.

**Key Takeaways from This Book:**
- **NumPy/SciPy**: Your workhorse for numerical probability and statistics
- **matplotlib/seaborn**: Visualization of distributions and results
- **SymPy** (Chapter 20): Exact symbolic computation and formula derivation
- **SageMath** (Chapter 21): Comprehensive mathematical system for theoretical work

**Next Steps:**
- Apply these tools to real-world problems in your field
- Explore advanced topics like stochastic processes, time series, or Bayesian statistics
- Contribute to open-source probability/statistics libraries
- Share your knowledge by teaching others!

Thank you for joining this hands-on journey through probability with Python. May your p-values be significant and your confidence intervals narrow! 🎲📊🐍
