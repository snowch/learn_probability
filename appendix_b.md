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
  - file: notebooks/appendix_b.ipynb
---

# Appendix B: Essential Library Reference

This appendix provides a quick reference for the most commonly used functions and concepts from the core Python libraries leveraged throughout this book. It is not exhaustive but covers the essentials for probability simulation, calculation, and visualization.

## 1. Python Standard Libraries

### `math`

- `math.factorial(n)`: Computes $n!$
- `math.comb(n, k)`: Computes binomial coefficient $\binom{n}{k}$ ('n choose k'). Requires Python 3.8+.
- `math.perm(n, k)`: Computes permutations $P(n, k) = \frac{n!}{(n-k)!}$. Requires Python 3.8+.
- `math.exp(x)`: Computes $e^x$.
- `math.log(x, [base])`: Computes the logarithm of $x$. Natural logarithm ($\ln x$) if base is omitted, otherwise $\log_{\text{base}} x$.
- `math.sqrt(x)`: Computes the square root $\sqrt{x}$.

## 2. NumPy (`import numpy as np`)

### Array Creation

- `np.array([list])`: Create a NumPy array from a Python list.
- `np.arange(start, stop, step)`: Create an array with evenly spaced values within a given interval (stop is exclusive).
- `np.linspace(start, stop, num)`: Create an array with `num` evenly spaced values between `start` and `stop` (inclusive).
- `np.zeros(shape)`, `np.ones(shape)`: Create arrays of a given shape filled with 0s or 1s.
- `np.eye(N)`: Create an $N \times N$ identity matrix.

### Random Sampling (`np.random`)

- `np.random.seed(integer)`: Set the random seed for reproducibility.
- `np.random.rand(d0, d1, ...)`: Generate random floats uniformly distributed over $[0, 1)$.
- `np.random.randn(d0, d1, ...)`: Generate random floats from the standard normal distribution ($N(0, 1)$).
- `np.random.randint(low, high, size)`: Generate random integers from `low` (inclusive) to `high` (exclusive).
- `np.random.choice(a, size, replace=True, p=None)`: Generate a random sample from a given 1-D array `a`. `replace` controls sampling with/without replacement. `p` allows specifying probabilities for each element in `a`.

### Array Operations & Math

- Standard arithmetic operators (`+`, `-`, `*`, `/`, `**`) operate element-wise.
- `np.sum(a)`, `np.mean(a)`, `np.std(a)`, `np.var(a)`: Calculate sum, mean, standard deviation ($\sigma$), and variance ($\sigma^2$) of array elements.
- `np.min(a)`, `np.max(a)`: Find minimum and maximum values.
- `np.argmin(a)`, `np.argmax(a)`: Find the indices of the minimum and maximum values.
- `np.sqrt(a)`, `np.exp(a)`, `np.log(a)`, `np.sin(a)`, etc.: Element-wise mathematical functions.
- `np.dot(a, b)` or `a @ b`: Matrix multiplication / dot product.
- `a.T`: Transpose of array `a`.

### Indexing and Slicing

- Standard Python slicing `a[start:stop:step]` works for each dimension.
- Boolean Indexing: `a[boolean_array]` or `a[a > 5]` selects elements where the condition is True.
- Fancy Indexing: `a[[1, 4, 0]]` selects specific rows/elements using a list of indices.

## 3. SciPy (`import scipy`)

### Special Functions (`scipy.special`)

- `scipy.special.perm(N, k)`: Computes permutations $P(N, k)$.
- `scipy.special.comb(N, k)`: Computes combinations $C(N, k) = \binom{N}{k}$.
- `scipy.special.gamma(z)`: Gamma function $\Gamma(z)$.
- `scipy.special.gammaln(z)`: Log of the absolute value of the Gamma function, $\ln|\Gamma(z)|$.

### Integration (`scipy.integrate`)

- `scipy.integrate.quad(func, a, b)`: Computes the definite integral $\int_a^b f(x) dx$. Returns the integral result and an estimated error.

### Statistics (`scipy.stats`)

Provides distribution objects (e.g., `norm`, `binom`, `poisson`) with common methods:

- `dist.rvs(...)`: Generate Random VariateS (samples).
- `dist.pmf(k, ...)`: Probability Mass Function $P(X=k)$ (for discrete distributions).
- `dist.pdf(x, ...)`: Probability Density Function $f(x)$ (for continuous distributions).
- `dist.cdf(x, ...)`: Cumulative Distribution Function $F(x) = P(X \le x)$.
- `dist.ppf(q, ...)`: Percent Point Function (inverse CDF or quantile function). Finds $x$ such that $F(x) = q$.
- `dist.sf(x, ...)`: Survival Function $S(x) = 1 - F(x) = P(X > x)$.
- `dist.isf(q, ...)`: Inverse Survival Function. Finds $x$ such that $S(x) = q$.
- `dist.stats(moments='mvsk')`: Computes Mean ('m'), Variance ('v'), Skewness ('s'), Kurtosis ('k').
- `dist.mean(...)`, `dist.median(...)`, `dist.var(...)`, `dist.std(...)`: Compute specific moments/statistics.
- `dist.interval(alpha, ...)`: Computes an interval containing `alpha` proportion of the probability density (e.g., `alpha=0.95` for a 95% interval).

Common Distributions (parameters might vary slightly from textbook definitions, check documentation):

- `scipy.stats.bernoulli(p)`: Bernoulli (probability of success `p`).
- `scipy.stats.binom(n, p)`: Binomial (number of trials `n`, probability of success `p`).
- `scipy.stats.geom(p)`: Geometric (probability of success `p`).
- `scipy.stats.nbinom(n, p)`: Negative Binomial (number of successes `n`, probability of success `p`).
- `scipy.stats.poisson(mu)`: Poisson (rate `mu`, $\lambda$).
- `scipy.stats.hypergeom(M, n, N)`: Hypergeometric (M=population size, n=items with feature, N=sample size).
- `scipy.stats.uniform(loc, scale)`: Uniform on $[loc, loc+scale]$.
- `scipy.stats.expon(scale)`: Exponential (scale = $1/\lambda$, where $\lambda$ is the rate parameter).
- `scipy.stats.norm(loc, scale)`: Normal (Gaussian) (loc=mean $\mu$, scale=standard deviation $\sigma$).
- `scipy.stats.gamma(a, loc, scale)`: Gamma (`a` is the shape parameter $\alpha$ or $k$).
- `scipy.stats.beta(a, b, loc, scale)`: Beta (`a`, `b` are shape parameters $\alpha, \beta$).

## 4. Matplotlib (`import matplotlib.pyplot as plt`)

### Basic Plotting

- `plt.plot(x, y, [fmt], ...)`: Line plot.
- `plt.scatter(x, y, ...)`: Scatter plot.
- `plt.bar(x, height, ...)`: Vertical bar chart.
- `plt.hist(data, bins=..., density=False, ...)`: Histogram. `density=True` normalizes the histogram to form a probability density.
- `plt.boxplot(data, ...)`: Box-and-whisker plot.

### Customization & Display

- `plt.title('...')`, `plt.xlabel('...')`, `plt.ylabel('...')`: Set plot title and axis labels.
- `plt.legend(['label1', 'label2'])`: Add a legend (use `label='...'` kwarg in plot commands).
- `plt.grid(True)`: Display grid lines.
- `plt.xlim(min, max)`, `plt.ylim(min, max)`: Set axis limits.
- `plt.figure(figsize=(width, height))`: Create a new figure object with specified size in inches.
- `plt.subplot(nrows, ncols, index)`: Create axes in a grid of subplots.
- `plt.tight_layout()`: Adjusts plot parameters for a tight layout.
- `plt.show()`: Display the current figure.
- `plt.savefig('filename.png')`: Save the current figure to a file.

## 5. Seaborn (`import seaborn as sns`)

Built on Matplotlib, providing higher-level interface for statistical graphics.

### Common Statistical Plots

- `sns.histplot(data=..., x=..., hue=..., kde=True)`: Histogram with optional Kernel Density Estimate (KDE).
- `sns.kdeplot(data=..., x=..., hue=...)`: Plot Kernel Density Estimate.
- `sns.ecdfplot(data=..., x=..., hue=...)`: Plot Empirical Cumulative Distribution Function.
- `sns.boxplot(data=..., x=..., y=..., hue=...)`: Box plot, easily handles grouping by categorical variables.
- `sns.violinplot(data=..., x=..., y=..., hue=...)`: Combines box plot with KDE.
- `sns.scatterplot(data=..., x=..., y=..., hue=..., size=...)`: Enhanced scatter plot.
- `sns.heatmap(data, annot=False, cmap=...)`: Visualize matrix data (e.g., covariance, transition matrices). `annot=True` shows values.
- `sns.pairplot(data, hue=...)`: Plot pairwise relationships between variables in a DataFrame.
- `sns.jointplot(data=..., x=..., y=..., kind='scatter'|'kde'|'hist')`: Scatter plot with marginal distributions on axes.

### Aesthetics

- `sns.set_theme(style=..., palette=..., context=...)`: Set the global aesthetic parameters (e.g., `style='whitegrid'`, `palette='viridis'`).

## 6. Pandas (`import pandas as pd`)

Primarily used for data loading, manipulation, and preliminary analysis.

### Core Data Structures

- `pd.Series(data, index=...)`: 1-dimensional labeled array.
- `pd.DataFrame(data, index=..., columns=...)`: 2-dimensional labeled table-like structure.

### Data Loading/Saving

- `pd.read_csv('filepath')`, `pd.read_excel('filepath')`: Load data from files.
- `df.to_csv('filepath')`, `df.to_excel('filepath')`: Save DataFrame to files.

### Selection & Inspection

- `df.head(n)`, `df.tail(n)`: View first/last n rows.
- `df.info()`: Summary of DataFrame (index dtype, columns, non-null values, memory usage).
- `df.describe()`: Generate descriptive statistics for numerical columns.
- `df['column']`, `df.column`: Select a column as a Series.
- `df[['col1', 'col2']]`: Select multiple columns as a DataFrame.
- `df.loc[row_label, col_label]`: Access group of rows/columns by label(s).
- `df.iloc[row_index, col_index]`: Access group of rows/columns by integer position(s).
- Boolean Indexing: `df[df['value'] > 0]`

### Common Operations & Statistics

- `df['column'].mean()`, `.std()`, `.var()`, `.median()`, `.min()`, `.max()`, `.sum()`: Column-wise statistics.
- `df.corr()`: Compute pairwise correlation of columns.
- `df.cov()`: Compute pairwise covariance of columns.
- `df['column'].value_counts()`: Count unique values in a Series.
- `df.groupby('col_name').agg({'other_col': 'mean'})`: Group data and apply aggregation functions.
- `df.apply(function, axis=0|1)`: Apply a function along an axis.

```{code-cell} ipython3

```
