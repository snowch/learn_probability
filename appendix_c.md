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

# Appendix C: Mathematical Notation Summary

This appendix provides a summary of the common mathematical notations used throughout this book. Familiarity with these symbols is helpful for understanding the theoretical underpinnings alongside the Python implementations.

## Set Theory and Probability Basics

| Notation              | Meaning                                     | Example                                                               | Chapter(s) |
| :-------------------- | :------------------------------------------ | :-------------------------------------------------------------------- | :--------- |
| $S$, $\Omega$          | Sample Space (the set of all possible outcomes) | $S = \{1, 2, 3, 4, 5, 6\}$ for a die roll.                             | 2          |
| $A, B, E, ...$        | Events (subsets of the sample space)        | $A = \{2, 4, 6\}$ (rolling an even number).                             | 2          |
| $\emptyset$           | Empty Set (impossible event)                 | Rolling a 7 on a standard die.                                        | 2          |
| $A \cup B$           | Union ('A or B' or both occur)              | $\{1, 2, 3\} \cup \{3, 4, 5\} = \{1, 2, 3, 4, 5\}$                     | 2          |
| $A \cap B$           | Intersection ('A and B' both occur)         | $\{1, 2, 3\} \cap \{3, 4, 5\} = \{3\}$                               | 2          |
| $A^c$, $\bar{A}$        | Complement ('not A')                       | If $S=\{1,2,3\}$, $A=\{1\}$, then $A^c = \{2, 3\}$.                      | 2          |
| $A \setminus B$        | Set Difference ('A but not B')              | $\{1, 2, 3\} \setminus \{3, 4, 5\} = \{1, 2\}$                          | 2          |
| $|A|$                 | Cardinality (number of elements in set A)   | $|\{2, 4, 6\}| = 3$                                                 | 2, 3       |
| $P(A)$                | Probability of event A occurring            | $P(\text{Heads}) = 0.5$ for a fair coin.                               | 2          |
| $P(A|B)$              | Conditional Probability (prob. of A given B) | $P(\text{Sum}>10 | \text{First roll is } 6)$                             | 4          |

## Counting Techniques

| Notation             | Meaning                                        | Example                                             | Chapter(s) |
| :------------------- | :--------------------------------------------- | :-------------------------------------------------- | :--------- |
| $n!$                 | Factorial ($n \times (n-1) \times ... \times 1$) | $5! = 5 \times 4 \times 3 \times 2 \times 1 = 120$               | 3          |
| $P(n, k)$, $^nP_k$    | Permutations (ordered arrangements of k from n) | Ways to award Gold, Silver, Bronze to 3 of 10 runners | 3          |
| $C(n, k)$, $^nC_k$, $\binom{n}{k}$ | Combinations (unordered selections of k from n) | Ways to choose a committee of 3 from 10 people    | 3          |

## Random Variables and Distributions

| Notation                       | Meaning                                                        | Example                                                             | Chapter(s) |
| :----------------------------- | :------------------------------------------------------------- | :------------------------------------------------------------------ | :--------- |
| $X, Y, Z$                      | Random Variables (variables whose values are numerical outcomes) | $X =$ Number of heads in 3 coin flips.                              | 6-12       |
| $x, y, z$                      | Specific values (realizations) of random variables             | $X$ could take the value $x=2$.                                     | 6-12       |
| $X \sim \text{Dist}(...)$      | 'X follows the distribution Dist with given parameters'         | $X \sim \text{Binomial}(n=10, p=0.5)$                              | 7, 9       |
| $p(x)$, $p_X(x)$, $P(X=x)$      | Probability Mass Function (PMF) of a discrete RV $X$           | $p_X(k) = P(X=k)$ for $k=0, 1, ..., n$ in a Binomial distribution. | 6, 7       |
| $f(x)$, $f_X(x)$                | Probability Density Function (PDF) of a continuous RV $X$      | The bell curve shape for a Normal distribution.                     | 8, 9       |
| $F(x)$, $F_X(x)$                | Cumulative Distribution Function (CDF) $P(X \le x)$           | $F_X(x) = P(X \le x)$                                                | 6, 8       |
| $E[X]$, $\mu$, $\mu_X$          | Expected Value (mean) of RV $X$                                | Average value expected from many trials.                            | 6, 8       |
| $Var(X)$, $\sigma^2$, $\sigma^2_X$ | Variance of RV $X$ (measure of spread)                         | $Var(X) = E[(X - \mu)^2]$                                          | 6, 8       |
| $SD(X)$, $\sigma$, $\sigma_X$    | Standard Deviation of RV $X$ ($\sqrt{Var(X)}$)                   | Spread measured in the same units as $X$.                           | 6, 8       |

## Multiple Random Variables

| Notation                           | Meaning                                                              | Chapter(s) |
| :--------------------------------- | :------------------------------------------------------------------- | :--------- |
| $(X, Y)$                           | A pair of random variables                                           | 10-12      |
| $p(x, y)$, $p_{X,Y}(x, y)$           | Joint PMF of discrete RVs $X, Y$                                     | 10         |
| $f(x, y)$, $f_{X,Y}(x, y)$           | Joint PDF of continuous RVs $X, Y$                                   | 10         |
| $F(x, y)$, $F_{X,Y}(x, y)$           | Joint CDF $P(X \le x, Y \le y)$                                      | 10         |
| $p_X(x)$, $f_X(x)$                  | Marginal PMF/PDF of $X$ (derived from joint distribution)          | 10         |
| $p(y|x)$, $p_{Y|X}(y|x)$            | Conditional PMF of $Y$ given $X=x$                                   | 10         |
| $f(y|x)$, $f_{Y|X}(y|x)$            | Conditional PDF of $Y$ given $X=x$                                   | 10         |
| $Cov(X, Y)$                        | Covariance between $X$ and $Y$ ($E[(X-\mu_X)(Y-\mu_Y)]$)             | 11         |
| $\rho(X, Y)$, $Corr(X, Y)$         | Correlation Coefficient between $X$ and $Y$ ($\frac{Cov(X,Y)}{\sigma_X \sigma_Y}$) | 11         |

## Limit Theorems and Convergence

| Notation           | Meaning                         | Chapter(s) |
| :----------------- | :------------------------------ | :--------- |
| $X_n \xrightarrow{p} X$ | Convergence in Probability      | 13         |
| $X_n \xrightarrow{d} X$ | Convergence in Distribution     | 14         |

## Bayesian Inference

| Notation           | Meaning                      | Chapter(s) |
| :----------------- | :--------------------------- | :--------- |
| $\theta$            | Parameter of interest        | 5, 15      |
| $\pi(\theta)$        | Prior distribution of $\theta$ | 15         |
| $L(\theta | x)$     | Likelihood function          | 15         |
| $p(\theta | x)$     | Posterior distribution of $\theta$ | 5, 15      |

## Markov Chains

| Notation  | Meaning                                     | Chapter(s) |
| :-------- | :------------------------------------------ | :--------- |
| $P_{ij}$  | Transition probability from state $i$ to $j$ | 16         |
| $\mathbf{P}$ | Transition Probability Matrix             | 16         |
| $\pi$    | Stationary distribution vector              | 16         |

## General Mathematical Symbols

| Notation           | Meaning                                                | Chapter(s) |
| :----------------- | :----------------------------------------------------- | :--------- |
| $\sum$            | Summation                                              | Throughout |
| $\int$            | Integral                                               | Throughout |
| $\approx$         | Approximately equal to                                 | Throughout |
| $\propto$         | Proportional to                                        | 5, 15      |
| $\mathbb{R}$      | Set of real numbers                                    | Throughout |
| $\mathbb{N}$      | Set of natural numbers (usually $\{1, 2, 3, ...\}$)      | Throughout |
| $\in$             | 'Element of' or 'belongs to'                           | 2          |
| $\forall$         | 'For all'                                              | Throughout |
| $\exists$         | 'There exists'                                         | Throughout |

```{code-cell} ipython3

```
