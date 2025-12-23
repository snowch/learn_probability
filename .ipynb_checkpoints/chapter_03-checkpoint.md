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

# Chapter 3: Counting Techniques: Permutations and Combinations

Welcome to Chapter 3! In the previous chapter, we established the fundamental language of probability using sets and explored the basic axioms and rules. Now, we dive into a crucial skill for calculating probabilities, especially when dealing with equally likely outcomes: **counting**.

Often, calculating a probability boils down to answering two questions:
1.  How many total possible outcomes are there in our sample space?
2.  How many of those outcomes correspond to the event we're interested in?

If all outcomes are equally likely, the probability is simply the ratio of these two counts. While this sounds simple, counting the number of possibilities can become complex very quickly. Imagine trying to list every possible 5-card poker hand!

This chapter introduces systematic methods for counting outcomes: the Multiplication Principle, Permutations, and Combinations. We'll see how these techniques allow us to tackle problems that would be tedious or impossible to solve by simple enumeration. We'll also use Python's `math` and `scipy.special` libraries to perform these calculations efficiently.

Let's start counting!

+++

## The Multiplication Principle

The most fundamental counting technique is the **Multiplication Principle** (also known as the rule of product).

**Principle:** If a procedure can be broken down into a sequence of $k$ steps, and
* there are $n_1$ ways to perform the first step,
* there are $n_2$ ways to perform the second step (regardless of the outcome of the first step),
* ...
* there are $n_k$ ways to perform the $k$-th step (regardless of the outcomes of the previous steps),

then the total number of ways to perform the entire procedure is the product $n_1 \times n_2 \times \dots \times n_k$.

**Example:** A restaurant offers a fixed-price dinner menu with 3 choices for starters, 4 choices for the main course, and 2 choices for dessert. How many different meal combinations are possible?

* Step 1: Choose a starter ($n_1 = 3$ ways)
* Step 2: Choose a main course ($n_2 = 4$ ways)
* Step 3: Choose a dessert ($n_3 = 2$ ways)

According to the Multiplication Principle, the total number of different meal combinations is $3 \times 4 \times 2$.

```{code-cell} ipython3
# Using Python for the meal combination example
num_starters = 3
num_mains = 4
num_desserts = 2
```

```{code-cell} ipython3
total_combinations = num_starters * num_mains * num_desserts
```

```{code-cell} ipython3
print(f"Total number of meal combinations: {total_combinations}")
```

This principle is the foundation upon which permutations and combinations are built.

+++

## Permutations: When Order Matters

A **permutation** is an arrangement of objects in a specific order. Consider arranging books on a shelf – swapping two books creates a different arrangement.

### Permutations without Repetition

This is the most common type of permutation. It involves arranging $k$ distinct objects chosen from a set of $n$ distinct objects, where order matters and objects cannot be reused.

**Formula:** The number of permutations of $n$ distinct objects taken $k$ at a time is denoted by $P(n, k)$, $_nP_k$, or $P^n_k$ and is calculated as:

$ P(n, k) = \frac{n!}{(n-k)!} $

where $n!$ (read "n factorial") is the product of all positive integers up to $n$ (i.e., $n! = n \times (n-1) \times \dots \times 2 \times 1$), and $0! = 1$ by definition.

**Example:** In a race with 8 runners, how many different ways can the 1st, 2nd, and 3rd place medals be awarded?

Here, we are choosing $k=3$ winners from $n=8$ runners, and the order matters (Gold is different from Silver). We cannot reuse a runner (no repetition).

Using the formula:

$ P(8, 3) = \frac{8!}{(8-3)!} = \frac{8!}{5!} = \frac{8 \times 7 \times 6 \times 5 \times 4 \times 3 \times 2 \times 1}{5 \times 4 \times 3 \times 2 \times 1} = 8 \times 7 \times 6 $

Let's calculate this using Python.

```{code-cell} ipython3
import math
from scipy.special import perm
```

```{code-cell} ipython3
# Using math.factorial
n_runners = 8
k_places = 3
```

```{code-cell} ipython3
p_8_3_math = math.factorial(n_runners) // math.factorial(n_runners - k_places) # Use // for integer division
print(f"Using math.factorial: P({n_runners}, {k_places}) = {p_8_3_math}")
```

```{code-cell} ipython3
# Using scipy.special.perm
# Note: scipy.special.perm(n, k) calculates P(n, k) directly
p_8_3_scipy = perm(n_runners, k_places, exact=True) # exact=True ensures integer result
print(f"Using scipy.special.perm: P({n_runners}, {k_places}) = {p_8_3_scipy}")
```

```{code-cell} ipython3
# Direct calculation based on the multiplication principle
p_8_3_direct = 8 * 7 * 6
print(f"Direct calculation: {p_8_3_direct}")
```

**Special Case:** The number of ways to arrange all $n$ distinct objects is $P(n, n) = \frac{n!}{(n-n)!} = \frac{n!}{0!} = n!$. For example, there are $3! = 3 \times 2 \times 1 = 6$ ways to arrange the letters A, B, C: (ABC, ACB, BAC, BCA, CAB, CBA).

+++

### Permutations with Repetition (Multinomial Coefficients)

Sometimes we need to arrange objects where some are identical.

**Formula:** The number of distinct permutations of $n$ objects where there are $n_1$ identical objects of type 1, $n_2$ identical objects of type 2, ..., and $n_k$ identical objects of type k (such that $n_1 + n_2 + \dots + n_k = n$) is:

$ \frac{n!}{n_1! n_2! \dots n_k!} $

**Example:** How many distinct ways can the letters in the word "MISSISSIPPI" be arranged?

* Total letters $n = 11$
* M: $n_1 = 1$
* I: $n_2 = 4$
* S: $n_3 = 4$
* P: $n_4 = 2$
(Check: $1 + 4 + 4 + 2 = 11$)

The number of distinct arrangements is:

$ \frac{11!}{1! 4! 4! 2!} $

```{code-cell} ipython3
import math
```

```{code-cell} ipython3
n = 11
n_M = 1
n_I = 4
n_S = 4
n_P = 2
```

```{code-cell} ipython3
# Calculate factorials
numerator = math.factorial(n)
denominator = math.factorial(n_M) * math.factorial(n_I) * math.factorial(n_S) * math.factorial(n_P)
```

```{code-cell} ipython3
distinct_arrangements = numerator // denominator # Use integer division
```

```{code-cell} ipython3
print(f"Number of distinct arrangements of 'MISSISSIPPI': {distinct_arrangements}")
```

## Combinations: When Order Doesn't Matter

A **combination** is a selection of objects where the order of selection does not matter. Consider choosing members for a committee – selecting Alice then Bob is the same as selecting Bob then Alice.

### Combinations without Repetition

This involves selecting $k$ distinct objects from a set of $n$ distinct objects, where order *does not* matter and objects cannot be reused.

**Formula:** The number of combinations of $n$ distinct objects taken $k$ at a time is denoted by $C(n, k)$, $_nC_k$, $C^n_k$, or $\binom{n}{k}$ (read "n choose k") and is calculated as:

$ C(n, k) = \binom{n}{k} = \frac{n!}{k!(n-k)!} $

Notice that $C(n, k) = \frac{P(n, k)}{k!}$. This is because for every combination of $k$ objects, there are $k!$ ways to order them (permutations). We divide the number of permutations $P(n,k)$ by $k!$ to remove the effect of order.

**Example:** How many ways can a committee of 3 people be chosen from a group of 10 people?

Here, we are choosing $k=3$ people from $n=10$, and the order in which they are chosen doesn't matter.

Using the formula:

$ C(10, 3) = \binom{10}{3} = \frac{10!}{3!(10-3)!} = \frac{10!}{3!7!} = \frac{10 \times 9 \times 8}{3 \times 2 \times 1} $

Let's calculate this using Python.

```{code-cell} ipython3
import math
from scipy.special import comb
```

```{code-cell} ipython3
# Using math.factorial
n_people = 10
k_committee = 3
```

```{code-cell} ipython3
c_10_3_math = math.factorial(n_people) // (math.factorial(k_committee) * math.factorial(n_people - k_committee))
print(f"Using math.factorial: C({n_people}, {k_committee}) = {c_10_3_math}")
```

```{code-cell} ipython3
# Using scipy.special.comb
# Note: scipy.special.comb(n, k) calculates C(n, k) directly
c_10_3_scipy = comb(n_people, k_committee, exact=True) # exact=True ensures integer result
print(f"Using scipy.special.comb: C({n_people}, {k_committee}) = {c_10_3_scipy}")
```

```{code-cell} ipython3
# Direct calculation
c_10_3_direct = (10 * 9 * 8) // (3 * 2 * 1)
print(f"Direct calculation: {c_10_3_direct}")
```

### Combinations with Repetition

This involves selecting $k$ objects from $n$ types of objects, where order doesn't matter and we can choose multiple objects of the same type (repetition is allowed). This is sometimes called "multiset coefficient" or "stars and bars" problem.

**Formula:** The number of combinations with repetition of $n$ types of objects taken $k$ at a time is:

$ \binom{n+k-1}{k} = \frac{(n+k-1)!}{k!(n-1)!} $

**Example:** A bakery offers 4 types of donuts (plain, chocolate, glazed, jelly). How many different ways can you select a dozen (12) donuts?

Here, $n=4$ (types of donuts) and we are choosing $k=12$ donuts. The order doesn't matter, and we can choose multiple donuts of the same type.

Using the formula:

$ \binom{4+12-1}{12} = \binom{15}{12} = \frac{15!}{12!(15-12)!} = \frac{15!}{12!3!} = \frac{15 \times 14 \times 13}{3 \times 2 \times 1} $

```{code-cell} ipython3
from scipy.special import comb
```

```{code-cell} ipython3
n_types = 4
k_donuts = 12
```

```{code-cell} ipython3
# Using the formula C(n+k-1, k)
combinations_with_repetition = comb(n_types + k_donuts - 1, k_donuts, exact=True)
```

```{code-cell} ipython3
print(f"Number of ways to choose {k_donuts} donuts from {n_types} types: {combinations_with_repetition}")
```

```{code-cell} ipython3
# Direct calculation
c_15_12_direct = (15 * 14 * 13) // (3 * 2 * 1)
print(f"Direct calculation: {c_15_12_direct}")
```

Note: 
- scipy.special.comb can also take repetition=True argument for this

```{code-cell} ipython3
# combinations_with_repetition_scipy = comb(n_types, k_donuts, exact=True, repetition=True)
# print(f"Using scipy.special.comb with repetition=True: {combinations_with_repetition_scipy}")
```

^^ Uncomment the above lines if your SciPy version supports repetition=True (relatively recent addition)

+++

## Applications to Probability Problems

Counting techniques are essential for calculating probabilities in scenarios with equally likely outcomes, often found in games of chance, sampling, and more.

The basic formula is:

$ P(\text{Event}) = \frac{\text{Number of outcomes favorable to the event}}{\text{Total number of possible outcomes}} $

Both the numerator and the denominator often require permutations or combinations to calculate.

**Example: UK National Lottery**

In the UK National Lottery's main "Lotto" game (as of early 2020s), a player chooses 6 distinct numbers from 1 to 59. The lottery machine then randomly selects 6 distinct numbers. What is the probability of winning the jackpot (matching all 6 numbers)?

1.  **Total number of possible outcomes:** This is the number of ways to choose 6 distinct numbers from 59, where order doesn't matter. This is a combination problem: $C(59, 6)$.
2.  **Number of favorable outcomes:** There is only 1 way to match the specific 6 numbers drawn by the machine.

The probability is $P(\text{Jackpot}) = \frac{1}{C(59, 6)}$.

```{code-cell} ipython3
from scipy.special import comb
```

```{code-cell} ipython3
# Total numbers to choose from
n_lotto = 59
# Numbers to choose
k_lotto = 6
```

```{code-cell} ipython3
# Calculate the total number of possible combinations
total_lotto_combinations = comb(n_lotto, k_lotto, exact=True)
print(f"Total possible UK Lotto combinations: {total_lotto_combinations:,}") # Format with commas
```

```{code-cell} ipython3
# Calculate the probability of winning the jackpot
prob_jackpot = 1 / total_lotto_combinations
print(f"Probability of winning the jackpot: 1 / {total_lotto_combinations:,}")
print(f"Probability (decimal): {prob_jackpot:.10f}") # Print with more decimal places
print(f"Probability (scientific notation): {prob_jackpot:e}")
```

**Example: Poker Hand Probability (Four of a Kind)**

What is the probability of being dealt "Four of a Kind" in a standard 5-card poker hand from a 52-card deck? (Four cards of one rank, plus one other card of a different rank).

1.  **Total number of possible outcomes:** The total number of ways to choose 5 cards from 52, where order doesn't matter. This is $C(52, 5)$.

2.  **Number of favorable outcomes (Four of a Kind):** We can use the Multiplication Principle to count this:
    * Step 1: Choose the rank for the four cards (e.g., four Aces, four Kings). There are $C(13, 1) = 13$ ways.
    * Step 2: Choose the four cards of that rank. There is only $C(4, 4) = 1$ way (you must take all four suits).
    * Step 3: Choose the rank for the fifth card (it must be different from the rank in Step 1). There are $C(12, 1) = 12$ remaining ranks.
    * Step 4: Choose the suit for the fifth card. There are $C(4, 1) = 4$ ways.
    Total favorable outcomes = $13 \times 1 \times 12 \times 4$.

The probability is $P(\text{Four of a Kind}) = \frac{13 \times 1 \times 12 \times 4}{C(52, 5)}$.

```{code-cell} ipython3
from scipy.special import comb
```

```{code-cell} ipython3
# Total cards in deck
n_deck = 52
# Cards in hand
k_hand = 5
```

```{code-cell} ipython3
# 1. Calculate the total number of possible 5-card hands
total_hands = comb(n_deck, k_hand, exact=True)
print(f"Total possible 5-card poker hands: {total_hands:,}")
```

```{code-cell} ipython3
# 2. Calculate the number of ways to get Four of a Kind
# Step 1: Choose rank for the four cards (13 ranks: A, 2, ..., 10, J, Q, K)
ways_choose_rank4 = comb(13, 1, exact=True)
# Step 2: Choose the 4 suits for that rank (only 1 way)
ways_choose_suits4 = comb(4, 4, exact=True)
# Step 3: Choose the rank for the fifth card (12 remaining ranks)
ways_choose_rank1 = comb(12, 1, exact=True)
# Step 4: Choose the suit for the fifth card (4 suits)
ways_choose_suit1 = comb(4, 1, exact=True)
```

```{code-cell} ipython3
favorable_outcomes_4kind = ways_choose_rank4 * ways_choose_suits4 * ways_choose_rank1 * ways_choose_suit1
print(f"Number of ways to get Four of a Kind: {favorable_outcomes_4kind}")
```

```{code-cell} ipython3
# 3. Calculate the probability
prob_4kind = favorable_outcomes_4kind / total_hands
print(f"Probability of being dealt Four of a Kind: {prob_4kind:.8f}")
print(f"Approximately 1 in {1/prob_4kind:,.0f}")
```

## Hands-on: Using Python for Counting

We've already seen how `math.factorial`, `scipy.special.perm`, and `scipy.special.comb` can be used. Let's solidify this.

**Key Functions:**
* `math.factorial(n)`: Computes $n!$. Requires `n` to be a non-negative integer.
* `scipy.special.perm(n, k, exact=True)`: Computes $P(n, k) = \frac{n!}{(n-k)!}$. `exact=True` is recommended for integer results.
* `scipy.special.comb(n, k, exact=True, repetition=False)`: Computes $C(n, k) = \binom{n}{k} = \frac{n!}{k!(n-k)!}$. `exact=True` is recommended. Set `repetition=True` for combinations with repetition.

**Remember to import them:**
```python
import math
from scipy.special import perm, comb
```

**Exercise Idea:** Calculate the probability of getting a "Full House" (three cards of one rank, two cards of another rank) in a 5-card poker hand.

*Hint:*
1.  Total hands: $C(52, 5)$ (calculated above).
2.  Favorable outcomes:
    * Choose the rank for the three cards: $C(13, 1)$ ways.
    * Choose 3 suits for that rank: $C(4, 3)$ ways.
    * Choose the rank for the two cards: $C(12, 1)$ ways (must be different from the first rank).
    * Choose 2 suits for that second rank: $C(4, 2)$ ways.
    * Use the Multiplication Principle.*

+++

Exercise: Calculate probability of a Full House

+++

Total hands (already calculated)
total_hands = comb(52, 5, exact=True)

```{code-cell} ipython3
# Favorable outcomes for Full House
# Step 1: Choose rank for the three cards
ways_choose_rank3 = comb(13, 1, exact=True)
# Step 2: Choose 3 suits for that rank
ways_choose_suits3 = comb(4, 3, exact=True)
# Step 3: Choose rank for the pair (from remaining 12 ranks)
ways_choose_rank2 = comb(12, 1, exact=True)
# Step 4: Choose 2 suits for that rank
ways_choose_suits2 = comb(4, 2, exact=True)
```

```{code-cell} ipython3
favorable_outcomes_fullhouse = ways_choose_rank3 * ways_choose_suits3 * ways_choose_rank2 * ways_choose_suits2
print(f"Number of ways to get a Full House: {favorable_outcomes_fullhouse}")
```

```{code-cell} ipython3
# Calculate the probability
prob_fullhouse = favorable_outcomes_fullhouse / total_hands
print(f"Probability of being dealt a Full House: {prob_fullhouse:.8f}")
print(f"Approximately 1 in {1/prob_fullhouse:,.0f}")
```

## Summary

In this chapter, we learned the fundamental counting techniques essential for calculating probabilities in many situations:
* **Multiplication Principle:** If a task has sequential steps, multiply the number of ways to do each step to get the total number of ways.
* **Permutations ($P(n, k)$):** Used when selecting $k$ items from $n$ **where order matters** and there is no repetition. Formula: $\frac{n!}{(n-k)!}$.
* **Combinations ($C(n, k)$ or $\binom{n}{k}$):** Used when selecting $k$ items from $n$ **where order does not matter** and there is no repetition. Formula: $\frac{n!}{k!(n-k)!}$.
* We also briefly touched upon permutations and combinations **with repetition**.
* These techniques are crucial for calculating probabilities of the form $P(E) = \frac{|E|}{|S|}$ where outcomes are equally likely.

We saw how to apply these concepts to practical examples like meal combinations, race outcomes, committee selections, lottery odds, and poker hands. We also leveraged Python's `math.factorial` and `scipy.special.perm`/`comb` functions to perform these calculations efficiently.

Mastering these counting techniques provides a powerful toolkit for tackling a wide range of probability problems. In the next chapter, we will move on to exploring probabilities when events are not independent, introducing the concept of Conditional Probability.

```{code-cell} ipython3

```
