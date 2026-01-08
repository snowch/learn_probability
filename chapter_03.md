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
  - file: notebooks/chapter_03.ipynb
---

# Chapter 3: Counting Techniques: Permutations and Combinations

Welcome to Chapter 3! In the previous chapter, we established the fundamental language of probability using sets and explored the basic axioms and rules. Now, we dive into a crucial skill for calculating probabilities, especially when dealing with equally likely outcomes: **counting**.

Often, calculating a probability boils down to answering two questions:
1.  How many total possible outcomes are there in our sample space?
2.  How many of those outcomes correspond to the event we're interested in?

If all outcomes are equally likely, the probability is simply the ratio of these two counts. While this sounds simple, counting the number of possibilities can become complex very quickly. Imagine trying to list every possible 5-card poker hand!

This chapter introduces systematic methods for counting outcomes: the Multiplication Principle, Permutations, and Combinations. We'll see how these techniques allow us to tackle problems that would be tedious or impossible to solve by simple enumeration. We'll also use Python's `math` and `scipy.special` libraries to perform these calculations efficiently.

Let's start counting!

:::{admonition} Understanding "without repetition" vs "with repetition"
:class: note

Throughout this chapter, you'll encounter the terms **"without repetition"** and **"with repetition"**. These phrases often confuse students because they sound like they describe **sampling methods** (like drawing balls from a bag with or without replacement), but they actually describe something different: **whether objects can occupy multiple positions**.

**What "without repetition" actually means:**
- Each object can only occupy **one position** in the arrangement or selection
- You don't use the same object multiple times
- Think: "each object appears at most once"

**What "with repetition" actually means:**
- The same object (or type of object) can occupy **multiple positions**
- Objects can be reused
- Think: "objects can appear more than once"

**Common confusion to avoid:**
- ❌ "Without repetition" does NOT mean "we're running out of objects to choose from"
- ❌ It's NOT the same as "sampling without replacement" from probability
- ✓ It means "each object is used in at most one position"

**Simple examples:**
- **Without repetition:** Awarding 3 medals to 8 runners — each runner gets at most one medal (can't give Gold AND Silver to same person)
- **With repetition:** A license plate with 3 letters — the same letter can appear multiple times (like AAA)
- **Without repetition:** Choosing 3 people for a committee from 10 — each person is either on the committee or not
- **With repetition:** Choosing 3 donuts from 4 flavors — you can choose chocolate 3 times

The sections below will show how this concept applies specifically to permutations and combinations, but the core idea remains the same: it's about whether objects can occupy multiple positions.
:::

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

:::{dropdown} Python Implementation
```{code-cell} ipython3
# Using Python for the meal combination example
num_starters = 3
num_mains = 4
num_desserts = 2

total_combinations = num_starters * num_mains * num_desserts
print(f"Total number of meal combinations: {total_combinations}")
```
:::

This principle is the foundation upon which permutations and combinations are built.

+++

## Permutations: When Order Matters

A **permutation** is an arrangement of objects in a specific order. Consider arranging books on a shelf – swapping two books creates a different arrangement.

### Permutations without Repetition

This is the most common type of permutation. It involves arranging $k$ distinct objects chosen from a set of $n$ distinct objects, where order matters and objects cannot be reused.

:::{admonition} Applying "without repetition" to permutations
:class: tip

Recall from the chapter introduction: "without repetition" means each object occupies at most one position.

**In permutations, this means:**
- Each object can only be used once in the arrangement
- For medals: each runner gets at most one medal (can't give Gold AND Silver to the same person)
- For arranging books: each book appears only once on the shelf

**Key point:** Since order matters in permutations, Runner A getting Gold ≠ Runner A getting Silver. We're counting different **arrangements**, where both the identity AND position matter.
:::

#### Building Intuition: The Multiplication Principle Approach

Before we introduce the general formula, let's understand permutations through the **Multiplication Principle** we learned earlier.

**Example:** In a race with 8 runners, how many different ways can the 1st, 2nd, and 3rd place medals be awarded?

Let's think through this step-by-step:
- **Step 1:** Choose who gets the Gold medal (1st place): 8 choices
- **Step 2:** Choose who gets the Silver medal (2nd place): 7 choices (can't give it to the Gold winner)
- **Step 3:** Choose who gets the Bronze medal (3rd place): 6 choices (can't give it to Gold or Silver winners)

By the Multiplication Principle:
$$\text{Total ways} = 8 \times 7 \times 6 = 336$$

This is a **permutation** problem because:
1. Order matters (Gold ≠ Silver ≠ Bronze)
2. We can't reuse runners (each runner gets at most one medal)

**Key insight:** Notice the pattern:
- We start with $n = 8$ runners
- We choose $k = 3$ medals
- The calculation is: $8 \times 7 \times 6$ — we multiply $k$ consecutive descending integers starting from $n$

#### The General Formula

This multiplication pattern holds for all permutation problems. The number of permutations of $n$ distinct objects taken $k$ at a time is denoted by $P(n, k)$, $_nP_k$, or $P^n_k$ and is calculated as:

$ P(n, k) = n \times (n-1) \times (n-2) \times \dots \times (n-k+1) $

This can be written more compactly using factorials:

$ P(n, k) = \frac{n!}{(n-k)!} $

where $n!$ (read "n factorial") is the product of all positive integers up to $n$ (i.e., $n! = n \times (n-1) \times \dots \times 2 \times 1$), and $0! = 1$ by definition.

**Why does this work?** The factorial formula gives us:
$$P(8, 3) = \frac{8!}{(8-3)!} = \frac{8!}{5!} = \frac{8 \times 7 \times 6 \times \cancel{5 \times 4 \times 3 \times 2 \times 1}}{\cancel{5 \times 4 \times 3 \times 2 \times 1}} = 8 \times 7 \times 6$$

The $(n-k)!$ in the denominator cancels out the unwanted terms, leaving us with exactly $k$ consecutive descending integers starting from $n$.

Let's calculate this using Python.

:::{dropdown} Python Implementation
```{code-cell} ipython3
import math
from scipy.special import perm

# Calculate P(8, 3) - race permutations
n_runners = 8
k_places = 3

# Using math.factorial
p_8_3_math = math.factorial(n_runners) // math.factorial(n_runners - k_places)
print(f"Using math.factorial: P({n_runners}, {k_places}) = {p_8_3_math}")

# Using scipy.special.perm
p_8_3_scipy = perm(n_runners, k_places, exact=True)
print(f"Using scipy.special.perm: P({n_runners}, {k_places}) = {p_8_3_scipy}")

# Direct calculation based on the multiplication principle
p_8_3_direct = 8 * 7 * 6
print(f"Direct calculation: {p_8_3_direct}")
```
:::

**Special Case:** The number of ways to arrange all $n$ distinct objects is $P(n, n) = \frac{n!}{(n-n)!} = \frac{n!}{0!} = n!$. For example, there are $3! = 3 \times 2 \times 1 = 6$ ways to arrange the letters A, B, C: (ABC, ACB, BAC, BCA, CAB, CBA).

+++

### Permutations with Repetition (Multinomial Coefficients)

Sometimes we need to arrange objects where some are identical.

:::{admonition} Applying "with repetition" to permutations
:class: tip

Recall from the chapter introduction: "with repetition" means the same object (or type) can occupy multiple positions.

**In this type of permutation, "with repetition" means:**
- We have **multiple identical copies** of the same object type
- Example: In "MISSISSIPPI", we have 4 identical I's, 4 identical S's, etc.
- These identical objects can occupy different positions (the 4 I's appear in positions 2, 5, 8, and 11)
- The challenge: How many **distinct arrangements** exist when some objects look the same?

**Important:** This is different from permutations **without** repetition, where every object is unique and distinguishable.
:::

:::{admonition} Terminology: Key terms in this section
:class: note

**Distinguishable vs. Distinct:**
- **Distinguishable objects**: Objects that can be told apart (like A₁ vs A₂). When we label identical objects with subscripts, we're treating them as distinguishable.
- **Distinct arrangements**: Unique arrangements that look different from each other (like AAB vs ABA vs BAA).

**Multinomial Coefficient:**
- The formula $\frac{n!}{n_1! \times n_2! \times \dots \times n_k!}$ is called the **multinomial coefficient**
- We explain why it has this name later in the section

The key question: How many **distinct arrangements** can we make when some objects are identical (not distinguishable)?
:::

#### Building Intuition: Starting Simple

**Simple Example:** How many distinct ways can you arrange the letters in "AAB"?

Let's list all possible arrangements:
1. **AAB**
2. **ABA**
3. **BAA**

Only **3 distinct arrangements**!

**But wait** – if all letters were distinguishable (say, A₁A₂B), how many arrangements would there be?

We'd have $3! = 6$ arrangements:
1. A₁A₂B
2. A₁BA₂
3. A₂A₁B ← looks the same as arrangement 1 when A's are identical
4. A₂BA₁ ← looks the same as arrangement 2 when A's are identical
5. BA₁A₂
6. BA₂A₁ ← looks the same as arrangement 5 when A's are identical

**Key insight:**
- When the two A's are **distinguishable**, we get 6 arrangements
- When the two A's are **identical**, arrangements 1&3 look the same (AAB), 2&4 look the same (ABA), and 5&6 look the same (BAA)
- Each distinct arrangement appears $2! = 2$ times (the number of ways to arrange the two identical A's)
- Therefore: Distinct arrangements = $\frac{3!}{2!} = \frac{6}{2} = 3$ ✓

**The pattern:**
$$\text{Distinct arrangements} = \frac{\text{Total if all were distinguishable}}{\text{Ways to rearrange identical objects}}$$

#### Scaling Up: MISSISSIPPI

Now let's apply this reasoning to a more complex problem: How many distinct ways can the letters in "MISSISSIPPI" be arranged?

**Step 1: Count the letters**
* Total letters: $n = 11$
* M: 1 (appears once)
* I: 4 (appears 4 times)
* S: 4 (appears 4 times)
* P: 2 (appears 2 times)

Check: $1 + 4 + 4 + 2 = 11$ ✓

**Step 2: Apply the pattern**

If all 11 letters were distinguishable, we'd have $11!$ arrangements.

But we're overcounting because:
- The 4 I's can be rearranged among themselves in $4!$ ways without creating a new word
- The 4 S's can be rearranged among themselves in $4!$ ways without creating a new word
- The 2 P's can be rearranged among themselves in $2!$ ways without creating a new word
- The 1 M appears only once ($1! = 1$, no overcounting)

Each distinct word is being counted $1! \times 4! \times 4! \times 2!$ times.

**Step 3: Calculate**

$$\text{Distinct arrangements} = \frac{11!}{1! \times 4! \times 4! \times 2!}$$

#### The General Formula

This pattern holds for all permutation-with-repetition problems. The number of distinct permutations of $n$ objects where there are $n_1$ identical objects of type 1, $n_2$ identical objects of type 2, ..., and $n_k$ identical objects of type k (where $n_1 + n_2 + \dots + n_k = n$) is:

$$\frac{n!}{n_1! \times n_2! \times \dots \times n_k!}$$

This is also called the **multinomial coefficient**.

:::{admonition} Why "multinomial coefficient"?
:class: note

The name has two parts to understand:

**"Multinomial"** means "many terms" (from Latin *multi* = many, *nomen* = name/term):
- We're dealing with **multiple types** of identical objects ($k$ different types)
- This generalizes the **binomial coefficient** $\binom{n}{k} = \frac{n!}{k!(n-k)!}$, which handles just **two types** (selected vs. not selected)
- For example, in MISSISSIPPI we have 4 types of letters (M, I, S, P), making this a truly "multi-nomial" problem

**"Coefficient"** refers to its role in algebra:
- A coefficient is a number that multiplies a term in an algebraic expression
- These values appear as the **coefficients in the multinomial theorem**: When you expand $(x_1 + x_2 + \dots + x_k)^n$, each term has a coefficient of the form $\frac{n!}{n_1! \times n_2! \times \dots \times n_k!}$
- For example, in $(a + b + c)^3$, the coefficient of $a^2bc$ is $\frac{3!}{2! \times 1! \times 1!} = 3$, meaning the term is $3a^2bc$
- So we call it a "coefficient" because it literally serves as a coefficient in polynomial expansions!
:::

:::{dropdown} Python Implementation
```{code-cell} ipython3
# Calculate distinct arrangements of MISSISSIPPI
n = 11
n_M = 1
n_I = 4
n_S = 4
n_P = 2

numerator = math.factorial(n)
denominator = math.factorial(n_M) * math.factorial(n_I) * math.factorial(n_S) * math.factorial(n_P)
distinct_arrangements = numerator // denominator

print(f"Number of distinct arrangements of 'MISSISSIPPI': {distinct_arrangements}")
```
:::

## Combinations: When Order Doesn't Matter

A **combination** is a selection of objects where the order of selection does not matter. Consider choosing members for a committee – selecting Alice then Bob is the same as selecting Bob then Alice.

### Combinations without Repetition

This involves selecting $k$ distinct objects from a set of $n$ distinct objects, where order *does not* matter and objects cannot be reused.

:::{admonition} Applying "without repetition" to combinations
:class: tip

Recall from the chapter introduction: "without repetition" means each object occupies at most one position.

**In combinations, this means:**
- Each object can only be selected once in the group
- For committees: each person is either on the committee or not (can't select Alice twice)
- For coin flips: we choose which **positions** get heads, and each position can only be chosen once

**Key point:** Since order doesn't matter in combinations, {Alice, Bob, Carol} = {Carol, Alice, Bob}. We're counting **selections**, where only the identity matters, not the arrangement.
:::

#### Building Intuition: From Permutations to Combinations

Before we introduce the general formula, let's understand combinations by building on what we learned about permutations.

**Example:** How many ways can a committee of 3 people be chosen from a group of 10 people?

Let's think through this step-by-step:

**Step 1: What if order mattered?**

Imagine we were choosing a President, Vice President, and Secretary (3 different roles):
- Using permutations: $P(10, 3) = 10 \times 9 \times 8 = 720$ ways

**Step 2: But order doesn't matter for a committee**

For a committee, these are all the **same selection**:
- Choose Alice, then Bob, then Carol
- Choose Alice, then Carol, then Bob
- Choose Bob, then Alice, then Carol
- Choose Bob, then Carol, then Alice
- Choose Carol, then Alice, then Bob
- Choose Carol, then Bob, then Alice

All 6 of these represent the committee {Alice, Bob, Carol}.

**Step 3: How many ways can we arrange the same 3 people?**

Any group of 3 people can be ordered in $3! = 3 \times 2 \times 1 = 6$ different ways.

**Step 4: Remove the overcounting**

Since each committee is being counted 6 times in our permutation count, we divide:

$$\text{Total committees} = \frac{P(10, 3)}{3!} = \frac{720}{6} = 120$$

**Key insight:** To convert from permutations (where order matters) to combinations (where order doesn't matter), we divide by $k!$ to eliminate all the different orderings of the same selection.

#### The General Formula

This pattern holds for all combination problems. The number of combinations of $n$ distinct objects taken $k$ at a time is denoted by $C(n, k)$, $_nC_k$, $C^n_k$, or $\binom{n}{k}$ (read "n choose k") and is calculated as:

$ C(n, k) = \binom{n}{k} = \frac{P(n, k)}{k!} = \frac{n!}{k!(n-k)!} $

**Why does this work?** Starting from the relationship to permutations:

$$C(10, 3) = \frac{P(10, 3)}{3!} = \frac{10 \times 9 \times 8}{3 \times 2 \times 1} = \frac{720}{6} = 120$$

Or using the factorial formula:

$$C(10, 3) = \frac{10!}{3!(10-3)!} = \frac{10!}{3! \times 7!} = \frac{10 \times 9 \times 8 \times \cancel{7!}}{3 \times 2 \times 1 \times \cancel{7!}} = \frac{720}{6} = 120$$

Let's calculate this using Python.

:::{dropdown} Python Implementation
```{code-cell} ipython3
import math
from scipy.special import comb

# Calculate C(10, 3) - committee combinations
n_people = 10
k_committee = 3

# Using math.factorial
c_10_3_math = math.factorial(n_people) // (math.factorial(k_committee) * math.factorial(n_people - k_committee))
print(f"Using math.factorial: C({n_people}, {k_committee}) = {c_10_3_math}")

# Using scipy.special.comb
c_10_3_scipy = comb(n_people, k_committee, exact=True)
print(f"Using scipy.special.comb: C({n_people}, {k_committee}) = {c_10_3_scipy}")

# Direct calculation
c_10_3_direct = (10 * 9 * 8) // (3 * 2 * 1)
print(f"Direct calculation: {c_10_3_direct}")
```
:::

### Combinations with Repetition

This involves selecting $k$ objects from $n$ types of objects, where order doesn't matter and we can choose multiple objects of the same type (repetition is allowed). This is sometimes called "multiset coefficient" or "stars and bars" problem.

:::{admonition} Applying "with repetition" to combinations
:class: tip

Recall from the chapter introduction: "with repetition" means the same object (or type) can occupy multiple positions.

**In combinations with repetition, this means:**
- We can **select the same type multiple times** in our group
- Example: Choosing 12 donuts from 4 flavors — we can choose chocolate 5 times, plain 3 times, etc.
- Example: Rolling a die 3 times and getting {6, 6, 2} — the outcome 6 appears twice
- The challenge: How many ways can we make selections when the same type can appear multiple times?

**Key point:** Order still doesn't matter {chocolate, chocolate, plain} = {plain, chocolate, chocolate}, but unlike combinations **without** repetition, we can now select the same item multiple times.
:::

#### Building Intuition: The "Stars and Bars" Visual Method

This is one of the most surprising formulas in counting, but there's a beautiful visual way to understand it! Let's start with a concrete example.

**Example:** A bakery offers 4 types of donuts (plain, chocolate, glazed, jelly). How many different ways can you select a dozen (12) donuts?

Here, $n=4$ (types of donuts) and we're choosing $k=12$ donuts. Order doesn't matter (choosing chocolate then plain is the same as plain then chocolate), and we can choose multiple donuts of the same type.

**Visual representation:** We can represent any selection using **stars** (★) for donuts and **bars** (|) as dividers between types.

For example, this arrangement:
```
★★|★★★★|★★★★★|★
```

Represents this selection:
- Type 1 (plain): **2** donuts (stars before the first bar)
- Type 2 (chocolate): **4** donuts (stars between first and second bar)
- Type 3 (glazed): **5** donuts (stars between second and third bar)
- Type 4 (jelly): **1** donut (stars after the third bar)
- Total: 2 + 4 + 5 + 1 = 12 ✓

**The counting problem becomes:** How many ways can we arrange 12 stars and 3 bars?

Let's analyze this:
- We have $k = 12$ stars (the donuts we're choosing)
- We need $n-1 = 3$ bars to create $n = 4$ sections (one for each type)
- Total objects to arrange: $12 + 3 = 15$

**Key insight:** Any arrangement of these 15 objects represents a valid donut selection! We just need to choose which 12 positions (out of 15 total) will have stars (the remaining 3 positions automatically get bars).

This is a **combination without repetition** problem we already know how to solve:

$$\text{Number of ways} = \binom{15}{12} = \binom{15}{3}$$

We can choose either the 12 star positions or the 3 bar positions — the result is the same!

**Generalizing the pattern:**
- $n = 4$ types → need $(n-1) = 3$ bars
- $k = 12$ objects → need $k = 12$ stars
- Total positions: $n + k - 1 = 4 + 12 - 1 = 15$
- Choose positions for stars: $\binom{n+k-1}{k} = \binom{15}{12}$

#### The General Formula

This pattern holds for all combinations-with-repetition problems. The number of combinations with repetition of $n$ types of objects taken $k$ at a time is:

$$\binom{n+k-1}{k} = \frac{(n+k-1)!}{k!(n-1)!}$$

**Why this formula?** We're arranging $k$ stars and $(n-1)$ bars, for a total of $(n+k-1)$ objects. We choose which $k$ positions get stars (or equivalently, which $(n-1)$ positions get bars).

**Calculating our donut example:**

$$\binom{4+12-1}{12} = \binom{15}{12} = \frac{15!}{12! \times 3!} = \frac{15 \times 14 \times 13}{3 \times 2 \times 1} = \frac{2730}{6} = 455$$

:::{dropdown} Python Implementation
```{code-cell} ipython3
# Calculate combinations with repetition - donut selection
n_types = 4
k_donuts = 12

# Using the formula C(n+k-1, k)
combinations_with_repetition = comb(n_types + k_donuts - 1, k_donuts, exact=True)
print(f"Number of ways to choose {k_donuts} donuts from {n_types} types: {combinations_with_repetition}")

# Direct calculation
c_15_12_direct = (15 * 14 * 13) // (3 * 2 * 1)
print(f"Direct calculation: {c_15_12_direct}")
```
:::

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

:::{dropdown} Python Implementation
```{code-cell} ipython3
# UK National Lottery - jackpot probability
n_lotto = 59  # Total numbers to choose from
k_lotto = 6   # Numbers to choose

# Calculate the total number of possible combinations
total_lotto_combinations = comb(n_lotto, k_lotto, exact=True)
print(f"Total possible UK Lotto combinations: {total_lotto_combinations:,}")

# Calculate the probability of winning the jackpot
prob_jackpot = 1 / total_lotto_combinations
print(f"Probability of winning the jackpot: 1 / {total_lotto_combinations:,}")
print(f"Probability (decimal): {prob_jackpot:.10f}")
print(f"Probability (scientific notation): {prob_jackpot:e}")
```
:::

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

:::{dropdown} Python Implementation
```{code-cell} ipython3
# Poker: Four of a Kind probability
n_deck = 52
k_hand = 5

# 1. Calculate the total number of possible 5-card hands
total_hands = comb(n_deck, k_hand, exact=True)
print(f"Total possible 5-card poker hands: {total_hands:,}")

# 2. Calculate the number of ways to get Four of a Kind
ways_choose_rank4 = comb(13, 1, exact=True)  # Choose rank for the four cards
ways_choose_suits4 = comb(4, 4, exact=True)  # Choose the 4 suits (only 1 way)
ways_choose_rank1 = comb(12, 1, exact=True)  # Choose rank for fifth card
ways_choose_suit1 = comb(4, 1, exact=True)   # Choose suit for fifth card

favorable_outcomes_4kind = ways_choose_rank4 * ways_choose_suits4 * ways_choose_rank1 * ways_choose_suit1
print(f"Number of ways to get Four of a Kind: {favorable_outcomes_4kind}")

# 3. Calculate the probability
prob_4kind = favorable_outcomes_4kind / total_hands
print(f"Probability of being dealt Four of a Kind: {prob_4kind:.8f}")
print(f"Approximately 1 in {1/prob_4kind:,.0f}")
```
:::

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

:::{dropdown} Python Implementation
```{code-cell} ipython3
# Poker: Full House probability
# Total hands (already calculated)
total_hands = comb(52, 5, exact=True)

# Step 1: Choose rank for the three cards
ways_choose_rank3 = comb(13, 1, exact=True)
# Step 2: Choose 3 suits for that rank
ways_choose_suits3 = comb(4, 3, exact=True)
# Step 3: Choose rank for the pair (from remaining 12 ranks)
ways_choose_rank2 = comb(12, 1, exact=True)
# Step 4: Choose 2 suits for that rank
ways_choose_suits2 = comb(4, 2, exact=True)

favorable_outcomes_fullhouse = ways_choose_rank3 * ways_choose_suits3 * ways_choose_rank2 * ways_choose_suits2
print(f"Number of ways to get a Full House: {favorable_outcomes_fullhouse}")

# Calculate the probability
prob_fullhouse = favorable_outcomes_fullhouse / total_hands
print(f"Probability of being dealt a Full House: {prob_fullhouse:.8f}")
print(f"Approximately 1 in {1/prob_fullhouse:,.0f}")
```
:::

+++

## Quick Reference: Which Counting Technique Should I Use?

One of the most common challenges is deciding which formula to apply. Use this decision guide:

### Decision Questions

**START HERE:** I need to count arrangements or selections

1. **Does ORDER matter?**
   - **YES** → Use **PERMUTATIONS**
     - Can items repeat? (e.g., same person in multiple positions?)
       - NO → Permutation without repetition: $P(n,k) = \frac{n!}{(n-k)!}$
       - YES → Permutation with repetition: $n^k$ or multinomial coefficient
   - **NO** → Use **COMBINATIONS**
     - Can items repeat? (e.g., multiple items of same type?)
       - NO → Combination without repetition: $C(n,k) = \binom{n}{k} = \frac{n!}{k!(n-k)!}$
       - YES → Combination with repetition: $\binom{n+k-1}{k}$

### Quick Reference Table

| Scenario | Order? | Repeat? | Technique | Formula |
|----------|--------|---------|-----------|---------|
| Race podium (1st, 2nd, 3rd from 8 runners) | YES | NO | Permutation | $P(8,3) = \frac{8!}{5!}$ |
| Committee of 3 from 10 people | NO | NO | Combination | $\binom{10}{3}$ |
| Arranging MISSISSIPPI | YES | YES | Perm. with rep. | $\frac{11!}{1!4!4!2!}$ |
| Choosing 12 donuts from 4 types | NO | YES | Comb. with rep. | $\binom{15}{12}$ |
| 5-card poker hand from 52 cards | NO | NO | Combination | $\binom{52}{5}$ |
| License plate: 3 letters, 4 digits | YES | YES | Multiplication | $26^3 \times 10^4$ |

### Common Examples by Type

**Permutations (order matters):**
- Arranging books on a shelf
- Assigning people to different roles/positions
- Creating a password where position matters
- Race results (who finishes 1st, 2nd, 3rd)

**Combinations (order doesn't matter):**
- Selecting a committee or team
- Choosing lottery numbers
- Dealing poker hands
- Selecting pizza toppings

**With repetition:**
- Rolling dice multiple times
- Choosing items where you can pick the same type multiple times
- Drawing cards with replacement

**Without repetition:**
- Dealing cards (can't deal same card twice)
- Choosing distinct committee members
- Assigning people to positions (one person per position)

+++

## Chapter Summary

### Key Takeaways

**The core insight:** Systematic counting techniques transform complex probability problems into manageable calculations. When outcomes are equally likely, $P(E) = \frac{|E|}{|S|}$ — but determining $|E|$ and $|S|$ requires methodical counting.

**The fundamental techniques:**

1. **Multiplication Principle:** Sequential choices multiply
   - If task has $k$ steps with $n_1, n_2, \ldots, n_k$ options each, total ways = $n_1 \times n_2 \times \cdots \times n_k$
   - Foundation for all other counting methods

2. **Permutations** ($P(n,k) = \frac{n!}{(n-k)!}$): **Order matters**, no repetition
   - Race podiums, passwords with distinct characters, arranging books
   - Special case: $P(n,n) = n!$ for arranging all $n$ objects

3. **Combinations** ($\binom{n}{k} = \frac{n!}{k!(n-k)!}$): **Order doesn't matter**, no repetition
   - Committees, lottery numbers, poker hands
   - Related to permutations: $C(n,k) = \frac{P(n,k)}{k!}$ (divide out ordering)

4. **With Repetition:**
   - **Permutations with repetition:** Multinomial coefficients for identical objects (MISSISSIPPI)
   - **Combinations with repetition:** Stars and bars method for choosing with replacement

### Why This Matters

Counting techniques are essential for:

- **Games and gambling:** Computing odds in poker, lottery, dice games
- **Cryptography:** Calculating keyspace sizes and brute-force attack complexity
- **Data science:** Understanding sample sizes, bootstrap methods, combinatorial optimization
- **Everyday decisions:** Evaluating risks when outcomes are equally likely

### Common Pitfalls to Avoid

1. **Confusing permutations and combinations:** Always ask "does order matter?"
2. **Misunderstanding "without repetition":** It means distinct positions/slots, not sampling without replacement
3. **Forgetting to divide by k!:** When converting permutations to combinations
4. **Overlooking repeated elements:** MISSISSIPPI needs multinomial, not simple $n!$

### Python Tools

```python
import math
from scipy.special import perm, comb

math.factorial(n)                  # n!
perm(n, k, exact=True)            # P(n,k)
comb(n, k, exact=True)            # C(n,k)
comb(n+k-1, k, exact=True)        # Combinations with repetition
```

Mastering these counting techniques provides a powerful toolkit for tackling a wide range of probability problems. In the next chapter, we will move on to exploring probabilities when events are not independent, introducing the concept of Conditional Probability.

+++

## Exercises

1. **Multiplication Principle:** A password must contain:
   - 3 letters (26 choices each, case-insensitive)
   - 2 digits (0-9)
   - 1 special character (!  @, #, $, %)

   How many different passwords are possible if:
   a) Characters can repeat
   b) All characters must be distinct

   ```{admonition} Answer
   :class: dropdown

   **a) With repetition allowed:**

   Using the Multiplication Principle:
   - Letters: $26 \times 26 \times 26 = 26^3$
   - Digits: $10 \times 10 = 10^2$
   - Special char: $5$ choices

   Total: $26^3 \times 10^2 \times 5 = 17{,}576 \times 100 \times 5 = 8{,}788{,}000$

   **b) All distinct:**

   - First letter: 26 choices
   - Second letter: 25 choices (can't reuse first)
   - Third letter: 24 choices
   - First digit: 10 choices
   - Second digit: 9 choices (can't reuse first digit)
   - Special char: 5 choices

   Total: $26 \times 25 \times 24 \times 10 \times 9 \times 5 = 7{,}020{,}000$
   ```

2. **Permutations:** A class has 12 students. In how many ways can:
   a) A president, vice president, and secretary be chosen (different roles)?
   b) An unordered committee of 3 students be formed?
   c) Verify that your answer to (a) equals your answer to (b) multiplied by 3!

   ```{admonition} Answer
   :class: dropdown

   **a) Ordered selection (different roles) — Permutation:**

   $$P(12, 3) = \frac{12!}{(12-3)!} = \frac{12!}{9!} = 12 \times 11 \times 10 = 1{,}320$$

   **b) Unordered selection (committee) — Combination:**

   $$C(12, 3) = \binom{12}{3} = \frac{12!}{3! \cdot 9!} = \frac{12 \times 11 \times 10}{3 \times 2 \times 1} = \frac{1{,}320}{6} = 220$$

   **c) Verification:**

   $C(12, 3) \times 3! = 220 \times 6 = 1{,}320 = P(12, 3)$ ✓

   This confirms that $P(n,k) = C(n,k) \times k!$ — permutations count all orderings of each combination.
   ```

3. **Permutations with Repetition:** How many distinct arrangements can be made from the letters in:
   a) STATISTICS
   b) PROBABILITY

   ```{admonition} Answer
   :class: dropdown

   **a) STATISTICS:**

   Total letters: 10
   - S: 3
   - T: 3
   - A: 1
   - I: 2
   - C: 1

   Number of distinct arrangements:
   $$\frac{10!}{3! \cdot 3! \cdot 1! \cdot 2! \cdot 1!} = \frac{3{,}628{,}800}{6 \times 6 \times 1 \times 2 \times 1} = \frac{3{,}628{,}800}{72} = 50{,}400$$

   **b) PROBABILITY:**

   Total letters: 11
   - P: 1
   - R: 1
   - O: 1
   - B: 2
   - A: 1
   - I: 2
   - L: 1
   - T: 1
   - Y: 1

   Number of distinct arrangements:
   $$\frac{11!}{1! \cdot 1! \cdot 1! \cdot 2! \cdot 1! \cdot 2! \cdot 1! \cdot 1! \cdot 1!} = \frac{39{,}916{,}800}{2 \times 2} = \frac{39{,}916{,}800}{4} = 9{,}979{,}200$$
   ```

4. **Combinations:** A standard deck has 52 cards. How many different 5-card poker hands:
   a) Are possible in total?
   b) Contain all hearts?
   c) Contain exactly 2 aces?

   ```{admonition} Answer
   :class: dropdown

   **a) Total 5-card hands:**

   $$\binom{52}{5} = \frac{52!}{5! \cdot 47!} = \frac{52 \times 51 \times 50 \times 49 \times 48}{120} = 2{,}598{,}960$$

   **b) All hearts:**

   Choose 5 from 13 hearts:
   $$\binom{13}{5} = \frac{13!}{5! \cdot 8!} = \frac{13 \times 12 \times 11 \times 10 \times 9}{120} = 1{,}287$$

   **c) Exactly 2 aces:**

   - Choose 2 aces from 4: $\binom{4}{2}$
   - Choose 3 non-aces from 48: $\binom{48}{3}$

   $$\binom{4}{2} \times \binom{48}{3} = 6 \times \frac{48 \times 47 \times 46}{6} = 6 \times 17{,}296 = 103{,}776$$
   ```

5. **Combinations with Repetition:** An ice cream shop offers 8 flavors. How many ways can you order:
   a) 3 scoops if each must be a different flavor?
   b) 3 scoops if flavors can repeat (stars and bars)?
   c) If you order 3 chocolate scoops, which formula applies?

   ```{admonition} Answer
   :class: dropdown

   **a) All different flavors (without repetition):**

   Choose 3 flavors from 8 (order doesn't matter for scoops):
   $$\binom{8}{3} = \frac{8!}{3! \cdot 5!} = \frac{8 \times 7 \times 6}{6} = 56$$

   **b) Flavors can repeat (with repetition):**

   Using stars and bars: $n = 8$ flavors, $k = 3$ scoops
   $$\binom{n+k-1}{k} = \binom{8+3-1}{3} = \binom{10}{3} = \frac{10 \times 9 \times 8}{6} = 120$$

   **c) Three chocolate scoops:**

   This is counted in (b) as one of the 120 possibilities. The "combinations with repetition" formula applies because we're choosing 3 items from 8 types where the same type can be chosen multiple times.
   ```

6. **Mixed Application:** You roll a fair die 4 times. What is the probability of getting exactly 2 sixes?

   *Hint: First count favorable outcomes using combinations to choose which 2 rolls are sixes, then calculate probability.*

   ```{admonition} Answer
   :class: dropdown

   **Step 1: Count favorable outcomes**

   - Choose which 2 of the 4 rolls are sixes: $\binom{4}{2} = 6$ ways
   - For each choice:
     - The 2 chosen positions must be 6: probability $(1/6)^2$
     - The 2 other positions must not be 6: probability $(5/6)^2$

   **Step 2: Calculate probability**

   Each specific sequence with exactly 2 sixes has probability:
   $$\left(\frac{1}{6}\right)^2 \times \left(\frac{5}{6}\right)^2 = \frac{1 \times 25}{36 \times 36} = \frac{25}{1{,}296}$$

   There are $\binom{4}{2} = 6$ such sequences, so:
   $$P(\text{exactly 2 sixes}) = 6 \times \frac{25}{1{,}296} = \frac{150}{1{,}296} = \frac{75}{648} = \frac{25}{216} \approx 0.1157$$

   **Interpretation:** This uses combinations without repetition to choose which positions are sixes (positions 1,2 vs 1,3 vs 1,4 etc. are different), not because we're sampling without replacement.
   ```

```{code-cell} ipython3

```
