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

:::{admonition} Common confusion: "Without repetition" in permutations
:class: warning

The phrase "without repetition" often confuses students because it sounds like "sampling without replacement" (drawing balls from a bag where each ball can only be drawn once).

**What it actually means:** Each **object can only occupy one position** — you don't use the same object in multiple positions.

**Example:** Awarding medals (Gold, Silver, Bronze) to 8 runners
- We use $P(8, 3)$ to choose which runner gets which medal
- "Without repetition" means each runner can only win one medal (you can't give Gold AND Silver to the same person)
- Order matters: Runner A getting Gold ≠ Runner A getting Silver
- We're not "running out of runners" — it's about distinct positions being filled by distinct objects

**Another example:** Arranging 5 books chosen from 10 on a shelf
- "Without repetition" means each book appears only once in the arrangement
- You can't put the same book in two different positions on the shelf
- It's about each position having a distinct object, not about depleting a supply
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

**Formula:** The number of distinct permutations of $n$ objects where there are $n_1$ identical objects of type 1, $n_2$ identical objects of type 2, ..., and $n_k$ identical objects of type k (such that $n_1 + n_2 + \dots + n_k = n$) is:

$ \frac{n!}{n_1! n_2! \dots n_k!} $

**Why divide by the factorials of repeated objects?**

If all $n$ objects were distinct, there would be $n!$ different arrangements. But when some objects are identical, many of these arrangements look the same:

- The $n_1$ identical objects of type 1 can be rearranged among themselves in $n_1!$ ways without creating a new distinct arrangement
- Similarly for type 2 ($n_2!$ ways), type 3 ($n_3!$ ways), etc.
- Each distinct arrangement we want to count is being overcounted by $n_1! \times n_2! \times \dots \times n_k!$ times

Therefore, we divide $n!$ by this product to get the number of truly distinct arrangements.

**Example:** How many distinct ways can the letters in the word "MISSISSIPPI" be arranged?

* Total letters $n = 11$
* M: $n_1 = 1$
* I: $n_2 = 4$
* S: $n_3 = 4$
* P: $n_4 = 2$
(Check: $1 + 4 + 4 + 2 = 11$)

The number of distinct arrangements is:

$ \frac{11!}{1! 4! 4! 2!} $

**Intuition:** If all letters were different, we'd have 11! arrangements. But swapping the 4 I's among themselves doesn't create a new word (neither does swapping the 4 S's or the 2 P's). We divide out this overcounting.

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

:::{admonition} Common confusion: "Without repetition" in combinations
:class: warning

The phrase "without repetition" often confuses students because it sounds like "sampling without replacement" (drawing balls from a bag where each ball can only be drawn once).

**What it actually means:** Each **object can only be selected once** — you don't include the same object multiple times in your selection.

**Example:** Choosing a committee of 3 from 10 people
- We use $\binom{10}{3}$ to select which 3 people form the committee
- "Without repetition" means each person can only be selected once (you can't have Alice appear twice on the committee)
- Order doesn't matter: {Alice, Bob, Carol} = {Carol, Alice, Bob}
- We're not "running out of people" — it's about selecting distinct objects for distinct slots

**Another example:** Flipping a coin 4 times and asking "how many ways can we get exactly 2 heads?"
- We use $\binom{4}{2}$ to choose which 2 **positions** (flip 1, 2, 3, or 4) will be heads
- "Without repetition" means we don't choose the same position twice (position 1 can't be both H and T)
- We're counting arrangements like HHTT, HTHT, HTTH, etc.
- We're not "running out of heads" — each flip is independent
- It's about distinct positions/slots, not about depleting a supply
:::

**Formula:** The number of combinations of $n$ distinct objects taken $k$ at a time is denoted by $C(n, k)$, $_nC_k$, $C^n_k$, or $\binom{n}{k}$ (read "n choose k") and is calculated as:

$ C(n, k) = \binom{n}{k} = \frac{n!}{k!(n-k)!} $

Notice that $C(n, k) = \frac{P(n, k)}{k!}$. This is because for every combination of $k$ objects, there are $k!$ ways to order them (permutations). We divide the number of permutations $P(n,k)$ by $k!$ to remove the effect of order.

**Why divide by k!?**

When choosing 3 people {Alice, Bob, Carol} from a group for a committee:
- **Permutations count** all orderings: ABC, ACB, BAC, BCA, CAB, CBA (6 different sequences)
- But for a committee, **all 6 of these represent the same combination** — it doesn't matter who we chose first
- Since there are 3! = 6 ways to arrange any 3 people, we divide the number of permutations by 6 to get the number of combinations
- This removes the overcounting caused by different orderings of the same group

**Example:** How many ways can a committee of 3 people be chosen from a group of 10 people?

Here, we are choosing $k=3$ people from $n=10$, and the order in which they are chosen doesn't matter.

Using the formula:

$ C(10, 3) = \binom{10}{3} = \frac{10!}{3!(10-3)!} = \frac{10!}{3!7!} = \frac{10 \times 9 \times 8}{3 \times 2 \times 1} $

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

**Formula:** The number of combinations with repetition of $n$ types of objects taken $k$ at a time is:

$ \binom{n+k-1}{k} = \frac{(n+k-1)!}{k!(n-1)!} $

:::{admonition} Intuition: The "Stars and Bars" method
:class: tip

This formula might look strange, but there's a beautiful visual way to understand it!

Imagine you want to distribute $k$ identical objects into $n$ different bins (or types). We can represent this using **stars** (★) for objects and **bars** (|) as dividers between bins.

**Example:** Choosing 12 donuts from 4 types is like arranging 12 stars and 3 bars:

```
★★|★★★★|★★★★★|★
```

This represents:
- Type 1 (plain): 2 donuts (stars before first bar)
- Type 2 (chocolate): 4 donuts (stars between first and second bar)
- Type 3 (glazed): 5 donuts (stars between second and third bar)
- Type 4 (jelly): 1 donut (stars after third bar)

**Key insight:**
- We have $k = 12$ stars (the donuts we're choosing)
- We need $n-1 = 3$ bars to create $n = 4$ sections (types)
- Total positions: $12 + 3 = 15$ objects to arrange
- We need to choose where to place the $k = 12$ stars (or equivalently, where to place the $n-1 = 3$ bars)
- Number of ways = $\binom{15}{12} = \binom{15}{3} = \binom{n+k-1}{k} = \binom{n+k-1}{n-1}$

This is why the formula is $\binom{n+k-1}{k}$ — we're choosing positions for $k$ items among $n+k-1$ total positions!
:::

**Example:** A bakery offers 4 types of donuts (plain, chocolate, glazed, jelly). How many different ways can you select a dozen (12) donuts?

Here, $n=4$ (types of donuts) and we are choosing $k=12$ donuts. The order doesn't matter, and we can choose multiple donuts of the same type.

Using the formula:

$ \binom{4+12-1}{12} = \binom{15}{12} = \frac{15!}{12!(15-12)!} = \frac{15!}{12!3!} = \frac{15 \times 14 \times 13}{3 \times 2 \times 1} $

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
