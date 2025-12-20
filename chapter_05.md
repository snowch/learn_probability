# Chapter 5: Bayes' Theorem and Independence

+++

In the previous chapter, we explored conditional probability – how the probability of an event changes given that another event has occurred. Now, we'll delve into one of the most powerful and widely applicable results stemming from conditional probability: **Bayes' Theorem**. This theorem provides a formal way to update our beliefs (probabilities) in light of new evidence. We will also formally define and explore the concept of **independence** between events, a crucial idea for simplifying probability calculations.

+++

## Learning Objectives:
* Understand the derivation and interpretation of Bayes' Theorem.
* Distinguish between prior and posterior probabilities.
* Apply Bayes' Theorem to solve problems, particularly diagnostic testing scenarios.
* Define and test for the independence of events.
* Understand the concept of conditional independence.
* Implement Bayesian updates and independence checks using Python simulations.

+++

## 1. Bayes' Theorem: Derivation and Interpretation

+++

Bayes' Theorem provides a way to "reverse" conditional probabilities. If we know $P(B|A)$, Bayes' Theorem helps us find $P(A|B)$. It's named after Reverend Thomas Bayes (1701-1761), who first provided an equation that allows new evidence to update beliefs.

**Derivation:**

Recall the definition of conditional probability:

1.  $P(A|B) = \frac{P(A \cap B)}{P(B)}$, provided $P(B) > 0$.
2.  $P(B|A) = \frac{P(B \cap A)}{P(A)}$, provided $P(A) > 0$.

Since $P(A \cap B) = P(B \cap A)$, we can rearrange these equations:

1.  $P(A \cap B) = P(A|B) P(B)$
2.  $P(B \cap A) = P(B|A) P(A)$

Setting them equal gives:

$P(A|B) P(B) = P(B|A) P(A)$

Dividing by $P(B)$ (assuming $P(B) > 0$), we get **Bayes' Theorem**:

$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$

**Interpretation:**

Let's think of A as an event or hypothesis we are interested in (e.g., "a patient has a specific disease," "a coin is biased") and B as new evidence or data observed (e.g., "the patient tested positive," "we observed 8 heads in 10 flips").

* $P(A)$: **Prior Probability**. Our initial belief about the probability of A *before* seeing the evidence B.
* $P(B|A)$: **Likelihood**. The probability of observing the evidence B *given* that our hypothesis A is true.
* $P(B)$: **Probability of Evidence**. The overall probability of observing the evidence B, regardless of whether A is true or not. This often requires using the Law of Total Probability (from Chapter 4): $P(B) = P(B|A)P(A) + P(B|A^c)P(A^c)$.
* $P(A|B)$: **Posterior Probability**. Our updated belief about the probability of A *after* observing the evidence B.

Bayes' Theorem tells us how to update our prior belief $P(A)$ to a posterior belief $P(A|B)$ based on the likelihood of the evidence $P(B|A)$ and the overall probability of the evidence $P(B)$.

+++

## 2. Updating Beliefs: Prior and Posterior Probabilities

+++

The core idea of Bayesian thinking is updating beliefs. We start with a prior belief, gather data (evidence), and update our belief to a posterior. This posterior can then become the prior for the next piece of evidence.

**Example:** Imagine you have a website and you're testing a new ad banner.

* **Hypothesis (A):** The new ad banner is effective (e.g., has a click-through rate > 5%).
* **Prior ($P(A)$):** Based on previous ad campaigns, you might initially believe there's a 30% chance the new ad is effective. So, $P(A) = 0.30$.
* **Evidence (B):** You observe a visitor's Browse history (e.g., they previously visited related product pages).
* **Likelihood ($P(B|A)$):** The probability that a visitor has this Browse history *given* the ad is effective. Perhaps effective ads are better targeted, so this might be high, say $P(B|A) = 0.70$.
* **Likelihood ($P(B|A^c)$):** The probability that a visitor has this Browse history *given* the ad is *not* effective. This might be lower, say $P(B|A^c) = 0.20$.
* **Probability of Evidence ($P(B)$):** Using the Law of Total Probability:
    $P(B) = P(B|A)P(A) + P(B|A^c)P(A^c)$
    $P(B) = (0.70)(0.30) + (0.20)(1 - 0.30)$
    $P(B) = 0.21 + (0.20)(0.70) = 0.21 + 0.14 = 0.35$
* **Posterior ($P(A|B)$):** Now apply Bayes' Theorem:
    $P(A|B) = \frac{P(B|A) P(A)}{P(B)} = \frac{(0.70)(0.30)}{0.35} = \frac{0.21}{0.35} = 0.60$

After observing the visitor's Browse history, your belief that the ad is effective increased from 30% (prior) to 60% (posterior).

+++

## 3. Applications: The Diagnostic Test Example

+++

One of the most classic and intuitive applications of Bayes' Theorem is in interpreting the results of medical diagnostic tests.

**Scenario:**
* A certain disease affects 1% of the population. (Prevalence)
* A test for the disease has 95% accuracy:
    * If a person *has* the disease, the test correctly identifies it 95% of the time. (Sensitivity)
    * If a person *does not have* the disease, the test correctly identifies it 95% of the time. (Specificity)

```{admonition} Sensitivity and Specificity 
:class: dropdown
Looking at the origins and definitions of the words "sensitivity" and "specificity" can definitely help reinforce their meanings in this context.

1. **Sensitivity:**  
   * **Origin:** Comes from the Latin word sentire, meaning "to feel" or "to perceive."  
   * **General Meaning:** The quality or condition of being sensitive; responsiveness to stimuli.  
   * **Connection to the Test:** Think of the test as needing to "feel" or "perceive" the presence of the disease. A highly **sensitive** test has a strong ability to *detect* the disease when it is actually there. It's responsive to the "stimulus" of the disease. If the disease is present, a sensitive test is likely to react (give a positive result). This aligns with its technical meaning of correctly identifying true positives.  
2. **Specificity:**  
   * **Origin:** Comes from the Latin word specificus, derived from species (meaning "kind" or "sort") and facere (meaning "to make"). Essentially, "making of a particular kind."  
   * **General Meaning:** The quality of being specific; restricted to a particular item, condition, or effect; being precise or exact.  
   * **Connection to the Test:** Think of the test as being designed for one *specific* target – the disease. A highly **specific** test is precise and only reacts to that *particular* target. It does *not* react to other things (like the absence of the disease or other conditions). It correctly identifies individuals who do *not* have the specific target disease (giving a negative result). This aligns with its technical meaning of correctly identifying true negatives.

**How it Helps Understanding:**

* **Sensitivity:** Relates to the test's ability to **sense** or **detect** the disease if it's present. High sensitivity means good detection.  
* **Specificity:** Relates to the test being **specific** or **precise** to only the disease in question. High specificity means the test only flags the *specific* condition it's looking for and avoids flagging healthy people.

So, the origins help frame the concepts: sensitivity is about *detection power*, while specificity is about *precision* and *target accuracy*.
```

**Question:** If a randomly selected person tests positive, what is the probability they actually have the disease?

**Let's define the events:**
* $D$: The person has the disease.
* $D^c$: The person does not have the disease.
* $Pos$: The person tests positive.
* $Neg$: The person tests negative.

**What we know:**
* $P(D) = 0.01$ (Prior probability of having the disease - Prevalence)
* $P(D^c) = 1 - P(D) = 0.99$
* $P(Pos|D) = 0.95$ (Probability of testing positive *given* you have the disease - Sensitivity)
* $P(Neg|D) = 1 - P(Pos|D) = 0.05$ (False Negative Rate)
* $P(Neg|D^c) = 0.95$ (Probability of testing negative *given* you don't have the disease - Specificity)
* $P(Pos|D^c) = 1 - P(Neg|D^c) = 0.05$ (False Positive Rate)

**What we want to find:** $P(D|Pos)$ (The probability of having the disease *given* a positive test result).

**Apply Bayes' Theorem:**

$P(D|Pos) = \frac{P(Pos|D) P(D)}{P(Pos)}$

We need to find $P(Pos)$. Use the Law of Total Probability:

$$
\begin{align*}
P(\text{Pos}) &= P(\text{Pos}|D)P(D) + P(\text{Pos}|D^c)P(D^c) \\
&= (0.95)(0.01) + (0.05)(0.99) \\
&= 0.0095 + 0.0495 \\
&= 0.0590
\end{align*}
$$

Now substitute into Bayes' Theorem:
$P(D|Pos) = \frac{(0.95)(0.01)}{0.0590} = \frac{0.0095}{0.0590} \approx 0.161$

**Interpretation:** Even with a positive test result from a 95% accurate test, the probability of actually having the disease is only about 16.1%! This seems counter-intuitive but highlights the strong influence of the low prior probability (prevalence) of the disease. Most positive tests come from the large group of healthy people who receive a false positive, rather than the small group of sick people who receive a true positive.

+++

## 4. Independence of Events

+++

Two events A and B are said to be **independent** if the occurrence (or non-occurrence) of one event does not affect the probability of the other event occurring.

I.e. Two events A and B are said to be **independent** if knowing whether one event happened tells you nothing about whether the other event will happen. Their probabilities are not linked.

### 4.1. Formal Definition

The formal mathematical definition of independence between two eventsis that Events A and B are independent if and only if:
$P(A \cap B) = P(A) P(B)$

```{admonition} Explanation
:class: dropdown

Events **A** and **B** are **independent** if and only if the probability that *both* events happen is equal to the product of their individual probabilities.

Mathematically:
$P(A \cap B) = P(A) \times P(B)$

* $P(A \cap B)$ means "the probability of both A AND B occurring" (the intersection of A and B).
* $P(A)$ is the probability of event A occurring.
* $P(B)$ is the probability of event B occurring.

**Why does this formula capture independence?**
Think about it this way: If the events truly don't influence each other, the chance of them *both* happening should just be a simple multiplication of their individual chances. If there *was* some influence (dependence), this multiplication wouldn't accurately reflect the combined probability.
```

```{admonition} Example: Flipping a Fair Coin Twice 🪙
:class: dropdown

Let's consider flipping a fair coin two times.

* **Event A**: Getting heads (H) on the **first flip**.
* **Event B**: Getting heads (H) on the **second flip**.

We want to know if these two events are independent.

1.  **Calculate $P(A)$**:
    The probability of getting heads on a single flip of a fair coin is $\frac{1}{2}$.
    So, $P(A) = \frac{1}{2}$.

2.  **Calculate $P(B)$**:
    The outcome of the second flip is not affected by the first flip. The coin has no memory. So, the probability of getting heads on the second flip is also $\frac{1}{2}$.
    So, $P(B) = \frac{1}{2}$.

3.  **Calculate $P(A \cap B)$**:
    This is the probability of getting heads on the first flip **AND** heads on the second flip (HH).
    The possible outcomes when flipping a coin twice are: HH, HT, TH, TT. There are 4 equally likely outcomes.
    Only one of these outcomes is HH.
    So, $P(A \cap B) = \frac{1}{4}$.

4.  **Check the Independence Formula**:
    Now we check if $P(A \cap B) = P(A) \times P(B)$.
    * $P(A) \times P(B) = \frac{1}{2} \times \frac{1}{2} = \frac{1}{4}$
    * We already found that $P(A \cap B) = \frac{1}{4}$.

5.  **Conclusion**:
    Since $P(A \cap B) = P(A) \times P(B)$ (because $\frac{1}{4} = \frac{1}{4}$), the events A (heads on the first flip) and B (heads on the second flip) are **independent**.

This makes intuitive sense: the result of the first coin flip doesn't change the probability of getting heads or tails on the second flip.
```

### 4.2. Alternative Definition (using conditional probability)

If $P(B) > 0$, A and B are independent if and only if:

$P(A|B) = P(A)$

Similarly, if $P(A) > 0$, independence means:

$P(B|A) = P(B)$

This definition aligns with the intuition: knowing B occurred doesn't change the probability of A.

```{admonition} Example: Fair Die Roll
:class: dropdown

| Event Definition                                  | Probability Calculation |
| :------------------------------------------------ | :---------------------- |
| **A**: "rolling an even number" = {2, 4, 6}       | $P(A) = 3/6 = 1/2$      |
| **B**: "rolling a number > 4" = {5, 6}            | $P(B) = 2/6 = 1/3$      |
| **A ∩ B**: "even number > 4" = {6}                | $P(A \cap B) = 1/6$     |

Let's check for independence:

Is $P(A \cap B) = P(A) P(B)$?

$$
\begin{align*}
P(A \cap B) &\stackrel{?}{=} P(A) P(B) \\
\frac{1}{6} &\stackrel{?}{=} \left(\frac{1}{2}\right) \times \left(\frac{1}{3}\right) \\
\frac{1}{6} &= \frac{1}{6} \quad \checkmark
\end{align*}
$$

Yes, the events A and B are independent. 

Knowing the roll is greater than 4 doesn't change the probability that it's even - it's still 1/2: 

$$
\begin{align*}
P(A|B) &= \frac{P(A \cap B)}{P(B)} \\
&= \frac{1/6}{1/3} \\
&= \frac{1}{6} \times 3 \\
&= \frac{3}{6} \\
&= \frac{1}{2} \\
&= P(A)
\end{align*}
$$

I.e. $P(A|B) = P(A)$
```

```{admonition} Example: Drawing Cards (Without Replacement)
:class: dropdown

Let A be the event "the first card drawn is an Ace". $P(A) = 4/52$.
Let B be the event "the second card drawn is an Ace".

Are A and B independent? Intuitively, no. If the first card was an Ace, the probability the second is an Ace changes.

Let's calculate $P(B)$. Using the Law of Total Probability:

$$
\begin{align*}
P(B) &= P(B|A)P(A) + P(B|A^c)P(A^c) \\
&= \left( \frac{3}{51} \right) \left( \frac{4}{52} \right) + \left( \frac{4}{51} \right) \left( \frac{48}{52} \right) \\
&= \frac{3 \times 4}{51 \times 52} + \frac{4 \times 48}{51 \times 52} \\
&= \frac{12}{2652} + \frac{192}{2652} \\
&= \frac{12 + 192}{2652} \\
&= \frac{204}{2652} \\
&= \frac{4}{52} \\
&= \frac{1}{13}
\end{align*}
$$

So, $P(B) = 1/13$.

Now let's calculate the intersection: $P(A \cap B) = P(\text{first is Ace AND second is Ace})$

$$
\begin{align*}
P(A \cap B) &= P(B|A)P(A) \\
&= \left( \frac{3}{51} \right) \left( \frac{4}{52} \right) \\
&= \frac{3 \times 4}{51 \times 52} \\
&= \frac{12}{2652} \\
&= \frac{1}{221}
\end{align*}
$$

Check for independence: 

Is $P(A \cap B) = P(A)P(B)$?

$$
\begin{align*}
\frac{1}{221} &\stackrel{?}{=} \left( \frac{4}{52} \right) \times \left( \frac{4}{52} \right) \\
&= \left( \frac{1}{13} \right) \times \left( \frac{1}{13} \right) \\
&= \frac{1}{169}
\end{align*}
$$

As expected, the events are **not** independent.
```

**Important Note:** Do not confuse independence with mutual exclusivity.
* **Mutually exclusive** events cannot happen together ($A \cap B = \emptyset$, so $P(A \cap B) = 0$).
* **Independent** events *can* happen together, but one doesn't affect the other's probability.
If two events A and B have non-zero probabilities, they *cannot* be both mutually exclusive and independent. If they were mutually exclusive, $P(A \cap B) = 0$. If they were independent, $P(A \cap B) = P(A)P(B) > 0$. This is a contradiction.

+++

## 5. Conditional Independence

+++

Sometimes, two events A and B might not be independent overall, but they become independent *given* some other event C. This is called **conditional independence**.

**Formal Definition:**

Events A and B are conditionally independent given event C (where $P(C) > 0$) if:

$P(A \cap B | C) = P(A|C) P(B|C)$

```{admonition} Reminder 
:class: tip dropdown

For independent events: $P(A \cap B) = P(A) P(B)$.

Note that we are adding $| C$ to each part.  Try to figure out why the formula changes for conditional independence.
```

**Alternative Definition:**
If $P(B|C) > 0$, conditional independence means:

$P(A | B \cap C) = P(A|C)$

```{admonition} Reminder 
:class: tip dropdown

For independent events: $P(A|B) = P(A)$.

Try to figure out why the formula changes for conditional independence.
```

Knowing B occurred provides no additional information about A *if we already know C occurred*.

**Example:** Consider two different coins, one fair (Coin F) and one biased to land heads 75% of the time (Coin B).
* Let $H_1$ be the event "the first flip is Heads".
* Let $H_2$ be the event "the second flip is Heads".

Are $H_1$ and $H_2$ independent? It depends on whether we know which coin we are flipping!

```{admonition} Scenario 1: We pick a coin at random (50% chance each) and flip it twice. 
:class: dropdown

Let's find $P(H_1)$ and $P(H_2)$.

$$
\begin{align*}
P(H_1) &= P(H_1|\text{Fair})P(\text{Fair}) + P(H_1|\text{Biased})P(\text{Biased}) \\
&= (0.5)(0.5) + (0.75)(0.5) \\
&= 0.25 + 0.375 \\
&= 0.625
\end{align*}
$$

By symmetry, $P(H_2) = 0.625$.

Now, let's find $P(H_1 \cap H_2) = P(\text{HH})$.

$P(\text{HH}) = P(\text{HH}|\text{Fair})P(\text{Fair}) + P(\text{HH}|\text{Biased})P(\text{Biased})$

Assuming flips are independent *given* the coin:

$$
\begin{align*}
P(\text{HH}) &= (0.5 \times 0.5)(0.5) + (0.75 \times 0.75)(0.5) \\
&= (0.25)(0.5) + (0.5625)(0.5) \\
&= 0.125 + 0.28125 \\
&= 0.40625
\end{align*}
$$

Check for independence: 

Is $P(H_1 \cap H_2) = P(H_1) P(H_2)$?

$$
\begin{align*}
0.40625 &\stackrel{?}{=} (0.625) \times (0.625) \\
&= 0.390625
\end{align*}
$$

They are **not** equal. $H_1$ and $H_2$ are **not** independent overall. If the first flip is heads, it slightly increases our belief we have the biased coin, thus increasing the probability the second flip is also heads.
```

```{admonition} Scenario 2: We know we are flipping the Fair coin (Event C = "Fair coin chosen").
:class: dropdown

* $ P(H_1 | C) = 0.5 $
* $ P(H_2 | C) = 0.5 $

$$
\begin{align*}
P(H_1 \cap H_2 | C) &= P(\text{HH} | \text{Fair}) \\
&= 0.5 \times 0.5 \\
&= 0.25 \quad \text{(assuming flips are independent for a given coin)}
\end{align*}
$$

Check for conditional independence: Is $P(H_1 \cap H_2 | C) = P(H_1|C) P(H_2|C)$?
$0.25 \stackrel{?}{=} (0.5) \times (0.5)$
$0.25 = 0.25$. Yes. $H_1$ and $H_2$ are **conditionally independent given** we chose the fair coin.
```

```{admonition} Scenario 3: We know we are flipping the Biased coin (Event C' = "Biased coin chosen").
:class: dropdown

* $P(H_1 | C') = 0.75$
* $P(H_2 | C') = 0.75$

$$
\begin{align*}
P(H_1 \cap H_2 | C') &= P(\text{HH} | \text{Biased}) \\
&= 0.75 \times 0.75 \\
&= 0.5625
\end{align*}
$$

Check for conditional independence: Is $P(H_1 \cap H_2 | C') = P(H_1|C') P(H_2|C')$?
$0.5625 \stackrel{?}{=} (0.75) \times (0.75)$
$0.5625 = 0.5625$. Yes. $H_1$ and $H_2$ are also **conditionally independent given** we chose the biased coin.
```

**Intuition:** Fuel efficiency might depend on tire pressure and engine size. These two factors might seem correlated overall (cars with bigger engines might tend to have specific tire pressure recommendations). However, *given a specific car model*, the effect of tire pressure on fuel efficiency might be independent of the effect of engine size (assuming the model already fixes the engine size).

+++

## 6. Hands-on Exercises

+++

### Exercise 5.1: Implementing Bayes' Theorem for Disease Test

+++

Let's verify the disease test calculation using Python. Define variables for the prior probability, sensitivity, and specificity, then implement the calculation for $P(D|Pos)$.

```{code-cell} ipython3
# Parameters
p_disease = 0.01        # P(D) - Prior probability (prevalence)
p_pos_given_disease = 0.95 # P(Pos|D) - Sensitivity
p_neg_given_no_disease = 0.95 # P(Neg|D^c) - Specificity

# Derived probabilities
p_no_disease = 1 - p_disease                 # P(D^c)
p_pos_given_no_disease = 1 - p_neg_given_no_disease # P(Pos|D^c) - False Positive Rate

# Calculate P(Pos) using Law of Total Probability
p_pos = (p_pos_given_disease * p_disease) + (p_pos_given_no_disease * p_no_disease)

# Calculate P(D|Pos) using Bayes' Theorem
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos

print(f"Prior P(Disease): {p_disease:.4f}")
print(f"Sensitivity P(Pos|Disease): {p_pos_given_disease:.4f}")
print(f"Specificity P(Neg|No Disease): {p_neg_given_no_disease:.4f}")
print(f"False Positive Rate P(Pos|No Disease): {p_pos_given_no_disease:.4f}")
print("-" * 30)
print(f"Overall P(Pos): {p_pos:.4f}")
print(f"Posterior P(Disease|Pos): {p_disease_given_pos:.4f}")

# What if the test is *negative*? Calculate P(Disease | Neg)
# P(Neg) = P(Neg|D)P(D) + P(Neg|D^c)P(D^c)
p_neg_given_disease = 1 - p_pos_given_disease # P(Neg|D) - False Negative Rate
p_neg = (p_neg_given_disease * p_disease) + (p_neg_given_no_disease * p_no_disease)

# P(D|Neg) = P(Neg|D)P(D) / P(Neg)
p_disease_given_neg = (p_neg_given_disease * p_disease) / p_neg

print("-" * 30)
print(f"Overall P(Neg): {p_neg:.4f}")
print(f"Posterior P(Disease|Neg): {p_disease_given_neg:.4f}")
print(f"Posterior P(No Disease|Neg) = {1 - p_disease_given_neg:.4f}")
```

### Exercise 5.2: Simulating Bayesian Updates

+++

Let's simulate the disease test scenario to build intuition. We'll create a population reflecting the disease prevalence, simulate their test results based on sensitivity/specificity, and then calculate the conditional probability directly from the simulated data.

```{code-cell} ipython3
import numpy as np
import pandas as pd

# Parameters
population_size = 1000000
p_disease = 0.01
p_pos_given_disease = 0.95
p_pos_given_no_disease = 0.05 # 1 - specificity

# Create population
# Assign actual disease status
has_disease = np.random.rand(population_size) < p_disease
num_diseased = np.sum(has_disease)
num_healthy = population_size - num_diseased

# Simulate test results
# Initialize test results array
tests_positive = np.zeros(population_size, dtype=bool)

# For those WITH the disease
tests_positive[has_disease] = np.random.rand(num_diseased) < p_pos_given_disease

# For those WITHOUT the disease
tests_positive[~has_disease] = np.random.rand(num_healthy) < p_pos_given_no_disease

# Create a DataFrame for easier analysis
data = pd.DataFrame({'Has_Disease': has_disease, 'Tests_Positive': tests_positive})
print(data.head())

# Calculate counts from the simulation
true_positives = np.sum(data['Has_Disease'] & data['Tests_Positive'])
false_positives = np.sum(~data['Has_Disease'] & data['Tests_Positive'])
total_positives = np.sum(data['Tests_Positive'])

# Calculate P(Disease | Positive) from simulation data
simulated_p_disease_given_pos = true_positives / total_positives

# Compare with theoretical calculation
print("\n--- Simulation Results ---")
print(f"Population Size: {population_size}")
print(f"Number Actually Diseased: {num_diseased}")
print(f"Number Actually Healthy: {num_healthy}")
print(f"Number Testing Positive: {total_positives}")
print(f"  - True Positives: {true_positives}")
print(f"  - False Positives: {false_positives}")
print("-" * 30)
print(f"Simulated P(Disease | Positive): {simulated_p_disease_given_pos:.4f}")
print(f"Theoretical P(Disease | Positive): {p_disease_given_pos:.4f}")
```

As the `population_size` increases, the simulated probability should converge to the theoretical probability calculated using Bayes' Theorem. This demonstrates how the theorem accurately reflects the underlying frequencies in a large population.

+++

### Exercise 5.3: Testing Independence from Data

+++

Let's simulate rolling two fair dice and check if the events "first die is even" and "sum is 7" are independent.

* Event A: First die is even. $P(A) = 1/2$.
* Event B: Sum is 7. The pairs are (1,6), (2,5), (3,4), (4,3), (5,2), (6,1). $P(B) = 6/36 = 1/6$.
* Event $A \cap B$: First die is even AND sum is 7. The pairs are (2,5), (4,3), (6,1). $P(A \cap B) = 3/36 = 1/12$.

Theoretical Check: Is $P(A \cap B) = P(A)P(B)$?
$1/12 \stackrel{?}{=} (1/2) \times (1/6)$
$1/12 = 1/12$. Yes, they are theoretically independent.

Now, let's check using simulation.

```{code-cell} ipython3
import numpy as np
import pandas as pd

num_rolls = 100000

# Simulate rolls
die1 = np.random.randint(1, 7, size=num_rolls)
die2 = np.random.randint(1, 7, size=num_rolls)
sums = die1 + die2

# Define events
event_A = (die1 % 2 == 0)  # First die is even
event_B = (sums == 7)     # Sum is 7

# Create DataFrame
rolls_data = pd.DataFrame({'Die1': die1, 'Die2': die2, 'Sum': sums, 'A': event_A, 'B': event_B})
print(rolls_data.head())

# Calculate probabilities from simulation
p_A_sim = np.mean(event_A)
p_B_sim = np.mean(event_B)
p_A_intersect_B_sim = np.mean(event_A & event_B)

# Check independence condition
print("\n--- Independence Check from Simulation ---")
print(f"Simulated P(A): {p_A_sim:.4f} (Theoretical: 0.5000)")
print(f"Simulated P(B): {p_B_sim:.4f} (Theoretical: {1/6:.4f})")
print(f"Simulated P(A intersect B): {p_A_intersect_B_sim:.4f} (Theoretical: {1/12:.4f})")
print("-" * 30)
print(f"P(A) * P(B) = {p_A_sim * p_B_sim:.4f}")
print(f"Is P(A intersect B) approx equal to P(A) * P(B)? {'Yes' if np.isclose(p_A_intersect_B_sim, p_A_sim * p_B_sim, atol=0.005) else 'No'}") # Use np.isclose for floating point comparison

# Alternative check: Is P(A|B) approx equal to P(A)?
# P(A|B) = P(A intersect B) / P(B)
if p_B_sim > 0:
    p_A_given_B_sim = p_A_intersect_B_sim / p_B_sim
    print(f"\nSimulated P(A|B): {p_A_given_B_sim:.4f}")
    print(f"Is P(A|B) approx equal to P(A)? {'Yes' if np.isclose(p_A_given_B_sim, p_A_sim, atol=0.01) else 'No'}")
else:
    print("\nCannot calculate P(A|B) as P(B) is zero in simulation.")
```

The simulation results should be close to the theoretical values, confirming the independence of these events. Small discrepancies are expected due to random sampling variation.

+++

## Chapter Summary

+++

* **Bayes' Theorem** $P(A|B) = \frac{P(B|A) P(A)}{P(B)}$ provides a fundamental rule for updating probabilities (beliefs) based on new evidence.
* It relates the **posterior probability** $P(A|B)$ to the **prior probability** $P(A)$ and the **likelihood** $P(B|A)$.
* The term $P(B)$ acts as a normalizing constant and can often be calculated using the **Law of Total Probability**.
* Bayes' Theorem is crucial in fields like medical diagnosis, machine learning (spam filtering, classification), and scientific reasoning.
* Two events A and B are **independent** if $P(A \cap B) = P(A)P(B)$, or equivalently, $P(A|B) = P(A)$ (assuming $P(B)>0$). The occurrence of one does not change the probability of the other.
* Events A and B are **conditionally independent** given C if $P(A \cap B | C) = P(A|C)P(B|C)$. They become independent once the outcome of C is known.
* Simulation is a valuable tool for building intuition about Bayes' Theorem and independence by observing frequencies in generated data.

+++

In the next part of the book, we will shift our focus from events to **Random Variables** – numerical outcomes of random phenomena – and explore their distributions. This will allow us to model and analyze probabilistic situations in a more structured way.
