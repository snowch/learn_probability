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

# Chapter 4: Conditional Probability

In the previous chapters, we laid the groundwork for probability, exploring sample spaces, events, and counting techniques. Now, we venture into one of the most fundamental and powerful concepts in probability theory: **conditional probability**.

Often, we are interested in the probability of an event occurring *given that* another event has already happened. Our knowledge or assumptions about one event can change our assessment of the probability of another. This is the essence of conditional probability. It allows us to update our beliefs in the face of new information.


## 1. Definition and Intuition

**Conditional Probability** measures the probability of an event $A$ occurring given that another event $B$ has already occurred (or is known to have occurred). We denote this as $P(A|B)$, read as "the probability of A given B".

**Intuition:** Imagine the entire sample space $S$. When we know that event $B$ has occurred, our focus effectively narrows down from the entire sample space $S$ to just the outcomes within $B$. We are now interested in the probability that $A$ occurs *within this new, reduced sample space* $B$. The outcomes favourable to "A given B" are those that belong to both $A$ and $B$, i.e., $A \cap B$.

**Formal Definition:**
For any two events $A$ and $B$ from a sample space $S$, where $P(B) > 0$, the conditional probability of $A$ given $B$ is defined as:

$$ P(A|B) = \frac{P(A \cap B)}{P(B)} $$

where:
* $P(A \cap B)$ is the probability that both events $A$ and $B$ occur.
* $P(B)$ is the probability that event $B$ occurs.

+++

%
% Example code
%

```{code-cell} python3
:tags: [remove-input, remove-output]

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles

fig = plt.figure(figsize=(7, 4.5))
v = venn2(subsets=(9, 2, 2), set_labels=('A: at least one 3', 'B: sum = 9'))
venn2_circles(subsets=(9, 2, 2), linestyle='solid', linewidth=1)

lab = v.get_label_by_id('11')
if lab is not None:
    lab.set_text('A ∩ B\n(3,6)\n(6,3)\n\n(A|B)')

lab = v.get_label_by_id('01')
if lab is not None:
    lab.set_text('B only\n(4,5)\n(5,4)')

lab = v.get_label_by_id('10')
if lab is not None:
    lab.set_text('A only\n(…9 outcomes…)')

plt.title('Given B, restrict to circle B; A|B is the overlap')
plt.annotate(
    'We already know B happened\n→ restrict attention to circle B\n\u2234 P(A|B) is a ratio of P(A n B) to P(B)',
    xy=(0.75, .75), xycoords='axes fraction',
    xytext=(1, 0.95), textcoords='axes fraction',
    arrowprops=dict(arrowstyle='->', lw=1),
    ha='left', va='top'
)
plt.tight_layout()

fig.savefig("venn-dice-a-given-b.svg", format="svg", bbox_inches="tight")

# plt.show()
# plt.close(fig)
```

%
% Example
% 


:::{admonition} Example
:class: tip dropdown

**Two Dice — “At least one 3” given “sum is 9”**

Roll two fair six‑sided dice.

- Let **A** be the event *“at least one die shows a 3”*.
- Let **B** be the event *“the sum of the two dice is 9”*.

We want the conditional probability $P(A\mid B)$.

**Step 1 — List the outcomes in $B$**
The outcomes (ordered pairs) that sum to 9 are:
$$B = \{(3,6),(4,5),(5,4),(6,3)\}.$$
So $|B|=4$ and therefore $P(B)=4/36$.

**Step 2 — Find which of those outcomes also lie in $A$**
Within $B$, the outcomes that include a 3 are:
$$A\cap B = \{(3,6),(6,3)\}.$$
So $|A\cap B|=2$ and therefore $P(A\cap B)=2/36$.

**Step 3 — Apply the definition**
$$P(A\mid B)=\frac{P(A\cap B)}{P(B)} = \frac{2/36}{4/36}=\frac12.$$

**Intuition:** once we’re told $B$ happened, the “new sample space” is just the 4 outcomes in $B$. In that restricted space, 2 of the 4 outcomes satisfy $A$, so $P(A\mid B)=2/4=1/2$.


```{figure} venn-dice-a-given-b.svg
```

:::


+++

## 2. The Multiplication Rule for Conditional Probability

Rearranging the definition of conditional probability gives us the **General Multiplication Rule**, which is useful for calculating the probability of the intersection of two events:

$$ P(A \cap B) = P(A|B) P(B) $$

Similarly, if $P(A) > 0$, we can write:

$$ P(A \cap B) = P(B|A) P(A) $$

This rule is particularly helpful when dealing with sequential events, where the outcome of the first event affects the probability of the second.


:::{admonition} Example
:class: tip dropdown
**Probability of drawing two Kings**

Probability of drawing two Kings from a standard 52-card deck without replacement.
Let $A$ be the event "the first card drawn is a King" and $B$ be the event "the second card drawn is a King".
We want to find $P(A \cap B)$.
Using the multiplication rule: $P(A \cap B) = P(B|A) P(A)$.
* $P(A)$: There are 4 Kings in 52 cards, so $P(A) = \frac{4}{52}$.
* $P(B|A)$: *Given* that the first card was a King, there are now 3 Kings left in the remaining 51 cards. So, $P(B|A) = \frac{3}{51}$.

Therefore,

$$
\begin{align*}
P(\text{Draw 2 Kings}) &= P(A \cap B) \\
&= P(B|A) P(A) \\
&= \frac{3}{51} \times \frac{4}{52} \\
&= \frac{12}{2652} \\
&\approx 0.0045
\end{align*}
$$

:::

The multiplication rule can be extended to more than two events. For three events $A, B, C$:

$$ P(A \cap B \cap C) = P(C | A \cap B) P(B | A) P(A) $$

:::{admonition} Derivation: The Chain Rule for Three Events
:class: tip dropdown

To find $P(A \cap B \cap C)$, we apply the multiplication rule in two stages:

**Step 1: Treat $(A \cap B)$ as a single event** Think of the first two events as one block. According to the standard multiplication rule:
$$P((A \cap B) \cap C) = P(A \cap B) \cdot P(C | A \cap B)$$
*(Logic: For all three to occur, the first two must happen, and then $C$ must happen given that $A$ and $B$ already occurred.)*

**Step 2: Break down the first block $P(A \cap B)$** Now, we apply the multiplication rule again to just the $A$ and $B$ part:
$$P(A \cap B) = P(A) \cdot P(B | A)$$

**Step 3: Combine the parts** Substitute the expression from Step 2 into the equation from Step 1:
$$P(A \cap B \cap C) = \underbrace{P(A) \cdot P(B | A)}_{P(A \cap B)} \cdot P(C | A \cap B)$$

**Final Result:**
$$P(A \cap B \cap C) = P(A) \cdot P(B | A) \cdot P(C | A \cap B)$$
:::

+++

## 3. The Law of Total Probability

Sometimes, calculating the probability of an event $A$ directly is difficult. However, we might know the conditional probabilities of $A$ occurring under various **mutually exclusive** and **exhaustive** scenarios. The **Law of Total Probability** lets us combine those scenario-based probabilities into one overall probability.

### 3.1 Definition

Let $B_1, B_2, \ldots, B_n$ be a **partition** of the sample space $S$. This means:

1. $B_i \cap B_j = \emptyset$ for all $i \neq j$ (the events are mutually exclusive),
2. $B_1 \cup B_2 \cup \cdots \cup B_n = S$ (they cover the whole sample space),
3. $P(B_i) > 0$ for all $i$ (so the conditional probabilities are well-defined).

Then, for any event $A$ in $S$, the Law of Total Probability states:

$$
P(A) = \sum_{i=1}^{n} P(A \mid B_i)\,P(B_i).
$$

Equivalently, written as an expanded sum:

$$
\begin{align*}
P(A) ={}& P(A\mid B_1)P(B_1) \\
& + P(A\mid B_2)P(B_2) \\
& + \ldots \\
& + P(A\mid B_n)P(B_n).
\end{align*}
$$

### 3.2 Why it works

The key idea is that the partition breaks $A$ into **disjoint pieces**:

$$
A = (A\cap B_1)\ \cup\ (A\cap B_2)\ \cup\ \cdots\ \cup\ (A\cap B_n),
$$

and these pieces do not overlap because the $B_i$ do not overlap.

So we can add their probabilities:

$$
P(A) = \sum_{i=1}^n P(A\cap B_i).
$$

Finally, apply the multiplication rule $P(A\cap B_i)=P(A\mid B_i)P(B_i)$ to each term:

$$
P(A) = \sum_{i=1}^n P(A\mid B_i)P(B_i).
$$

### 3.3 Intuition

Think of the $B_i$ as “which scenario we are in.” First, one scenario $B_i$ happens (with probability $P(B_i)$). Then, within that scenario, $A$ happens with probability $P(A\mid B_i)$. The overall probability $P(A)$ is a **weighted average** of the conditional probabilities $P(A\mid B_i)$, weighted by how likely each scenario is.

### 3.4 Visual intuition: area model

**How to read the diagram**

- The sample space $S$ is split into disjoint strips $B_1,\dots,B_n$ (a partition), so exactly one $B_i$ occurs.
- Each strip’s **width** represents $P(B_i)$.
- The shaded piece inside strip $i$ represents the piece of $A$ that lies in that strip, i.e. $A\cap B_i$.
- The (true) area of that piece is $P(A\cap B_i)=P(A\mid B_i)P(B_i)$.
- Adding the disjoint shaded pieces gives $P(A)$.

```{figure} total-probability-area.svg
---
width: 100%
figclass: full-width
---
Area model: $P(A)$ is the sum of the disjoint pieces $A\cap B_i$.
````

### 3.5 Visual intuition: probability tree (same idea, different view)

A tree diagram shows the same logic: first choose which scenario $B_i$ occurs, then (within that scenario) whether $A$ occurs.

```{mermaid}
graph TD
    S((Start))
    
    S --> B1["B1: P_B1"]
    S --> Bd["... (B2..B(n-1))"]
    S --> Bn["Bn: P_Bn"]
    
    B1 --> A1["A: P_A_given_B1"]
    B1 --> NA1["not A: 1 - P_A_given_B1"]
    
    Bn --> An["A: P_A_given_Bn"]
    Bn --> NAn["not A: 1 - P_A_given_Bn"]
```


On the branch $S \to B_i \to A$, the probability is the product $P(B_i),P(A\mid B_i)$. Summing those “$A$” leaves over all scenarios gives $P(A)$.

:::{admonition} Example
:class: tip dropdown

**Two manufacturing lines**

A factory makes parts on two lines:

* $B_1$: the part came from Line 1
* $B_2$: the part came from Line 2

Suppose:

* $P(B_1)=0.6$, $P(B_2)=0.4$
* $P(A\mid B_1)=0.02$ (2% defective on Line 1)
* $P(A\mid B_2)=0.05$ (5% defective on Line 2)

Then:

$$
P(A)=P(A\mid B_1)P(B_1)+P(A\mid B_2)P(B_2)
=0.02\cdot 0.6 + 0.05\cdot 0.4
=0.012+0.020
=0.032.
$$

So about **3.2%** of all parts are defective overall.

:::

+++


## 4. Tree Diagrams

Tree diagrams are a useful visualization tool for problems involving sequences of events, especially when conditional probabilities are involved.

* Each branch represents an event.
* The probability of each event is written on the branch.
* Branches emanating from a single point represent mutually exclusive outcomes of a stage, and their probabilities should sum to 1.
* The probability of reaching a specific endpoint (a sequence of events) is found by multiplying the probabilities along the path leading to that endpoint (using the Multiplication Rule).
* The probability of an event that can occur via multiple paths is found by summing the probabilities of those paths (related to the Law of Total Probability).


:::{admonition} Example
:class: tip dropdown
Visualizing the probabilities of outcomes in a sequence of two potentially biased coin flips.
Suppose a coin has $P(\text{Heads}) = 0.6$ and $P(\text{Tails}) = 0.4$. We flip it twice. The outcomes are independent.

```{mermaid}
graph TD
    Start((Start))
    
    Start -- 0.6 --> H1(H)
    Start -- 0.4 --> T1(T)
    
    H1 -- 0.6 --> H2(H)
    H1 -- 0.4 --> T2(T)
    
    T1 -- 0.6 --> H3(H)
    T1 -- 0.4 --> T3(T)
    
    subgraph Flip 1
    H1
    T1
    end
    
    subgraph Flip 2
    H2
    T2
    H3
    T3
    end
```

* **Path 1 (HH):** 

$$
\begin{align*}
P(H_1 \cap H_2) &= P(H_1) \times P(H_2 | H_1) \\
&= 0.6 \times 0.6 \\
&= 0.36
\end{align*}
$$
$$
\text{(Since flips are independent, } P(H_2|H_1) = P(H_2) = 0.6 \text{)}
$$

* **Path 2 (HT):** $P(\text{H on 1st} \cap \text{T on 2nd}) = 0.6 \times 0.4 = 0.24$.
* **Path 3 (TH):** $P(\text{T on 1st} \cap \text{H on 2nd}) = 0.4 \times 0.6 = 0.24$.
* **Path 4 (TT):** $P(\text{T on 1st} \cap \text{T on 2nd}) = 0.4 \times 0.4 = 0.16$.

Note that the probabilities of all possible outcomes sum to 1: $0.36 + 0.24 + 0.24 + 0.16 = 1.0$.

We can use this to find probabilities of combined events, e.g., 

$$
\begin{align*}
P(\text{Exactly one Head}) &= P(HT) + P(TH) \\
&= 0.24 + 0.24 \\
&= 0.48
\end{align*}
$$

:::

+++

## 5. Tips for differentiating between $P(A \cap B)$ and $P(A | B)$

It can be challenging to differentiate between $P(A \cap B)$ and $P(A | B)$ in probability problems.

**$P(A \cap B)$** represents the **probability that both event A AND event B occur**. Look for keywords like "**and**," "**both**," or phrases indicating a direct overlap between two characteristics. For example, "the probability that a student is an engineering major **and** is female."

**$P(A | B)$** signifies the **probability of event A occurring GIVEN that event B has already occurred**. This is a **conditional probability**, focusing on a subset of the population. Phrases such as "**given that**," "**of those who**," or "**if a [characteristic B] is selected**" are strong indicators. For instance, "Of the students who study engineering, 20% are female" is an example of $P(\text{Female} | \text{Engineering})$.

The key distinction lies in whether the problem describes the likelihood of two events happening simultaneously (intersection) or the likelihood of one event happening *under the condition* that another event has already happened (conditional).

+++

## Exercises

1.  **Two Dice:** If you roll two fair six-sided dice, what is the conditional probability that the sum is 8, given that the first die shows a 3? What is the conditional probability that the first die shows a 3, given that the sum is 8?

    ```{admonition} Answer
    :class: dropdown

    Let D1 be the result of the first die and D2 be the result of the second die. The total number of possible outcomes when rolling two fair six-sided dice is $6 \times 6 = 36$. Each outcome (D1, D2) is equally likely.

    **Part 1: Conditional probability that the sum is 8, given that the first die shows a 3.**
    Let A be the event that the sum is 8.
    Let B be the event that the first die shows a 3.
    We want to find $P(A | B)$.

    The outcomes for event B (first die is 3) are:
    {(3,1), (3,2), (3,3), (3,4), (3,5), (3,6)}. There are 6 such outcomes.
    Given that the first die is 3, for the sum to be 8, the second die (D2) must be $8 - 3 = 5$.
    So, the only outcome where the first die is 3 and the sum is 8 is (3,5).
    Thus, within the reduced sample space where the first die is 3, there is 1 outcome where the sum is 8.
    Therefore, $P(\text{Sum}=8 | \text{First die}=3) = 1/6$.

    Alternatively, using the formula $P(A | B) = P(A \cap B) / P(B)$:
    $P(B) = 6/36 = 1/6$.
    $P(A \cap B)$ (sum is 8 and first die is 3) corresponds to the outcome (3,5), so $P(A \cap B) = 1/36$.
    $P(A | B) = (1/36) / (6/36) = 1/6$.

    **Part 2: Conditional probability that the first die shows a 3, given that the sum is 8.**
    Let A be the event that the sum is 8.
    Let B be the event that the first die shows a 3.
    We want to find $P(B | A)$.

    The outcomes for event A (sum is 8) are:
    {(2,6), (3,5), (4,4), (5,3), (6,2)}. There are 5 such outcomes.
    Given that the sum is 8, we look for outcomes where the first die is 3. The only such outcome is (3,5).
    Thus, within the reduced sample space where the sum is 8, there is 1 outcome where the first die is 3.
    Therefore, $P(\text{First die}=3 | \text{Sum}=8) = 1/5$.

    Alternatively, using the formula $P(B | A) = P(B \cap A) / P(A)$:
    $P(A) = 5/36$.
    $P(B \cap A)$ (first die is 3 and sum is 8) corresponds to the outcome (3,5), so $P(B \cap A) = 1/36$.
    $P(B | A) = (1/36) / (5/36) = 1/5$.
    ```

2.  **Medical Test:** A disease affects 1 in 1000 people. A test for the disease is 99% accurate (i.e., P(Positive | Disease) = 0.99) and has a 2% false positive rate (i.e., P(Positive | No Disease) = 0.02). Use the Law of Total Probability to calculate the overall probability that a randomly selected person tests positive. (We will revisit this in the Bayes' Theorem chapter).

    ```{admonition} Answer
    :class: dropdown

    Let D be the event that a person has the disease, and $\neg D$ be the event that a person does not have the disease.
    Let + be the event that a person tests positive.

    We are given:
    * $P(D) = 1/1000 = 0.001$
    * $P(\text{Positive} | D) = P(+|D) = 0.99$ (test accuracy for those with the disease)
    * $P(\text{Positive} | \neg D) = P(+|\neg D) = 0.02$ (false positive rate)

    From $P(D)$, we can find $P(\neg D)$:
    * $P(\neg D) = 1 - P(D) = 1 - 0.001 = 0.999$

    We need to calculate the overall probability that a randomly selected person tests positive, $P(+)$. We use the Law of Total Probability:
    $P(+) = P(+|D) \cdot P(D) + P(+|\neg D) \cdot P(\neg D)$
    $P(+) = (0.99 \cdot 0.001) + (0.02 \cdot 0.999)$
    $P(+) = 0.00099 + 0.01998$
    $P(+) = 0.02097$

    So, the overall probability that a randomly selected person tests positive is 0.02097, or about 2.097%.
    ```

3.  **Card Simulation:** Modify the card drawing simulation to calculate the probability of drawing two cards of the *same rank* (e.g., two 7s, two Kings, etc.). Compare the simulation result to the theoretical probability. (Hint: The first card can be anything. What's the probability the second matches its rank?).

    ```{admonition} Answer
    :class: dropdown

    **Theoretical Probability:**
    We want to find the probability of drawing two cards of the *same rank* from a standard 52-card deck without replacement.

    * Consider the first card drawn. It can be any card, and its rank is now fixed.
    * For the second card to match the rank of the first, it must be one of the remaining 3 cards of that same rank.
    * There are 51 cards remaining in the deck after the first draw.
    * So, the probability that the second card's rank matches the first card's rank is $3/51$.
    * $3/51 = 1/17$.

    The theoretical probability is $1/17 \approx 0.0588$.

    **Modifying a Card Drawing Simulation:**
    To calculate this probability via simulation:
    1.  **Represent Deck and Cards:** Create a representation of a 52-card deck where each card has a suit and a rank (e.g., 'King of Hearts', '7 of Spades'). Ensure you can easily extract the rank (e.g., 'King', '7').
    2.  **Simulation Loop:** Repeat the following steps for a large number of trials (e.g., 10,000 or 100,000 times):
        a.  **Shuffle and Draw:** Shuffle the deck and draw two cards without replacement.
        b.  **Extract Ranks:** Get the rank of the first card and the rank of the second card.
        c.  **Compare Ranks:** Check if the two ranks are identical.
        d.  **Count Successes:** If the ranks are the same, increment a counter for successful trials.
    3.  **Calculate Simulated Probability:** After all trials are complete, the simulated probability is the number of successful trials divided by the total number of trials.
        $P(\text{same rank}) \approx \text{count of same ranks} / \text{total trials}$
    4.  **Compare:** Compare this simulated result to the theoretical probability of $1/17$. As the number of trials increases, the simulated probability should converge towards the theoretical one.
    ```

4.  **Data Analysis:** Load a real dataset (e.g., the Titanic dataset often used in machine learning introductions) using Pandas. Calculate the conditional probability of survival given the passenger's class (e.g., P(Survived | Class=1st), P(Survived | Class=3rd)). What do these probabilities tell you?

    ```{admonition} Answer
    :class: dropdown

    This exercise involves using a dataset like the Titanic dataset to calculate conditional probabilities with Pandas.

    **Steps:**
    1.  **Load Data:**
        * Import the Pandas library: `import pandas as pd`
        * Load the Titanic dataset. This dataset is often available in libraries like Seaborn (`import seaborn as sns; df = sns.load_dataset('titanic')`) or can be loaded from a CSV file (`df = pd.read_csv('path_to_titanic.csv')`).
        * The DataFrame (`df`) typically contains columns like 'survived' (0 = No, 1 = Yes) and 'pclass' (passenger class: 1, 2, 3).

    2.  **Calculate P(Survived | Class=1st):**
        * This is the probability of a passenger surviving, given they were in 1st class.
        * **Filter for 1st Class:** Create a subset of the DataFrame containing only 1st class passengers:
            `df_class1 = df[df['pclass'] == 1]`
        * **Calculate Survival Rate:** Within this subset, find the proportion of passengers who survived. If 'survived' is coded as 1 for survived and 0 for not, the mean of the 'survived' column gives this probability:
            `P_survived_given_class1 = df_class1['survived'].mean()`
            Alternatively, count survivors and divide by the total in that class:
            `survived_class1_count = df_class1['survived'].sum()`
            `total_class1_count = len(df_class1)`
            `P_survived_given_class1 = survived_class1_count / total_class1_count`

    3.  **Calculate P(Survived | Class=3rd):**
        * This is the probability of a passenger surviving, given they were in 3rd class.
        * **Filter for 3rd Class:** Create a subset for 3rd class passengers:
            `df_class3 = df[df['pclass'] == 3]`
        * **Calculate Survival Rate:**
            `P_survived_given_class3 = df_class3['survived'].mean()`
            (Or using the count and divide method as above).

    **What do these probabilities tell you?**
    * $P(\text{Survived} | \text{Class=1st})$ tells you the likelihood (or proportion) of survival specifically for passengers who travelled in first class.
    * $P(\text{Survived} | \text{Class=3rd})$ tells you the likelihood of survival specifically for passengers who travelled in third class.
    * **Comparison is Key:** By comparing these two probabilities, you can infer the impact of passenger class on survival chances during the Titanic disaster. Historically, one would expect $P(\text{Survived} | \text{Class=1st})$ to be significantly higher than $P(\text{Survived} | \text{Class=3rd})$. This difference would highlight the socio-economic disparities of the era, as first-class passengers generally had better accommodations, were located on higher decks closer to lifeboats, and potentially received preferential treatment during the evacuation.
    * These conditional probabilities provide quantitative evidence of how different subgroups within a population (passengers in this case) experienced different outcomes based on a specific condition (their travel class).
    ```

+++
