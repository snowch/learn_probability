# Exercises for Chapter 4.

```{note}
It can definitely be tricky to tell the difference between $P(A \cap B)$ and $P(A | B)$ just from the wording of a problem! Let's break down how to identify each one.

**$P(A \cap B)$ (The Probability of A and B)**

This represents the probability that **both** event A **and** event B occur. Keywords and phrases that often indicate $P(A \cap B)$ include:

* "the probability that someone is a [characteristic A] **and** [characteristic B]"
* "the proportion of items that are [type A] **and** [have property B]"
* "the chance of [event A] **and at the same time** [event B]"
* Statements that directly give you a count or proportion of the overlap between two categories.

**Example:** "The probability that a student studies engineering **and** is female is 0.05." This directly tells you $P(\text{Engineering} \cap \text{Female}) = 0.05$.

**$P(A | B)$ (The Probability of A Given B)**

This represents the probability that event A occurs **given that** event B has already occurred. It's a *conditional* probability. Keywords and phrases that often indicate $P(A | B)$ include:

* "the probability that someone is [characteristic A] **given that** they are [characteristic B]"
* "the proportion of [type B] items that **are also** [type A]"
* "if a [characteristic B] is selected, what is the probability they are also [characteristic A]?"
* Statements that provide information about a subset of the population.

**Example:** "Of the students who study engineering, 20% are female." This tells you $P(\text{Female} | \text{Engineering}) = 0.20$. Notice how the focus is on the subset of engineering students.

**Key Takeaway:**

Pay close attention to the phrasing that indicates a condition or a restriction to a specific group. "Of those who...", "given that...", and "among the..." are strong indicators of conditional probability $P(A | B)$. Phrases using "and" or describing an overlap between two characteristics often point towards the probability of the intersection $P(A \cap B)$.
```

1.  **Single Die Roll Variance:** You roll a fair six-sided die.

      * What is the probability of rolling an even number?
      * What is the probability of rolling a number greater than 4?
      * What is the probability of rolling a prime number? (Consider 1 not to be prime)

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    Let S be the sample space for rolling a fair six-sided die: $S = \{1, 2, 3, 4, 5, 6\}$. The total number of outcomes is $|S| = 6$.

    **Part 1: Probability of rolling an even number.**
    Let E be the event of rolling an even number. The outcomes for E are $\{2, 4, 6\}$.
    The number of outcomes for E is $|E| = 3$.

    $$
    \begin{align*}
    P(E) &= \frac{|E|}{|S|} \\
    &= \frac{3}{6} \\
    &= \frac{1}{2}
    \end{align*}
    $$

    **Part 2: Probability of rolling a number greater than 4.**
    Let G be the event of rolling a number greater than 4. The outcomes for G are $\{5, 6\}$.
    The number of outcomes for G is $|G| = 2$.
    
    $$
    \begin{align*}
    P(G) &= \frac{|G|}{|S|} \\
    &= \frac{2}{6} \\
    &= \frac{1}{3}
    \end{align*}
    $$

    **Part 3: Probability of rolling a prime number.**
    Let P be the event of rolling a prime number. Prime numbers in S are $\{2, 3, 5\}$. (1 is not prime by convention).
    The number of outcomes for P is $|P| = 3$.
    
    $$
    \begin{align*}
    P(P) &= \frac{|P|}{|S|} \\
    &= \frac{3}{6} \\
    &= \frac{1}{2}
    \end{align*}
    $$
    ```

2.  **Deck of Cards - Specific Properties:** You draw one card from a standard 52-card deck.

      * What is the probability of drawing a face card (Jack, Queen, King)?
      * What is the probability of drawing a card that is NOT a Spade?
      * What is the probability of drawing a red Ace (Ace of Hearts or Ace of Diamonds)?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    A standard deck has 52 cards.

    **Part 1: Probability of drawing a face card.**
    There are 3 face cards (Jack, Queen, King) in each of the 4 suits.
    Number of face cards = $3 \times 4 = 12$.
    
    $$
    \begin{align*}
    P(\text{Face Card}) &= \frac{12}{52} \\
    &= \frac{3}{13}
    \end{align*}
    $$

    **Part 2: Probability of drawing a card that is NOT a Spade.**
    There are 13 Spade cards in the deck.
    Number of cards that are not Spades = $52 - 13 = 39$.
    
    $$
    \begin{align*}
    P(\text{Not a Spade}) &= \frac{39}{52} \\
    &= \frac{3}{4}
    \end{align*}
    $$

    Alternatively, $P(\text{Spade}) = \frac{13}{52} = \frac{1}{4}$.
    
    $$
    \begin{align*}
    P(\text{Not a Spade}) &= 1 - P(\text{Spade}) \\
    &= 1 - \frac{1}{4} \\
    &= \frac{3}{4}
    \end{align*}
    $$

    **Part 3: Probability of drawing a red Ace.**
    There are two red Aces: Ace of Hearts and Ace of Diamonds.
    Number of red Aces = 2.
    
    $$
    \begin{align*}
    P(\text{Red Ace}) &= \frac{2}{52} \\
    &= \frac{1}{26}
    \end{align*}
    $$
    ```

3.  **Two Fair Coin Flips:** You flip two fair coins.

      * What is the sample space for this experiment?
      * What is the probability of getting at least one Tail (T)?
      * What is the probability of getting two Heads (HH)?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    Let H denote Heads and T denote Tails.

    **Part 1: Sample space.**
    The possible outcomes when flipping two fair coins are:
    $S = \{\text{HH, HT, TH, TT}\}$
    There are $2 \times 2 = 4$ possible outcomes.

    **Part 2: Probability of getting at least one Tail.**
    Let A be the event of getting at least one Tail. The outcomes for A are $\{\text{HT, TH, TT}\}$.
    There are 3 such outcomes.
    $P(A) = \frac{3}{4}$.
    Alternatively, this is the complement of getting no Tails (i.e., getting HH).
    $P(\text{No Tails}) = P(\text{HH}) = \frac{1}{4}$.
    
    $$
    \begin{align*}
    P(\text{At least one Tail}) &= 1 - P(\text{No Tails}) \\
    &= 1 - \frac{1}{4} \\
    &= \frac{3}{4}
    \end{align*}
    $$

    **Part 3: Probability of getting two Heads.**
    Let B be the event of getting two Heads. The only outcome for B is $\{\text{HH}\}$.
    There is 1 such outcome.
    $P(B) = \frac{1}{4}$.
    ```

4.  **Rainy Days Probability:** The probability of rain on any given day in a city is 0.3. Assume that the weather on any day is independent of the weather on other days.

      * What is the probability that it does not rain on a given day?
      * What is the probability that it rains on Monday AND Tuesday?
      * What is the probability that it rains on Monday OR Tuesday (or both)?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    Let R be the event that it rains on a given day. We are given $P(R) = 0.3$.
    Events are independent.

    **Part 1: Probability that it does not rain on a given day.**
    Let $\neg R$ be the event that it does not rain. This is the complement of R.
    
    $$
    \begin{align*}
    P(\neg R) &= 1 - P(R) \\
    &= 1 - 0.3 \\
    &= 0.7
    \end{align*}
    $$

    **Part 2: Probability that it rains on Monday AND Tuesday.**
    Let $R_M$ be rain on Monday, $R_T$ be rain on Tuesday.
    Since the events are independent, $P(R_M \text{ and } R_T) = P(R_M) \times P(R_T)$.
    
    $$
    \begin{align*}
    P(R_M \text{ and } R_T) &= 0.3 \times 0.3 \\
    &= 0.09
    \end{align*}
    $$

    **Part 3: Probability that it rains on Monday OR Tuesday (or both).**
    
    We use the formula:

    $$P(R_M \text{ or } R_T) = P(R_M) + P(R_T) - P(R_M \text{ and } R_T)$$
    
    Calculation:
    
    $$
    \begin{align*}
    P(R_M \text{ or } R_T) &= 0.3 + 0.3 - 0.09 \\
    &= 0.6 - 0.09 \\
    &= 0.51
    \end{align*}
    $$
    
    Alternatively, this is $1 - P(\text{no rain on Monday AND no rain on Tuesday})$.
    
    $$
    \begin{align*}
    P(\neg R_M \text{ and } \neg R_T) &= P(\neg R_M) \times P(\neg R_T) \\
    &= 0.7 \times 0.7 \\
    &= 0.49
    \end{align*}
    $$

    So, $P(R_M \text{ or } R_T) = 1 - 0.49 = 0.51$.
    ```

5.  **Dice Sum Condition:** You roll two fair six-sided dice.

      * What is the probability that the sum of the dice is 6?
      * Given that the first die shows a 2, what is the probability that the sum is 6?
      * Are the events "sum is 6" and "first die shows a 2" independent?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    Total possible outcomes = $6 \times 6 = 36$. Each outcome is equally likely.

    **Part 1: Probability that the sum of the dice is 6.**
    Let S6 be the event that the sum is 6.
    Outcomes for S6: {(1,5), (2,4), (3,3), (4,2), (5,1)}. There are 5 such outcomes.
    $P(S6) = \frac{5}{36}$.

    **Part 2: Given that the first die shows a 2, what is the probability that the sum is 6?**

    Let D1_2 be the event that the first die shows a 2.

    Outcomes for D1_2: {(2,1), (2,2), (2,3), (2,4), (2,5), (2,6)}. There are 6 such outcomes.

    We want to find $P(S6 | D1\_2)$.

    If the first die is 2, for the sum to be 6, the second die must be $6 - 2 = 4$.

    The outcome (2,4) is the only one that satisfies both conditions.

    Within the reduced sample space where the first die is 2, there is 1 outcome where the sum is 6.

    $P(S6 | D1\_2) = \frac{1}{6}$.

    Alternatively, using the formula

    $$P(S6 | D1\_2) = \frac{P(S6 \cap D1\_2)}{P(D1\_2)}$$

    $P(D1\_2) = \frac{6}{36} = \frac{1}{6}$.
    $P(S6 \cap D1\_2)$ (sum is 6 and first die is 2) is the outcome (2,4), so $P(S6 \cap D1\_2) = \frac{1}{36}$.
    
    $$
    \begin{align*}
    P(S6 | D1\_2) &= \frac{1/36}{6/36} \\
    &= \frac{1}{6}
    \end{align*}
    $$

    **Part 3: Are the events "sum is 6" and "first die shows a 2" independent?**

    Two events A and B are independent if $P(A \cap B) = P(A) \times P(B)$.

    Here, $A = S6$ and $B = D1\_2$.
    $P(S6) = \frac{5}{36}$.
    $P(D1\_2) = \frac{1}{6}$.

    $$P(S6) \times P(D1\_2) = \frac{5}{36} \times \frac{1}{6} = \frac{5}{216}$$

    We found $P(S6 \cap D1\_2) = \frac{1}{36}$.
    Since $\frac{1}{36} \neq \frac{5}{216}$, the events are NOT independent.

    Alternatively, events are independent if $P(A|B) = P(A)$. Here, $P(S6 | D1\_2) = \frac{1}{6}$ and $P(S6) = \frac{5}{36}$. Since $\frac{1}{6} \neq \frac{5}{36}$, the events are not independent.
    ```

6.  **Drawing Two Cards Consecutively:** You draw two cards from a standard 52-card deck without replacement.

      * What is the probability of drawing a King first, then a Queen second?
      * What is the probability of drawing a King and a Queen in any order?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    **Part 1: Probability of drawing a King first, then a Queen second (K then Q).**
    Probability of drawing a King first: $P(K1) = \frac{4}{52}$.
    Given a King was drawn first, there are 51 cards left. There are still 4 Queens.
    Probability of drawing a Queen second, given a King was first: $P(Q2 | K1) = \frac{4}{51}$.
    
    $$
    \begin{align*}
    P(K1 \text{ then } Q2) &= P(K1) \times P(Q2 | K1) \\
    &= \frac{4}{52} \times \frac{4}{51} \\
    &= \frac{1}{13} \times \frac{4}{51} \\
    &= \frac{4}{663}
    \end{align*}
    $$

    **Part 2: Probability of drawing a King and a Queen in any order.**
    This can happen in two ways:
    1.  King first, then Queen (K then Q): $P(K1 \text{ then } Q2) = \frac{4}{663}$ (calculated above).
    2.  Queen first, then King (Q then K):
        $P(Q1) = \frac{4}{52}$.
        $P(K2 | Q1) = \frac{4}{51}$.

        $$
        \begin{align*}
        P(Q1 \text{ then } K2) &= P(Q1) \times P(K2 | Q1) \\
        &= \frac{4}{52} \times \frac{4}{51} \\
        &= \frac{4}{663}
        \end{align*}
        $$

    The probability of drawing a King and a Queen in any order is the sum of these probabilities:

    $$
    \begin{align*}
    P(\text{King and Queen}) &= P(K1 \text{ then } Q2) + P(Q1 \text{ then } K2) \\
    &= \frac{4}{663} + \frac{4}{663} \\
    &= \frac{8}{663}
    \end{align*}
    $$
    ```

7.  **Checking Independence of Events:** Let A and B be two events. Suppose $P(A) = 0.4$, $P(B) = 0.5$, and $P(A \\cup B) = 0.7$.

      * Find $P(A \\cap B)$.
      * Are events A and B independent?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    **Part 1: Find $P(A \cap B)$.**
    We use the formula for the probability of the union of two events:
    $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
    We are given:
    $P(A) = 0.4$
    $P(B) = 0.5$
    $P(A \cup B) = 0.7$
    So,

    $$
    \begin{align*}
    0.7 &= 0.4 + 0.5 - P(A \cap B) \\
    0.7 &= 0.9 - P(A \cap B) \\
    P(A \cap B) &= 0.9 - 0.7 \\
    P(A \cap B) &= 0.2
    \end{align*}
    $$

    **Part 2: Are events A and B independent?**
    Two events A and B are independent if $P(A \cap B) = P(A) \times P(B)$.
    We calculated $P(A \cap B) = 0.2$.
    Let's calculate $P(A) \times P(B)$:
    $P(A) \times P(B) = 0.4 \times 0.5 = 0.2$.
    Since $P(A \cap B) = P(A) \times P(B)$ (both are 0.2), the events A and B are independent.
    ```

8.  **Family with Two Children:** A family has two children. Assume the probability of having a boy (B) or a girl (G) is equal (0.5) and independent for each child.

      * What is the probability that both children are boys?
      * Given that at least one child is a boy, what is the probability that both children are boys?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    The possible gender combinations for two children (oldest to youngest) are:
    $S = \{\text{BB, BG, GB, GG}\}$. Each outcome has a probability of $0.5 \times 0.5 = 0.25$.

    **Part 1: Probability that both children are boys.**
    Let E be the event that both children are boys. The outcome is {BB}.
    $P(E) = P(\text{BB}) = 0.25$.

    **Part 2: Given that at least one child is a boy, what is the probability that both children are boys?**
    Let A be the event that at least one child is a boy.
    The outcomes for A are $\{\text{BB, BG, GB}\}$. So $P(A) = \frac{3}{4} = 0.75$.
    Let B be the event that both children are boys. The outcome for B is $\{\text{BB}\}$. So $P(B) = \frac{1}{4} = 0.25$.
    We want to find $P(B | A) = P(\text{both boys } | \text{ at least one boy})$.
    The event "B and A" (both boys AND at least one boy) is simply the event "both boys", which is $\{\text{BB}\}$.
    So, $P(B \cap A) = P(\text{BB}) = 0.25$.

    Using the formula for conditional probability:
    $$P(B | A) = \frac{P(B \cap A)}{P(A)}$$

    $$
    \begin{align*}
    P(B | A) &= \frac{0.25}{0.75} \\
    &= \frac{1}{3}
    \end{align*}
    $$

    Alternatively, using the reduced sample space:
    The event "at least one child is a boy" means the possible outcomes are $\{\text{BB, BG, GB}\}$. This is our new sample space, with 3 equally likely outcomes.
    Out of these 3 outcomes, only 1 outcome is "both children are boys" (BB).
    So, the conditional probability is $\frac{1}{3}$.
    ```

9.  **Survey Data: Coffee and Productivity:** A survey of 100 office workers found:

      * 60 drink coffee.
      * 40 of those who drink coffee report feeling productive in the morning.
      * 30 of those who do NOT drink coffee report feeling productive in the morning.
        Let C be the event a worker drinks coffee, and P be the event a worker feels productive.
      * Find $P(C)$.
      * Find $P(P | C)$.
      * Find $P(P | \\neg C)$.

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    Total workers = 100.
    Number of coffee drinkers = 60.
    Number of non-coffee drinkers = $100 - 60 = 40$.
    Number of coffee drinkers who feel productive = 40.
    Number of non-coffee drinkers who feel productive = 30.

    **Part 1: Find $P(C)$.**
    C is the event a worker drinks coffee.

    $$
    \begin{align*}
    P(C) &= \frac{\text{Number of coffee drinkers}}{\text{Total workers}} \\
    &= \frac{60}{100} \\
    &= 0.6
    \end{align*}
    $$

    **Part 2: Find $P(P | C)$.**
    This is the probability a worker feels productive, GIVEN they drink coffee.

    $$
    \begin{align*}
    P(P | C) &= \frac{\text{Number of coffee drinkers who feel productive}}{\text{Number of coffee drinkers}} \\
    &= \frac{40}{60} \\
    &= \frac{2}{3} \\
    &\approx 0.667
    \end{align*}
    $$

    **Part 3: Find $P(P | \neg C)$.**
    This is the probability a worker feels productive, GIVEN they do NOT drink coffee.
    $\neg C$ is the event a worker does not drink coffee.

    $$
    \begin{align*}
    P(P | \neg C) &= \frac{\text{Number of non-coffee drinkers who feel productive}}{\text{Number of non-coffee drinkers}} \\
    &= \frac{30}{40} \\
    &= \frac{3}{4} \\
    &= 0.75
    \end{align*}
    $$
    ```

10. **Defective Parts from Machines:** A factory has two machines, A and B.

      * Machine A produces 60% of the daily output, and 5% of its products are defective.
      * Machine B produces 40% of the daily output, and 3% of its products are defective.
      * What is the probability that a randomly selected part was made by Machine A and is defective?
      * What is the probability that a randomly selected part was made by Machine B and is NOT defective?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    Let A be the event that a part is made by Machine A, and B be the event it's made by Machine B.
    Let D be the event that a part is defective, and $\neg D$ be the event it's not defective.

    Given:
    $P(A) = 0.60$
    $P(B) = 0.40$
    $P(D | A) = 0.05$ (defect rate for Machine A)
    $P(D | B) = 0.03$ (defect rate for Machine B)

    From this, we can find:

    $$
    \begin{align*}
    P(\neg D | A) &= 1 - P(D | A) \\
    &= 1 - 0.05 \\
    &= 0.95
    \end{align*}
    $$

    $$
    \begin{align*}
    P(\neg D | B) &= 1 - P(D | B) \\
    &= 1 - 0.03 \\
    &= 0.97
    \end{align*}
    $$

    **Part 1: Probability that a part was made by Machine A AND is defective.**
    We want to find $P(A \cap D)$.
    Using the multiplication rule: $P(A \cap D) = P(D | A) \times P(A)$
    $P(A \cap D) = 0.05 \times 0.60 = 0.03$.

    **Part 2: Probability that a part was made by Machine B AND is NOT defective.**
    We want to find $P(B \cap \neg D)$.
    Using the multiplication rule: $P(B \cap \neg D) = P(\neg D | B) \times P(B)$
    $P(B \cap \neg D) = 0.97 \times 0.40 = 0.388$.
    ```

11. **Balls in an Urn (Two Draws):** An urn contains 5 red balls and 3 blue balls. You draw two balls from the urn *without* replacement.

      * What is the probability that both balls drawn are red?
      * What is the probability that the first ball is red and the second ball is blue?
      * What is the probability that one ball is red and one ball is blue (in any order)?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    Total balls initially = $5 \text{ Red} + 3 \text{ Blue} = 8$ balls.

    **Part 1: Probability that both balls drawn are red (R1 and R2).**
    $P(R1) = \text{Probability first ball is red} = \frac{5}{8}$.
    Given the first was red, there are 4 red balls left and 7 total balls.
    $P(R2 | R1) = \text{Probability second ball is red, given first was red} = \frac{4}{7}$.

    $$
    \begin{align*}
    P(R1 \text{ and } R2) &= P(R1) \times P(R2 | R1) \\
    &= \frac{5}{8} \times \frac{4}{7} \\
    &= \frac{20}{56} \\
    &= \frac{5}{14}
    \end{align*}
    $$

    **Part 2: Probability that the first ball is red and the second ball is blue (R1 and B2).**
    $P(R1) = \frac{5}{8}$.
    Given the first was red, there are 3 blue balls left and 7 total balls.
    $P(B2 | R1) = \text{Probability second ball is blue, given first was red} = \frac{3}{7}$.

    $$
    \begin{align*}
    P(R1 \text{ and } B2) &= P(R1) \times P(B2 | R1) \\
    &= \frac{5}{8} \times \frac{3}{7} \\
    &= \frac{15}{56}
    \end{align*}
    $$

    **Part 3: Probability that one ball is red and one ball is blue (any order).**
    This can happen in two ways:
    1.  Red first, then Blue (R1 and B2): $P(R1 \text{ and } B2) = \frac{15}{56}$ (from Part 2).
    2.  Blue first, then Red (B1 and R2):
        $P(B1) = \text{Probability first ball is blue} = \frac{3}{8}$.
        Given the first was blue, there are 5 red balls left and 7 total balls.
        $P(R2 | B1) = \text{Probability second ball is red, given first was blue} = \frac{5}{7}$.

        $$
        \begin{align*}
        P(B1 \text{ and } R2) &= P(B1) \times P(R2 | B1) \\
        &= \frac{3}{8} \times \frac{5}{7} \\
        &= \frac{15}{56}
        \end{align*}
        $$

    The total probability is the sum of these two mutually exclusive events:

    $$
    \begin{align*}
    P(\text{one red, one blue}) &= P(R1 \text{ and } B2) + P(B1 \text{ and } R2) \\
    &= \frac{15}{56} + \frac{15}{56} \\
    &= \frac{30}{56} \\
    &= \frac{15}{28}
    \end{align*}
    $$
    ```

12. **Course Prerequisites and Passing:** To take Course B, a student must first pass Course A.

      * The probability a student passes Course A is $P(A\_p) = 0.7$.
      * If a student passes Course A, the probability they also pass Course B is $P(B\_p | A\_p) = 0.8$.
      * What is the probability a student passes both Course A and Course B?
      * What is the probability a student passes Course A but fails Course B? (Assume $P(B\_f | A\_p) = 1 - P(B\_p | A\_p)$)

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    Let $A_p$ be the event a student passes Course A.
    Let $B_p$ be the event a student passes Course B.
    Let $B_f$ be the event a student fails Course B.

    Given:
    $P(A_p) = 0.7$
    $P(B_p | A_p) = 0.8$

    **Part 1: Probability a student passes both Course A and Course B.**
    We want to find $P(A_p \cap B_p)$.
    Using the multiplication rule for conditional probability:
    $P(A_p \cap B_p) = P(B_p | A_p) \times P(A_p)$
    $P(A_p \cap B_p) = 0.8 \times 0.7 = 0.56$.

    **Part 2: Probability a student passes Course A but fails Course B.**
    We are looking for $P(A_p \cap B_f)$.
    If a student passed Course A, they either pass Course B or fail Course B.
    So,

    $$
    \begin{align*}
    P(B_f | A_p) &= 1 - P(B_p | A_p) \\
    &= 1 - 0.8 \\
    &= 0.2
    \end{align*}
    $$

    Then, using the multiplication rule:
    $P(A_p \cap B_f) = P(B_f | A_p) \times P(A_p)$
    $P(A_p \cap B_f) = 0.2 \times 0.7 = 0.14$.
    ```

13. **Alternative Medical Test Scenario (Law of Total Probability):** A different disease affects 2% of the population. A new test has a 95% chance of correctly identifying an infected person (sensitivity) and a 10% chance of incorrectly identifying a healthy person as infected (false positive rate). What is the overall probability that a randomly selected person tests positive?

    ```{admonition} Answer
    :class: dropdown

    Let D be the event that a person has the disease, and $\neg D$ be the event that a person does not have the disease.
    Let + be the event that a person tests positive.

    We are given:
    * $P(D) = 0.02$ (prevalence of the disease)
    * $P(+ | D) = 0.95$ (sensitivity: test is positive given disease)
    * $P(+ | \neg D) = 0.10$ (false positive rate: test is positive given no disease)

    From $P(D)$, we can find $P(\neg D)$:
    \begin{align*}
    P(\neg D) &= 1 - P(D) \\
    &= 1 - 0.02 \\
    &= 0.98
    \end{align*}

    We need to calculate the overall probability that a randomly selected person tests positive, $P(+)$. We use the Law of Total Probability:

    $$
    \begin{align*}
    P(+) &= P(+|D) \cdot P(D) + P(+|\neg D) \cdot P(\neg D) \\
    &= (0.95 \cdot 0.02) + (0.10 \cdot 0.98) \\
    &= 0.019 + 0.098 \\
    &= 0.117
    \end{align*}
    $$

    So, the overall probability that a randomly selected person tests positive is 0.117, or 11.7%.
    ```

14. **Bayes' Theorem Application:** Using the information from the "Alternative Medical Test Scenario" (Exercise 13): if a randomly selected person tests positive, what is the probability they actually have the disease?

    ```{admonition} Answer
    :class: dropdown

    From Exercise 13, we have:
    * $P(D) = 0.02$
    * $P(\neg D) = 0.98$
    * $P(+ | D) = 0.95$
    * $P(+ | \neg D) = 0.10$
    * $P(+) = 0.117$ (overall probability of testing positive)

    We want to find $P(D | +)$, the probability that a person has the disease given they tested positive.
    Using Bayes' Theorem:
    $$P(D | +) = \frac{P(+ | D) \cdot P(D)}{P(+)}$$
    Calculating the value:

    $$
    \begin{align*}
    P(D | +) &= \frac{0.95 \cdot 0.02}{0.117} \\
    &= \frac{0.019}{0.117} \\
    &\approx 0.16239
    \end{align*}
    $$

    So, if a person tests positive, the probability they actually have the disease is approximately 0.1624, or about 16.24%.
    Notice that even with a positive test, the probability of actually having the disease is still relatively low due to the low prevalence of the disease and the false positive rate.
    ```

15. **Three Printers Error Rates:** A company has three printers: P1, P2, and P3, which print 30%, 50%, and 20% of all documents, respectively. The error rates for these printers are 1%, 2%, and 3%, respectively. If a randomly selected document has an error, what is the probability it came from P1?

    ```{admonition} Answer
    :class: dropdown

    Let P1, P2, P3 be the events that a document was printed by printer 1, 2, or 3, respectively.
    Let E be the event that a document has an error.

    We are given:
    $P(P1) = 0.30$
    $P(P2) = 0.50$
    $P(P3) = 0.20$

    And the conditional probabilities of an error given the printer:
    $P(E | P1) = 0.01$
    $P(E | P2) = 0.02$
    $P(E | P3) = 0.03$

    First, we calculate the overall probability of an error, $P(E)$, using the Law of Total Probability:

    $$
    \begin{align*}
    P(E) &= P(E | P1)P(P1) + P(E | P2)P(P2) + P(E | P3)P(P3) \\
    &= (0.01 \cdot 0.30) + (0.02 \cdot 0.50) + (0.03 \cdot 0.20) \\
    &= 0.003 + 0.010 + 0.006 \\
    &= 0.019
    \end{align*}
    $$

    Now, we want to find the probability that the document came from P1, given it has an error: $P(P1 | E)$.
    Using Bayes' Theorem:
    $$P(P1 | E) = \frac{P(E | P1) \cdot P(P1)}{P(E)}$$
    Calculating the value:

    $$
    \begin{align*}
    P(P1 | E) &= \frac{0.01 \cdot 0.30}{0.019} \\
    &= \frac{0.003}{0.019} \\
    &\approx 0.15789
    \end{align*}
    $$

    So, if a document has an error, the probability it came from Printer 1 is approximately 0.1579, or about 15.79%.
    ```

16. **Simplified Email Spam Filter:** Suppose 70% of emails are legitimate (ham) and 30% are spam.

      * The word "free" appears in 10% of spam emails.
      * The word "free" appears in 1% of ham emails.
        If an email contains the word "free", what is the probability it is spam?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    Let S be the event that an email is spam, and H be the event that an email is ham (legitimate).
    Let F be the event that an email contains the word "free".

    We are given:
    $P(S) = 0.30$
    $P(H) = 0.70$ (since $P(H) = 1 - P(S)$)
    $P(F | S) = 0.10$ (probability "free" appears given spam)
    $P(F | H) = 0.01$ (probability "free" appears given ham)

    We want to find $P(S | F)$, the probability an email is spam given it contains "free".
    First, we need $P(F)$, the overall probability an email contains "free". Using the Law of Total Probability:

    $$
    \begin{align*}
    P(F) &= P(F | S)P(S) + P(F | H)P(H) \\
    &= (0.10 \cdot 0.30) + (0.01 \cdot 0.70) \\
    &= 0.030 + 0.007 \\
    &= 0.037
    \end{align*}
    $$

    Now, using Bayes' Theorem:
    $$P(S | F) = \frac{P(F | S) \cdot P(S)}{P(F)}$$
    Calculating the value:

    $$
    \begin{align*}
    P(S | F) &= \frac{0.10 \cdot 0.30}{0.037} \\
    &= \frac{0.030}{0.037} \\
    &\approx 0.8108
    \end{align*}
    $$

    So, if an email contains the word "free", the probability it is spam is approximately 0.8108, or about 81.08%.
    ```

17. **Simple Lottery Probability:** In a mini-lottery, you pick 2 distinct numbers from the set ${1, 2, ..., 10}$. The lottery also picks 2 distinct numbers from this set. What is the probability your two chosen numbers exactly match the lottery's two numbers?

    ```{admonition} Answer
    :class: dropdown

    First, let's determine the total number of ways the lottery can pick 2 distinct numbers from 10. The order in which the lottery picks them doesn't matter for a match, so we use combinations.

    $$
    \begin{align*}
    \text{Total possible pairs} &= C(10, 2) = \binom{10}{2} \\
    &= \frac{10!}{2!(10-2)!} \\
    &= \frac{10 \times 9}{2 \times 1} \\
    &= 45
    \end{align*}
    $$

    There are 45 possible unique pairs of numbers the lottery can draw.

    You pick one specific pair of numbers.
    There is only 1 way for your chosen pair to be the exact pair drawn by the lottery.

    So, the probability of your two numbers matching the lottery's two numbers is:

    $$
    \begin{align*}
    P(\text{match}) &= \frac{\text{Number of your successful pairs}}{\text{Total possible lottery pairs}} \\
    &= \frac{1}{45}
    \end{align*}
    $$
    ```

18. **Committee Selection with Specific Roles:** A committee of 3 people is to be selected from a group of 5 men and 4 women.

      * What is the total number of ways to form the committee?
      * What is the probability that the committee consists of exactly 2 men and 1 woman?
      * If the committee must have a Chair, a Secretary, and a Treasurer, and these roles are assigned after the 3 people are selected, how many ways can the roles be assigned to a specific committee of 3?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    Total people = $5 \text{ men} + 4 \text{ women} = 9$ people.

    **Part 1: Total number of ways to form the committee of 3.**
    The order of selection for the committee doesn't matter, so we use combinations.

    $$
    \begin{align*}
    \text{Total ways} &= C(9, 3) = \binom{9}{3} \\
    &= \frac{9!}{3!(9-3)!} \\
    &= \frac{9 \times 8 \times 7}{3 \times 2 \times 1} \\
    &= 3 \times 4 \times 7 = 84
    \end{align*}
    $$

    There are 84 possible committees.

    **Part 2: Probability that the committee consists of exactly 2 men and 1 woman.**
    Number of ways to choose 2 men from 5:
    $$C(5, 2) = \binom{5}{2} = \frac{5 \times 4}{2 \times 1} = 10$$
    Number of ways to choose 1 woman from 4:
    $$C(4, 1) = \binom{4}{1} = \frac{4}{1} = 4$$
    Number of ways to form a committee with 2 men and 1 woman = $C(5, 2) \times C(4, 1) = 10 \times 4 = 40$.

    $$
    \begin{align*}
    P(\text{2 men, 1 woman}) &= \frac{\text{Ways to choose 2 men and 1 woman}}{\text{Total ways to choose 3 people}} \\
    &= \frac{40}{84} \\
    &= \frac{10}{21}
    \end{align*}
    $$

    **Part 3: If the committee must have a Chair, a Secretary, and a Treasurer, how many ways can the roles be assigned to a specific committee of 3?**
    Once a specific committee of 3 people (say Person A, Person B, Person C) has been selected, we need to assign 3 distinct roles to these 3 people. This is a permutation problem.
    Number of ways to assign roles = $P(3, 3) = 3! = 3 \times 2 \times 1 = 6$.
    (Chair can be any of 3, Secretary any of remaining 2, Treasurer is the last one).
    ```

19. **Marble Selection Probability:** An urn contains 4 red, 3 green, and 2 blue marbles. You randomly select 3 marbles without replacement. What is the probability that you select exactly 1 of each color (1 red, 1 green, 1 blue)?

    ```{admonition} Answer
    :class: dropdown

    Total marbles = $4 (\text{R}) + 3 (\text{G}) + 2 (\text{B}) = 9$ marbles.
    We are selecting 3 marbles.

    First, find the total number of ways to select 3 marbles from 9:
    
    $$
    \begin{align*}
    \text{Total combinations} &= C(9, 3) = \binom{9}{3} \\
    &= \frac{9 \times 8 \times 7}{3 \times 2 \times 1} \\
    &= 3 \times 4 \times 7 = 84
    \end{align*}
    $$

    Next, find the number of ways to select exactly 1 red, 1 green, and 1 blue marble:
    * Ways to choose 1 red marble from 4 = $C(4, 1) = 4$.
    * Ways to choose 1 green marble from 3 = $C(3, 1) = 3$.
    * Ways to choose 1 blue marble from 2 = $C(2, 1) = 2$.

    The number of ways to select 1 of each color is the product of these:
    Ways (1R, 1G, 1B) = $C(4, 1) \times C(3, 1) \times C(2, 1) = 4 \times 3 \times 2 = 24$.

    The probability of selecting 1 of each color is:

    $$
    \begin{align*}
    P(\text{1R, 1G, 1B}) &= \frac{\text{Ways (1R, 1G, 1B)}}{\text{Total combinations}} \\
    &= \frac{24}{84} \\
    &= \frac{2}{7}
    \end{align*}
    $$
    ```

20. **Simple Dice Game Expected Value:** You play a game where you roll one fair six-sided die.

      * If you roll a 6, you win $10.
      * If you roll a 1, you lose $4.
      * If you roll any other number (2, 3, 4, 5), you win $0.
        What is the expected value of playing this game once?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    Let X be the random variable representing the winnings from playing the game.
    The possible outcomes for X are $10, -$4, and $0.
    We need the probabilities of each outcome:
    * $P(X = 10)$ (rolling a 6) = $1/6$.
    * $P(X = -4)$ (rolling a 1) = $1/6$.
    * $P(X = 0)$ (rolling a 2, 3, 4, or 5) = $4/6 = 2/3$.

    The expected value $E(X)$ is calculated as:
    $$E(X) = \sum [x \cdot P(X=x)]$$

    $$
    \begin{align*}
    E(X) &= (10 \cdot P(X=10)) + (-4 \cdot P(X=-4)) + (0 \cdot P(X=0)) \\
    &= \left(10 \cdot \frac{1}{6}\right) + \left(-4 \cdot \frac{1}{6}\right) + \left(0 \cdot \frac{4}{6}\right) \\
    &= \frac{10}{6} - \frac{4}{6} + 0 \\
    &= \frac{6}{6} \\
    &= 1
    \end{align*}
    $$

    The expected value of playing this game once is $1. This means on average, you would expect to win $1 per game if you played many times.
    ```

21. **Raffle Ticket Expected Value:** A charity sells 500 raffle tickets for $2 each. There is one grand prize of $300 and two second prizes of $50 each.

      * What is the expected value of buying one ticket from the perspective of the buyer?
      * Is this a "fair" game for the buyer?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    Cost of one ticket = \$2.
    Total tickets sold = 500.

    Prizes:
    * 1 grand prize of \$300.
    * 2 second prizes of \$50 each (total \$100 in second prizes).

    Let X be the net gain from buying one ticket. The possible values for X are:
    
    * Win grand prize: Gain = \$300 (prize) - \$2 (cost) = \$298.
        Probability = $\frac{1}{500}$.
    * Win second prize: Gain = \$50 (prize) - \$2 (cost) = \$48.
        Probability = $\frac{2}{500}$.
    * Win nothing: Gain = \$0 (prize) - \$2 (cost) = -\$2.
        Number of losing tickets = $500 - 1 - 2 = 497$.
        Probability = $\frac{497}{500}$.

    **Part 1: Expected value of buying one ticket.**

    $$
    \begin{align*}
    E(X) &= \left(298 \cdot \frac{1}{500}\right) + \left(48 \cdot \frac{2}{500}\right) + \left(-2 \cdot \frac{497}{500}\right) \\
    &= \frac{298}{500} + \frac{96}{500} - \frac{994}{500} \\
    &= \frac{298 + 96 - 994}{500} \\
    &= \frac{394 - 994}{500} \\
    &= \frac{-600}{500} \\
    &= -\frac{6}{5} = -\$1.20
    \end{align*}
    $$

    The expected value of buying one ticket is -\$1.20. This means on average, a buyer expects to lose \$1.20 per ticket.

    **Part 2: Is this a "fair" game for the buyer?**
    A "fair" game is one where the expected value is 0. Since the expected value is -\$1.20 (which is negative), this is not a fair game for the buyer. It is favorable to the seller (the charity). This is typical for raffles and lotteries, as they are designed to raise money.
    ```

22. **Investment Decision Expected Value:** You have $1000 to invest.

      * Investment A: 70% chance to return \$1200 (profit \$200), 30% chance to return \$800 (loss \$200).
      * Investment B: 40% chance to return \$1500 (profit \$500), 60% chance to return \$900 (loss \$100).
        Calculate the expected *profit* for each investment. Which investment has a higher expected profit?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    Let $X_A$ be the profit from Investment A, and $X_B$ be the profit from Investment B.

    **Investment A:**
    * Profit if success: \$1200 - \$1000 = \$200. Probability = 0.70.
    * Profit if failure (loss): \$800 - \$1000 = -\$200. Probability = 0.30.
    Expected profit for Investment A, $E(X_A)$:

    $$
    \begin{align*}
    E(X_A) &= (200 \cdot 0.70) + (-200 \cdot 0.30) \\
    &= 140 - 60 \\
    &= \$80
    \end{align*}
    $$

    **Investment B:**
    * Profit if success: \$1500 - \$1000 = \$500. Probability = 0.40.
    * Profit if failure (loss): \$900 - \$1000 = -\$100. Probability = 0.60.
    Expected profit for Investment B, $E(X_B)$:

    $$
    \begin{align*}
    E(X_B) &= (500 \cdot 0.40) + (-100 \cdot 0.60) \\
    &= 200 - 60 \\
    &= \$140
    \end{align*}
    $$

    **Comparison:**
    Expected profit for Investment A = \$80.
    Expected profit for Investment B = \$140.

    Investment B has a higher expected profit (\$140) compared to Investment A (\$80). Based solely on expected profit, Investment B would be the preferred choice.
    ```

23. **Biased Coin Flip Simulation:** A coin is biased such that it lands on Heads (H) with a probability of 0.6 and Tails (T) with a probability of 0.4.

      * Describe how you would simulate flipping this coin 1000 times.
      * After the simulation, how would you verify if the observed frequencies of Heads and Tails are close to their theoretical probabilities?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    **Part 1: Describing the simulation.**
    To simulate flipping this biased coin 1000 times:
    1.  **Initialize Counters:** Set a counter for Heads ( `count_H` ) to 0 and a counter for Tails ( `count_T` ) to 0.
    2.  **Loop for Trials:** Repeat the following steps 1000 times (for each flip):
        a.  **Generate Random Number:** Generate a random number `r` uniformly distributed between 0 and 1.
        b.  **Determine Outcome:**
            * If `r < 0.6`, consider the outcome to be Heads. Increment `count_H`.
            * Else (if `r >= 0.6`), consider the outcome to be Tails. Increment `count_T`.
    3.  **Record Results:** After 1000 flips, `count_H` will hold the total number of Heads observed, and `count_T` will hold the total number of Tails observed.

    **Part 2: Verifying observed frequencies.**
    After the simulation:
    1.  **Calculate Observed Frequencies (Proportions):**
        * $$\text{Observed frequency of Heads} = \frac{\text{count\_H}}{1000}$$
        * $$\text{Observed frequency of Tails} = \frac{\text{count\_T}}{1000}$$
    2.  **Compare with Theoretical Probabilities:**
        * Compare the observed frequency of Heads with the theoretical probability $P(H) = 0.6$.
        * Compare the observed frequency of Tails with the theoretical probability $P(T) = 0.4$.
    3.  **Assess Closeness:**
        * The observed frequencies should be "close" to the theoretical probabilities. Due to the randomness of the simulation, they are unlikely to be exactly equal.
        * The Law of Large Numbers suggests that as the number of trials (flips) increases, the observed frequencies will converge towards the theoretical probabilities. For 1000 flips, we would expect the observed proportion of Heads to be around 0.6 (e.g., between 0.57 and 0.63 might be typical).
        * You can calculate the absolute difference: $| \text{Observed Freq(H)} - 0.6 |$ and $| \text{Observed Freq(T)} - 0.4 |$. Smaller differences indicate better agreement.
    ```

24. **Birthday Problem Simulation (Approximate):** The "Birthday Problem" asks for the probability that in a group of N people, at least two share a birthday.

      * Describe how you would simulate this for $N=23$ people to estimate this probability. Assume 365 days in a year and equal likelihood for each birthday.
      * How would you calculate the estimated probability from many simulation trials?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    **Part 1: Describing one simulation trial for N=23 people.**
    To simulate one trial for a group of 23 people:
    1.  **Generate Birthdays:** Create a list or array to store the birthdays of the 23 people. For each person, randomly assign a birthday, which is an integer from 1 to 365 (inclusive). Each day should have an equal chance of being selected.
    2.  **Check for Duplicates:** Examine the list of 23 birthdays. If there is at least one pair of identical birthdays in the list, then this trial results in a "match" (at least two people share a birthday). Otherwise, it's a "no match."
        * A simple way to check for duplicates is to add the birthdays to a set. If the size of the set is less than 23, it means there was at least one duplicate birthday.

    **Part 2: Calculating the estimated probability from many simulation trials.**
    To estimate the probability:
    1.  **Initialize Counters:** Set a counter for the number of trials with a match (`match_count`) to 0. Define the total number of simulation trials to run (e.g., `total_trials = 10000`).
    2.  **Run Simulations:** Repeat the single trial simulation (described in Part 1) for `total_trials` times.
        * For each trial, if it results in a "match," increment `match_count`.
    3.  **Calculate Estimated Probability:** After all trials are complete, the estimated probability of at least two people sharing a birthday in a group of 23 is:
        $$P(\text{shared birthday}) \approx \frac{\text{match\_count}}{\text{total\_trials}}$$

    For $N=23$, the theoretical probability is just over 50%. The simulation should yield a result close to this as `total_trials` increases.
    ```

25. **Simulating Drawing Specific Cards:** You draw 3 cards from a standard 52-card deck without replacement.

      * Describe how you would modify a card drawing simulation to estimate the probability of drawing exactly 2 Hearts out of the 3 cards drawn.
      * How would you calculate this estimated probability?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    **Part 1: Modifying a card drawing simulation for one trial.**
    To simulate drawing 3 cards and checking for exactly 2 Hearts:
    1.  **Represent Deck:** Create a representation of a standard 52-card deck. Each card should have a suit (Hearts, Diamonds, Clubs, Spades) and a rank. There are 13 Hearts in the deck.
    2.  **Shuffle and Draw:**
        a.  Shuffle the deck thoroughly to randomize the order of cards.
        b.  Draw the top 3 cards from the shuffled deck.
    3.  **Count Hearts:** Examine the 3 cards drawn. Count how many of them are Hearts.
    4.  **Check Condition:** If the count of Hearts is exactly 2, then this trial is a "success." Otherwise, it's a "failure."

    **Part 2: Calculating the estimated probability.**
    To estimate the probability of drawing exactly 2 Hearts in 3 cards:
    1.  **Initialize Counters:** Set a counter for successful trials (`success_count`) to 0. Define the total number of simulation trials to run (e.g., `total_trials = 10000` or more for better accuracy).
    2.  **Run Simulations:** Repeat the single trial simulation (described in Part 1) for `total_trials` times.
        * For each trial, if it results in a "success" (exactly 2 Hearts were drawn), increment `success_count`.
    3.  **Calculate Estimated Probability:** After all trials are complete, the estimated probability is:
        $$P(\text{exactly 2 Hearts in 3 draws}) \approx \frac{\text{success\_count}}{\text{total\_trials}}$$

    **Theoretical Approach (for comparison):**
    The theoretical probability can be calculated using combinations:
    * Ways to choose 2 Hearts from 13: $C(13, 2) = \binom{13}{2} = \frac{13 \times 12}{2} = 78$.
    * Ways to choose 1 non-Heart from the remaining 39 non-Heart cards: $C(39, 1) = \binom{39}{1} = 39$.
    * Total ways to choose 3 cards from 52:

    $$
    \begin{align*}
    C(52, 3) &= \binom{52}{3} \\
    &= \frac{52 \times 51 \times 50}{3 \times 2 \times 1} \\
    &= 22100
    \end{align*}
    $$

    *

    $$
    \begin{align*}
    P(\text{exactly 2 Hearts}) &= \frac{C(13, 2) \times C(39, 1)}{C(52, 3)} \\
    &= \frac{78 \times 39}{22100} \\
    &= \frac{3042}{22100} \\
    &\approx 0.1376
    \end{align*}
    $$

    The simulation result should converge towards this theoretical probability.
    ```

26. **Titanic Dataset: Survival by Sex:** Using the Titanic dataset (commonly available in libraries like Seaborn or as a CSV):

      * Calculate $P(\text{Survived} | \text{Sex='female'})$ (the probability of survival given the passenger was female).
      * Calculate $P(\text{Survived} | \text{Sex='male'})$ (the probability of survival given the passenger was male).
      * What do these probabilities suggest about survival likelihood based on sex?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    This exercise involves using a dataset like the Titanic dataset to calculate conditional probabilities with Pandas.

    **Steps using Pandas:**
    1.  **Load Data:**
        * Import Pandas: `import pandas as pd`
        * Load the Titanic dataset (e.g., `df = sns.load_dataset('titanic')` if using Seaborn, or from a CSV).
        * The DataFrame (`df`) typically contains a 'survived' column (0 = No, 1 = Yes) and a 'sex' column (e.g., 'male', 'female').

    2.  **Calculate $P(\text{Survived} | \text{Sex='female'})$:**
        * **Filter for Females:** Create a subset of the DataFrame for female passengers:
            `df_female = df[df['sex'] == 'female']`
        * **Calculate Survival Rate for Females:** Within this subset, find the proportion who survived. The mean of the 'survived' column (if coded 0/1) gives this probability:
            `P_survived_given_female = df_female['survived'].mean()`

    3.  **Calculate $P(\text{Survived} | \text{Sex='male'})$:**
        * **Filter for Males:** Create a subset for male passengers:
            `df_male = df[df['sex'] == 'male']`
        * **Calculate Survival Rate for Males:**
            `P_survived_given_male = df_male['survived'].mean()`

    **What do these probabilities suggest?**
    * $P(\text{Survived} | \text{Sex='female'})$ indicates the proportion of female passengers who survived the Titanic disaster.
    * $P(\text{Survived} | \text{Sex='male'})$ indicates the proportion of male passengers who survived.
    * **Comparison:** By comparing these two probabilities, we can infer the influence of a passenger's sex on their chance of survival. Historically, and as reflected in the data, $P(\text{Survived} | \text{Sex='female'})$ is typically significantly higher than $P(\text{Survived} | \text{Sex='male'})$. This reflects the "women and children first" protocol that was often (though not perfectly) followed during the evacuation. These conditional probabilities provide quantitative evidence of differing survival rates between these two groups.
    ```

27. **Titanic Dataset: Child Survivors:** Using the Titanic dataset, define a "child" as someone with an age less than 18.

      * Calculate $P(\text{Child} | \text{Survived}=1)$ (the probability a survivor was a child).
      * Calculate $P(\text{Child} | \text{Survived}=0)$ (the probability a non-survivor was a child).
      * What might these probabilities indicate about the survival priority of children?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    This exercise uses the Titanic dataset and Pandas. Assume 'age' and 'survived' columns are present.

    **Steps using Pandas:**
    1.  **Load Data & Define Child:**
        * Load the Titanic dataset as in the previous exercise.
        * Handle missing 'age' values (e.g., by dropping rows with missing age or imputation, though for this exercise, dropping might be simpler for a cleaner calculation on known ages). `df.dropna(subset=['age'], inplace=True)`
        * Create a 'is_child' boolean column: `df['is_child'] = df['age'] < 18`

    2.  **Calculate $P(\text{Child} | \text{Survived}=1)$:**
        * This is the proportion of children among those who survived.
        * **Filter for Survivors:** Create a subset of the DataFrame for passengers who survived:
            `df_survived = df[df['survived'] == 1]`
        * **Calculate Proportion of Children among Survivors:**
            `P_child_given_survived = df_survived['is_child'].mean()` (The mean of a boolean column (True=1, False=0) gives the proportion of True values).

    3.  **Calculate $P(\text{Child} | \text{Survived}=0)$:**
        * This is the proportion of children among those who did not survive.
        * **Filter for Non-Survivors:** Create a subset for passengers who did not survive:
            `df_notsurvived = df[df['survived'] == 0]`
        * **Calculate Proportion of Children among Non-Survivors:**
            `P_child_given_notsurvived = df_notsurvived['is_child'].mean()`

    **What might these probabilities indicate?**
    * $P(\text{Child} | \text{Survived}=1)$ tells us, of the group of people who survived, what fraction were children.
    * $P(\text{Child} | \text{Survived}=0)$ tells us, of the group of people who did not survive, what fraction were children.
    * **Interpretation:** If $P(\text{Child} | \text{Survived}=1)$ is notably higher than the overall proportion of children on board, and potentially higher than $P(\text{Child} | \text{Survived}=0)$, it might suggest that children were given some priority in evacuation efforts. However, it's important to compare this with $P(\text{Survived} | \text{Child})$ as well for a fuller picture. A higher proportion of children among survivors than among non-survivors would lend support to the idea that children had a better chance of surviving relative to adults within the same outcome group.
    ```

28. **Iris Dataset: Species Prediction based on Petal Width:** Load the Iris dataset (e.g., via scikit-learn or Seaborn).

      * Choose a threshold $X$ for 'petal width (cm)' (e.g., $X=1.5 \text{ cm}$).
      * Calculate $P(\text{Species='virginica'} | \text{Petal Width} \> X)$.
      * Calculate $P(\text{Species='setosa'} | \text{Petal Width} \< 0.5 \text{ cm})$. (Note: Setosa typically has small petal width).
      * What do these conditional probabilities suggest about using petal width to help identify species?

    <!-- end list -->

    ```{admonition} Answer
    :class: dropdown

    This exercise uses the Iris dataset, common in machine learning.

    **Steps using Pandas (and potentially scikit-learn for data loading):**
    1.  **Load Data:**
        * `from sklearn.datasets import load_iris`
        * `import pandas as pd`
        * `iris = load_iris()`
        * `df = pd.DataFrame(data=iris.data, columns=iris.feature_names)`
        * `df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)`
        (The columns would include 'petal width (cm)' and 'species' with values like 'setosa', 'versicolor', 'virginica').

    2.  **Calculate $P(\text{Species='virginica'} | \text{Petal Width} > 1.5 \text{ cm})$:**
        * Let $X = 1.5$.
        * **Filter for Petal Width > X:** Create a subset of the DataFrame:
            `df_petal_gt_X = df[df['petal width (cm)'] > 1.5]`
        * **Calculate Proportion of Virginica:** Within this subset, find the proportion of 'virginica' species:
            `P_virginica_given_petal_gt_X = (df_petal_gt_X['species'] == 'virginica').mean()`

    3.  **Calculate $P(\text{Species='setosa'} | \text{Petal Width} < 0.5 \text{ cm})$:**
        * **Filter for Petal Width < 0.5:** Create a subset:
            `df_petal_lt_0_5 = df[df['petal width (cm)'] < 0.5]`
        * **Calculate Proportion of Setosa:** Within this subset, find the proportion of 'setosa' species:
            `P_setosa_given_petal_lt_0_5 = (df_petal_lt_0_5['species'] == 'setosa').mean()`

    **What do these conditional probabilities suggest?**
    * $P(\text{Species='virginica'} | \text{Petal Width} > 1.5 \text{ cm})$: If this probability is high (e.g., close to 1), it suggests that if an Iris flower has a petal width greater than 1.5 cm, it is very likely to be of the 'virginica' species. This indicates that large petal width is a strong indicator for 'virginica'.
    * $P(\text{Species='setosa'} | \text{Petal Width} < 0.5 \text{ cm})$: If this probability is very high (likely close to 1, as Setosa usually has small petal widths around 0.2-0.3 cm), it suggests that a petal width less than 0.5 cm is a very strong indicator that the species is 'setosa'.
    * **Overall:** These conditional probabilities demonstrate how a specific feature measurement (petal width) can be used to predict or classify the species of an Iris flower. High conditional probabilities indicate a strong relationship between the feature value and the class label, forming the basis of many classification algorithms.
    ```

29. **Gambler's Fallacy Explanation:** Explain the Gambler's Fallacy using the example of flipping a fair coin. If you flip a fair coin 5 times and get Heads each time (HHHHH), what is the probability of getting Heads on the 6th flip? Why do some people incorrectly believe the probability changes?

    ```{admonition} Answer
    :class: dropdown

    **The Gambler's Fallacy:**
    The Gambler's Fallacy is the mistaken belief that if something happens more frequently than normal during a given period, it will happen less frequently in the future (or vice versa). It's an error in reasoning that assumes past independent events can influence the outcome of future independent events.

    **Example: Flipping a Fair Coin**
    Suppose you flip a fair coin 5 times and get Heads each time: HHHHH.
    * **Question:** What is the probability of getting Heads on the 6th flip?
    * **Correct Answer:** The probability of getting Heads on the 6th flip is still $\frac{1}{2}$ (or 0.5).

    **Why the probability remains 1/2:**
    * **Independence:** Each coin flip is an independent event. The outcome of one flip does not affect the outcome of any other flip. The coin has no "memory" of past results.
    * **Fair Coin:** A fair coin, by definition, has an equal probability of landing on Heads or Tails on any given flip ($P(H) = 0.5$, $P(T) = 0.5$).

    **Why some people incorrectly believe the probability changes:**
    1.  **Misunderstanding of "Law of Averages":** People might think that to "even out" the results or get closer to the expected 50/50 distribution, a Tail is "due." They expect the long-run frequencies to correct themselves in the short term. While it's true that over a very large number of flips the proportion of Heads will tend towards 0.5, this doesn't mean future flips are dependent on past ones to achieve this balance.
    2.  **Pattern Seeking:** Humans are prone to seeing patterns, even in random sequences. A string of HHHHH might seem like a strong pattern that "needs" to be broken by a Tail.
    3.  **Representativeness Heuristic:** People may judge the probability of an event by how representative it is of a typical sequence. A sequence like HHHHHH seems less representative of randomness than, say, HHTHTH, leading them to believe the former is less likely to continue.

    The probability of getting 6 Heads in a row (HHHHHH) from the start is indeed low ($\left(\frac{1}{2}\right)^6 = \frac{1}{64}$). However, *given that the first 5 flips were already Heads*, the probability of the *next* flip being Heads is simply the basic probability of a single flip, which is $\frac{1}{2}$. The previous outcomes are "sunk history" and do not influence the independent next event.
    ```

30. **The Monty Hall Problem:** State the Monty Hall problem. Explain the optimal strategy and why it works, referring to conditional probabilities if possible.

    ```{admonition} Answer
    :class: dropdown

    **Stating the Monty Hall Problem:**
    Suppose you're on a game show, and you're given the choice of three doors. Behind one door is a car; behind the others, goats.
    1.  You pick a door (say, Door #1).
    2.  The host, who knows what's behind the doors, opens another door (say, Door #3), which has a goat. (The host will always open a door with a goat and will never open your chosen door).
    3.  The host then asks you: "Do you want to switch your choice to the remaining closed door (Door #2), or do you want to stay with your original choice (Door #1)?"

    **Is it to your advantage to switch your choice?**

    **Optimal Strategy:** Yes, you should switch. Switching doors doubles your probability of winning the car from $\frac{1}{3}$ to $\frac{2}{3}$.

    **Explanation and Conditional Probabilities:**

    Let C1, C2, C3 be the events that the car is behind Door 1, Door 2, or Door 3, respectively.
    Initially, $P(C1) = P(C2) = P(C3) = \frac{1}{3}$.

    Assume you initially pick Door #1.

    **Scenario 1: You Stay with Door #1.**
    * You win if the car is actually behind Door #1.
    * The probability of this was $\frac{1}{3}$ from the start. The host opening another door with a goat doesn't change the fact that your initial $\frac{1}{3}$ chance was tied to Door #1 having the car.
    * So, $P(\text{Win} | \text{Stay}) = \frac{1}{3}$.

    **Scenario 2: You Switch Doors.**
    Consider what happens based on the car's initial location:
    * **Case A: Car is behind Door #1 (your initial pick).** Probability = $\frac{1}{3}$.
        If you switch, you will lose (because the host will open either Door #2 or Door #3, both having goats, and you switch to the other goat).
    * **Case B: Car is behind Door #2.** Probability = $\frac{1}{3}$.
        You initially picked Door #1. The host *must* open Door #3 (the other goat). If you switch, you switch to Door #2 and win.
    * **Case C: Car is behind Door #3.** Probability = $\frac{1}{3}$.
        You initially picked Door #1. The host *must* open Door #2 (the other goat). If you switch, you switch to Door #3 and win.

    If you switch, you win if the car was *not* behind your initial choice.
    The probability that the car was *not* behind your initial choice (Door #1) is

    $$
    \begin{align*}
    P(\neg C1) &= P(C2) + P(C3) \\
    &= \frac{1}{3} + \frac{1}{3} \\
    &= \frac{2}{3}
    \end{align*}
    $$

    When the car is not behind your initial choice, the host is forced to open the *other* door that has a goat, leaving the door with the car as the only one to switch to.
    So, $P(\text{Win} | \text{Switch}) = \frac{2}{3}$.

    **Conditional Probability Perspective (simplified):**
    Let $D_i$ be the event you initially choose door $i$. Let $C_j$ be the event the car is behind door $j$. $P(C_j)=\frac{1}{3}$.
    Assume you pick Door 1. $P(C_1) = \frac{1}{3}$. The probability the car is behind one of the other doors (Door 2 or Door 3) is $P(C_2 \cup C_3) = \frac{2}{3}$.
    The host's action of opening a door with a goat from doors {2, 3} provides information. It essentially concentrates the initial $\frac{2}{3}$ probability (that the car was behind Door 2 or Door 3) onto the single remaining closed door that you didn't initially pick.

    If your initial door was correct (1/3 chance), switching loses.
    If your initial door was incorrect (2/3 chance), switching wins.
    Therefore, switching is the better strategy.
    ```