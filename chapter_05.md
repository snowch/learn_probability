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

Let's think of $A$ as an event or hypothesis we are interested in (e.g., "a patient has a specific disease," "a coin is biased") and $B$ as new evidence or data observed (e.g., "the patient tested positive," "we observed 8 heads in 10 flips").

- $P(A)$: **Prior probability** — our belief about $A$ *before* seeing the evidence $B$.
- $P(B\mid A)$: **Likelihood** — the probability of observing the evidence $B$ *given that* $A$ is true.
- $P(B)$: **Probability of the evidence** — the overall probability of observing $B$, regardless of whether $A$ is true or not.  
  Using the Law of Total Probability with the partition $\{A, A^c\}$:
  
$$
P(B)=P(B\mid A)P(A)+P(B\mid A^c)P(A^c).
$$
  
- $P(A\mid B)$: **Posterior probability** — our updated belief about $A$ *after* observing the evidence $B$.
Bayes' Theorem tells us how to update our prior belief $P(A)$ to a posterior belief $P(A|B)$ based on the likelihood of the evidence $P(B|A)$ and the overall probability of the evidence $P(B)$.

### Visual intuition: Bayes’ Theorem (area model)

You can read Bayes’ theorem directly from the picture below as an **area ratio**:

- **Numerator** = the shaded overlap area $A \cap B$
- **Denominator** = the total shaded $B$ area

So by the definition of conditional probability:

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}.
$$

Now rewrite the overlap using the multiplication rule:

$$
P(A\cap B)=P(B\mid A)\,P(A),
$$

which gives the compact “Bayes form”:

$$
P(A\mid B)=\frac{P(B\mid A)\,P(A)}{P(B)}.
$$

To connect this *directly* to the area model, expand the denominator by splitting \(B\) into the part inside \(A\) and the part inside \(A^c\):

$$
P(B)=P(B\cap A)+P(B\cap A^c)=P(B\mid A)P(A)+P(B\mid A^c)P(A^c).
$$

Substitute into the Bayes form:

$$
P(A\mid B)
=\frac{P(B\mid A)P(A)}{P(B\mid A)P(A)+P(B\mid A^c)P(A^c)}.
$$

```{code-cell} python3
:tags: [remove-input, remove-output]

from pathlib import Path

def save_bayes_area_svg(
    filename="bayes-area.svg",
    pA=0.35,
    pB_given_A=0.70,
    pB_given_Ac=0.20,
    font_scale=2.0,
):
    pAc = 1 - pA
    pB = pB_given_A * pA + pB_given_Ac * pAc
    pA_given_B = (pB_given_A * pA) / pB

    def fmt(x):
        return f"{x:.4f}".rstrip("0").rstrip(".")

    # --- sizing ---
    L = 70
    box_w, box_h = 1200, 340
    W = box_w + 2 * L

    outline = "#111827"
    strip_fill = "#f8fafc"
    shade_fill = "#ef4444"
    shade_stroke = "#b91c1c"
    accentA = "#2563eb"
    accentAc = "#64748b"

    title_sz = int(22 * font_scale)
    text_sz  = int(14 * font_scale)
    num_sz   = int(13 * font_scale)
    big_sz   = int(18 * font_scale)

    def gap(sz, mult=1.25):
        return int(sz * mult)

    # layout
    y = 70
    title_y = y; y += gap(title_sz, 1.10)
    sub1_y  = y; y += gap(text_sz, 1.15)

    y0 = y + int(24 * font_scale)
    x0 = L
    y1 = y0 + box_h

    bottom1_y = y1 + int(44 * font_scale)
    bottom2_y = bottom1_y + gap(num_sz, 1.35)
    bottom3_y = bottom2_y + gap(big_sz, 1.10)
    bottom4_y = bottom3_y + gap(text_sz, 1.20)
    H = bottom4_y + int(40 * font_scale)

    # widths (to scale)
    wA  = box_w * pA
    wAc = box_w * pAc

    # heights (to scale)
    hA  = box_h * pB_given_A
    hAc = box_h * pB_given_Ac
    yA_shade  = y1 - hA
    yAc_shade = y1 - hAc

    cxA  = x0 + wA / 2
    cxAc = x0 + wA + wAc / 2

    area_A_and_B  = pB_given_A  * pA
    area_Ac_and_B = pB_given_Ac * pAc

    def txt(x, y, s, size, weight=400, fill=outline, anchor="middle", opacity=1.0):
        return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                f'font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial" '
                f'font-size="{size}" font-weight="{weight}" fill="{fill}" opacity="{opacity}">{s}</text>')

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')

    # header
    parts.append(txt(L, title_y, "Bayes’ theorem — area model", title_sz, 800, outline, "start"))
    parts.append(txt(L, sub1_y,
                     "A is the entire vertical strip (including shaded). B is the shaded overlay.",
                     text_sz, 400, outline, "start"))

    # outer box
    parts.append(f'<rect x="{x0}" y="{y0}" width="{box_w}" height="{box_h}" fill="none" stroke="{outline}" stroke-width="2"/>')

    # strips
    parts.append(f'<rect x="{x0}" y="{y0}" width="{wA}" height="{box_h}" fill="{strip_fill}" stroke="{outline}" stroke-width="1"/>')
    parts.append(f'<rect x="{x0+wA}" y="{y0}" width="{wAc}" height="{box_h}" fill="{strip_fill}" stroke="{outline}" stroke-width="1"/>')

    # watermarks (smaller + lighter)
    parts.append(txt(cxA,  y0 + box_h*0.56, "A",  int(52*font_scale), 800, outline, opacity=0.05))
    parts.append(txt(cxAc, y0 + box_h*0.56, "Aᶜ", int(52*font_scale), 800, outline, opacity=0.05))

    # shaded B overlay
    parts.append(f'<rect x="{x0}" y="{yA_shade}" width="{wA}" height="{hA}" fill="{shade_fill}" fill-opacity="0.16" stroke="{shade_stroke}" stroke-width="1"/>')
    parts.append(f'<rect x="{x0+wA}" y="{yAc_shade}" width="{wAc}" height="{hAc}" fill="{shade_fill}" fill-opacity="0.16" stroke="{shade_stroke}" stroke-width="1"/>')

    # strong outlines for full strips (clarifies inclusion)
    parts.append(f'<rect x="{x0}" y="{y0}" width="{wA}" height="{box_h}" fill="none" stroke="{accentA}" stroke-width="3"/>')
    parts.append(f'<rect x="{x0+wA}" y="{y0}" width="{wAc}" height="{box_h}" fill="none" stroke="{accentAc}" stroke-width="3"/>')

    # corner labels (instead of big bold top labels)
    pad = int(14 * font_scale)
    parts.append(txt(x0 + wA - pad, y0 + int(26*font_scale), "A", int(15*font_scale), 800, outline, "end"))
    parts.append(txt(x0 + wA - pad, y0 + int(45*font_scale), f"P(A) = {fmt(pA)}", num_sz, 400, outline, "end"))

    parts.append(txt(x0 + box_w - pad, y0 + int(26*font_scale), "Aᶜ", int(15*font_scale), 800, outline, "end"))
    parts.append(txt(x0 + box_w - pad, y0 + int(45*font_scale), f"P(Aᶜ) = {fmt(pAc)}", num_sz, 400, outline, "end"))

    y_anchor = y1 - int(22 * font_scale)   # move up/down by changing 90
    
    line1_y = y_anchor
    line2_y = y_anchor + int(18 * font_scale)
    
    parts.append(txt(cxA,  line1_y, "A ∩ B", num_sz, 900, shade_stroke))
    parts.append(txt(cxA,  line2_y, f"P(B|A) = {fmt(pB_given_A)}", num_sz, 500, shade_stroke))
    
    parts.append(txt(cxAc, line1_y, "Aᶜ ∩ B", num_sz, 900, shade_stroke))
    parts.append(txt(cxAc, line2_y, f"P(B|Aᶜ) = {fmt(pB_given_Ac)}", num_sz, 500, shade_stroke))


    # bottom explanations
    parts.append(txt(cxA,  bottom1_y, f"area(A∩B) = P(B|A)·P(A) = {fmt(area_A_and_B)}", num_sz, 400, outline))
    parts.append(txt(cxAc, bottom1_y, f"area(Aᶜ∩B) = P(B|Aᶜ)·P(Aᶜ) = {fmt(area_Ac_and_B)}", num_sz, 400, outline))

    parts.append(txt(x0 + box_w/2, bottom3_y,
                     f"P(A|B) = area(A∩B) / area(B) = {fmt(pA_given_B)}",
                     big_sz, 900, accentA))
    parts.append(txt(x0 + box_w/2, bottom4_y,
                     f"area(B) = area(A∩B) + area(Aᶜ∩B) = P(B) = {fmt(pB)}",
                     text_sz, 400, outline))

    parts.append("</svg>")
    Path(filename).write_text("\n".join(parts), encoding="utf-8")
    return filename

save_bayes_area_svg("bayes-area.svg", pA=0.35, pB_given_A=0.70, pB_given_Ac=0.20, font_scale=2.0)
```

```{figure} bayes-area.svg
---
width: 100%
figclass: full-width
---
Area model: $P(A\mid B)$ is “the share of the shaded $B$ region that falls inside the $A$ strip”.
```

**How to read the diagram**

* The outer rectangle is the sample space $S$ (all possible outcomes).
* The two vertical strips form a partition of $S$: either you are in $A$ or in $A^c$ (never both, and no gaps).

  * The strip widths are proportional to their probabilities: width$(A)=P(A)$ and width$(A^c)=P(A^c)$.
* The shaded overlay represents the evidence event $B$.
* Within each strip, the **shaded height** encodes the conditional probability of $B$ in that case:

  * In the $A$ strip the height is $P(B\mid A)$, so the shaded area is
    $$
    \text{area}(A\cap B)=P(B\mid A)\,P(A).
    $$
  * In the $A^c$ strip the height is $P(B\mid A^c)$, so the shaded area is
    $$
    \text{area}(A^c\cap B)=P(B\mid A^c)\,P(A^c).
    $$
* Adding the two shaded areas gives the total shaded region:
  $$
  \text{area}(B)=\text{area}(A\cap B)+\text{area}(A^c\cap B)=P(B).
  $$
  This is the **Law of Total Probability** using the partition $\{A, A^c\}$.
* Bayes’ theorem is the corresponding **area ratio**:
  $$
  P(A\mid B)=\frac{\text{area}(A\cap B)}{\text{area}(B)}.
  $$
  Read it as: *“given that we are in the shaded $B$ region, what fraction of that region lies inside $A$?”*

+++

## 2. Updating Beliefs: Prior and Posterior Probabilities

+++

The core idea of Bayesian thinking is updating beliefs. We start with a prior belief, gather data (evidence), and update our belief to a posterior. This posterior can then become the prior for the next piece of evidence.

**Example:** Imagine you have a website and you're testing a new ad banner.

* **Hypothesis (A):** The new ad banner is effective (e.g., has a click-through rate > 5%).
* **Prior ( $P(A)$ ):** Based on previous ad campaigns, you might initially believe there's a 30% chance the new ad is effective. So, $P(A) = 0.30$.
* **Evidence (B):** You observe a visitor's Browse history (e.g., they previously visited related product pages).
* **Likelihood ( $P(B|A) $):** The probability that a visitor has this Browse history *given* the ad is effective. Perhaps effective ads are better targeted, so this might be high, say $P(B|A) = 0.70$.
* **Likelihood ( $P(B|A^c)$ ):** The probability that a visitor has this Browse history *given* the ad is *not* effective. This might be lower, say $P(B|A^c) = 0.20$.
* **Probability of Evidence ( $P(B)$ ):** Using the Law of Total Probability:
    $P(B) = P(B|A)P(A) + P(B|A^c)P(A^c)$
    $P(B) = (0.70)(0.30) + (0.20)(1 - 0.30)$
    $P(B) = 0.21 + (0.20)(0.70) = 0.21 + 0.14 = 0.35$
* **Posterior ( $P(A|B)$ ):** Now apply Bayes' Theorem:
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

Sometimes two events look related **overall**, but become independent once we “zoom in” on a relevant condition.

Think of **$C$** as a *context switch*: if you fix the context, $A$ and $B$ stop giving each other information.

---

### Definition and notation

We use the symbol **$\perp$** (read “is independent of”).

* **Unconditional independence**
  $$
  A \perp B
  \quad\Longleftrightarrow\quad
  P(A\cap B)=P(A)\,P(B).
  $$

* **Conditional independence**
  $$
  A \perp B \mid C
  \quad\Longleftrightarrow\quad
  P(A\cap B \mid C)=P(A\mid C)\,P(B\mid C),
  \qquad P(C)>0.
  $$

**How to read it:** “Within the world where $C$ is known to be true, $A$ and $B$ behave like independent events.”

---

:::{admonition} A more intuitive equivalent check (optional, but useful)
:class: tip dropdown

If $P(B\cap C)>0$, then
$$
A \perp B \mid C
\quad\Longleftrightarrow\quad
P(A\mid B\cap C)=P(A\mid C).
$$

Likewise (symmetrically), if $P(A\cap C)>0$ then
$$
P(B\mid A\cap C)=P(B\mid C).
$$

**Interpretation:** once you already know $C$, learning $B$ gives you **no further update** about $A$ (and vice versa).
:::

---

:::{admonition} Warning: conditional independence is not the same as independence
:class: warning dropdown

$A \perp B \mid C$ does **not** imply $A \perp B$.

A very common pattern is:

* independent **within each fixed** value of $C$
* dependent **after mixing** (when $C$ is hidden)

So conditional independence is about what happens **inside** a fixed context, not after you average over contexts.
:::

---

### A visual mini-example: two flips of a randomly chosen coin

To make conditional independence concrete, we’ll use a simple example.

We have two coins:

* Fair coin (F): $P(H)=0.5$
* Biased coin (B): $P(H)=0.75$

Pick a coin uniformly at random, then flip it twice.

Let:

* $H_1$ = “first flip is Heads”
* $H_2$ = “second flip is Heads”
* $C$ = “we chose the fair coin” (so $C^c$ = “we chose the biased coin”)

**How to read the figure:** each top panel fixes the context (fair vs biased coin). Inside a panel, the shaded overlap represents $P(H_1\cap H_2\mid \text{context})$, and the strip width/height represent $P(H_1\mid \text{context})$ and $P(H_2\mid \text{context})$.

```{code-cell} python3
:tags: [remove-input, remove-output]

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

def draw_context(ax, p, title):
    '''
    Unit box for a fixed context (area = 1).
    Here p = P(H) within that context, so:
      P(H1|context)=p, P(H2|context)=p, P(H1∩H2|context)=p^2
    '''
    p = max(0.0, min(1.0, float(p)))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # Outer box
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, linewidth=2))

    # Light grayscale fills (clean in SVG and print-friendly)
    ax.add_patch(Rectangle((0, 0), p, 1, facecolor="#d9d9d9", edgecolor="none"))     # H1 strip
    ax.add_patch(Rectangle((0, 1-p), 1, p, facecolor="#c7c7c7", edgecolor="none"))   # H2 strip
    ax.add_patch(Rectangle((0, 1-p), p, p, facecolor="#9e9e9e", edgecolor="none"))   # overlap

    # Subtle outlines
    ax.add_patch(Rectangle((0, 0), p, 1, fill=False, linewidth=1.0))
    ax.add_patch(Rectangle((0, 1-p), 1, p, fill=False, linewidth=1.0))
    ax.add_patch(Rectangle((0, 1-p), p, p, fill=False, linewidth=1.2))

    # Labels
    ax.text(p/2, 0.03, r"$H_1$", ha="center", va="bottom", fontsize=12)
    ax.text(0.03, 1-p/2, r"$H_2$", ha="left", va="center", fontsize=12)
    ax.text(p/2, 1-p/2, r"$H_1\cap H_2$", ha="center", va="center", fontsize=12, color="white")

    # Compact numbers (no arrows)
    ax.text(
        0.0, -0.14,
        rf"$P(H_1\mid\cdot)={p:.2f}$   $P(H_2\mid\cdot)={p:.2f}$   $P(H_1\cap H_2\mid\cdot)={p*p:.4f}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=11
    )

def draw_mixture(ax, w_fair, w_biased, p_fair, p_biased):
    '''
    Mixture panel + the overall (unconditional) independence check.
    '''
    w_fair = max(0.0, float(w_fair))
    w_biased = max(0.0, float(w_biased))
    tot = (w_fair + w_biased) if (w_fair + w_biased) > 0 else 1.0
    wf, wb = w_fair / tot, w_biased / tot

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(r"If you do NOT know $C$ (mixture)", fontsize=13, fontweight="bold", pad=10)

    # Outer container
    ax.add_patch(Rectangle((0, 0.28), 1, 0.60, fill=False, linewidth=2))

    # Two stacked context bands (light fills)
    ax.add_patch(Rectangle((0, 0.28 + 0.60*(1-wf)), 1, 0.60*wf, facecolor="#e6e6e6", edgecolor="none"))
    ax.add_patch(Rectangle((0, 0.28), 1, 0.60*wb, facecolor="#d1d1d1", edgecolor="none"))

    ax.text(0.02, 0.28 + 0.60*(1-wf/2), rf"Fair context ($C$), weight $P(C)={w_fair:.2f}$",
            ha="left", va="center", fontsize=12)
    ax.text(0.02, 0.28 + 0.60*(wb/2), rf"Biased context ($C^c$), weight $P(C^c)={w_biased:.2f}$",
            ha="left", va="center", fontsize=12)

    # Overall numbers (the punchline)
    P_H1 = w_fair*p_fair + w_biased*p_biased
    P_H2 = P_H1
    P_HH = w_fair*(p_fair*p_fair) + w_biased*(p_biased*p_biased)
    prod = P_H1 * P_H2
    P_H2_given_H1 = P_HH / P_H1

    ax.text(0.5, 0.17,
            rf"$P(H_1\cap H_2)={P_HH:.5f}$  vs  $P(H_1)P(H_2)={prod:.6f}$",
            ha="center", va="center", fontsize=13)

    ax.text(0.5, 0.07,
            rf"Update check:  $P(H_2\mid H_1)={P_H2_given_H1:.2f}$  but  $P(H_2)={P_H2:.3f}$",
            ha="center", va="center", fontsize=12)

# --- Coin example numbers ---
p_fair, p_biased = 0.50, 0.75
w_fair, w_biased = 0.50, 0.50

fig = plt.figure(figsize=(12.5, 7.4))
gs = GridSpec(2, 2, height_ratios=[1.15, 0.85], hspace=0.40, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

draw_context(ax1, p_fair,   title=r"Given $C$ = Fair coin ($P(H)=0.5$)")
draw_context(ax2, p_biased, title=r"Given $C^c$ = Biased coin ($P(H)=0.75$)")
draw_mixture(ax3, w_fair, w_biased, p_fair, p_biased)

fig.suptitle(
    "Two flips of a randomly chosen coin: factorizes within each context; mixing creates dependence",
    fontsize=14, fontweight="bold", y=0.98
)

out_svg = "conditional-independence-coin-mix.svg"
fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0)
plt.close(fig)
```

```{figure} conditional-independence-coin-mix.svg
---
width: 100%
figclass: full-width
---
Top row: within each fixed context ($C$ or $C^c$), $H_1$ and $H_2$ factorize.  
Bottom: when $C$ is hidden, the mixture can create dependence overall.
```

---

:::{admonition} Worked calculations (optional)
:class: tip dropdown

#### Scenario 1: we do *not* know which coin was chosen

By total probability:
$$
P(H_1)=0.5\cdot 0.5 + 0.5\cdot 0.75 = 0.625,
\qquad
P(H_2)=0.625.
$$

And
$$
P(H_1\cap H_2)=0.5\cdot 0.25 + 0.5\cdot 0.5625 = 0.40625.
$$

If $H_1$ and $H_2$ were independent overall, we’d have
$$
P(H_1)P(H_2)=0.625^2=0.390625 \neq 0.40625.
$$

An “update” check shows the same thing:
$$
P(H_2\mid H_1)=\frac{0.40625}{0.625}=0.65
\quad\text{but}\quad
P(H_2)=0.625.
$$

#### Scenario 2: we know we chose the fair coin ($C$)

$$
P(H_1\mid C)=0.5,\quad P(H_2\mid C)=0.5,\quad P(H_1\cap H_2\mid C)=0.25,
$$
and
$$
P(H_1\cap H_2\mid C)=P(H_1\mid C)\,P(H_2\mid C).
$$

#### Scenario 3: we know we chose the biased coin ($C^c$)

$$
P(H_1\mid C^c)=0.75,\quad P(H_2\mid C^c)=0.75,\quad P(H_1\cap H_2\mid C^c)=0.5625,
$$
and
$$
P(H_1\cap H_2\mid C^c)=P(H_1\mid C^c)\,P(H_2\mid C^c).
$$

So $H_1$ and $H_2$ are **conditionally independent** given which coin was chosen:
$$
H_1 \perp H_2 \mid C.
$$
:::

---

### Intuition in one sentence

**Conditioning on $C$ “locks in the context”; within that context, $H_1$ and $H_2$ don’t update each other—hiding $C$ mixes contexts and can create dependence.**

+++

## Chapter Summary


* **Bayes' Theorem** $P(A|B) = \frac{P(B|A) P(A)}{P(B)}$ provides a fundamental rule for updating probabilities (beliefs) based on new evidence.
* It relates the **posterior probability** $P(A|B)$ to the **prior probability** $P(A)$ and the **likelihood** $P(B|A)$.
* The term $P(B)$ acts as a normalizing constant and can often be calculated using the **Law of Total Probability**.
* Bayes' Theorem is crucial in fields like medical diagnosis, machine learning (spam filtering, classification), and scientific reasoning.
* Two events A and B are **independent** if $P(A \cap B) = P(A)P(B)$, or equivalently, $P(A|B) = P(A)$ (assuming $P(B)>0$). The occurrence of one does not change the probability of the other.
* Events A and B are **conditionally independent** given C if $P(A \cap B | C) = P(A|C)P(B|C)$. They become independent once the outcome of C is known.
* Simulation is a valuable tool for building intuition about Bayes' Theorem and independence by observing frequencies in generated data.

+++

In the next part of the book, we will shift our focus from events to **Random Variables** – numerical outcomes of random phenomena – and explore their distributions. This will allow us to model and analyze probabilistic situations in a more structured way.
