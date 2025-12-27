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

### 1.1 Visual intuition: Bayes’ Theorem (area model)

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
\begin{align*}
P(B) &= P(B\cap A)+P(B\cap A^c) \\
&= P(B\mid A)P(A)+P(B\mid A^c)P(A^c).
\end{align*}
$$

Substitute into the Bayes form:

$$
\begin{align*}
P(A\mid B)
&=\frac{P(B\mid A)P(A)}{P(B\mid A)P(A)+P(B\mid A^c)P(A^c)}.
\end{align*}
$$

```{code-cell} ipython3
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
  
    $$
    \begin{align*}
    P(B) &= P(B|A)P(A) + P(B|A^c)P(A^c) \\
    &= (0.70)(0.30) + (0.20)(1 - 0.30) \\
    &= 0.21 + (0.20)(0.70) \\
    &= 0.21 + 0.14 = 0.35
    \end{align*}
    $$
  
* **Posterior ( $P(A|B)$ ):** Now apply Bayes' Theorem:
  
    $$
    \begin{align*}
    P(A|B) &= \frac{P(B|A) P(A)}{P(B)} \\
    &= \frac{(0.70)(0.30)}{0.35} \\
    &= \frac{0.21}{0.35} = 0.60
    \end{align*}
    $$

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

$$
\begin{align*}
P(D|Pos) &= \frac{(0.95)(0.01)}{0.0590} \\
&= \frac{0.0095}{0.0590} \\
&\approx 0.161
\end{align*}
$$

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
&= \left( \frac{3}{51} \right) \left( \frac{4}{52} \right)
   + \left( \frac{4}{51} \right) \left( \frac{48}{52} \right) \\
&= \frac{3 \times 4}{51 \times 52} + \frac{4 \times 48}{51 \times 52} \\
&= \frac{12}{2652} + \frac{192}{2652} \\
&= \frac{12 + 192}{2652} \\
&= \frac{204}{2652} = \frac{4}{52} = \frac{1}{13}
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
\frac{1}{221} &\stackrel{?}{=} \left( \frac{4}{52} \right)
   \times \left( \frac{4}{52} \right) \\
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

```{code-cell} ipython3
:tags: [remove-input, remove-output]

from pathlib import Path

def save_common_cause_svg():
    """Create two diagrams showing common cause pattern."""

    # Colors
    node_fill = "#1e293b"
    node_stroke = "#0f172a"
    arrow_color = "#3b82f6"
    arrow_blocked = "#94a3b8"
    text_color = "#ffffff"
    bg_color = "#ffffff"
    label_color = "#111827"
    highlight_color = "#ef4444"

    # Dimensions
    node_r = 60
    font_size = 18
    label_font = 14
    arrow_width = 3

    def make_arrow(x1, y1, x2, y2, color, dashed=False):
        """Create an arrow path."""
        # Shorten to stop at node edge
        dx, dy = x2 - x1, y2 - y1
        length = (dx**2 + dy**2)**0.5
        if length == 0:
            return ""
        ux, uy = dx/length, dy/length
        x1_adj = x1 + ux * node_r
        y1_adj = y1 + uy * node_r
        x2_adj = x2 - ux * (node_r + 15)
        y2_adj = y2 - uy * (node_r + 15)

        style = f'stroke-dasharray="8,4"' if dashed else ''

        return f'''
        <defs>
            <marker id="arrowhead-{color.replace("#","")}" markerWidth="10" markerHeight="10"
                    refX="9" refY="3" orient="auto">
                <polygon points="0 0, 10 3, 0 6" fill="{color}" />
            </marker>
        </defs>
        <line x1="{x1_adj}" y1="{y1_adj}" x2="{x2_adj}" y2="{y2_adj}"
              stroke="{color}" stroke-width="{arrow_width}" {style}
              marker-end="url(#arrowhead-{color.replace("#","")})" />
        '''

    def make_node(x, y, label, sublabel=""):
        """Create a circular node."""
        sub_part = f'<text x="{x}" y="{y+22}" text-anchor="middle" font-size="{label_font}" fill="{text_color}" opacity="0.8">{sublabel}</text>' if sublabel else ''
        return f'''
        <circle cx="{x}" cy="{y}" r="{node_r}" fill="{node_fill}" stroke="{node_stroke}" stroke-width="2"/>
        <text x="{x}" y="{y+6}" text-anchor="middle" font-size="{font_size}" font-weight="bold" fill="{text_color}">{label}</text>
        {sub_part}
        '''

    def make_block_symbol(x, y):
        """Create a blocking symbol (circled X)."""
        return f'''
        <circle cx="{x}" cy="{y}" r="25" fill="{highlight_color}" stroke="#b91c1c" stroke-width="2"/>
        <text x="{x}" y="{y+8}" text-anchor="middle" font-size="28" font-weight="bold" fill="#ffffff">⊗</text>
        '''

    # Image 1: Without context (apparent dependence)
    w1, h1 = 800, 300
    cx1, cy1 = w1 // 2, h1 // 2

    svg1_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w1}" height="{h1}" viewBox="0 0 {w1} {h1}">']
    svg1_parts.append(f'<rect width="{w1}" height="{h1}" fill="{bg_color}"/>')

    # Nodes
    x_h1, x_h2 = 200, 600
    svg1_parts.append(make_node(x_h1, cy1, "H₁", "Umbrellas"))
    svg1_parts.append(make_node(x_h2, cy1, "H₂", "Flashlights"))

    # Arrow from H1 to H2
    svg1_parts.append(make_arrow(x_h1, cy1, x_h2, cy1, arrow_color))

    # Label
    svg1_parts.append(f'<text x="{cx1}" y="40" text-anchor="middle" font-size="16" font-weight="bold" fill="{label_color}">Without Context: Information Appears to Flow</text>')
    svg1_parts.append(f'<text x="{cx1}" y="260" text-anchor="middle" font-size="14" fill="{label_color}">P(H₂ | H₁) ≠ P(H₂) — The events appear dependent</text>')

    svg1_parts.append('</svg>')

    # Image 2: With context (conditional independence)
    w2, h2 = 800, 400

    svg2_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w2}" height="{h2}" viewBox="0 0 {w2} {h2}">']
    svg2_parts.append(f'<rect width="{w2}" height="{h2}" fill="{bg_color}"/>')

    # Nodes - arranged in V shape
    cx2 = w2 // 2
    y_top = 130
    y_bottom = 310
    x_left = 200
    x_right = 600

    svg2_parts.append(make_node(cx2, y_top, "C", "Storm Warning"))
    svg2_parts.append(make_node(x_left, y_bottom, "H₁", "Umbrellas"))
    svg2_parts.append(make_node(x_right, y_bottom, "H₂", "Flashlights"))

    # Arrows from C to both H1 and H2
    svg2_parts.append(make_arrow(cx2, y_top, x_left, y_bottom, arrow_color))
    svg2_parts.append(make_arrow(cx2, y_top, x_right, y_bottom, arrow_color))

    # Blocked arrow between H1 and H2 (dashed and grayed)
    svg2_parts.append(make_arrow(x_left, y_bottom, x_right, y_bottom, arrow_blocked, dashed=True))

    # Block symbol in the middle
    svg2_parts.append(make_block_symbol(cx2, y_bottom))

    # Labels
    svg2_parts.append(f'<text x="{cx2}" y="40" text-anchor="middle" font-size="16" font-weight="bold" fill="{label_color}">With Context: Common Cause Blocks Information Flow</text>')
    svg2_parts.append(f'<text x="{cx2}" y="394" text-anchor="middle" font-size="14" fill="{label_color}">P(H₂ | H₁, C) = P(H₂ | C) — Conditionally independent given C</text>')

    svg2_parts.append('</svg>')

    # Save both files
    Path("common-cause-without-context.svg").write_text("\n".join(svg1_parts), encoding="utf-8")
    Path("common-cause-with-context.svg").write_text("\n".join(svg2_parts), encoding="utf-8")

    return "common-cause-without-context.svg", "common-cause-with-context.svg"

save_common_cause_svg()
```

```{admonition} Why this section matters (and why it's tricky)
:class: tip

Conditional independence is one of the most subtle but important concepts in probability. It explains many real-world phenomena that seem paradoxical at first:

* Why a treatment might appear effective overall but ineffective (or even harmful) within specific patient groups
* Why two variables might seem correlated in your data but are actually unrelated once you account for a hidden factor
* How mixing data from different sources can create spurious relationships

**The key insight:** Two events can be *independent* when you know the context, but *dependent* when the context is hidden. This is counter-intuitive because we're used to thinking of independence as an absolute property, not something that depends on what else we know.

**Take your time with this section.** The concepts are subtle and you will probably need to read this section multiple times. This is completely normal - conditional independence takes time to internalize, but the payoff is enormous for understanding statistics, causality, and data analysis.
```

Sometimes two events appear related overall (in the same experiment), but become independent once we condition on a relevant context **$C$**.

Think of **$C$** as a *context switch*: if you fix the context, $A$ and $B$ stop giving each other information.

---

:::{admonition} Example: The Grocery Store (Common Cause)
:class: tip

This example illustrates how external "shocks" affect behavior and create apparent connections between events.

**Scenario:**
- Variable $H_1$: Sales of umbrellas increase.
- Variable $H_2$: Sales of flashlights increase.
- Condition $C$: A severe storm warning is issued.

**Why it works:**

You notice that every time people buy umbrellas, they also seem to buy flashlights. The two events appear linked. However, the umbrella purchase doesn't *cause* the flashlight purchase. Both are independent responses to the storm warning ($C$).

Once you know the storm is coming, seeing someone grab an umbrella tells you nothing new about the flashlight stock—the storm already told you everything you needed to know.

**Mathematically:**
$$
P(H_2 \mid H_1, C) = P(H_2 \mid C)
$$

```{admonition} Notation note
:class: note
The notation $P(H_2 \mid H_1, C)$ means "the probability of $H_2$ given both $H_1$ and $C$"—the comma is shorthand for "and". We'll explain this notation in more detail in Section 5.1.
```

This equation says: "Given that we know a storm warning was issued ($C$), learning that umbrella sales increased ($H_1$) gives us no additional information about whether flashlight sales increased ($H_2$)."

```{figure} common-cause-without-context.svg
---
width: 80%
figclass: full-width
---
**Without knowing the context:** When we don't know about the storm warning, umbrella sales ($H_1$) and flashlight sales ($H_2$) appear to be dependent. Observing one gives us information about the other.
```

```{figure} common-cause-with-context.svg
---
width: 80%
figclass: full-width
---
**With context revealed:** Once we know about the storm warning ($C$), the connection between $H_1$ and $H_2$ is blocked. The storm warning is the common cause of both events. Given $C$, learning about umbrella sales tells us nothing new about flashlight sales—they are conditionally independent.
```

**The key insight:** This pattern—where a common cause creates apparent dependence between effects—is one of the most important concepts in conditional independence. We'll formalize this idea in the sections that follow.

:::

+++

### 5.1. Notation and Definition

Before we explore conditional independence, we need to understand how to work with conditional probabilities involving multiple conditions.

#### Conditioning on Multiple Events

When we write $P(A \mid B, C)$, we mean the probability of event $A$ given that *both* events $B$ and $C$ have occurred. This is equivalent to conditioning on the intersection:

$$
P(A \mid B, C) = P(A \mid B \cap C)
$$

The comma in the conditioning clause is simply a convenient shorthand for the intersection. Both notations are used interchangeably in probability and statistics.

```{admonition} Reading the notation
:class: note

$P(A \mid B, C)$ reads as "the probability of $A$ given $B$ and $C$"

It represents our updated belief about $A$ when we know that both $B$ and $C$ have occurred.
```

```{admonition} Example: Coin Flips
:class: dropdown

Consider flipping a coin twice after choosing which coin to use:
- Let $H_1$ = "first flip is heads"
- Let $H_2$ = "second flip is heads"
- Let $C$ = "we chose the fair coin"

Then $P(H_2 \mid H_1, C)$ means: "What is the probability the second flip is heads, given that the first flip was heads AND we chose the fair coin?"

For a fair coin, knowing the first flip doesn't help predict the second flip, so:
$$
P(H_2 \mid H_1, C) = P(H_2 \mid C) = 0.5
$$

This equation says: "Given we have the fair coin, learning about the first flip gives us no additional information about the second flip." This is an example of conditional independence, which we'll explore in detail below.
```

```{admonition} Important: Order doesn't matter
:class: tip

The order of events after the conditioning bar doesn't matter:
$$
P(A \mid B, C) = P(A \mid C, B) = P(A \mid B \cap C)
$$
```

#### Formal Definition of Conditional Independence

Before we dive into the formal definition, recall that we've already seen independence in Section 4. **Conditional independence** is a related but distinct concept: it's about independence that holds *within* a specific context, even though the events might be dependent overall when contexts are mixed.

We use the symbol **$\perp$** (read "is independent of"). We also use the symbol **$\Longleftrightarrow$** (if and only if) to indicate that both statements are equivalent—each implies the other.

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

**How to read it:** "Within the world where $C$ is known to be true, $A$ and $B$ behave like independent events."

+++

#### Visual representation: Conditional Independence

The Venn diagram below illustrates conditional independence. When we condition on event $C$ having occurred, we restrict our attention to the region $C$. Within that region, events $A$ and $B$ are independent, meaning the overlap of $A$ and $B$ within $C$ equals what we'd expect from the product of their conditional probabilities.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles

# Create figure with three panels side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Function to create a Venn diagram with specific highlighting
def create_venn_panel(ax, highlight_mode, title_text):
    """
    highlight_mode: 'A_and_C', 'B_and_C', or 'A_and_B_and_C'
    """
    # Create three-set Venn diagram
    v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels=('', '', ''), ax=ax)

    # Default: all regions in C are light, regions outside C are very light
    if v.get_patch_by_id('100'):  # A only
        v.get_patch_by_id('100').set_color('#f5f5f5')
        v.get_patch_by_id('100').set_alpha(0.5)
    if v.get_patch_by_id('010'):  # B only
        v.get_patch_by_id('010').set_color('#f5f5f5')
        v.get_patch_by_id('010').set_alpha(0.5)
    if v.get_patch_by_id('110'):  # A ∩ B only (not in C)
        v.get_patch_by_id('110').set_color('#e0e0e0')
        v.get_patch_by_id('110').set_alpha(0.4)

    # Color C regions based on highlight mode
    if highlight_mode == 'A_and_C':
        # Highlight all A ∩ C regions
        if v.get_patch_by_id('001'):  # C only
            v.get_patch_by_id('001').set_color('#ffe0b2')
            v.get_patch_by_id('001').set_alpha(0.5)
        if v.get_patch_by_id('011'):  # B ∩ C (not in A)
            v.get_patch_by_id('011').set_color('#ffe0b2')
            v.get_patch_by_id('011').set_alpha(0.5)
        if v.get_patch_by_id('101'):  # A ∩ C (not in B) - HIGHLIGHT
            v.get_patch_by_id('101').set_color('#ff9800')
            v.get_patch_by_id('101').set_alpha(0.85)
        if v.get_patch_by_id('111'):  # A ∩ B ∩ C - HIGHLIGHT
            v.get_patch_by_id('111').set_color('#ff9800')
            v.get_patch_by_id('111').set_alpha(0.85)
    elif highlight_mode == 'B_and_C':
        # Highlight all B ∩ C regions
        if v.get_patch_by_id('001'):  # C only
            v.get_patch_by_id('001').set_color('#ffe0b2')
            v.get_patch_by_id('001').set_alpha(0.5)
        if v.get_patch_by_id('101'):  # A ∩ C (not in B)
            v.get_patch_by_id('101').set_color('#ffe0b2')
            v.get_patch_by_id('101').set_alpha(0.5)
        if v.get_patch_by_id('011'):  # B ∩ C (not in A) - HIGHLIGHT
            v.get_patch_by_id('011').set_color('#ff9800')
            v.get_patch_by_id('011').set_alpha(0.85)
        if v.get_patch_by_id('111'):  # A ∩ B ∩ C - HIGHLIGHT
            v.get_patch_by_id('111').set_color('#ff9800')
            v.get_patch_by_id('111').set_alpha(0.85)
    else:  # 'A_and_B_and_C'
        # Highlight only the center region
        if v.get_patch_by_id('001'):  # C only
            v.get_patch_by_id('001').set_color('#ffe0b2')
            v.get_patch_by_id('001').set_alpha(0.5)
        if v.get_patch_by_id('101'):  # A ∩ C (not in B)
            v.get_patch_by_id('101').set_color('#ffe0b2')
            v.get_patch_by_id('101').set_alpha(0.5)
        if v.get_patch_by_id('011'):  # B ∩ C (not in A)
            v.get_patch_by_id('011').set_color('#ffe0b2')
            v.get_patch_by_id('011').set_alpha(0.5)
        if v.get_patch_by_id('111'):  # A ∩ B ∩ C - HIGHLIGHT
            v.get_patch_by_id('111').set_color('#ff6d00')
            v.get_patch_by_id('111').set_alpha(0.9)

    # Draw circles
    venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='solid', linewidth=2, ax=ax)

    # Add set labels
    label_A = v.get_label_by_id('A')
    if label_A:
        label_A.set_text('A')
        label_A.set_fontsize(16)

    label_B = v.get_label_by_id('B')
    if label_B:
        label_B.set_text('B')
        label_B.set_fontsize(16)

    label_C = v.get_label_by_id('C')
    if label_C:
        label_C.set_text('C')
        label_C.set_fontsize(16)

    # Add title below the diagram
    ax.text(0.5, -0.15, title_text,
            transform=ax.transAxes,
            fontsize=13, ha='center', va='top',
            fontweight='bold')

    return v

# Panel 1: P(A|C)
create_venn_panel(ax1, 'A_and_C', 'P(A | C)\nProportion of C that is in A')

# Panel 2: P(B|C)
create_venn_panel(ax2, 'B_and_C', 'P(B | C)\nProportion of C that is in B')

# Panel 3: P(A∩B|C)
create_venn_panel(ax3, 'A_and_B_and_C', 'P(A ∩ B | C)\nProportion of C in both A and B')

# Add overall title
fig.suptitle('Conditional Independence Formula Components: P(A ∩ B | C) = P(A | C) × P(B | C)',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
fig.savefig("venn-conditional-independence.svg", format="svg", bbox_inches="tight", pad_inches=0.3)
```

```{figure} venn-conditional-independence.svg
---
width: 100%
:figclass: full-width
---
Three-panel visualization of the conditional independence formula. Left panel: $P(A \mid C)$ highlights all regions in both $A$ and $C$. Middle panel: $P(B \mid C)$ highlights all regions in both $B$ and $C$. Right panel: $P(A \cap B \mid C)$ highlights only the region in all three sets. The formula states these proportions satisfy: $P(A \cap B \mid C) = P(A \mid C) \times P(B \mid C)$.
```

**Key observation from the three panels:**

The three panels above show how each term in the conditional independence formula corresponds to different regions within $C$:

**Breaking down the formula:** $P(A \cap B \mid C) = P(A \mid C) \times P(B \mid C)$

* **Left panel - $P(A \mid C)$:** Shows the proportion of region $C$ that lies in $A$
  * The dark orange regions represent all parts of $A$ that overlap with $C$

* **Middle panel - $P(B \mid C)$:** Shows the proportion of region $C$ that lies in $B$
  * The dark orange regions represent all parts of $B$ that overlap with $C$

* **Right panel - $P(A \cap B \mid C)$:** Shows the proportion of region $C$ that lies in *both* $A$ and $B$
  * The dark orange region is the central intersection of all three sets

:::{admonition} Important: We're multiplying proportions, not adding areas!
:class: warning

Notice that the center region $(A \cap B \cap C)$ appears highlighted in both the left and middle panels. This might look like we're "double counting," but we're not adding these areas—we're **multiplying proportions**.

When we compute $P(A \mid C) \times P(B \mid C)$, we're multiplying two fractions: (orange area in left panel ÷ total $C$ area) × (orange area in middle panel ÷ total $C$ area). This multiplication gives us the proportion shown in the right panel.

Under conditional independence, this multiplication of proportions equals exactly the proportion of $C$ that lies in both $A$ and $B$. The fact that the center region appears in both left and middle panels is precisely what makes the multiplication work out to match the right panel.
:::

**The independence relationship:** Conditional independence means that when we restrict our view to region $C$, these proportions satisfy the multiplication rule. The proportion in both $A$ and $B$ (right panel) equals the product of the individual proportions (left panel × middle panel). This is the visual embodiment of $P(A \cap B \mid C) = P(A \mid C) P(B \mid C)$.

This is different from looking at $A$ and $B$ in the entire sample space, where they might be dependent. Conditional independence means they become independent *once we fix the context* $C$.

+++

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
:class: warning

**This is a critical point that students often miss:** $A \perp B \mid C$ does **not** imply $A \perp B$.

A very common pattern is:

* independent **within each fixed** value of $C$
* dependent **after mixing** (when $C$ is hidden)

So conditional independence is about what happens **inside** a fixed context, not after you average over contexts.
:::

---

### 5.2. A visual mini-example: two flips of a randomly chosen coin

To make conditional independence concrete, we’ll use a simple example.

We have two coins:

* Fair coin (F): $P(H)=0.5$
* Biased coin (B): $P(H)=0.75$

Pick a coin uniformly at random, then flip it twice.

Let:

* $H_1$ = “first flip is Heads”
* $H_2$ = “second flip is Heads”
* $C$ = “we chose the fair coin” (so $C^c$ = “we chose the biased coin”)

#### Part 1: Independence within each context

First, let's see what happens when we **know which coin we have**. The key insight is that once you fix the context (know the coin), the two flips become independent.

**What to notice:**

If you **fix the coin** (you know $C$ or $C^c$), then the two flips are independent: knowing $H_1$ doesn't change the probability of $H_2$. Mathematically:

$$
P(H_2\mid H_1, C) = P(H_2\mid C)
\quad\text{and}\quad
P(H_2\mid H_1, C^c) = P(H_2\mid C^c)
$$

This means the joint probability factorizes (splits into a product) within each context:

(factorization-formula)=
$$
P(H_1\cap H_2\mid C) = P(H_1\mid C)\,P(H_2\mid C)
$$

and similarly for $C^c$. Let's visualize this:

```{code-cell} ipython3
:tags: [remove-input, remove-output]

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_context(ax, p, title, cond_tex):
    """Draw a single context panel showing conditional independence."""
    p = max(0.0, min(1.0, float(p)))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, linewidth=2))

    # Draw strips for H1 and H2
    ax.add_patch(Rectangle((0, 0), p, 1, facecolor="#d9d9d9", edgecolor="none"))     # H1 strip
    ax.add_patch(Rectangle((0, 1-p), 1, p, facecolor="#c7c7c7", edgecolor="none"))   # H2 strip
    ax.add_patch(Rectangle((0, 1-p), p, p, facecolor="#9e9e9e", edgecolor="none"))   # overlap

    ax.add_patch(Rectangle((0, 0), p, 1, fill=False, linewidth=1.0))
    ax.add_patch(Rectangle((0, 1-p), 1, p, fill=False, linewidth=1.0))
    ax.add_patch(Rectangle((0, 1-p), p, p, fill=False, linewidth=1.2))

    # Labels
    ax.text(p/2, 0.03, r"$H_1$", ha="center", va="bottom", fontsize=12)
    ax.text(0.03, 1-p/2, r"$H_2$", ha="left", va="center", fontsize=12)
    ax.text(p/2, 1-p/2, r"$H_1\cap H_2$", ha="center", va="center", fontsize=12, color="white")

# Coin parameters
p_fair, p_biased = 0.50, 0.75

# Create figure with two panels side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

draw_context(ax1, p_fair,   title=r"Given $C$ = Fair coin ($P(H)=0.5$)",    cond_tex=r"C")
draw_context(ax2, p_biased, title=r"Given $C^c$ = Biased coin ($P(H)=0.75$)", cond_tex=r"C^c")

fig.suptitle(
    "Conditional independence: within each fixed context, the flips are independent",
    fontsize=14, fontweight="bold", y=1.02
)

plt.tight_layout()
fig.savefig("conditional-independence-contexts.svg", format="svg", bbox_inches="tight", pad_inches=0.3)
```

:::{figure} conditional-independence-contexts.svg
:width: 100%
:figclass: full-width

**Conditional independence within each context.** Each panel fixes the coin type. Within a panel, the shaded overlap represents $P(H_1\cap H_2\mid \text{context})$, and the strip dimensions show $P(H_1\mid \text{context})$ and $P(H_2\mid \text{context})$.
:::

**Numerical verification:**

For the **fair coin** (left panel):
- $P(H_1\mid C) = 0.50$
- $P(H_2\mid C) = 0.50$
- $P(H_1\cap H_2\mid C) = 0.25$
- **Factorization check:** $P(H_1\mid C) \times P(H_2\mid C) = 0.50 \times 0.50 = 0.25$ ✓

For the **biased coin** (right panel):
- $P(H_1\mid C^c) = 0.75$
- $P(H_2\mid C^c) = 0.75$
- $P(H_1\cap H_2\mid C^c) = 0.5625$
- **Factorization check:** $P(H_1\mid C^c) \times P(H_2\mid C^c) = 0.75 \times 0.75 = 0.5625$ ✓

In both panels, the joint probability equals the product of the marginals. This is what independence looks like.

---

#### Part 2: What happens when the context is hidden (mixing)

Now comes the surprising part: **when we don't know which coin was chosen**, the flips are no longer independent!

**Why dependence emerges:**

If you **don't know the coin**, then observing $H_1$ gives you information about *which coin you probably have*. For example:
- Seeing Heads on the first flip makes the biased coin more likely
- This makes Heads on the second flip more likely
- So $H_1$ and $H_2$ are dependent when the context is hidden

**Mathematical setup:**

To find the overall probability of both flips being heads when we don't know which coin was chosen, we apply the Law of Total Probability using the partition $\{C, C^c\}$:

$$
\begin{align*}
P(H_1\cap H_2) &= P(H_1\cap H_2\mid C)P(C) \\
&\quad + P(H_1\cap H_2\mid C^c)P(C^c)
\end{align*}
$$

This is the same principle we used earlier for single events (like $P(B) = P(B|A)P(A) + P(B|A^c)P(A^c)$), but now applied to the intersection $H_1 \cap H_2$. We're splitting the joint event into two mutually exclusive cases (fair coin vs. biased coin) and adding their weighted probabilities.

Let's visualize how mixing the two contexts creates dependence:

```{code-cell} ipython3
:tags: [remove-input, remove-output]

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_mixture(ax, w_fair, w_biased, p_fair, p_biased):
    w_fair = max(0.0, float(w_fair))
    w_biased = max(0.0, float(w_biased))
    tot = (w_fair + w_biased) if (w_fair + w_biased) > 0 else 1.0
    wf, wb = w_fair / tot, w_biased / tot

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(r"Context hidden (mixture)", fontsize=13, fontweight="bold", pad=10)

    # Center the box
    y0, h = 0.25, 0.50
    ax.add_patch(Rectangle((0, y0), 1, h, fill=False, linewidth=2))

    ax.add_patch(Rectangle((0, y0 + h*(1-wf)), 1, h*wf, facecolor="#e6e6e6", edgecolor="none"))
    ax.add_patch(Rectangle((0, y0),            1, h*wb, facecolor="#d1d1d1", edgecolor="none"))

    ax.text(0.02, y0 + h*(1-wf/2),
            rf"Fair context ($C$), weight $P(C)={w_fair:.2f}$",
            ha="left", va="center", fontsize=12, clip_on=False)

    ax.text(0.02, y0 + h*(wb/2),
            rf"Biased context ($C^c$), weight $P(C^c)={w_biased:.2f}$",
            ha="left", va="center", fontsize=12, clip_on=False)

# Coin parameters
p_fair, p_biased = 0.50, 0.75
w_fair, w_biased = 0.50, 0.50  # Equal probability of choosing each coin

# Create figure with just the mixture panel
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

draw_mixture(ax, w_fair, w_biased, p_fair, p_biased)

plt.tight_layout()
fig.savefig("conditional-independence-mixture.svg", format="svg", bbox_inches="tight", pad_inches=0.4)
```

:::{figure} conditional-independence-mixture.svg
:width: 70%

**The mixing effect.** When we don't know which coin was chosen, we must combine the two contexts (fair and biased) using their probabilities as weights.
:::

**Understanding the calculation:**

When the context is hidden, we use the **Law of Total Probability** to combine both scenarios, weighting each by how likely it is to occur (note that $P(C) + P(C^c) = 1$):

$$
\begin{align*}
P(H_1\cap H_2) &= P(H_1\cap H_2\mid C)P(C) \\
&\quad + P(H_1\cap H_2\mid C^c)P(C^c)
\end{align*}
$$

**Numerical verification:**

Recall from our setup that we choose each coin with equal probability, so $P(C) = P(C^c) = 0.5$. The fair coin gives heads with probability 0.5, and the biased coin gives heads with probability 0.75. Since each flip has the same probability regardless of whether it's first or second, we have $P(H_1\mid C) = P(H_2\mid C) = 0.5$ and $P(H_1\mid C^c) = P(H_2\mid C^c) = 0.75$.

Now let's calculate the individual probabilities:

$$
\begin{align*}
P(H_1) &= P(H_1\mid C)P(C) + P(H_1\mid C^c)P(C^c) \\
&= (0.50)(0.50) + (0.75)(0.50) \\
&= 0.625
\end{align*}
$$

$$
P(H_2) = 0.625 \quad \text{(by the same calculation)}
$$

For the intersection, we combine two ideas:

1. **Law of Total Probability** (shown above) gives us the structure:

   $$
   \begin{align*}
   P(H_1\cap H_2) &= P(H_1\cap H_2\mid C)P(C) \\
   &\quad + P(H_1\cap H_2\mid C^c)P(C^c)
   \end{align*}
   $$

2. **Conditional independence** ([from Part 1](#factorization-formula)) lets us factorize within each context:

   For the fair coin:

   $$
   \begin{align*}
   P(H_1\cap H_2\mid C) &= P(H_1\mid C) \times P(H_2\mid C) \\
   &= 0.5 \times 0.5 \\
   &= 0.25
   \end{align*}
   $$

   For the biased coin:

   $$
   \begin{align*}
   P(H_1\cap H_2\mid C^c) &= P(H_1\mid C^c) \times P(H_2\mid C^c) \\
   &= 0.75 \times 0.75 \\
   &= 0.5625
   \end{align*}
   $$

3. **Putting it together**:

   $$
   \begin{align*}
   P(H_1\cap H_2) &= P(H_1\cap H_2\mid C)P(C) + P(H_1\cap H_2\mid C^c)P(C^c) \\
   &= (0.25)(0.50) + (0.5625)(0.50) \\
   &= 0.125 + 0.28125 \\
   &= 0.40625
   \end{align*}
   $$

Now let's check for independence:

$$
P(H_1\cap H_2) = 0.40625
$$

$$
\begin{align*}
P(H_1) \times P(H_2) &= 0.625 \times 0.625 \\
&= 0.390625
\end{align*}
$$

Since $0.40625 \neq 0.390625$, the joint probability does **not** equal the product. This means **the events are dependent** when the context is hidden.

**Update check (alternative verification):**

We can also verify dependence by checking whether observing $H_1$ updates our belief about $H_2$:

$$
\begin{align*}
P(H_2\mid H_1) &= \frac{P(H_1\cap H_2)}{P(H_1)} \\
&= \frac{0.40625}{0.625} \\
&= 0.65
\end{align*}
$$

But $P(H_2) = 0.625$. Since $P(H_2\mid H_1) = 0.65 \neq 0.625 = P(H_2)$, observing $H_1$ **does** update our belief about $H_2$, confirming they are dependent.

**The key insight:** The flips are independent within each context, but dependent overall. This is because observing $H_1$ changes our belief about which coin we have, which in turn affects our belief about $H_2$.

:::{admonition} Summary of all three scenarios
:class: tip dropdown

The calculations above showed what happens when the context is hidden (Scenario 1). Here's a complete summary of all three cases:

**Scenario 1: Context hidden (we do NOT know which coin)**
- $P(H_1) = 0.625$, $P(H_2) = 0.625$
- $P(H_1\cap H_2) = 0.40625 \neq P(H_1)P(H_2) = 0.390625$
- **Not independent** (observing $H_1$ updates belief about $H_2$)

**Scenario 2: We know we chose the fair coin ($C$)**

- $P(H_1\mid C) = 0.5$, $P(H_2\mid C) = 0.5$
- $P(H_1\cap H_2\mid C) = 0.25 = P(H_1\mid C) \times P(H_2\mid C)$
- **Independent** within this context

**Scenario 3: We know we chose the biased coin ($C^c$)**
- $P(H_1\mid C^c) = 0.75$, $P(H_2\mid C^c) = 0.75$
- $P(H_1\cap H_2\mid C^c) = 0.5625 = P(H_1\mid C^c) \times P(H_2\mid C^c)$
- **Independent** within this context

**Conclusion:** $H_1$ and $H_2$ are **conditionally independent** given which coin was chosen ($H_1 \perp H_2 \mid C$), but **not independent** when the coin is unknown.
:::

---

**Connecting back to the general principle:**

Our coin example perfectly illustrates the key insight from Section 5.1:
- We have $H_1 \perp H_2 \mid C$ (the flips are conditionally independent given the coin)
- But we do NOT have $H_1 \perp H_2$ (the flips are dependent overall when the coin is unknown)

This demonstrates that **conditional independence does not imply unconditional independence**. The dependence emerges when we mix contexts (average over the hidden variable $C$). This pattern appears everywhere in statistics and data analysis: relationships that disappear within subgroups but appear in the overall data, or vice versa.

---

### 5.3. Key takeaways and real-world applications

**The core insight in one sentence:**

Conditioning on $C$ "locks in the context"—given $C$, events $A$ and $B$ don't update each other. When $C$ is hidden, mixing contexts can create dependence (or mask independence).

**Why this matters in practice:**

Conditional independence is the idea behind **controlling for confounders** in real experiments and data analysis:

1. **Medical research:** An apparent relationship between a treatment and outcome might weaken, disappear, or even reverse once you control for age, sex, or baseline severity.

2. **Data analysis:** Many "false discoveries" come from ignoring hidden grouping variables. Mixing data from different batches, sites, or time periods can create spurious correlations that look like real effects.

3. **Machine learning:** Understanding when features are conditionally independent given others is crucial for building accurate models and avoiding confounding.

**The practical lesson:** Always ask "what context am I in?" When analyzing relationships between variables, consider whether there's a hidden factor $C$ that, once accounted for, changes the picture entirely. This is one of the most important concepts for moving from probability theory to real-world statistical reasoning.

---

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

+++

## Exercises

1.  **Two urns (Bayes):** You pick an urn at random:

    * $U_1$ with probability $0.6$ (contains 3 red, 2 blue)
    * $U_2$ with probability $0.4$ (contains 1 red, 4 blue)

    You draw one ball and it is **red**. What is $P(U_1\mid R)$?

    ```{admonition} Answer
    :class: dropdown

    We are given:

    * $P(U_1)=0.6$, $P(U_2)=0.4$
    * $P(R\mid U_1)=3/5=0.6$
    * $P(R\mid U_2)=1/5=0.2$

    First compute $P(R)$ by total probability:

    $$
    \begin{align*}
    P(R) &= P(R\mid U_1)P(U_1)+P(R\mid U_2)P(U_2) \\
    &= (0.6)(0.6)+(0.2)(0.4) \\
    &= 0.44.
    \end{align*}
    $$

    Now apply Bayes' theorem:

    $$
    \begin{align*}
    P(U_1\mid R) &= \frac{P(R\mid U_1)P(U_1)}{P(R)} \\
    &= \frac{0.6\cdot 0.6}{0.44} \\
    &= \frac{0.36}{0.44} \\
    &= \frac{9}{11}\approx 0.818.
    \end{align*}
    $$
    ```

2.  **Diagnostic test (posterior probability):** A disease has prevalence $P(D)=0.005$ (0.5%). A test has:

    * Sensitivity $P(	ext{Pos}\mid D)=0.98$
    * False positive rate $P(	ext{Pos}\mid D^c)=0.03$

    If someone tests positive, what is $P(D\mid 	ext{Pos})$?

    ```{admonition} Answer
    :class: dropdown

    First find $P(	ext{Pos})$:

    $$
    \begin{align*}
    P(\text{Pos})
      &= P(\text{Pos}\mid D)P(D) + P(\text{Pos}\mid D^c)P(D^c) \\
      &= 0.98\cdot 0.005 + 0.03\cdot (1-0.005) \\
      &= 0.0049 + 0.02985 \\
      &= 0.03475.
    \end{align*}
    $$

    Then Bayes' theorem:

    $$
    \begin{align*}
    P(D\mid \text{Pos})
    &= \frac{P(\text{Pos}\mid D)P(D)}{P(\text{Pos})} \\
    &= \frac{0.98\cdot 0.005}{0.03475} \\
    &\approx 0.141.
    \end{align*}
    $$

    So even with a positive result, the chance of actually having the disease is about **14.1%** (because the disease is rare).
    ```

3.  **Spam filter (Bayes):** Suppose 20% of emails are spam:

    * $P(S)=0.20$
    * The word “FREE” appears in 50% of spam emails: $P(F\mid S)=0.50$
    * The word “FREE” appears in 2% of non-spam emails: $P(F\mid S^c)=0.02$

    If an email contains “FREE”, what is $P(S\mid F)$?

    ```{admonition} Answer
    :class: dropdown

    First compute $P(F)$:

    $$
    \begin{align*}
    P(F) &= P(F\mid S)P(S)+P(F\mid S^c)P(S^c) \\
    &= 0.50\cdot 0.20 + 0.02\cdot 0.80 \\
    &= 0.10 + 0.016 \\
    &= 0.116.
    \end{align*}
    $$

    Then Bayes' theorem:

    $$
    \begin{align*}
    P(S\mid F) &= \frac{P(F\mid S)P(S)}{P(F)} \\
    &= \frac{0.50\cdot 0.20}{0.116} \\
    &= \frac{0.10}{0.116} \\
    &= \frac{25}{29}\approx 0.862.
    \end{align*}
    $$

    So $P(S\mid F)\approx 86.2\%$.
    ```

4.  **Are these events independent?** Roll a fair six-sided die.

    * $A$ = “the roll is even” = {2, 4, 6}
    * $B$ = “the roll is prime” = {2, 3, 5}

    Are $A$ and $B$ independent?

    ```{admonition} Answer
    :class: dropdown

    Compute:

    * $P(A)=3/6=1/2$
    * $P(B)=3/6=1/2$
    * $A\cap B$ = {2}, so $P(A\cap B)=1/6$

    If $A$ and $B$ were independent, we would have
    $P(A\cap B)=P(A)P(B)=(1/2)(1/2)=1/4$.

    But $1/6 \ne 1/4$, so the events are **not independent**.
    ```

5.  **Mutually exclusive vs independent:** Roll a fair six-sided die.

    * $A$ = “the roll is 1”
    * $B$ = “the roll is 2”

    Are $A$ and $B$ independent?

    ```{admonition} Answer
    :class: dropdown

    They are **mutually exclusive**: $A\cap B=\emptyset$, so $P(A\cap B)=0$.

    But $P(A)=1/6$ and $P(B)=1/6$, so $P(A)P(B)=1/36$.

    Since $P(A\cap B) \ne P(A)P(B)$, the events are **not independent**.
    ```

6.  **Conditional independence (coin mixture):** You choose a coin:

    * Fair with probability $P(C)=0.4$ (so $P(H\mid C)=0.5$)
    * Biased with probability $P(C^c)=0.6$ (so $P(H\mid C^c)=0.8$)

    Then you flip it twice. Let $H_1$ be “first flip is Heads” and $H_2$ be “second flip is Heads”.

    1. Compute $P(H_2)$ and $P(H_2\mid H_1)$ and decide whether $H_1$ and $H_2$ are independent overall.
    2. Show that $H_1 \perp H_2 \mid C$.

    ```{admonition} Answer
    :class: dropdown

    **1) Overall (context hidden).**

    By total probability:

    $$
    \begin{align*}
    P(H_2) &= P(H\mid C)P(C)+P(H\mid C^c)P(C^c) \\
    &= 0.5\cdot 0.4 + 0.8\cdot 0.6 \\
    &= 0.68.
    \end{align*}
    $$

    Also,

    $$
    \begin{align*}
    P(H_1\cap H_2) &= P(HH\mid C)P(C) \\
    &\quad +P(HH\mid C^c)P(C^c) \\
    &= (0.5^2)\cdot 0.4 + (0.8^2)\cdot 0.6 \\
    &= 0.25\cdot 0.4 + 0.64\cdot 0.6 \\
    &= 0.484.
    \end{align*}
    $$

    So

    $$
    \begin{align*}
    P(H_2\mid H_1) &= \frac{P(H_1\cap H_2)}{P(H_1)} \\
    &= \frac{0.484}{0.68} \\
    &\approx 0.712.
    \end{align*}
    $$

    Since $P(H_2\mid H_1)\approx 0.712 \ne P(H_2)=0.68$, the flips are **not independent overall**.

    **2) Within a fixed context.**

    If you condition on which coin you chose:

    * Given $C$ (fair coin), the flips are independent, so
      $$
      P(H_2\mid H_1, C)=P(H_2\mid C)=0.5.
      $$
    * Given $C^c$ (biased coin), similarly,
      $$
      P(H_2\mid H_1, C^c)=P(H_2\mid C^c)=0.8.
      $$

    That is exactly the “no extra update” condition, so

    $$
    H_1 \perp H_2 \mid C.
    $$
    ```
