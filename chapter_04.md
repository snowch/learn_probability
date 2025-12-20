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


```{code-cell} python3
:tags: [remove-input, remove-output]

# create and save visualisation total-probability-area.svg

from pathlib import Path
import random

def save_total_probability_svg(
    filename="total-probability-area.svg",
    n=6,
    p_B=None,
    p_A_given_B=None,
    magnify=8.0,
    font_scale=2.0,
    line_gap=2.0,
    jitter=True,
    jitter_seed=7,
):
    if p_B is None:
        p_B = [1/n] * n
    if p_A_given_B is None:
        p_A_given_B = [0.10, 0.35, 0.18, 0.06, 0.28, 0.12][:n]

    if len(p_B) != n or len(p_A_given_B) != n:
        raise ValueError("p_B and p_A_given_B must have length n.")

    # normalise widths
    s = sum(p_B)
    p_B = [x / s for x in p_B]

    def fmt(x):
        return f"{x:.3f}".rstrip("0").rstrip(".")

    pA = sum(pb * pab for pb, pab in zip(p_B, p_A_given_B))

    # ---------- sizing ----------
    W = 1600
    L = 60
    box_w, box_h = 1320, 330

    outline = "#111827"
    strip_fill = "#f8fafc"
    shade_fill = "#ef4444"
    shade_stroke = "#b91c1c"

    # fonts
    title_sz = int(22 * font_scale)
    note_sz  = int(15 * font_scale)
    B_sz     = int(14 * font_scale)
    num_sz   = int(14 * font_scale)   # "smallest" numbers
    inside_sz = int(15 * font_scale)
    center1  = int(20 * font_scale)
    center2  = int(15 * font_scale)

    # helper: line spacing in px (baseline-to-baseline)
    def dy(px): 
        return int(px * line_gap)

    # ---------- top-down layout (prevents overlaps) ----------
    title_y = 60
    note1_y = title_y + dy(title_sz)
    note2_y = note1_y + dy(note_sz)

    header_bottom = note2_y + int(note_sz * 0.9)

    # Put B-label band BELOW header
    y_B  = header_bottom + dy(B_sz)        # B_i
    y_PB = y_B + dy(num_sz)                # P(B_i)

    # Put box BELOW B-label band
    y0 = y_PB + int(num_sz * 1.8)
    x0 = L
    y1 = y0 + box_h

    # Bottom labels below box
    y_bottom1 = y1 + int(num_sz * 1.8)
    y_bottom2 = y_bottom1 + dy(num_sz)

    # Total height
    H = y_bottom2 + int(num_sz * 4.0)

    # ---------- SVG ----------
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')

    # Header as separate text lines (cleaner than giant tspans at huge spacing)
    parts.append(f'''
<text x="{L}" y="{title_y}"
      font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
      font-size="{title_sz}" font-weight="700" fill="{outline}">
  Law of Total Probability — area model
</text>
<text x="{L}" y="{note1_y}"
      font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
      font-size="{note_sz}" fill="{outline}">
  Widths are to scale (P(Bᵢ)). Shaded height is ×{magnify:g} for visibility (not to scale).
</text>
<text x="{L}" y="{note2_y}"
      font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
      font-size="{note_sz}" fill="{outline}">
  P(A)=Σᵢ P(A|Bᵢ)P(Bᵢ) (example numbers here give P(A)={fmt(pA)})
</text>
'''.strip())

    # Box
    parts.append(f'<rect x="{x0}" y="{y0}" width="{box_w}" height="{box_h}" fill="none" stroke="{outline}" stroke-width="2"/>')

    # Inside label
    parts.append(f'''
<text x="{x0+14}" y="{y0+int(inside_sz*1.2)}" text-anchor="start"
      font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
      font-size="{inside_sz}" font-weight="800" fill="{outline}">S</text>
<text x="{x0+34}" y="{y0+int(inside_sz*1.2)}" text-anchor="start"
      font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
      font-size="{inside_sz-1}" font-weight="600" fill="{outline}">partitioned into Bᵢ</text>
'''.strip())

    rng = random.Random(jitter_seed)

    x = x0
    for i, (pb, pab) in enumerate(zip(p_B, p_A_given_B), start=1):
        w = box_w * pb
        h_mag = min(box_h, box_h * pab * magnify)
        contrib = pb * pab
        cx = x + w/2

        if i == 1:
            lab = "B₁"
        elif i == n:
            lab = "Bₙ"
        else:
            lab = f"B{i}"

        # strip
        parts.append(
            f'<rect x="{x:.2f}" y="{y0}" width="{w:.2f}" height="{box_h}" '
            f'fill="{strip_fill}" stroke="{outline}" stroke-width="1"/>'
        )

        # shaded piece y-position (jittered)
        if jitter:
            max_top = y0
            max_bottom = y1 - h_mag
            y_shade = rng.uniform(max_top, max_bottom) if max_bottom > max_top else (y1 - h_mag)
        else:
            y_shade = y1 - h_mag

        parts.append(
            f'<rect x="{x:.2f}" y="{y_shade:.2f}" width="{w:.2f}" height="{h_mag:.2f}" '
            f'fill="{shade_fill}" fill-opacity="0.18" stroke="{shade_stroke}" stroke-width="1"/>'
        )

        # top labels (now safely below header)
        parts.append(
            f'<text x="{cx:.2f}" y="{y_B}" text-anchor="middle" '
            f'font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial" '
            f'font-size="{B_sz}" font-weight="800" fill="{outline}">{lab}</text>'
        )
        parts.append(
            f'<text x="{cx:.2f}" y="{y_PB}" text-anchor="middle" '
            f'font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial" '
            f'font-size="{num_sz}" fill="{outline}">P({lab})={fmt(pb)}</text>'
        )

        # bottom two lines with your requested bigger spacing
        parts.append(
            f'<text x="{cx:.2f}" y="{y_bottom1}" text-anchor="middle" '
            f'font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial" '
            f'font-size="{num_sz}" fill="{shade_stroke}">P(A|{lab})={fmt(pab)}</text>'
        )
        parts.append(
            f'<text x="{cx:.2f}" y="{y_bottom2}" text-anchor="middle" '
            f'font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial" '
            f'font-size="{num_sz}" fill="{outline}">P(A∩{lab})={fmt(contrib)}</text>'
        )

        x += w

    # Center red explanation
    center_x = x0 + box_w/2
    center_y = y0 + box_h/2
    parts.append(
        f'<text x="{center_x:.2f}" y="{center_y-10:.2f}" text-anchor="middle" '
        f'font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial" '
        f'font-size="{center1}" font-weight="900" fill="{shade_stroke}">P(A) = total shaded area</text>'
    )
    parts.append(
        f'<text x="{center_x:.2f}" y="{center_y + dy(center2):.2f}" text-anchor="middle" '
        f'font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial" '
        f'font-size="{center2}" font-weight="800" fill="{shade_stroke}">= Σᵢ P(A|Bᵢ)P(Bᵢ)</text>'
    )

    # Note for jitter placement (inside box, bottom-left)
    if jitter:
        parts.append(
            f'<text x="{x0+12}" y="{y1-12}" text-anchor="start" '
            f'font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial" '
            f'font-size="{num_sz}" fill="{outline}">'
            f'Note: vertical placement of shaded pieces is arbitrary — only the areas matter.</text>'
        )

    parts.append("</svg>")
    Path(filename).write_text("\n".join(parts), encoding="utf-8")
    return filename

# Your call (works now without overlaps even at huge spacing):
save_total_probability_svg(
    "total-probability-area.svg",
    n=6,
    magnify=8.0,
    jitter=True,
    line_gap=2.0,
    font_scale=2.0
)
```

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

- Let $A$ be the event *“the part is defective”*.
- Let $B_1$ be the event *“the part came from Line 1”*.
- Let $B_2$ be the event *“the part came from Line 2”*.

Suppose:

* $P(B_1)=0.6$, $P(B_2)=0.4$
* $P(A\mid B_1)=0.02$ (2% defective on Line 1)
* $P(A\mid B_2)=0.05$ (5% defective on Line 2)

Then:

$$
\begin{align*}
P(A) &= P(A\mid B_1)P(B_1)+P(A\mid B_2)P(B_2) \\
     &= 0.02\cdot 0.6 + 0.05\cdot 0.4 \\
     &= 0.012 + 0.020 \\
     &= 0.032.
\end{align*}
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

3.  **Two Cards — Same Rank:** Draw two cards from a standard 52-card deck **without replacement**. What is the probability that the two cards have the **same rank** (e.g., two 7s, two Kings)?

    ```{admonition} Answer
    :class: dropdown

    Let \(A\) be the event “the two cards have the same rank.”

    **Conditional probability approach (matches the hint):**
    - The first card can be anything; after drawing it, its rank is fixed.
    - In a 52-card deck there are 4 cards of each rank.
    - After drawing the first card, there are **3** remaining cards of that same rank.
    - There are **51** cards left in total.

    So,
    
    $$
    P(A)=P(\text{2nd card has same rank as 1st}\mid \text{1st card drawn}).
    $$
    
    Recall the definition of conditional probability:
    
    $$
    P(A\mid B)=\frac{P(A\cap B)}{P(B)} \quad (P(B)>0).
    $$
    
    To make this concrete, let $B$ be the event “the first card is an Ace” and let $A$ be the event “the second card is an Ace”.
    Then $A\cap B$ is the event “the first two cards are both Aces”.
    
    We have:
    - $P(B)=\frac{4}{52}=\frac{1}{13}$.
    - $P(A\cap B)=\frac{4}{52}\cdot\frac{3}{51}$ (4 ways to draw an Ace first, then 3 Aces remain out of 51 cards).
    
    So,
    
    $$
    P(A\mid B)=\frac{P(A\cap B)}{P(B)}
    =\frac{\frac{4}{52}\cdot\frac{3}{51}}{\frac{4}{52}}
    =\frac{3}{51}
    =\frac{1}{17}\approx 0.0588.
    $$
    
    By symmetry, this is the probability that the second card matches the rank of the first card.




    **(Optional check using counting)**
    - Total 2-card hands: (binomial coefficient) $\binom{52}{2}$.
    - Favourable: choose the rank (13 ways), then choose 2 suits out of 4: $\binom{4}{2}$.

    $$
    P(A)=\frac{13\binom{4}{2}}{\binom{52}{2}}
        =\frac{13\cdot 6}{1326}
        =\frac{78}{1326}
        =\frac{1}{17}.
    $$

    ```

4.  **Choosing a Coin — Total Probability:** A bag contains **two fair coins** and **one biased coin**.  
    - If a coin is fair, \(P(H)=0.5\).  
    - If a coin is biased, \(P(H)=0.8\).  
    You randomly pick **one** coin from the bag and flip it **twice**. What is the probability of getting **exactly one Head**?

    ```{admonition} Answer
    :class: dropdown

    Let:
    - \(A\) be the event “exactly one Head in two flips.”
    - \(B_1\) be “a fair coin was chosen.”
    - \(B_2\) be “the biased coin was chosen.”

    These form a partition: you choose either a fair coin or the biased coin.

    **Step 1 — Probabilities of the scenarios**
    There are 3 coins total, 2 are fair:
    
    $$
    P(B_1)=\frac{2}{3}, \qquad P(B_2)=\frac{1}{3}.
    $$

    **Step 2 — Compute the conditional probabilities**
    - Given a fair coin, exactly one Head can happen as HT or TH:
    
      $$
      P(A\mid B_1)=P(HT)+P(TH)=(0.5)(0.5)+(0.5)(0.5)=0.5.
      $$
      
    - Given the biased coin, $P(H)=0.8$ and $P(T)=0.2$:
      
      $$
      P(A\mid B_2)=P(HT)+P(TH)=(0.8)(0.2)+(0.2)(0.8)=0.32.
      $$

    **Step 3 — Apply the Law of Total Probability**
    
    $$
    P(A)=P(A\mid B_1)P(B_1)+P(A\mid B_2)P(B_2)
        =(0.5)\left(\frac{2}{3}\right)+(0.32)\left(\frac{1}{3}\right).
    $$

    Compute:
    $$
    (0.5)\left(\frac{2}{3}\right)=\frac{1}{3}, \qquad
    (0.32)\left(\frac{1}{3}\right)=\frac{0.32}{3}=\frac{8}{75}.
    $$

    So,
    
    $$
    P(A)=\frac{1}{3}+\frac{8}{75}=\frac{25}{75}+\frac{8}{75}=\frac{33}{75}=\frac{11}{25}=0.44.
    $$

    **Answer:** $P(\text{exactly one Head})=\frac{11}{25}=0.44$.
    ```

+++
