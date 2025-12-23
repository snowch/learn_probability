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

The definition above is the *formal* one. In practice, people often use an equivalent “no extra information” form because it matches how we *think* about conditioning.

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

The example below shows this happening explicitly.
:::

---

### A visual that matches the example

In the example that follows:

* $H_1$ = “first flip is Heads”
* $H_2$ = “second flip is Heads”
* $C$ = “the chosen coin is fair”

The figure uses an **area model** *inside each fixed context*:

* Within a box (e.g. “Given $C$ = Fair”), the box represents probability **1**.
* The **width** is $P(H_1\mid\text{context})$ and the **height** is $P(H_2\mid\text{context})$.
* If $H_1$ and $H_2$ are independent **within that context**, the overlap area is the product:
  $$
  P(H_1\cap H_2\mid \text{context})=P(H_1\mid \text{context})\,P(H_2\mid \text{context}).
  $$

When we *do not* know which coin was chosen, we are mixing two contexts. That mixing can create dependence overall.

```{code-cell} python3
:tags: [remove-input, remove-output]

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

def draw_context(ax, pH1, pH2, title):
    '''
    Unit 'given context' box (area = 1):
      - H1 strip: vertical band of width pH1
      - H2 strip: horizontal band of height pH2
      - overlap: H1 ∩ H2 (hatched)
    '''
    pH1 = max(0.0, min(1.0, float(pH1)))
    pH2 = max(0.0, min(1.0, float(pH2)))

    ax.set_xlim(-0.15, 1.25)
    ax.set_ylim(-0.25, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, linewidth=2))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # H2 strip (top)
    ax.add_patch(Rectangle((0, 1 - pH2), 1, pH2, alpha=0.12, linewidth=0))
    ax.text(0.02, 1 - pH2/2, r"$H_2$", va="center", ha="left", fontsize=12)

    # H1 strip (left)
    ax.add_patch(Rectangle((0, 0), pH1, 1, alpha=0.18, linewidth=0))
    ax.text(pH1/2, 0.02, r"$H_1$", va="bottom", ha="center", fontsize=12)

    # Overlap H1 ∩ H2
    ax.add_patch(Rectangle((0, 1 - pH2), pH1, pH2, fill=False, hatch="///", linewidth=1.5))
    ax.text(pH1 + 0.03, 1 - pH2 + 0.02, r"$H_1\cap H_2$", va="bottom", ha="left", fontsize=12)

    # Dimension callouts
    ax.annotate("", xy=(0, -0.08), xytext=(pH1, -0.08), arrowprops=dict(arrowstyle="<->", linewidth=1.3))
    ax.text(pH1/2, -0.14, rf"width = $P(H_1\mid\cdot)$ = {pH1:.2f}", ha="center", va="top", fontsize=11)

    ax.annotate("", xy=(1.08, 1 - pH2), xytext=(1.08, 1), arrowprops=dict(arrowstyle="<->", linewidth=1.3))
    ax.text(1.12, 1 - pH2/2, rf"height = $P(H_2\mid\cdot)$ = {pH2:.2f}", ha="left", va="center", fontsize=11)

    ax.text(0.5, -0.22,
            r"$P(H_1\cap H_2\mid\cdot)=P(H_1\mid\cdot)\,P(H_2\mid\cdot)$",
            ha="center", va="top", fontsize=12)

def draw_mixture(ax, w_fair, w_biased, title=r"If you do NOT know $C$ (mixture)"):
    '''
    Stacked mixture panel: shows you're mixing contexts with weights.
    '''
    w_fair = max(0.0, float(w_fair))
    w_biased = max(0.0, float(w_biased))
    tot = (w_fair + w_biased) if (w_fair + w_biased) > 0 else 1.0
    t_fair = w_fair / tot
    t_biased = w_biased / tot

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.20, 1.05)
    ax.axis("off")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, linewidth=2))

    ax.add_patch(Rectangle((0, 1 - t_fair), 1, t_fair, alpha=0.10, linewidth=0))
    ax.text(0.02, 1 - t_fair/2, rf"Fair context ($C$)   weight $P(C)$ = {w_fair:.2f}",
            va="center", ha="left", fontsize=12)

    ax.add_patch(Rectangle((0, 0), 1, t_biased, alpha=0.18, linewidth=0))
    ax.text(0.02, t_biased/2, rf"Biased context ($C^c$) weight $P(C^c)$ = {w_biased:.2f}",
            va="center", ha="left", fontsize=12)

    ax.text(0.5, -0.10,
            "Mixing contexts can create overall dependence.",
            ha="center", va="top", fontsize=12)

# --- Numbers from the coin example ---
p_fair = 0.50
p_biased = 0.75
w_fair = 0.50
w_biased = 0.50

fig = plt.figure(figsize=(12.5, 8.0))
gs = GridSpec(2, 2, height_ratios=[1.0, 0.62], hspace=0.35, wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

draw_context(ax1, p_fair,   p_fair,   title=r"Given $C$ = Fair coin ($P(H)=0.5$)")
draw_context(ax2, p_biased, p_biased, title=r"Given $C^c$ = Biased coin ($P(H)=0.75$)")
draw_mixture(ax3, w_fair, w_biased)

fig.suptitle(
    r"Conditional independence in the coin example: independent within each context; mixing creates dependence",
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
**Area model tied to the coin example.**  
Top row: within each fixed context (Fair vs Biased), $H_1$ and $H_2$ factorize, so the overlap area equals $P(H_1\mid\cdot)\,P(H_2\mid\cdot)$.  
Bottom: if we *don’t* observe which coin was chosen, we mix contexts, and dependence can appear overall.
```

---

:::{admonition} Example
:class: tip dropdown

### Example: two flips of a randomly chosen coin

We have two coins:

* Fair coin (F): $P(H)=0.5$
* Biased coin (B): $P(H)=0.75$

Pick a coin uniformly at random, then flip it twice.

Let:

* $H_1$ = “first flip is Heads”
* $H_2$ = “second flip is Heads”
* $C$ = “we chose the fair coin” (so $C^c$ = “we chose the biased coin”)

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
So $H_1$ and $H_2$ are **not** independent.

A “no extra information” check makes the reason concrete:
$$
P(H_2\mid H_1)=\frac{0.40625}{0.625}=0.65
\quad\text{but}\quad
P(H_2)=0.625.
$$
Seeing $H_1$ makes $H_2$ more likely because it increases our belief we picked the biased coin.

#### Scenario 2: we know we chose the fair coin ($C$)

Given $C$ (fair coin), flips are independent:
$$
P(H_1\mid C)=0.5,\quad P(H_2\mid C)=0.5,\quad P(H_1\cap H_2\mid C)=0.25,
$$
and indeed
$$
P(H_1\cap H_2\mid C)=P(H_1\mid C)\,P(H_2\mid C).
$$

#### Scenario 3: we know we chose the biased coin ($C^c$)

Similarly,
$$
P(H_1\mid C^c)=0.75,\quad P(H_2\mid C^c)=0.75,\quad P(H_1\cap H_2\mid C^c)=0.5625,
$$
and again
$$
P(H_1\cap H_2\mid C^c)=P(H_1\mid C^c)\,P(H_2\mid C^c).
$$

So the flips are **conditionally independent** given which coin was chosen:
$$
H_1 \perp H_2 \mid C.
$$
:::

---

### Intuition in one sentence

**Conditioning on $C$ “locks in the context”; within that context, $H_1$ and $H_2$ don’t update each other—hiding $C$ mixes contexts and can create dependence.**

+++
