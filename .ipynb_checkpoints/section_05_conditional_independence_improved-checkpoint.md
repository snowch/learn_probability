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

### A visual you can read without the example

This figure is a self-contained story:

1. **Top-left:** assume we *know* we are in one context (e.g. a “fair coin” world).  
2. **Top-right:** assume we *know* we are in a different context (e.g. a “biased coin” world).  
3. **Bottom:** if we *don’t* know the context, we are mixing those worlds.

Inside each fixed context, we draw a **unit box** (total probability = 1):

* the **left strip** has width $P(H_1\mid\text{context})$
* the **top strip** has height $P(H_2\mid\text{context})$
* the **overlap** has area $P(H_1\cap H_2\mid\text{context})$

When $H_1$ and $H_2$ are conditionally independent in that context, the overlap area equals the product:
$$
P(H_1\cap H_2\mid\text{context}) = P(H_1\mid\text{context})\,P(H_2\mid\text{context}).
$$

The bottom panel then shows the key warning: once the context is hidden, mixing can create
$$
P(H_1\cap H_2)\neq P(H_1)\,P(H_2).
$$

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
Within each fixed context (fair vs biased), $H_1$ and $H_2$ factorize:
$P(H_1\cap H_2\mid\cdot)=P(H_1\mid\cdot)P(H_2\mid\cdot)$.
When $C$ is hidden, we mix contexts and can get $P(H_1\cap H_2)\neq P(H_1)P(H_2)$.
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
