# Learn Probability - Style and Structure Guide

This document describes the writing style, structure patterns, and code organization for the Learn Probability course.

## Chapter Structure

### Opening Pattern
1. **Welcoming introduction** - Start with context linking to previous chapters
2. **Why this matters** - Explain the importance and real-world relevance
3. **Learning objectives** (optional) - Bullet list of what readers will learn
4. **Progressive structure** - Build from simple to complex

### Section Organization
1. **Numbered sections** - Use clear hierarchical numbering (e.g., "## 1. Topic Name")
2. **Subsections** - Use "### 1.1 Subtopic" format
3. **Visual separators** - Use `+++` to separate major sections

### Chapter Closing
1. **Summary section** - Recap key takeaways in bullet format
2. **Exercises** - 4-6 exercises with answers in dropdown admonitions
3. **Forward link** - Brief mention of what's next

## Content Development Pattern

### For Each Major Concept

Follow this sequence:

1. **Intuitive introduction**
   - Start with a concrete, relatable example
   - Use plain language before formal terminology

2. **Visual representation**
   - Create diagrams, area models, or tree diagrams
   - Use hidden code blocks for visualization code
   - Include figure captions with clear descriptions

3. **Formal definition**
   - Present mathematical definition in a clear box or section
   - Use LaTeX for all mathematical notation
   - Include all necessary conditions

4. **Worked examples**
   - Provide step-by-step solutions
   - Show multiple examples of increasing complexity
   - Use admonition boxes for longer examples

5. **Python implementation**
   - Show code in appropriate format (see Code Organization below)

## Code Organization

### Visualization Code (NOT part of learning)

**Purpose**: Creates figures, plots, diagrams that support concepts

**Format**:
```python
:::{code-cell} ipython3
:tags: [remove-input, remove-output]

# Code to create visualization
# This code is HIDDEN from readers
# Saves to SVG or displays plot
:::
```

**When to use**:
- Creating Venn diagrams
- Generating area models
- Building tree diagrams
- Making illustrative plots that aren't teaching code skills

### Learning Code (part of the pedagogy)

**Purpose**: Demonstrates calculations, implementations readers should understand

**Format** - Use dropdown:
```markdown
:::{dropdown} Python Implementation
\`\`\`{code-cell} ipython3
# Code that readers should see and understand
# Shows how to calculate or implement concept
\`\`\`
:::
```

**When to use**:
- Demonstrating how to use scipy.stats functions
- Showing probability calculations
- Implementing formulas
- Calculating examples from the text

### Exploration Code (hands-on practice)

**Purpose**: Simulation, experimentation, comparison to theory

**Format** - Visible, split into logical cells:
```markdown
\`\`\`{code-cell} ipython3
# Setup and parameters
num_simulations = 10000
\`\`\`

\`\`\`{code-cell} ipython3
# Run simulation
results = simulate(...)
\`\`\`

\`\`\`{code-cell} ipython3
# Analyze results
print(f"Mean: {np.mean(results)}")
\`\`\`
```

**When to use**:
- Simulations that verify theoretical results
- Hands-on sections
- Interactive explorations

### Code Cell Splitting Guidelines

**Split cells when**:
1. Moving from setup to execution to analysis
2. Each output should be separate (one print per cell generally)
3. Separating conceptually different operations
4. The cell output is large or takes time to compute

**Keep cells together when**:
1. Tightly coupled operations (import statements)
2. Defining a helper function
3. Setup that has no meaningful output

**Example - Good splitting**:
```markdown
\`\`\`{code-cell} ipython3
# Define parameters
n = 52
k = 5
\`\`\`

\`\`\`{code-cell} ipython3
# Calculate combinations
total_hands = comb(n, k, exact=True)
print(f"Total hands: {total_hands:,}")
\`\`\`

\`\`\`{code-cell} ipython3
# Visualize
plt.figure(figsize=(8, 4))
plt.bar(...)
plt.show()
\`\`\`
```

**Example - Bad splitting**:
```markdown
\`\`\`{code-cell} ipython3
import numpy as np
\`\`\`

\`\`\`{code-cell} ipython3
import matplotlib.pyplot as plt
\`\`\`

\`\`\`{code-cell} ipython3
from scipy.special import comb
\`\`\`
```

Better as:
```markdown
\`\`\`{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
\`\`\`
```

### Code Cell Combining Guidelines

**Combine cells when**:
1. All imports should be in one cell at the start
2. Defining multiple related helper functions
3. Setting up parameters that belong together
4. Multiple calculations that produce one logical output

## Admonition Usage

### Types and When to Use

**Examples** - `:::{admonition} Example` with `:class: tip dropdown`
- Use for detailed worked examples
- Use for supplementary examples that might distract from main flow

**Notes** - `:::{admonition} Note` with `:class: note`
- Terminology clarifications
- Important points that aren't warnings
- Connections to other concepts

**Tips** - `:::{admonition} Tip` with `:class: tip`
- Study strategies
- Memory aids
- Key insights

**Warnings** - `:::{admonition} Warning` with `:class: warning`
- Common mistakes
- Confusing distinctions
- Critical points not to miss

**Dropdown Examples** - `:class: dropdown`
- Long examples that interrupt flow
- Supplementary content
- Derivations that some readers may skip

## Mathematical Notation

### Formatting Standards

**Inline math**: Use single `$...$` for inline: `$P(A \cap B)$`

**Display math**: Use `$$...$$` for centered equations:
```markdown
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$
```

**Multi-line derivations**: Use `align` blocks:
```markdown
$$
\begin{align*}
P(A \cap B) &= P(A|B) \times P(B) \\
&= 0.5 \times 0.3 \\
&= 0.15
\end{align*}
$$
```

**Sets**: Use `\{` and `\}`: `$\{1, 2, 3\}$`

**Conditional probability**: Always use `|` (not colon): `$P(A|B)$`

**Expected value**: Use square brackets: `$E[X]$`

## Visual Elements

### Figures

**SVG preferred** for:
- Venn diagrams
- Tree diagrams
- Custom visualizations
- Area models

**Format**:
```markdown
\`\`\`{figure} filename.svg
---
width: 80%
figclass: full-width (optional)
---
Clear caption describing what the figure shows.
\`\`\`
```

### Plots

**Use matplotlib** with:
- Consistent style: `plt.style.use('seaborn-v0_8-whitegrid')`
- Clear labels: xlabel, ylabel, title
- Appropriate figure size: `figsize=(8, 4)` or similar
- Grid for readability: `plt.grid(...)`

## Writing Style

### Voice and Tone
- **Welcoming and encouraging** - "Let's explore..."
- **Clear and direct** - Avoid unnecessary complexity
- **Build confidence** - "Notice that..." "We can see..."
- **Connect concepts** - Link to previous learning frequently

### Example Language Patterns

**Good**:
- "Let's think through this step-by-step"
- "Notice how this connects to..."
- "The key insight is..."
- "This makes intuitive sense because..."

**Avoid**:
- "Obviously..." (might not be obvious!)
- "Simply..." (might not be simple!)
- "Trivially..." (condescending)
- "As everyone knows..." (assumption)

### Mathematical Exposition

**Pattern for introducing formulas**:
1. Motivate why we need it
2. Show the formula
3. Explain each component
4. Work through an example
5. Show Python implementation

**Example**:
```markdown
The number of permutations is given by:

$$ P(n, k) = \frac{n!}{(n-k)!} $$

where:
* $n$ is the total number of objects
* $k$ is the number of objects selected
* $n!$ (read "n factorial") is the product...

Let's calculate $P(8, 3)$...
```

## Mobile Formatting Considerations

### Math Display

**Blank lines before math blocks**: Essential for mobile rendering
```markdown
The probability is calculated as follows:

$$
P(A) = \frac{1}{6}
$$

This shows that...
```

**Avoid long equations**: Break into multiple lines using `align`

### Lists and Spacing

**Blank lines in numbered lists**: Add before math or code blocks
```markdown
1. First step

   $$
   P(A) = 0.5
   $$

2. Second step
```

## Chapter-Specific Patterns

### Counting Chapter (Ch 3)
- Heavy use of step-by-step multiplication principle
- "Building Intuition" then "General Formula" pattern
- Quick reference table at end
- Decision trees for choosing methods

### Conditional Probability (Ch 4)
- Strong emphasis on visual representation (Venn diagrams)
- Tree diagrams for sequential events
- "Tips for differentiating" sections

### Bayes & Independence (Ch 5)
- Area models for Bayes' theorem
- Careful distinction between related concepts
- Multiple visual representations of same concept
- "Why this section matters" boxes

### Random Variables (Ch 6)
- Progression: Definition → PMF → CDF → Expected Value → Variance
- Strong connection to simulation
- "Hands-on" sections for empirical verification

## Quality Checklist

Before finalizing a chapter:

- [ ] Introduction links to previous content
- [ ] Each major concept follows: intuition → visual → formal → example → code
- [ ] Visualization code is hidden with tags
- [ ] Learning code is in dropdowns
- [ ] Code cells are appropriately split/combined
- [ ] All math has blank lines before/after for mobile
- [ ] Figures have descriptive captions
- [ ] Exercises have dropdown solutions
- [ ] Summary section captures key points
- [ ] Forward link to next chapter
- [ ] Consistent notation throughout
- [ ] No orphaned concepts (everything is explained or linked)
