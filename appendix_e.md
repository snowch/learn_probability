# Appendix E: Summary of Formulas

This appendix provides a summary of the key formulas introduced in Chapters 1–8.

## **Chapter 2: The Language of Probability: Sets, Sample Spaces, and Events**

### **Axioms of Probability**

Let $S$ be a sample space, and $P(A)$ denote the probability of an event $A$.

1. **Non-negativity**: For any event $A$, the probability of $A$ is greater than or equal to zero. $P(A)\ge 0$
2. **Normalization**: The probability of the entire sample space $S$ is equal to $1$. $P(S)=1$
3. **Additivity for Disjoint Events**: If $A_1,A_2,A_3,\dots$ is a sequence of *mutually exclusive* (disjoint) events (i.e., $A_i\cap A_j=\emptyset$ for all $i\ne j$), then the probability of their union is the sum of their individual probabilities.
   $$P(A_1\cup A_2\cup A_3\cup \cdots)=P(A_1)+P(A_2)+P(A_3)+\cdots$$

   * For a finite number of disjoint events, say $A$ and $B$: If $A\cap B=\emptyset$, then $P(A\cup B)=P(A)+P(B)$

* **Probability of Impossible Event**: The probability of an impossible event (the empty set, $\emptyset$) is $0$. $P(\emptyset)=0$

### **Basic Probability Rules**

1. **Probability Range**: For any event $A$: $0\le P(A)\le 1$
2. **Complement Rule**: The probability that event $A$ does *not* occur is $1$ minus the probability that it *does* occur. $P(A^c)=1-P(A)$
3. **Addition Rule (General)**: For any two events $A$ and $B$ (not necessarily disjoint), the probability that $A$ *or* $B$ (or both) occurs is:
   $$P(A\cup B)=P(A)+P(B)-P(A\cap B)$$

### **Empirical Probability**

The empirical probability of an event $A$ is estimated from simulations:
$$
P_{\text{empirical}}(A)=\frac{\text{Number of times event $A$ occurred}}{\text{Total number of trials}}.
$$

## **Chapter 3: Counting Techniques: Permutations and Combinations**

### **The Multiplication Principle**

If a procedure can be broken down into a sequence of $k$ steps, with $n_1$ ways for the first step, $n_2$ for the second, $\dots$, $n_k$ for the $k$-th step, then the total number of ways to perform the entire procedure is:
$$
\text{Total ways}=n_1\times n_2\times \cdots \times n_k.
$$

### **Permutations (Order Matters)**

1. **Permutations without Repetition**: The number of permutations of $n$ distinct objects taken $k$ at a time:
   $$P(n,k)=\frac{n!}{(n-k)!}$$

   * Special Case: Arranging all $n$ distinct objects: $P(n,n)=n!$
2. **Permutations with Repetition (Multinomial Coefficients)**: The number of distinct permutations of $n$ objects where there are $n_1$ identical objects of type 1, $n_2$ of type 2, $\dots$, $n_k$ of type $k$ (such that $n_1+n_2+\cdots+n_k=n$):
   $$\frac{n!}{n_1!,n_2!\cdots n_k!}$$

### **Combinations (Order Doesn't Matter)**

1. **Combinations without Repetition**: The number of combinations of $n$ distinct objects taken $k$ at a time (also “$n$ choose $k$”):
   $$C(n,k)=\binom{n}{k}=\frac{n!}{k!(n-k)!}$$

   * Relationship to permutations:
     $$C(n,k)=\frac{P(n,k)}{k!}$$
2. **Combinations with Repetition**: The number of combinations with repetition of $n$ types of objects taken $k$ at a time:
   $$\binom{n+k-1}{k}=\frac{(n+k-1)!}{k!(n-1)!}$$

### **Probability with Equally Likely Outcomes**

The probability of an event $E$ when all outcomes in the sample space $S$ are equally likely:
$$
P(E)=\frac{\text{Number of outcomes favorable to }E}{\text{Total number of possible outcomes in }S}
=\frac{|E|}{|S|}.
$$

## **Chapter 4: Conditional Probability**

### **Definition of Conditional Probability**

For any two events $A$ and $B$ from a sample space $S$, where $P(B)>0$, the conditional probability of $A$ given $B$ is defined as:
$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}.
$$

### **The Multiplication Rule for Conditional Probability**

Rearranging the definition of conditional probability gives:
$$
P(A\cap B)=P(A\mid B)P(B).
$$
Similarly, if $P(A)>0$:
$$
P(A\cap B)=P(B\mid A)P(A).
$$
For three events $A,B,C$:
$$
P(A\cap B\cap C)=P(C\mid A\cap B),P(B\mid A),P(A).
$$

### **The Law of Total Probability**

Let $B_1,B_2,\dots,B_n$ be a partition of the sample space $S$. Then, for any event $A$ in $S$:
$$
P(A)=\sum_{i=1}^{n} P(A\mid B_i),P(B_i).
$$
Expanded form:
$$
P(A)=P(A\mid B_1)P(B_1)+P(A\mid B_2)P(B_2)+\cdots+P(A\mid B_n)P(B_n).
$$

## **Chapter 5: Bayes' Theorem and Independence**

### **Bayes' Theorem**

Provides a way to “reverse” conditional probabilities. If $P(B)>0$:
$$
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}.
$$
Where $P(B)$ can often be calculated using the Law of Total Probability (e.g., with a partition ${A,A^c}$):
$$
P(B)=P(B\mid A)P(A)+P(B\mid A^c)P(A^c).
$$

### **Independence of Events**

1. **Formal Definition**: Events $A$ and $B$ are independent if and only if:
   $$P(A\cap B)=P(A)P(B)$$
2. **Alternative Definition (using conditional probability)**:

   * If $P(B)>0$, $A$ and $B$ are independent if and only if: $P(A\mid B)=P(A)$
   * Similarly, if $P(A)>0$, independence means: $P(B\mid A)=P(B)$

### **Conditional Independence**

Events $A$ and $B$ are conditionally independent given event $C$ (where $P(C)>0$) if:
$$
P(A\cap B\mid C)=P(A\mid C)P(B\mid C).
$$
Alternative Definition: If $P(B\mid C)>0$, conditional independence means:
$$
P(A\mid B\cap C)=P(A\mid C).
$$

## **Chapter 6: Discrete Random Variables**

### **Probability Mass Function (PMF)**

For a discrete random variable X, the PMF pX​(x) is:  
pX​(x)=P(X=x)  
Properties of a PMF:

1. pX​(x)≥0 for all possible values x.  
2. ∑x​pX​(x)=1 (sum over all possible values x).

### **Cumulative Distribution Function (CDF)**

For a random variable X, the CDF FX​(x) is:  
FX​(x)=P(X≤x)  
For a discrete random variable X:  
FX​(x)=∑k≤x​pX​(k)  
Properties of a CDF:

1. 0≤FX​(x)≤1 for all x.  
2. If a\<b, then FX​(a)≤FX​(b) (non-decreasing).  
3. limx→−∞​FX​(x)=0  
4. limx→+∞​FX​(x)=1  
5. P(X\>x)=1−FX​(x)  
6. P(a\<X≤b)=FX​(b)−FX​(a) for a\<b.  
7. P(X=x)=FX​(x)−limy→x−​FX​(y) (for discrete RV, this is the jump at x).

### **Expected Value (Mean)**

For a discrete random variable X:  
E\[X\]=μX​=∑x​x⋅pX​(x)

### **Variance**

For a random variable X with mean μX​:  
Var(X)=σX2​=E\[(X−μX​)2\]  
For a discrete random variable X:  
Var(X)=∑x​(x−μX​)2⋅pX​(x)  
Computational formula for variance:  
Var(X)=E\[X2\]−(E\[X\])2  
Where E\[X2\] for a discrete random variable is:  
E\[X2\]=∑x​x2⋅pX​(x)

### **Standard Deviation**

The positive square root of the variance:  
SD(X)=σX​=Var(X)​

### **Functions of a Random Variable**

If Y=g(X):

1. **PMF of Y (for discrete X)**: pY​(y)=P(Y=y)=P(g(X)=y)=∑x:g(x)=y​pX​(x)  
2. **Expected Value of Y=g(X) (LOTUS - Law of the Unconscious Statistician)**: For a discrete random variable X: E\[Y\]=E\[g(X)\]=∑x​g(x)⋅pX​(x)

## **Chapter 7: Common Discrete Distributions**

### **Bernoulli Distribution**

Models a single trial with two outcomes (success=1, failure=0).  
Parameter: p (probability of success).

* **PMF**: P(X=k)=pk(1−p)1−kfor k∈{0,1}  
  * Alternatively: P(X=k)=⎩⎨⎧​p1−p0​if k=1if k=0otherwise​  
* **Mean**: E\[X\]=p  
* **Variance**: Var(X)=p(1−p)

### **Binomial Distribution**

Models the number of successes in n independent Bernoulli trials.  
Parameters: n (number of trials), p (probability of success on each trial).

* **PMF**: P(X=k)=(kn​)pk(1−p)n−kfor k=0,1,…,n  
* **Mean**: E\[X\]=np  
* **Variance**: Var(X)=np(1−p)

### **Geometric Distribution**

Models the number of trials (k) needed to get the first success.  
Parameter: p (probability of success on each trial).

* **PMF** (for X= trial number of first success): P(X=k)=(1−p)k−1pfor k=1,2,3,…  
* **Mean** (trial number of first success): E\[X\]=p1​  
* **Variance** (trial number of first success): Var(X)=p21−p​

### **Negative Binomial Distribution**

Models the number of trials (k) needed to achieve r successes.  
Parameters: r (target number of successes), p (probability of success on each trial).

* **PMF** (for X= trial number of r-th success): P(X=k)=(r−1k−1​)pr(1−p)k−rfor k=r,r+1,r+2,…  
* **Mean** (trial number of r-th success): E\[X\]=pr​  
* **Variance** (trial number of r-th success): Var(X)=p2r(1−p)​

### **Poisson Distribution**

Models the number of events occurring in a fixed interval of time or space.  
Parameter: λ (average number of events in the interval).

* **PMF**: P(X=k)=k\!e−λλk​for k=0,1,2,…  
* **Mean**: E\[X\]=λ  
* **Variance**: Var(X)=λ

### **Hypergeometric Distribution**

Models the number of successes in a sample of size n drawn without replacement from a finite population of size N containing1 K successes.  
Parameters: N (population size), K (total successes in population), n (sample size).

* **PMF**: P(X=k)=(nN​)(kK​)(n−kN−K​)​ for k such that max(0,n−(N−K))≤k≤min(n,K).  
* **Mean**: E\[X\]=nNK​  
* **Variance**: Var(X)=nNK​(1−NK​)(N−1N−n​)  
  * **Finite Population Correction Factor**: N−1N−n​

## **Chapter 8: Continuous Random Variables**

### **Probability Density Function (PDF)**

For a continuous random variable X, the PDF fX​(x) describes the relative likelihood of X.  
Properties of a PDF:

1. fX​(x)≥0 for all x.  
2. ∫−∞∞​fX​(x)dx=1 (total area under curve is 1).  
3. Probability as Area: P(a≤X≤b)=∫ab​fX​(x)dx.  
4. For any specific value c: P(X=c)=∫cc​fX​(x)dx=0.

### **Cumulative Distribution Function (CDF)**

For a continuous random variable X, the CDF FX​(x) is:  
FX​(x)=P(X≤x)=∫−∞x​fX​(t)dt  
Properties of a CDF:

1. FX​(x) is non-decreasing.  
2. limx→−∞​FX​(x)=0.  
3. limx→∞​FX​(x)=1.  
4. P(a\<X≤b)=FX​(b)−FX​(a).  
5. fX​(x)=dxd​FX​(x) (where the derivative exists).

### **Expected Value (Mean)**

For a continuous random variable X:  
E\[X\]=μ=∫−∞∞​xfX​(x)dx

### **Variance**

For a continuous random variable X with mean μ:  
Var(X)=σ2=E\[(X−μ)2\]=∫−∞∞​(x−μ)2fX​(x)dx  
Computational formula:  
Var(X)=E\[X2\]−(E\[X\])2  
Where E\[X2\]=∫−∞∞​x2fX​(x)dx.

### **Standard Deviation**

The positive square root of the variance:  
σ=Var(X)​

### **Percentiles and Quantiles**

The p-th percentile xp​ is the value such that FX​(xp​)=P(X≤xp​)=p.  
The quantile function Q(p) is the inverse of the CDF: Q(p)=FX−1​(p)=xp​.

### **Functions of a Continuous Random Variable**

If Y=g(X):

1. **CDF of Y**: FY​(y)=P(Y≤y)=P(g(X)≤y).  
2. **PDF of Y (Change of Variables Formula)**: If g(x) is monotonic with inverse x=g−1(y), then: fY​(y)=fX​(g−1(y))​dydx​​  
3. **Expected Value of Y=g(X) (LOTUS)**: E\[Y\]=E\[g(X)\]=∫−∞∞​g(x)fX​(x)dx

## **Chapter 9: Common Continuous Distributions**

### **1\. Uniform Distribution**

X∼U(a,b)

* **PDF (Probability Density Function):** $$ f(x; a, b) = \begin{cases} \frac{1}{b-a} & \text{for } a \le x \le b \\ 0 & \text{otherwise} \end{cases} $$  
* **CDF (Cumulative Distribution Function):** $$ F(x; a, b) = P(X \le x) = \begin{cases} 0 & \text{for } x < a \\ \frac{x-a}{b-a} & \text{for } a \le x \le b \\ 1 & \text{for } x > b \end{cases} $$  
* **Expected Value:** E\[X\]=2a+b​  
* **Variance:** Var(X)=12(b−a)2​

### **2\. Exponential Distribution**

T∼Exp(λ)

* **PDF (Probability Density Function):** $$ f(t; \lambda) = \begin{cases} \lambda e^{-\lambda t} & \text{for } t \ge 0 \\ 0 & \text{for } t < 0 \end{cases} $$  
* **CDF (Cumulative Distribution Function):** $$ F(t; \lambda) = P(T \le t) = \begin{cases} 1 - e^{-\lambda t} & \text{for } t \ge 0 \\ 0 & \text{for } t < 0 \end{cases} $$  
* **Survival Function:** P(T\>t)=1−F(t)=e−λt  
* **Expected Value:** E\[T\]=λ1​  
* **Variance:** Var(T)=λ21​  
* **Memoryless Property:** P(T\>s+t∣T\>s)=P(T\>t) for any s,t≥0.

### **3\. Normal (Gaussian) Distribution**

X∼N(μ,σ2)

* **PDF (Probability Density Function):** $$ f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{ - \frac{(x-\mu)^2}{2\sigma^2} } $$ for −∞\<x\<∞.  
* **Expected Value:** E\[X\]=μ  
* **Variance:** Var(X)=σ2  
* **Standardization (Z-score):** Z=σX−μ​ where Z∼N(0,1).

### **4\. Gamma Distribution**

X∼Gamma(k,λ) (using shape k and rate λ) or X∼Gamma(k,θ) (using shape k and scale θ=1/λ)  
The Gamma function is Γ(k)=∫0∞​xk−1e−xdx. For positive integers k, Γ(k)=(k−1)\!.

* **PDF (Probability Density Function):** Using shape k and rate λ: $$ f(x; k, \lambda) = \frac{\lambda^k x^{k-1} e^{-\lambda x}}{\Gamma(k)} \quad \text{for } x \ge 0 $$ Using shape k and scale θ=1/λ: $$ f(x; k, \theta) = \frac{1}{\Gamma(k)\theta^k} x^{k-1} e^{-x/\theta} \quad \text{for } x \ge 0 $$  
* **Expected Value:** E\[X\]=λk​=kθ  
* **Variance:** Var(X)=λ2k​=kθ2

### **5\. Beta Distribution**

X∼Beta(α,β)  
The Beta function is B(α,β)=∫01​tα−1(1−t)β−1dt=Γ(α+β)Γ(α)Γ(β)​.

* **PDF (Probability Density Function):** $$ f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha-1} (1-x)^{\beta-1} = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1} (1-x)^{\beta-1} $$ for 0≤x≤1.  
* **Expected Value:** E\[X\]=α+βα​  
* **Variance:** Var(X)=(α+β)2(α+β+1)αβ​

## Chapter 10: Joint Distributions

### Joint Probability Mass Functions (PMFs)

For two discrete random variables $X$ and $Y$:
* **Joint PMF Definition:**
    $$p_{X,Y}(x, y) = P(X=x, Y=y)$$
* **Conditions:**
    1.  $p_{X,Y}(x, y) \ge 0$ for all $(x, y)$
    2.  $\sum_{x} \sum_{y} p_{X,Y}(x, y) = 1$

### Joint Probability Density Functions (PDFs)

For two continuous random variables $X$ and $Y$:
* **Probability over a Region A:**
    $$P((X, Y) \in A) = \iint_A f_{X,Y}(x, y) \,dx \,dy$$
* **Conditions:**
    1.  $f_{X,Y}(x, y) \ge 0$ for all $(x, y)$
    2.  $\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{X,Y}(x, y) \,dx \,dy = 1$

### Marginal Distributions

* **Marginal PMF of X (Discrete):**
    $$p_X(x) = P(X=x) = \sum_{y} P(X=x, Y=y) = \sum_{y} p_{X,Y}(x, y)$$
* **Marginal PMF of Y (Discrete):**
    $$p_Y(y) = P(Y=y) = \sum_{x} P(X=x, Y=y) = \sum_{x} p_{X,Y}(x, y)$$
* **Marginal PDF of X (Continuous):**
    $$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \,dy$$
* **Marginal PDF of Y (Continuous):**
    $$f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \,dx$$

### Conditional Distributions

* **Conditional PMF of Y given X=x (Discrete):**
    $$p_{Y|X}(y|x) = P(Y=y | X=x) = \frac{P(X=x, Y=y)}{P(X=x)} = \frac{p_{X,Y}(x, y)}{p_X(x)}$$
    (provided $p_X(x) > 0$)
* **Conditional PDF of Y given X=x (Continuous):**
    $$f_{Y|X}(y|x) = \frac{f_{X,Y}(x, y)}{f_X(x)}$$
    (provided $f_X(x) > 0$)

### Joint Cumulative Distribution Functions (CDFs)

* **Joint CDF Definition:**
    $$F_{X,Y}(x, y) = P(X \le x, Y \le y)$$
* **Discrete Case:**
    $$F_{X,Y}(x, y) = \sum_{x_i \le x} \sum_{y_j \le y} p_{X,Y}(x_i, y_j)$$
* **Continuous Case:**
    $$F_{X,Y}(x, y) = \int_{-\infty}^{x} \int_{-\infty}^{y} f_{X,Y}(u, v) \,dv \,du$$
* **Properties:**
    1.  $0 \le F_{X,Y}(x, y) \le 1$
    2.  $F_{X,Y}(x, y)$ is non-decreasing in both $x$ and $y$.
    3.  $\lim_{x \to \infty, y \to \infty} F_{X,Y}(x, y) = 1$
    4.  $\lim_{x \to -\infty} F_{X,Y}(x, y) = 0$ and $\lim_{y \to -\infty} F_{X,Y}(x, y) = 0$

## Chapter 11: Independence, Covariance, and Correlation

### Independence of Random Variables

Two random variables $X$ and $Y$ are **independent** if for any sets $A$ and $B$:
$$P(X \in A, Y \in B) = P(X \in A) P(Y \in B)$$

This is equivalent to:

* **Discrete:**
    $$P(X=x, Y=y) = P(X=x) P(Y=y)$$
    (Joint PMF = Product of Marginal PMFs)

* **Continuous:**
    $$f_{X,Y}(x,y) = f_X(x) f_Y(y)$$
    (Joint PDF = Product of Marginal PDFs)

### Covariance

The **covariance** between two random variables $X$ and $Y$:
* **Definition:**
    $$\mathrm{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])]$$
* **Computational Formula:**
    $$\mathrm{Cov}(X, Y) = E[XY] - E[X]E[Y]$$
* **Properties:**
    1.  $\mathrm{Cov}(X, X) = \mathrm{Var}(X)$
    2.  $\mathrm{Cov}(X, Y) = \mathrm{Cov}(Y, X)$
    3.  $\mathrm{Cov}(aX + b, cY + d) = ac \mathrm{Cov}(X, Y)$
    4.  $\mathrm{Cov}(X+Y, Z) = \mathrm{Cov}(X, Z) + \mathrm{Cov}(Y, Z)$
    5.  If $X$ and $Y$ are independent, then $\mathrm{Cov}(X, Y) = 0$.

### Correlation Coefficient

The **Pearson correlation coefficient** between two random variables $X$ and $Y$:
* **Definition:**
    $$\rho(X, Y) = \frac{\mathrm{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\mathrm{Cov}(X, Y)}{\sqrt{\mathrm{Var}(X) \mathrm{Var}(Y)}}$$
* **Properties:**
    1.  $-1 \le \rho(X, Y) \le 1$
    2.  $\rho(aX + b, cY + d) = \mathrm{sign}(ac) \rho(X, Y)$, (assuming $a \ne 0, c \ne 0$)

### Variance of Sums of Random Variables

For any two random variables $X$ and $Y$, and constants $a$ and $b$:
* **General Formula:**
    $$\mathrm{Var}(aX + bY) = a^2 \mathrm{Var}(X) + b^2 \mathrm{Var}(Y) + 2ab \mathrm{Cov}(X, Y)$$
* **Sum of Variables ($a=1, b=1$):**
    $$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2 \mathrm{Cov}(X, Y)$$
* **Difference of Variables ($a=1, b=-1$):**
    $$\mathrm{Var}(X - Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) - 2 \mathrm{Cov}(X, Y)$$
* **If $X$ and $Y$ are independent ($\mathrm{Cov}(X, Y) = 0$):**
    $$\mathrm{Var}(aX + bY) = a^2 \mathrm{Var}(X) + b^2 \mathrm{Var}(Y)$$   $$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$$   $$\mathrm{Var}(X - Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$$
* **Extension to Multiple Variables ($X_1, X_2, ..., X_n$):**
    $$\mathrm{Var}\left(\sum_{i=1}^n a_i X_i\right) = \sum_{i=1}^n a_i^2 \mathrm{Var}(X_i) + \sum_{i \ne j} a_i a_j \mathrm{Cov}(X_i, X_j)$$
    or
    $$\mathrm{Var}\left(\sum_{i=1}^n a_i X_i\right) = \sum_{i=1}^n a_i^2 \mathrm{Var}(X_i) + 2 \sum_{i < j} a_i a_j \mathrm{Cov}(X_i, X_j)$$
* **If all $X_i$ are independent:**
    $$\mathrm{Var}\left(\sum_{i=1}^n a_i X_i\right) = \sum_{i=1}^n a_i^2 \mathrm{Var}(X_i)$$

## Chapter 12: Functions of Multiple Random Variables

### Sums of Independent Random Variables (Convolution)

Let $X$ and $Y$ be two random variables, and $Z = X+Y$.

* **Discrete Case (PMF of Z):**
    $$P(Z=z) = \sum_{k} P(X=k, Y=z-k)$$
    If $X$ and $Y$ are independent:
    $$P(Z=z) = \sum_{k} P(X=k)P(Y=z-k)$$
    This is the discrete convolution of the PMFs.

* **Continuous Case (PDF of Z):**
    $$f_Z(z) = \int_{-\infty}^{\infty} f_{X,Y}(x, z-x)dx$$
    If $X$ and $Y$ are independent:
    $$f_Z(z) = \int_{-\infty}^{\infty} f_X(x)f_Y(z-x)dx = (f_X * f_Y)(z)$$
    This is the convolution of the PDFs.

### General Transformations (Jacobian Method for PDFs)

If $Y_1 = g_1(X_1, X_2)$ and $Y_2 = g_2(X_1, X_2)$ are transformations of random variables $X_1, X_2$, and these transformations are invertible such that $X_1 = h_1(Y_1, Y_2)$ and $X_2 = h_2(Y_1, Y_2)$.

* **Joint PDF of $Y_1, Y_2$:**
    $$f_{Y_1,Y_2}(y_1, y_2) = f_{X_1,X_2}(h_1(y_1,y_2), h_2(y_1,y_2)) |J|$$
    Where $|J|$ is the absolute value of the determinant of the Jacobian matrix.

* **Jacobian Determinant (J):**
    $$
    J = \det \begin{pmatrix}
    \frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} \\
    \frac{\partial x_2}{\partial y_1} & \frac{\partial x_2}{\partial y_2}
    \end{pmatrix}
    $$

### Order Statistics

Let $X_1, X_2, \dots, X_n$ be $n$ independent and identically distributed (i.i.d.) random variables with CDF $F_X(x)$ and PDF $f_X(x)$. Let $X_{(1)}, X_{(2)}, \dots, X_{(n)}$ be the order statistics (sorted values).

* **CDF of the Maximum ($Y_n = X_{(n)}$):**
    $$F_{Y_n}(y) = P(X_{(n)} \le y) = [F_X(y)]^n$$

* **PDF of the Maximum ($Y_n = X_{(n)}$):**
    $$f_{Y_n}(y) = n[F_X(y)]^{n-1}f_X(y)$$

* **CDF of the Minimum ($Y_1 = X_{(1)}$):**
    $$F_{Y_1}(y) = P(X_{(1)} \le y) = 1 - [1-F_X(y)]^n$$

* **PDF of the Minimum ($Y_1 = X_{(1)}$):**
    $$f_{Y_1}(y) = n[1-F_X(y)]^{n-1}f_X(y)$$

* **PDF of the $k$-th Order Statistic ($Y_k = X_{(k)}$):**
    $$f_{Y_k}(y) = \frac{n!}{(k-1)!(n-k)!} [F_X(y)]^{k-1} [1-F_X(y)]^{n-k} f_X(y)$$

## Chapter 13: The Law of Large Numbers (LLN)

### Chebyshev's Inequality

For a random variable $X$ with mean $\mu$ and finite variance $\sigma^2$:

* **Form 1:**
    $$P(|X - \mu| \ge k\sigma) \le \frac{1}{k^2}$$
    (where $k$ is the number of standard deviations)

* **Form 2:**
    $$P(|X - \mu| \ge \epsilon) \le \frac{\sigma^2}{\epsilon^2}$$
    (where $\epsilon > 0$ is any positive number)

### Weak Law of Large Numbers (WLLN)

For a sequence of i.i.d. random variables $X_1, X_2, \dots, X_n$ with common mean $E[X_i] = \mu$ and common finite variance $Var(X_i) = \sigma^2$. Let $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ be the sample mean.

* **Statement:** For any $\epsilon > 0$,
    $$\lim_{n \to \infty} P(|\bar{X}_n - \mu| \ge \epsilon) = 0$$
    or equivalently,
    $$\lim_{n \to \infty} P(|\bar{X}_n - \mu| < \epsilon) = 1$$

* **Formulas used in WLLN proof via Chebyshev's Inequality:**
    * Expected Value of Sample Mean:
        $$E[\bar{X}_n] = \mu$$
    * Variance of Sample Mean (for i.i.d. variables):
        $$Var(\bar{X}_n) = \frac{\sigma^2}{n}$$
    * Application of Chebyshev's Inequality to $\bar{X}_n$:
        $$P(|\bar{X}_n - \mu| \ge \epsilon) \le \frac{Var(\bar{X}_n)}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2}$$

### Strong Law of Large Numbers (SLLN)

For a sequence of i.i.d. random variables $X_1, X_2, \dots, X_n$ with common mean $E[X_i] = \mu$.

* **Statement:**
    $$P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$$
    (The sample mean converges almost surely to the population mean).

## Chapter 14: The Central Limit Theorem (CLT)

* **Statement of CLT (Lindeberg-Lévy CLT):**
    Let $X_1, X_2, \dots, X_n$ be i.i.d. random variables with mean $\mu$ and variance $\sigma^2$. Let $\bar{X}_n$ be the sample mean.
    $$Z_n = \frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1)$$
    (where $\xrightarrow{d}$ denotes convergence in distribution)

* **Convergence in Distribution (for $Z_n$):**
    $$\lim_{n \to \infty} P(Z_n \le z) = \Phi(z)$$
    where $\Phi(z)$ is the CDF of the standard Normal distribution $N(0, 1)$.

* **Approximation for Sample Mean $\bar{X}_n$:**
    $$P(\bar{X}_n \le x) \approx \Phi\left(\frac{x - \mu}{\sigma/\sqrt{n}}\right)$$

* **CLT for Sums ($S_n = \sum_{i=1}^{n} X_i$):**
    $E[S_n] = n\mu$, $Var(S_n) = n\sigma^2$.
    $$\frac{S_n - n\mu}{\sqrt{n}\sigma} \xrightarrow{d} N(0, 1)$$

* **Normal Approximation to Binomial Distribution:**
    For $X \sim \text{Binomial}(n, p)$:
    * Mean: $E[X] = np$
    * Variance: $Var(X) = np(1-p)$
    * Approximation: $X \approx N(np, np(1-p))$ (if $np \ge 5$ and $n(1-p) \ge 5$ is a common rule of thumb).

* **Continuity Correction (for approximating discrete with continuous):**
    * To approximate $P(X \le k)$, use $P(Y \le k + 0.5)$
    * To approximate $P(X \ge k)$, use $P(Y \ge k - 0.5)$
    * To approximate $P(X = k)$, use $P(k - 0.5 \le Y \le k + 0.5)$
    * To approximate $P(a \le X \le b)$, use $P(a - 0.5 \le Y \le b + 0.5)$

## Chapter 15: Introduction to Bayesian Inference

* **Bayes' Theorem for Distributions:**
    $$p(\theta | D) = \frac{p(D | \theta) p(\theta)}{p(D)}$$
    Where:
    * $p(\theta | D)$ is the **posterior probability** of parameter $\theta$ given data $D$.
    * $p(D | \theta)$ is the **likelihood** of data $D$ given parameter $\theta$.
    * $p(\theta)$ is the **prior probability** of parameter $\theta$.
    * $p(D)$ is the **evidence** (or marginal likelihood of data).

* **Proportionality Form:**
    $$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$$

* **Evidence Calculation:**
    * For continuous $\theta$: $p(D) = \int p(D|\theta) p(\theta) d\theta$
    * For discrete $\theta$: $p(D) = \sum_{\theta} p(D|\theta) p(\theta)$

* **Beta-Binomial Conjugate Prior Update:**
    If prior is $\text{Beta}(\alpha_{prior}, \beta_{prior})$ and data is $k$ successes in $n$ trials (Binomial likelihood):
    * Posterior is $\text{Beta}(\alpha_{posterior}, \beta_{posterior}) = \text{Beta}(\alpha_{prior} + k, \beta_{prior} + n - k)$

* **Point Estimates from Posterior:**
    * **Maximum a Posteriori (MAP) Estimate:**
        $$\hat{\theta}_{MAP} = \arg \max_{\theta} p(\theta | D)$$
        For a Beta$(\alpha, \beta)$ posterior (if $\alpha > 1, \beta > 1$):
        $$\hat{\theta}_{MAP} = \frac{\alpha - 1}{\alpha + \beta - 2}$$
    * **Posterior Mean:**
        $$\hat{\theta}_{Mean} = E[\theta | D] = \int \theta p(\theta | D) d\theta$$
        For a Beta$(\alpha, \beta)$ posterior:
        $$\hat{\theta}_{Mean} = \frac{\alpha}{\alpha + \beta}$$

* **Credible Interval:**
    An interval $[L, U]$ such that:
    $$P(L \le \theta \le U | D) = \int_L^U p(\theta | D) d\theta = 1 - \gamma$$
    (where $1-\gamma$ is the credibility level, e.g., 95%)

## Chapter 16: Introduction to Markov Chains

* **Transition Probability (from state $i$ to state $j$):**
    $$P_{ij} = P(X_{t+1} = s_j | X_t = s_i)$$

* **n-Step Transition Probability:**
    The $(i, j)$-th entry of the matrix $P^n$ (the transition matrix $P$ raised to the power of $n$):
    $$P^{(n)}_{ij} = P(X_{t+n} = s_j | X_t = s_i) = (P^n)_{ij}$$

* **Stationary Distribution ($\pi$):**
    A row vector $\pi = [\pi_1, \pi_2, ..., \pi_k]$ such that:
    $$\pi P = \pi$$
    and
    $$\sum_{j=1}^{k} \pi_j = 1$$

## Chapter 17: Monte Carlo Methods

* **Estimating Probability $P(A)$:**
    $$P(A) \approx \frac{N_A}{N}$$
    (where $N_A$ is the number of times event A occurred in $N$ simulations)

* **Estimating Expected Value $E[g(X)]$:**
    $$E[g(X)] \approx \frac{1}{N} \sum_{i=1}^{N} g(X_i)$$
    (where $X_i$ are samples from the distribution of $X$)

* **Monte Carlo Integration (Hit-or-Miss for Area):**
    $$\text{Area}(A) \approx \text{Area}(B) \times \frac{N_{hit}}{N}$$

* **Monte Carlo Integration (Using Expected Values for $I = \int_a^b g(x) dx$):**
    If $X \sim \text{Uniform}(a, b)$:
    $$I \approx (b-a) \times \frac{1}{N} \sum_{i=1}^{N} g(X_i)$$

* **Inverse Transform Method for Generating Random Variables:**
    If $U \sim \text{Uniform}(0, 1)$, then $X = F^{-1}(U)$ has CDF $F(x)$.
    * For Exponential($\lambda$): $F^{-1}(u) = -\frac{1}{\lambda} \ln(1 - u)$ or $F^{-1}(u) = -\frac{1}{\lambda} \ln(u)$.

* **Acceptance-Rejection Method:**
    To sample from target PDF $f(x)$ using proposal PDF $g(x)$ where $f(x) \le c \cdot g(x)$:
    1.  Sample $y$ from $g(x)$.
    2.  Sample $u$ from $\text{Uniform}(0, 1)$.
    3.  Accept $y$ if $u \le \frac{f(y)}{c \cdot g(y)}$.

* **Buffon's Needle Problem ($L \le D$):**
    Probability of needle crossing a line:
    $$P(\text{cross}) = \frac{2L}{\pi D}$$
    For $L=1, D=2$:
    $$\pi \approx \frac{1}{P(\text{cross})}$$

## Chapter 18: (Optional) Further Explorations

* **Entropy $H(X)$ (for discrete random variable $X$ with PMF $p(x)$):**
    $$H(X) = - \sum_{x} p(x) \log_b p(x)$$
    (Base $b$ is often 2 for bits, or $e$ for nats)

* **Kullback-Leibler (KL) Divergence (for discrete distributions $P$ and $Q$):**
    $$D_{KL}(P || Q) = \sum_{x} P(x) \log_b \frac{P(x)}{Q(x)}$$

* **Geometric Brownian Motion (GBM) $S(t)$:**
    * Stochastic Differential Equation: $dS(t) = \mu S(t) dt + \sigma S(t) dW(t)$
    * Solution: $S(t) = S(0) \exp\left( \left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W(t) \right)$

* **Probability Generating Function (PGF) $G_X(z)$ (for non-negative integer-valued RV $X$):**
    $$G_X(z) = E[z^X] = \sum_{k=0}^{\infty} P(X=k) z^k$$
    * $E[X] = G'_X(1)$
    * $Var(X) = G''_X(1) + G'_X(1) - [G'_X(1)]^2$

* **Moment Generating Function (MGF) $M_X(t)$:**
    $$M_X(t) = E[e^{tX}]$$
    * $E[X^n] = M_X^{(n)}(0)$ (n-th derivative evaluated at $t=0$)
    * For independent $X, Y$: $M_{X+Y}(t) = M_X(t) M_Y(t)$