# SBBTS — Theory, Mathematics, and Implementation

> **Paper:** *SBBTS: A Unified Schrödinger–Bass Framework for Synthetic Financial Time Series*
> Alouadi, Loeper, Marsala, Mazhar, Pham — arXiv:2604.07159 (April 2026)

This document is my personal working-through of the theory behind this project. The goal is to understand not just *what* the paper proves, but *why* the problem is set up the way it is, what each formula is actually saying, and how all of it maps to the code. I wrote it as I would want to find it myself: starting from scratch on the concepts, going through the math at a level where every symbol is explained, and linking everything to the actual implementation at every step.

It is long by design. If you already know optimal transport, skip to Part III. If you already know the SBB framework, skip to Part IV.

---

## Table of Contents

**Part I — The Problem**
1. [Why financial time series is genuinely hard](#1-why-financial-time-series-is-genuinely-hard)
2. [What existing methods get wrong](#2-what-existing-methods-get-wrong)
3. [What we need and why it's non-trivial](#3-what-we-need-and-why-its-non-trivial)

**Part II — Mathematical Foundations**
4. [Probability, processes, and stochastic differential equations](#4-probability-processes-and-stochastic-differential-equations)
5. [Optimal transport — moving distributions like sand](#5-optimal-transport--moving-distributions-like-sand)
6. [Schrödinger Bridges — the entropy perspective](#6-schrödinger-bridges--the-entropy-perspective)
7. [Bass Martingales — calibrating only the volatility](#7-bass-martingales--calibrating-only-the-volatility)
8. [Why we need both and neither is enough alone](#8-why-we-need-both-and-neither-is-enough-alone)

**Part III — The Schrödinger-Bass Bridge**
9. [The SBB cost functional](#9-the-sbb-cost-functional)
10. [The β parameter in depth](#10-the-β-parameter-in-depth)
11. [The optimal solution and what it looks like](#11-the-optimal-solution-and-what-it-looks-like)
12. [The auxiliary process Y_t — the key to everything](#12-the-auxiliary-process-y_t--the-key-to-everything)
13. [The large-β approximation](#13-the-large-β-approximation)

**Part IV — From Two Marginals to Time Series**
14. [The SBBTS problem formulation](#14-the-sbbts-problem-formulation)
15. [Theorem 3.2 — the decomposition result](#15-theorem-32--the-decomposition-result)
16. [What the decomposition means for implementation](#16-what-the-decomposition-means-for-implementation)

**Part V — The Neural Algorithm**
17. [Score matching — what we are actually learning](#17-score-matching--what-we-are-actually-learning)
18. [The Brownian bridge as training distribution](#18-the-brownian-bridge-as-training-distribution)
19. [The loss function, broken down](#19-the-loss-function-broken-down)
20. [Algorithm 1 — step by step with code](#20-algorithm-1--step-by-step-with-code)

**Part VI — Neural Architecture**
21. [What the score network must do](#21-what-the-score-network-must-do)
22. [The trajectory encoder (causal transformer)](#22-the-trajectory-encoder-causal-transformer)
23. [The drift head (score network)](#23-the-drift-head-score-network)
24. [Why these architectural choices](#24-why-these-architectural-choices)

**Part VII — Sampling**
25. [Euler-Maruyama — simulating the SDE](#25-euler-maruyama--simulating-the-sde)
26. [Recovering X from Y at each boundary](#26-recovering-x-from-y-at-each-boundary)
27. [The ξ offset and why division-by-zero is a real concern](#27-the-ξ-offset-and-why-division-by-zero-is-a-real-concern)

**Part VIII — Implementation Insights**
28. [Input normalization — why returns at scale 0.01 break training](#28-input-normalization--why-returns-at-scale-001-break-training)
29. [Bridge time normalization — the numerical stability fix](#29-bridge-time-normalization--the-numerical-stability-fix)
30. [The β·Δt condition in practice](#30-the-βδt-condition-in-practice)
31. [Low-β InverseNet mode](#31-low-β-inversenet-mode)

**Part IX — Experiments**
32. [Heston model recovery](#32-heston-model-recovery)
33. [S&P 500 data augmentation](#33-sp-500-data-augmentation)

**Part X — Code Map**
34. [File structure and code cross-reference](#34-file-structure-and-code-cross-reference)

---

## Part I — The Problem

### 1. Why financial time series is genuinely hard

Financial returns look like noise. At first glance, daily log returns of a stock index are just a sequence of numbers oscillating around zero with no obvious pattern. And yet these numbers are far from random in the simple sense — they have a very particular statistical texture that any competent generative model must reproduce if it wants to be useful.

This texture is called **stylized facts** — empirically robust properties that have been documented across decades, asset classes, and geographies. They are not theoretical predictions; they are observed facts about how markets behave.

**Fat tails.**
A Gaussian (normal) distribution assigns negligibly small probability to returns more extreme than 3–4 standard deviations. In real markets, such events happen far more often than that — daily moves of 5%, 8%, or even 20% in a single day are rare but not once-in-a-century events. Formally, the return distribution has *excess kurtosis* (heavier tails than a Gaussian). This matters enormously for risk: a model that assumes normality will consistently underestimate the probability of catastrophic losses.

**Volatility clustering.**
When markets are calm, they tend to stay calm. When they are turbulent, they tend to stay turbulent. If today's return is large in absolute terms, tomorrow's is likely to be large too. This is not correlation of returns (which is weak) but correlation of *squared* returns — the autocorrelation function of |r_t| or r_t² is positive and decays slowly. A model where daily volatility is constant (like GBM) completely misses this.

**Leverage effect.**
Negative returns today predict higher volatility tomorrow more than positive returns do. If the S&P 500 drops 3% today, tomorrow's volatility is likely higher than if it had gained 3%. This asymmetry — named after the financial concept of leverage — means that drawdowns are more dangerous than equivalent gains in a way that goes beyond the simple magnitude of the move.

**Stochastic volatility.**
Volatility is not constant. It has its own random dynamics — mean-reverting, with clustering, and correlated with price moves (leverage effect). This is the heart of models like Heston and Rough Heston, and it's what makes options pricing hard. A model that fixes volatility at a constant level (like Black-Scholes or plain GBM) will systematically misprice options and misestimate risk.

**Cross-asset correlations.**
In a portfolio of many assets, the correlation structure is time-varying and crisis-dependent. During normal times, correlations between assets in different sectors might be 0.3. During a crash (2008, March 2020), correlations across all assets jump toward 1 — exactly when diversification would matter most. Any model that assumes fixed correlations will fail to capture this regime-switching behavior.

Here is how the most important generative approaches score against these:

| Method | Fat tails | Vol clustering | Leverage | Stoch. vol | Cross-asset |
|---|:---:|:---:|:---:|:---:|:---:|
| Geometric Brownian Motion | ✗ | ✗ | ✗ | ✗ | ✗ |
| GAN / VAE | ~ | ~ | ~ | ~ | ~ |
| Schrödinger Bridge (SB only) | ✓ | ✓ | ✓ | **✗** | ~ |
| Bass martingale | ~ | ~ | ✗ | ✓ | ~ |
| **SBBTS** | ✓ | ✓ | ✓ | ✓ | ✓ |

The `~` marks mean "depends heavily on architecture and data." The hard failures (`✗`) are structural — they cannot be fixed by training harder, because the model class itself rules out that behavior.

---

### 2. What existing methods get wrong

**GBM (Black-Scholes baseline)**

The Geometric Brownian Motion assumes:
```
dX_t = μ X_t dt + σ X_t dW_t
```
where μ and σ are *constants*. This gives log returns that are exactly i.i.d. Gaussian at every timescale. Every single stylized fact is violated by construction. GBM is still widely used for option pricing (in its risk-neutral form), but it is universally acknowledged to be wrong as a description of real dynamics.

**Diffusion models and Score-Based Generative Models**

The modern ML approach learns to reverse a noising process. These models are extremely powerful for images and text but applying them naively to financial time series faces a structural limitation: they tend to learn marginal distributions well but struggle with the precise temporal dependency structure (the autocorrelation of |r_t|, the leverage effect). The cross-time correlations are subtle and the signal-to-noise ratio is extremely low in finance.

**Schrödinger Bridges (pure SB, e.g., SBTS from the 2025 JMLR paper)**

This is the closest predecessor. The SB approach finds the diffusion process $dX_t = \alpha_t dt + dW_t$ that starts at distribution $\mu_0$, ends at distribution $\mu_T$, and is closest to Brownian motion in relative entropy.

It works remarkably well for capturing marginal distributions and temporal structure. The critical failure mode is: **the diffusion coefficient is fixed at $\sigma_t = I_d$ by construction.**

This matters because in finance, the quadratic variation of a process — the total amount of "squared movement" — encodes its volatility. If $\sigma_t = I_d$, then the process $X_t$ has quadratic variation $\langle X \rangle_t = t$, which grows linearly at rate 1. This means: constant volatility, no stochastic volatility.

Concretely: the SB framework cannot generate paths where volatility itself changes randomly. It can learn *how returns are distributed* and *how they correlate in time*, but it cannot capture the fact that the variance of returns varies day to day in a stochastic, path-dependent way.

The paper demonstrates this precisely: when fitting an SB-based model to Heston trajectories and then recovering Heston parameters, the "vol of vol" parameter ξ and the price-vol correlation ρ are both wrong — they collapse to a single point rather than spanning their true range. These parameters are precisely the ones that encode *how much volatility itself varies* (ξ) and *in what direction volatility moves when price moves* (ρ). Both are inaccessible to a model with fixed $\sigma_t$.

**Bass martingales (pure volatility calibration)**

The opposite approach: set drift to zero, and use the volatility alone to match the target distribution. A Bass martingale constructs $X_t = f_t(W_t)$ such that $X_T \sim \mu_T$. It has no drift. It can match stochastic volatility (because $f_t$ shapes the diffusion coefficient) but it cannot capture temporal predictive structure — the drift is by definition always zero, so any tendency of the process to move in a particular direction over time is lost.

In finance terms: a pure Bass martingale is a risk-neutral model (appropriate for option pricing) but useless for capturing the directional dynamics, momentum, or mean-reversion that real time series exhibit.

---

### 3. What we need and why it's non-trivial

The ideal generative model for financial time series needs to:
1. Match the marginal distribution of returns at every time step (fat tails, skewness)
2. Capture temporal dependencies (vol clustering, autocorrelation of |r|)
3. Allow for stochastic volatility (the diffusion coefficient must be random and path-dependent)
4. Model cross-asset correlations (multi-dimensional case)
5. Be tractable to train with limited data (~thousands of windows)

The fundamental insight of SBBTS is that **both drift and diffusion must be free to adapt**, and the way to do this efficiently is through optimal transport applied to the joint distribution of the entire path.

---

## Part II — Mathematical Foundations

### 4. Probability, processes, and stochastic differential equations

Before the main course, let's establish the vocabulary. I'll assume you know what a probability distribution is but not necessarily what a stochastic process or SDE means.

**Stochastic process.** A stochastic process $(X_t)_{t \geq 0}$ is a family of random variables indexed by time. Think of it as a function $t \mapsto X_t(\omega)$ where $\omega$ is the outcome of an underlying random experiment. For each fixed $\omega$, the function $t \mapsto X_t(\omega)$ is called a **trajectory** or **path**. The collection of all possible paths is what we care about.

**Brownian motion (Wiener process).** The canonical stochastic process. $W_t$ is defined by:
- $W_0 = 0$
- Increments $W_t - W_s$ for disjoint intervals are independent
- $W_t - W_s \sim \mathcal{N}(0, t-s)$ — increments are Gaussian with variance proportional to time
- Paths are continuous (but nowhere differentiable)

Brownian motion is the continuous-time analogue of a random walk where you take infinitely many infinitely small steps. Its key property: it is *memoryless* in the increments (independent increments) but not memoryless in its level (the value at time $t$ depends on where it started).

**Stochastic Differential Equation (SDE).** We want to describe more general processes than pure Brownian motion. The Itô SDE:
$$dX_t = \alpha_t \, dt + \sigma_t \, dW_t$$
says: at each instant, $X_t$ moves by a deterministic drift $\alpha_t$ (direction and speed) plus a random diffusion term $\sigma_t \, dW_t$ (a scaled Brownian increment). This is not an equation you integrate like a normal differential equation — $dW_t$ is not differentiable — but it has a rigorous mathematical meaning (the Itô integral).

If $\alpha_t$ and $\sigma_t$ are functions of $(t, X_t)$ only (no memory), the process is **Markov** — its future depends only on its current value, not its history. If they depend on the full path $X_{0:t}$, it is **path-dependent** (or non-Markov). Financial time series are almost certainly path-dependent (vol clustering means today's vol depends on the recent history), which is why SBBTS uses a path-encoder.

**Quadratic variation.** For an Itô process $dX_t = \alpha_t dt + \sigma_t dW_t$, the **quadratic variation** is:
$$\langle X \rangle_t = \int_0^t \|\sigma_s\|^2 \, ds$$

This is the total "squared fluctuation" accumulated up to time $t$. For pure Brownian motion, $\sigma_t = I_d$, so $\langle W \rangle_t = t$ — it grows linearly. The rate of quadratic variation at any instant is $\|\sigma_t\|^2$, which is directly the variance (squared volatility) of the process.

This is why fixing $\sigma_t = I_d$ (as classical SB does) fixes the quadratic variation and hence the volatility structure. Stochastic volatility requires $\sigma_t$ to be random.

**Martingale.** A process $M_t$ is a martingale if $\mathbb{E}[M_t | M_s] = M_s$ for all $s < t$. In words: the best forecast of a martingale's future value is its current value — there is no predictable drift. Asset prices under the risk-neutral measure (used for option pricing) are martingales. This is the no-arbitrage condition: if prices were predictably drifting, you could trade profitably without risk.

A Bass martingale is specifically a martingale that is *continuous* and matches prescribed endpoint distributions — its diffusion coefficient (volatility) does all the work of shaping the distribution.

**KL divergence (relative entropy).** For two probability measures $P$ and $Q$, the KL divergence is:
$$\text{KL}(P \| Q) = \mathbb{E}_P\left[\log \frac{dP}{dQ}\right]$$
where $dP/dQ$ is the Radon-Nikodym derivative (the ratio of densities if they exist). KL divergence is always $\geq 0$, equals 0 if and only if $P = Q$, and is **not symmetric** ($\text{KL}(P\|Q) \neq \text{KL}(Q\|P)$ in general). It measures how "surprised" you would be by samples from $P$ if you thought the distribution was $Q$.

In the context of stochastic processes, $\text{KL}(P \| W)$ where $W$ is Wiener measure measures how far a process $P$ is from pure Brownian motion.

---

### 5. Optimal transport — moving distributions like sand

**The basic problem.** You have a pile of sand shaped like probability distribution $\mu_0$ and you want to reshape it into distribution $\mu_T$. What is the cheapest way to do it, if moving a unit of sand from position $x$ to position $y$ costs $\|x - y\|^2$?

Formally, you are looking for a **coupling** — a joint distribution $\pi(x, y)$ on pairs $(x, y)$ such that the marginals are $\mu_0$ (distribution of $x$) and $\mu_T$ (distribution of $y$). The problem is to minimize the average cost:

$$W_2^2(\mu_0, \mu_T) = \inf_{\pi \in \Pi(\mu_0, \mu_T)} \mathbb{E}_\pi[\|X - Y\|^2]$$

The infimum over all couplings with the right marginals is called the **2-Wasserstein distance** between $\mu_0$ and $\mu_T$. The optimal coupling $\pi^*$ tells you: if a grain of sand is at position $x$ under $\mu_0$, it should go to position $y$ under $\mu_T$, where the pairing $x \mapsto y$ minimizes total work.

For example, if $\mu_0 = \mathcal{N}(0, I_d)$ and $\mu_T = \mathcal{N}(m, \Sigma)$, the optimal map is simply $y = \Sigma^{1/2} x + m$ — stretch and shift. This makes intuitive sense: the cheapest way to reshape a centered Gaussian into a shifted/scaled Gaussian is to linearly transform every particle simultaneously, not shuffle them around.

**Why the squared distance?** You could use $|x-y|$ (Wasserstein-1) or $\|x-y\|^p$ for other $p$. The squared distance ($W_2$) is special because:
1. When the distributions are Gaussian, $W_2$ has a closed form
2. The optimal map $T: \mathbb{R}^d \to \mathbb{R}^d$ is the gradient of a convex function (Brenier's theorem)
3. It connects naturally to Schrödinger bridges via the Girsanov change of measure

**The static vs. dynamic problem.** Classic OT is *static*: it matches two snapshots of the distribution without specifying how to get from one to the other. For time series, we need *dynamic* OT: we need to specify an entire trajectory connecting $\mu_0$ and $\mu_T$, not just the endpoint mapping.

This is where the connection to stochastic processes enters: a continuous-time process $X_t$ with $X_0 \sim \mu_0$ and $X_T \sim \mu_T$ provides a dynamic transport plan. The problem is to choose among all such processes the one that is "most natural" or "cheapest" in some sense.

---

### 6. Schrödinger Bridges — the entropy perspective

**The question.** Among all continuous-time processes that start with distribution $\mu_0$ at time 0 and end with distribution $\mu_T$ at time $T$, which one is closest to Brownian motion?

"Closest" here means minimum relative entropy (KL divergence) against Wiener measure $\mathcal{W}$ (the law of a standard $d$-dimensional Brownian motion):

$$\text{SB}(\mu_0, \mu_T) = \inf_{\substack{P : \\ P \circ X_0^{-1} = \mu_0 \\ P \circ X_T^{-1} = \mu_T}} \text{KL}(P \| \mathcal{W})$$

Think of it this way: Brownian motion is the "default" or "most random" process. The Schrödinger Bridge asks: what is the minimum amount of structure (departure from pure randomness) needed to connect $\mu_0$ to $\mu_T$?

**The answer.** The solution exists and is unique under mild regularity conditions. It is a diffusion process of the form:

$$dX_t = \underbrace{\nabla_x \log h_t(X_t)}_{\text{score drift}} \, dt + dW_t$$

where $h_t$ solves a backward heat equation. The key word is the gradient of the log: this is the **score function** of the distribution of $X_t$ given what we need to happen at $T$. Intuitively, the drift is "pulling" the process toward configurations that are consistent with the target distribution.

**What SB gives us:**
- A principled, probabilistic way to interpolate between two distributions
- The entire path, not just endpoints
- A unique solution that is well-defined mathematically

**What SB cannot give us:**
- Any control over the diffusion coefficient — it is always $\sigma_t = I_d$ (unit Brownian motion)
- Stochastic volatility — the quadratic variation $\langle X \rangle_t = t$ is deterministic and linear
- Variance that changes over time in any path-dependent way

The constraint $P \ll \mathcal{W}$ (absolute continuity against Wiener measure) forces the diffusion coefficient to be exactly the identity. This is not a limitation of the particular algorithm — it is fundamental to the formulation.

---

### 7. Bass Martingales — calibrating only the volatility

**The opposite extreme.** What if we set drift to zero and let volatility do all the work?

A Bass martingale constructs a process of the form $X_t = f_t(W_t)$ where $f_t : \mathbb{R}^d \to \mathbb{R}^d$ is a time-dependent map applied to Brownian motion. The result has zero drift and diffusion coefficient $\sigma_t = \nabla f_t(W_t)$ (the Jacobian of $f_t$).

By choosing $f_T$ appropriately, we can make $X_T \sim \mu_T$ for any target distribution $\mu_T$ — the volatility "reshapes" the process to hit the right endpoint distribution. The relevant cost here is:

$$\text{Bass}(\mu_0, \mu_T) = \inf_{\substack{M \text{ martingale} \\ M_0 \sim \mu_0, M_T \sim \mu_T}} \mathbb{E}\left[\int_0^T \|\sigma_t - I_d\|^2 \, dt\right]$$

The martingale constraint ($\alpha_t = 0$) and the quadratic penalty on $\sigma_t - I_d$ mean: find the martingale with marginals $\mu_0$ and $\mu_T$ whose diffusion coefficient is closest to the identity.

**Why martingales appear in finance.** In a frictionless market with no arbitrage, asset prices (discounted by the risk-free rate) must be martingales under the risk-neutral measure. Bass martingales are thus natural for calibrating local volatility models to option prices — they encode the option market's implied distribution of future prices while respecting the martingale constraint.

**What Bass gives us:**
- Full control over the diffusion coefficient (stochastic volatility is possible)
- Exact matching of endpoint distributions
- A principled calibration framework for volatility

**What Bass cannot give us:**
- Any temporal drift dynamics — the process has zero drift by construction
- Predictive structure, momentum, or mean-reversion in returns
- The leverage effect (which involves the correlation between drift innovations and vol innovations)

---

### 8. Why we need both and neither is enough alone

Here is the core issue laid out plainly:

- SB fixes $\sigma_t = I_d$ (constant vol) and learns the drift. It captures temporal dependencies and distributional structure, but cannot model stochastic volatility.
- Bass sets $\alpha_t = 0$ (zero drift) and learns the volatility. It can match distributions with stochastic volatility, but loses all temporal predictive structure.

Financial time series needs both:
- **Stochastic volatility** (requires $\sigma_t$ to be non-trivial and random)
- **Temporal structure** (requires $\alpha_t$ to be non-zero and path-dependent)

The SBB framework interpolates between these two regimes in a principled way by putting both under a single cost functional that penalizes both non-trivial drift and non-trivial diffusion. The parameter $\beta$ controls the trade-off. This is the paper's central contribution.

---

## Part III — The Schrödinger-Bass Bridge

### 9. The SBB cost functional

The Schrödinger-Bass Bridge (SBB) problem was introduced in [Henry-Labordère et al., 2026] and extended to time series in this paper. Here is the full setup.

**The process.** We consider a general diffusion process:
$$X_t = X_0 + \int_0^t \alpha_s \, ds + \int_0^t \sigma_s \, dW_s, \quad t \in [0, T]$$

where $\alpha_t$ is the drift (allowed to be path-dependent) and $\sigma_t$ is the diffusion coefficient (matrix), also allowed to be path-dependent.

**The cost functional.** Given two distributions $\mu_0$ and $\mu_T$, we want to find the process with marginals $P \circ X_0^{-1} = \mu_0$ and $P \circ X_T^{-1} = \mu_T$ that minimizes:

$$J(P) = \mathbb{E}_P\left[\int_0^T \|\alpha_t\|^2 + \beta \|\sigma_t - I_d\|^2 \, dt\right]$$

Let us read each piece:

- $\|\alpha_t\|^2$: the squared norm of the drift at time $t$. This penalizes large drifts. A process with zero drift has no cost from this term. A process with large drift (strong directional movement) pays a high cost.

- $\|\sigma_t - I_d\|^2$: the squared deviation of the diffusion coefficient from the identity. A process with $\sigma_t = I_d$ (pure Brownian motion diffusion) pays no cost from this term. A process with non-trivial diffusion pays in proportion to how far it deviates from standard Brownian motion.

- $\beta$: controls the relative weight of the two penalties. Large $\beta$ means deviating from Brownian diffusion is very expensive, so the model will prefer to use drift to match the target distribution (→ SB limit). Small $\beta$ means drift is very expensive relative to diffusion, so the model will prefer to match using volatility (→ Bass limit).

- The integral over $t$: we are penalizing the *total* use of non-trivial drift and diffusion over the entire time interval, not just the endpoints.

**The SBB problem is then:**

$$\text{SBB}(\mu_0, \mu_T) = \inf_{\substack{P : \\ P \circ X_0^{-1} = \mu_0 \\ P \circ X_T^{-1} = \mu_T}} J(P)$$

The optimal value $\text{SBB}(\mu_0, \mu_T)$ is the minimum cost to connect $\mu_0$ to $\mu_T$ when both drift and diffusion are available as tools.

---

### 10. The β parameter in depth

**The interpolation.** $\beta$ genuinely interpolates between SB and Bass:

**As $\beta \to \infty$:**
The second term $\beta\|\sigma_t - I_d\|^2$ dominates. Any deviation of $\sigma_t$ from $I_d$ becomes infinitely costly. The optimizer is forced to set $\sigma_t = I_d$ everywhere, and all the work of connecting $\mu_0$ to $\mu_T$ must be done by the drift $\alpha_t$. The cost becomes:
$$J(P) \approx \mathbb{E}_P\left[\int_0^T \|\alpha_t\|^2 \, dt\right]$$
Minimizing this over processes with $\sigma_t = I_d$ and the right marginals is exactly the Schrödinger Bridge problem (the relative entropy against Wiener measure equals half the expected squared drift norm — this is Girsanov's theorem).

**As $\beta \to 0$:**
Dividing $J(P)$ by $\beta$ and sending $\beta \to 0$, the cost from drift $\|\alpha_t\|^2 / \beta$ becomes infinitely expensive relative to $\|\sigma_t - I_d\|^2$. The optimizer is forced to set $\alpha_t = 0$, and all transport must be done by the diffusion coefficient. This is the Bass martingale.

**In between:**
For finite $\beta$, both drift and diffusion are active. The model can choose: some combination of drift and volatility that together achieves the marginal constraints at minimum total cost.

**In practice:**
In finance, $\beta$ is typically large (not in the sense of $\to \infty$, but $\beta \cdot \Delta t \gg 1$). There are two reasons:

1. The existence condition for the SBB solution is $\beta \cdot \Delta t > 1$ (Theorem 3.2). This constrains the minimum value of $\beta$ given the time step $\Delta t$ of the data.

2. Financial time series at daily frequency have small $\Delta t$ (in the normalized $[0,1]$ scale, each step is $\approx 1/(T-1)$ which for $T=252$ is about $0.004$). For $\beta \cdot \Delta t > 1$, we need $\beta > 250$. In practice we use safety factors and end up with $\beta$ in the range of 1000–5000 for $T=252$.

At large $\beta$, the model is close to the SB regime, but with just enough diffusion freedom to capture stochastic volatility.

---

### 11. The optimal solution and what it looks like

The SBB problem (when the existence condition $\beta T > 1$ holds) has a unique solution characterized by a triple $(h, \nu, Y)$:

$$h_t = h_T * \mathcal{N}_{T-t}$$
$$\nu_t = \nu_0 * \mathcal{N}_t$$
$$Y_t = (\nabla_y \Phi_t)^{-1}, \quad \Phi_t(y) = \frac{|y|^2}{2} + \frac{1}{\beta} \log h_t(y)$$

Let me explain each piece:

**$h_t$:** A positive function that encodes information about the target distribution $\mu_T$ propagated backwards in time. It is the *backward* part of the Schrödinger factorization. The $*\mathcal{N}_{T-t}$ notation means convolution with a Gaussian of variance $T-t$ — this is the heat kernel. Intuitively, $h_T$ "knows" about $\mu_T$, and $h_t$ for $t < T$ is this knowledge blurred by the time remaining.

The score function $s^* = \nabla_y \log h_t$ appears in both the optimal drift and the transport map — it is the fundamental object the neural network is learning to approximate.

**$\nu_t$:** A positive measure that encodes the "forward" part — information about $\mu_0$ propagated forward. $\nu_0 * \mathcal{N}_t$ is the convolution of the initial distribution with a Gaussian that grows with time.

**$\Phi_t$:** A potential function — the sum of a quadratic term $|y|^2/2$ (which creates the identity map when you take its gradient, i.e., the Brownian motion baseline) and a correction term $\frac{1}{\beta} \log h_t(y)$ (which encodes the target information). The gradient $\nabla_y \Phi_t$ is a map from the auxiliary $y$-space to the $x$-space.

**$Y_t = (\nabla_y \Phi_t)^{-1}$:** The transport map from $x$-space to $y$-space. It inverts $\nabla_y \Phi_t$.

**The optimal drift and diffusion:**
$$\alpha_t^* = \nabla_y \log h_t(Y_t(X_t))$$
$$\sigma_t^* = D^2_y \Phi_t(Y_t(X_t))$$

The optimal drift is the score of $h_t$ evaluated at the transported position $Y_t(X_t)$. The optimal diffusion is the Hessian (second derivative matrix) of the potential $\Phi_t$ at the transported position. The Hessian of $\Phi_t$ measures how curved the potential is — in regions where the target distribution has high density, the potential is flat (Hessian ≈ identity), and in regions where the density is low, it is curved (Hessian ≠ identity). This curvature is what creates the non-trivial diffusion.

---

### 12. The auxiliary process Y_t — the key to everything

The most important conceptual step in making SBB tractable is the introduction of an **auxiliary process** $Y_t$ defined by:

$$Y_t = Y_t(X_t) = X_t - \frac{1}{\beta} \nabla_y \log h_t(Y_t(X_t))$$

This is an implicit equation relating $Y_t$ and $X_t$ through the score $s^* = \nabla_y \log h_t$. The inverse relationship (recovering $X_t$ from $Y_t$) is:

$$X_t = Y_t + \frac{1}{\beta} \nabla_y \log h_t(Y_t) = Y_t + \frac{1}{\beta} s^*(t, Y_t)$$

**Why does this help?**

Under a change of measure $d\hat{Q}/dP^{\text{SBB}} = 1/h_t(Y_t)$, the process $Y_t$ becomes a **standard Brownian motion** under $\hat{Q}$.

This is profound: instead of working with the complicated SBB process $X_t$ (which has non-trivial drift and diffusion), we can work with the simple $Y_t$ (which is just Brownian motion under $\hat{Q}$). The map $X_t \leftrightarrow Y_t$ is the "transport map" that converts between the two pictures.

Under the original measure $P^{\text{SBB}}$, the process $Y_t$ satisfies the SDE:
$$dY_t = \nabla_y \log h_t(Y_t) \, dt + dW_t$$

This is a **diffusion Schrödinger bridge** — the same structure as the classical SB solution, but for the auxiliary process $Y_t$. The drift is the score function of $h_t$.

So the strategy is:
1. Work with $Y_t$ (which solves a classical SB, something we know how to do)
2. Learn the score function $s^* = \nabla_y \log h_t$ (the drift of $Y_t$)
3. Convert back to $X_t$ using $X_t = Y_t + \frac{1}{\beta} s^*(t, Y_t)$ (the `y_to_x` function in the code)

The neural network $s_\theta$ is training to approximate this score function $s^*(t, Y_t, \text{context})$.

**In code:** `sbbts/transport/transport_map.py`

```python
# x_to_y: X → Y  (used in training to transform data to Y-space)
Y = X - (1/β) * s_θ(t, X, context)

# y_to_x: Y → X  (used in sampling to recover X from Y)  
X = Y + (1/β) * s_θ(t, Y, context)
```

---

### 13. The large-β approximation

For large $\beta$ (which is the relevant regime in finance), the score function is small relative to $X$ itself. More precisely, the transport map $Y_t(x) = x - \frac{1}{\beta} \nabla_y \log h_t(Y_t(x))$ admits a first-order approximation:

$$Y_t(x) \approx x - \frac{1}{\beta} \nabla_y \log h_t(x)$$

The approximation replaces $Y_t(x)$ on the right-hand side of the implicit equation with $x$ itself. This is valid when $\frac{1}{\beta}|s^*|$ is small relative to $|x|$ — the transport map is close to the identity.

**Why this matters:** The exact equation $Y_t(x) = x - \frac{1}{\beta} s^*(t, Y_t(x))$ is implicit — you need to solve a fixed-point problem to find $Y_t(x)$ given $x$. The approximation makes it explicit: you just evaluate $s_\theta$ at $x$ directly, with no iteration needed. This is the LightSB-M approach (from Alouadi et al., 2026), and it's what makes the algorithm tractable.

The paper notes that in the financial time series regime (daily data, T=252, β·Δt ≈ 5–20), this approximation is accurate and the algorithm converges in K=5 outer iterations.

---

## Part IV — From Two Marginals to Time Series

### 14. The SBBTS problem formulation

The SBB framework handles two distributions: $\mu_0$ (at time 0) and $\mu_T$ (at time $T$). Real financial data has structure at every time step — you have not just two snapshots but an entire **joint distribution** over trajectories.

Given a time grid $0 = t_0 < t_1 < \cdots < t_n = T$, the data gives us a joint distribution $\mu \in \mathcal{P}((\mathbb{R}^d)^{n+1})$ — the law of the full discretized path $(X_{t_0}, X_{t_1}, \ldots, X_{t_n})$.

The SBBTS problem is:

$$\text{SBBTS}(\mu) = \inf_{\substack{P \in \mathcal{P} : \\ P \circ (X_{t_0}, \ldots, X_{t_n})^{-1} = \mu}} J(P)$$

where $J(P) = \mathbb{E}_P\left[\int_0^T \|\alpha_t\|^2 + \beta\|\sigma_t - I_d\|^2 \, dt\right]$ as before.

The constraint is stronger than just matching marginals at $t_0$ and $t_n$: we require the **entire joint law** of the path at all $n+1$ time points to match $\mu$. This means matching not just the distribution of returns on day 1 and day 252, but also the distribution of (day 1 return, day 2 return, ..., day 252 return) simultaneously — including all their temporal dependencies.

**Why this is harder.** The two-marginal SBB problem has a unique solution characterized by two "potentials" (for the forward and backward directions). The multi-marginal problem has $n+1$ marginal constraints and requires $n+1$ potentials, coupled in a complicated way. Directly solving this would be intractable.

This is where Theorem 3.2 saves us.

---

### 15. Theorem 3.2 — the decomposition result

This is the central theoretical result of the paper. Let me state it, then explain it, then prove the intuition.

**Notation:**
- $\mu_i$: the marginal law of $(X_{t_0}, \ldots, X_{t_i})$ (the first $i+1$ steps of the path under $\mu$)
- $\mu_{i+1|0:i}(\cdot | x_{0:i})$: the conditional distribution of $X_{t_{i+1}}$ given that $(X_{t_0}, \ldots, X_{t_i}) = x_{0:i}$
- $\Delta t_i = t_{i+1} - t_i$: the length of the $i$-th interval
- $V_i(x_{0:i})$: the SBB cost from the point $x_i$ to the conditional distribution $\mu_{i+1|0:i}(\cdot | x_{0:i})$ on the interval $[t_i, t_{i+1}]$:
  $$V_i(x_{0:i}) = \text{SBB}\!\left(\delta_{x_i},\; \mu_{i+1|0:i}(\cdot | x_{0:i})\right)$$

**Theorem 3.2 (Decomposition).** Assume $\beta \cdot \Delta t_i > 1$ for all $i = 0, \ldots, n-1$. Then:

$$\boxed{\text{SBBTS}(\mu) = \sum_{i=0}^{n-1} \int V_i(x_{0:i}) \, \mu_i(dx_{0:i})}$$

Equivalently: $\text{SBBTS}(\mu) = \mathbb{E}_\mu\!\left[\sum_{i=0}^{n-1} V_i(X_{t_0:t_i})\right]$.

**What does this say?** The global multi-marginal optimal transport problem — matching the entire joint law of the path — decomposes into a sum of local two-marginal problems. At each time step $i$, given the path so far $x_{0:i}$, we solve the SBB problem from the current point $x_i$ to the conditional next-step distribution $\mu_{i+1|0:i}(\cdot | x_{0:i})$. The total cost is the average of these local costs.

**Intuition for the decomposition.**

Think of it this way. The cost $J(P) = \int_0^T \|\alpha_t\|^2 + \beta\|\sigma_t - I_d\|^2 \, dt$ decomposes naturally over time intervals:

$$J(P) = \sum_{i=0}^{n-1} \int_{t_i}^{t_{i+1}} \left(\|\alpha_t\|^2 + \beta\|\sigma_t - I_d\|^2\right) dt$$

Now consider: at time $t_i$, if we already know the past $x_{0:i}$, the remaining constraint is to reach $\mu_{i+1|0:i}(\cdot | x_{0:i})$ at time $t_{i+1}$ before doing anything else. These are separable optimization problems — what happens in $[t_i, t_{i+1}]$ affects only the cost on that interval, given that we start at $x_i$.

The formal proof (in Appendix A.2 of the paper) uses a measurable selection argument to show that the infimum over all processes with the right joint marginals equals the average of infima over individual intervals. The key is that optimal "continuation policies" can be concatenated consistently.

**Why the condition $\beta \cdot \Delta t_i > 1$?**

This is the existence condition for the SBB solution on each interval. Each sub-problem $V_i$ is a SBB from a single point $\delta_{x_i}$ to a distribution $\mu_{i+1|0:i}$. The SBB solution exists when $\beta \cdot \Delta t_i > 1$ (this is the condition for the Gaussian convolution $\nu_0 * \mathcal{N}_{\Delta t_i}$ to be non-degenerate relative to $\mu_{i+1|0:i}$). If any interval violates this condition, the decomposition doesn't hold and the algorithm is invalid.

**Practical consequence:**

We cannot freely choose $\beta$ and $T$ independently. For a time grid with $n$ intervals of equal length $\Delta t = T_{\text{total}} / n$ (after normalizing $T_{\text{total}} = 1$), the condition becomes:

$$\beta > \frac{1}{\Delta t} = \frac{n}{T_{\text{total}}} = n = T - 1 \quad (\text{for } T \text{ time points})$$

For $T = 252$ (one trading year), we need $\beta > 251$. In the code, `suggest_beta(n_time_points=252, safety_factor=5)` returns $\beta = 5 \times 251 = 1255$.

---

### 16. What the decomposition means for implementation

The decomposition reduces the SBBTS problem to learning a single function: the **conditional score** $s^*(t, Y_t, \text{past path})$.

At each interval $[t_i, t_{i+1}]$, we need to solve the SBB from $\delta_{x_i}$ to $\mu_{i+1|0:i}(\cdot | x_{0:i})$. By the SBB theory, this requires knowing the score function of the backward factor $h^i_t$ for that interval. This score depends on:
1. The current time $t \in [t_i, t_{i+1}]$
2. The current auxiliary state $Y_t$
3. The conditioning history $x_{0:i}$ (which determines which conditional distribution $\mu_{i+1|0:i}$ we are targeting)

So we need to learn a single function:
$$s_\theta(t, Y_t, \underbrace{\Phi_\theta(Y_{t_0:t_i})}_{\text{encoded history}}) \approx \nabla_y \log h^i_t(Y_t)$$

The architecture is then natural: a trajectory encoder $\Phi_\theta$ reads the past and produces a context vector; a score head $s_\theta$ uses the context vector, the current time, and the current state to predict the score.

The chain rule of conditional distributions lets us write:
$$\mu = \mu_0 \cdot \prod_{i=0}^{n-1} \mu_{i+1|0:i}$$

Each factor $\mu_{i+1|0:i}(\cdot | x_{0:i})$ is the "next-step" distribution given the full past. By learning to predict scores for all these conditional distributions simultaneously (over randomly sampled paths and time points), we learn a single score network that implicitly covers all $n$ local SBB problems.

---

## Part V — The Neural Algorithm

### 17. Score matching — what we are actually learning

The neural network is learning to approximate the score function $s^*(t, y) = \nabla_y \log h_t(y)$ — the gradient of the log of the backward factor.

Why is this called "score matching"? The term "score" in statistics refers to $\nabla_x \log p(x)$ — the gradient of the log-density. Score matching methods train neural networks to predict this gradient without needing to compute the (often intractable) normalizing constant of $p(x)$.

In our setting, the score of the target distribution at the auxiliary process level appears naturally as the drift of the $Y_t$ process:
$$dY_t = \underbrace{\nabla_y \log h_t(Y_t)}_{\text{score } s^*} dt + dW_t$$

If we knew $s^*$ exactly, we could simulate $Y_t$ directly and then recover $X_t$ via the transport map. Since we don't know $s^*$, we approximate it with a neural network trained to minimize a loss that makes the network's output close to $s^*$.

---

### 18. The Brownian bridge as training distribution

**The key insight for training.** To train the network, we need samples from the correct distribution at intermediate times $t \in (t_i, t_{i+1})$. But the correct distribution is the law of $Y_t$ under the SBB process — which depends on the score network we are trying to learn (circular dependency).

The resolution: **under the auxiliary measure $\hat{Q}$, $Y_t$ is a Brownian motion.** The conditional law of a Brownian motion between two fixed endpoints $y_{t_i}$ and $y_{t_{i+1}}$ is a **Brownian bridge**.

The Brownian bridge from $y_a$ to $y_b$ on $[a, b]$ has the following law at time $t \in (a, b)$:

$$Y_t | Y_a = y_a, Y_b = y_b \;\sim\; \mathcal{N}\!\left(\frac{b-t}{b-a} y_a + \frac{t-a}{b-a} y_b,\; \frac{(t-a)(b-t)}{b-a}\right)$$

That is:
- **Mean:** a linear interpolation between the two endpoints
- **Variance:** $\sigma_t^2 = \frac{(t-t_i)(t_{i+1}-t)}{t_{i+1}-t_i}$ — zero at both endpoints (the bridge must pass through them) and maximized at the midpoint

A concrete example: if $y_{t_i} = 0$ and $y_{t_{i+1}} = 1$ and the interval is $[0, 1]$, then at $t = 0.5$ the bridge has $\mathbb{E}[Y_{0.5}] = 0.5$ and $\text{Var}(Y_{0.5}) = 0.25$. The bridge is "pulled" toward the endpoint but has uncertainty about how it gets there.

**In practice:** We compute the endpoints $Y_{t_i}$ and $Y_{t_{i+1}}$ from data (via the transport map $x_\text{to\_y}$), then sample a random time $t$ and a random bridge point $Y_t$ using the formula above. This gives us an unbiased training sample from the correct distribution.

**In code:** `sbbts/transport/brownian_bridge.py`

```python
def sample_brownian_bridge(y_start, y_end, t_mid, t_start, t_end):
    alpha = (t_mid - t_start) / (t_end - t_start)   # linear weight
    mean  = (1 - alpha) * y_start + alpha * y_end
    std   = sqrt(alpha * (1 - alpha) * (t_end - t_start))
    return mean + std * randn_like(mean)
```

---

### 19. The loss function, broken down

The training loss (Eq. 4.1 in the paper) is:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=0}^{N-1} \mathbb{E}_{t \sim \mathcal{U}([t_i, t_{i+1})),\; Y_{t_{i+1}} \sim \mu_{i+1|0:i},\; Y_t \sim \mathcal{W}|_{Y_{t_i}, Y_{t_{i+1}}}} \!\!\!\left[\left\|s_\theta\!\left(t, Y_t, \Phi_\theta(Y_{t_0:t_i})\right) - \frac{Y_{t_{i+1}} - Y_t}{t_{i+1} - t}\right\|^2\right]$$

Let us dissect every piece:

**$s_\theta(t, Y_t, \Phi_\theta(Y_{t_0:t_i}))$:** The score network prediction. It takes:
- $t$: the current time (scalar, embedded via sinusoidal functions)
- $Y_t$: the current auxiliary state (vector in $\mathbb{R}^d$)
- $\Phi_\theta(Y_{t_0:t_i})$: the encoded trajectory up to interval $i$ (context vector in $\mathbb{R}^{d_\text{model}}$)

**$\frac{Y_{t_{i+1}} - Y_t}{t_{i+1} - t}$:** The **score target** — the direction and magnitude of movement from the current bridge point $Y_t$ toward the endpoint $Y_{t_{i+1}}$, normalized by the remaining time.

Why is this the right target? Because for a Brownian bridge from $y_a$ to $y_b$, the score of the distribution at time $t$ (pointing toward where the path is going) is exactly $\frac{y_b - y_t}{t_b - t}$. This is the Doob's $h$-transform: to make a Brownian motion end up at $y_b$, you need to apply a drift of $\frac{y_b - y_t}{t_b - t}$ at each moment. This drift "pulls" the process toward the target, more aggressively as time runs out.

**Intuition for the target formula:** Imagine driving to an appointment at a fixed time. If you have lots of time left, you don't need to rush ($\frac{y_b - y_t}{t_b - t}$ is small when $t_b - t$ is large). If you are almost late, you need to hurry ($\frac{y_b - y_t}{t_b - t}$ becomes large as $t \to t_b$). The Brownian bridge follows exactly this logic.

**The squared norm loss:** We minimize the expected squared difference between the prediction and the target. When the loss is zero, the network has learned the score function exactly, and $Y_t$ simulated with this drift will be a perfect Brownian bridge.

---

### 20. Algorithm 1 — step by step with code

Here is the complete training algorithm, with every step cross-referenced to the code.

**Initialization:** $s_\theta^0 \equiv 0$ (all weights to zero in the final output layer). This means $Y^0 = X$ — in the first iteration, the data is used directly as if the transport map were the identity.

**Outer loop, iteration $k$:**

```
For k = 0, 1, ..., K-1:
```

**Step 1. Sample a mini-batch.**

Draw $B$ trajectories $(X^b_{t_0}, X^b_{t_1}, \ldots, X^b_{t_n})$ from the training data.

```python
# sbbts_solver.py — inside fit() inner loop
for items in dataloader:
    batch = items[0]   # shape (B, N+1, d)
```

**Step 2. Transform X → Y at interval endpoints.**

For each interval $i$, compute the auxiliary process values at both endpoints using the *current* score network $s^k_\theta$:

$$\tilde{Y}^b_{t_i} = X^b_{t_i} - \frac{1}{\beta} s^k_\theta\!\left(t_i, X^b_{t_i}, \Phi^k_\theta(X^b_{t_0:t_i})\right)$$
$$\tilde{Y}^b_{t_{i+1}} = X^b_{t_{i+1}} - \frac{1}{\beta} s^k_\theta\!\left(t_{i+1}, X^b_{t_{i+1}}, \Phi^k_\theta(X^b_{t_0:t_i})\right)$$

At $k=0$, since $s_\theta \equiv 0$, we have $\tilde{Y} = X$.

```python
# sbbts_solver.py — inside _compute_training_loss()
contexts = self.score_net.encode_all_prefixes(batch, covariates=covariates)
score_at_start = self.score_net.forward_batched(s_start, x_ti,  contexts)
score_at_end   = self.score_net.forward_batched(s_end,   x_ti1, contexts)
y_ti  = x_ti  - score_at_start / self.beta
y_ti1 = x_ti1 - score_at_end   / self.beta
```

**Step 3. Sample from the Brownian bridge.**

Given the endpoints $\tilde{Y}_{t_i}$ and $\tilde{Y}_{t_{i+1}}$, sample a random intermediate point:

$$t \sim \mathcal{U}([t_i, t_{i+1})), \quad Z \sim \mathcal{N}(0, I_d)$$
$$Y_t = \frac{t_{i+1} - t}{\Delta t_i} \tilde{Y}_{t_i} + \frac{t - t_i}{\Delta t_i} \tilde{Y}_{t_{i+1}} + \sqrt{\frac{(t-t_i)(t_{i+1}-t)}{\Delta t_i}} Z$$

```python
# Brownian bridge in normalised time s ∈ [0, 1] per interval
alpha       = (s / T_bridge).unsqueeze(-1)
bridge_mean = (1.0 - alpha) * y_ti + alpha * y_ti1
bridge_std  = torch.sqrt((s * (1.0 - s / T_bridge)).clamp(min=0.0)).unsqueeze(-1)
y_s = bridge_mean + bridge_std * torch.randn_like(bridge_mean)
```

Note: the code uses normalized time $s \in [0, 1]$ per interval (not actual calendar time). See Part VIII §29 for why this is crucial.

**Step 4. Compute the score target and loss.**

$$\text{target} = \frac{\tilde{Y}_{t_{i+1}} - Y_t}{t_{i+1} - t}$$

```python
denom  = (T_bridge - s).clamp(min=safe_s).unsqueeze(-1)
target = (y_ti1 - y_s) / denom
score_pred = self.score_net.forward_batched(s, y_s, contexts)
loss = ((score_pred - target) ** 2).sum(dim=-1).mean()
```

**Step 5. Backpropagation and optimizer step.**

```python
scaler.scale(loss).backward()
if self.grad_clip > 0.0:
    scaler.unscale_(optimizer)
    gnorm = float(nn.utils.clip_grad_norm_(self.score_net.parameters(), self.grad_clip))
scaler.step(optimizer)
scaler.update()
```

**Step 6. After `n_epochs` epochs, carry $\theta^k$ forward to $\theta^{k+1}$.**

The weights are not reset between outer iterations — each iteration refines the previous result.

**Why iterate?** At iteration $k=0$, the bridge endpoints are the raw data $(X_{t_i}, X_{t_{i+1}})$. The learned score $s^1_\theta$ is an approximation to the true score. At iteration $k=1$, we re-compute the endpoints using this approximation: $Y_{t_i} = X_{t_i} - \frac{1}{\beta} s^1_\theta(\ldots)$, which are now closer to the true auxiliary process. Training on these refined endpoints gives a better score approximation. Each iteration "tightens" the self-consistency between the score and the transport map. In practice, $K=5$ is sufficient.

---

## Part VI — Neural Architecture

### 21. What the score network must do

The score network $s_\theta(t, Y_t, \Phi_\theta(Y_{t_0:t_i}))$ has four responsibilities:

1. **Read time** $t \in [0, T_{\text{bridge}}]$ — where are we in the current interval?
2. **Read the current state** $Y_t \in \mathbb{R}^d$ — where is the auxiliary process now?
3. **Read the history** $(Y_{t_0}, \ldots, Y_{t_i})$ — what happened in all previous intervals?
4. **Output the predicted drift** $s_\theta \in \mathbb{R}^d$ — in which direction should $Y_t$ move?

The architecture separates these cleanly: a trajectory encoder handles (3), and a drift head handles (1), (2), (4).

---

### 22. The trajectory encoder (causal transformer)

The encoder $\Phi_\theta$ takes the past trajectory $(Y_{t_0}, Y_{t_1}, \ldots, Y_{t_i}) \in (\mathbb{R}^d)^{i+1}$ and produces a context vector $c_i \in \mathbb{R}^{d_\text{model}}$.

**Architecture:**

```
Input: (Y_{t_0}, Y_{t_1}, ..., Y_{t_i})   shape: (i+1, d)
   ↓
Linear projection: (i+1, d) → (i+1, d_model)
   ↓
Add positional encoding (sinusoidal, to give the encoder a sense of position in the sequence)
   ↓
Causal Transformer Encoder (n_encoder_layers=1, n_heads=16)
   ↓
Take the last token (position i): (d_model,) = c_i
```

**Causal masking.** The transformer uses a mask that prevents position $j$ from attending to positions $k > j$. This means when computing the context for interval $i$, only positions $0, 1, \ldots, i$ contribute. The encoder cannot "see the future."

Why is this critical? During training, we process a full trajectory and compute contexts at all positions simultaneously. If we didn't use causal masking, the context at position $i$ would use information from positions $> i$ (future observations), giving unrealistically easy training. At test time, the future doesn't exist. The causal mask ensures the training and test regimes are consistent.

**Why a transformer?** Financial time series can have complex, long-range dependencies. A transformer with self-attention can learn which parts of the past trajectory are most relevant to the current conditional distribution, without being restricted to a fixed window or a Markov assumption. The paper tried fixed Gaussian mixture weights (LightSB approach) and found it "insufficiently flexible for time series data, as the weights of the Gaussian mixture are fixed."

An alternative (also implemented in our code): the **path signature encoder** (`sbbts/nn/signature_encoder.py`), which encodes the trajectory using iterated integrals (the path signature). The signature is a canonical description of the "shape" of a path and captures features like total variation, area swept, and higher-order dependencies.

---

### 23. The drift head (score network)

Given the context vector $c_i \in \mathbb{R}^{d_\text{model}}$, current time $t$, and current state $Y_t$, the drift head predicts the score:

```
t        → FNN_t → h_t   ∈ ℝ^{d_model}
Y_t      → FNN_y → h_y   ∈ ℝ^{d_model}
c_i      ────────────────── (passed through directly)
           ↓ concatenate
        [h_t ; h_y ; c_i]  ∈ ℝ^{3·d_model}
           ↓
        FNN_out            → s_θ ∈ ℝ^d
```

Each `FNN` block is: `Linear → LayerNorm → SiLU → Linear`.

**Time embedding.** Rather than feeding $t$ as a raw scalar, we embed it using sinusoidal functions (as in the original Transformer paper and diffusion model literature). This gives the network a structured, continuous representation of time that generalizes smoothly between training points.

**SiLU activation.** SiLU (Sigmoid Linear Unit) = $x \cdot \sigma(x)$ where $\sigma$ is the sigmoid. Compared to ReLU, SiLU is smooth everywhere (no kink at zero), bounded below, and has a non-zero gradient even for negative inputs. It tends to work better than ReLU in regression tasks where the output can be negative (as the score function can be).

**LayerNorm.** Layer normalization normalizes the activations within each token (across features, not across the batch). This stabilizes training, especially when inputs have varying scales across features.

**Zero initialization of the final layer.** The output linear layer `Linear(3·d_model, d)` is initialized with weights set to zero. This means at epoch 0, $s_\theta \equiv 0$ and the transport map is the identity: $Y = X$. This is crucial for stable training — the first outer iteration trains on the actual data without any distortion from a poorly initialized score.

---

### 24. Why these architectural choices

| Choice | Why |
|---|---|
| Encoder-only transformer | We need a context vector from past observations; no generation/decoding needed |
| Causal masking | Training and test-time conditions must match; the future is unknown at test time |
| Separate FNNs for $t$ and $Y_t$ | Time and state have very different scales and semantics; joint embedding risks one dominating |
| Concatenation (not addition) | Concatenation preserves all three signals independently; addition would conflate them |
| Zero output init | Stable starting point; identity transport is a reasonable initial condition at large $\beta$ |
| $d_\text{model} = 128$, 16 heads | Large enough to capture complex financial dependencies; 16 heads = 8-dimensional attention per head |
| 1 encoder layer | Financial time series are not grammatically complex; 1 layer captures local and global structure with self-attention |

---

## Part VII — Sampling

### 25. Euler-Maruyama — simulating the SDE

After training, generating a new trajectory works as follows. Within each interval $[t_i, t_{i+1}]$, the auxiliary process $Y_t$ follows:

$$dY_t = s_\theta(t, Y_t, c_i) \, dt + dW_t$$

We cannot solve this SDE analytically (the drift $s_\theta$ is a neural network, not a simple function). Instead, we use the **Euler-Maruyama** scheme — the simplest numerical method for SDEs, analogous to Euler's method for ODEs.

Divide the interval $[t_i, t_{i+1}]$ into $N_\pi = 50$ sub-steps of size $\delta t = (t_{i+1} - t_i)/N_\pi$:

$$Y_{t+\delta t} = Y_t + s_\theta(t, Y_t, c_i) \cdot \delta t + \sqrt{\delta t} \cdot Z, \quad Z \sim \mathcal{N}(0, I_d)$$

Each sub-step:
1. Evaluates the score (drift direction) at the current position
2. Takes a small step in that direction
3. Adds Gaussian noise (the Brownian increment)

With more sub-steps ($N_\pi$ larger), the discretization is more accurate. The paper uses $N_\pi = 50$ (Table 2). More than that gives diminishing returns for the cost increase.

**In code:** `sbbts_solver.py — sample() method`

```python
_dt_bridge = 1.0 / self.n_euler_steps   # in normalised [0,1] bridge time
for step in range(self.n_euler_steps):
    s_cur    = step * _dt_bridge
    s_tensor = torch.full((n,), s_cur, device=self.device)
    drift    = self.score_net.forward_with_context(s_tensor, y_current, context)
    y_current = (y_current
                 + drift * _dt_bridge
                 + torch.randn_like(y_current) * math.sqrt(_dt_bridge))
```

---

### 26. Recovering X from Y at each boundary

After simulating $Y_{t_{i+1}}$ via Euler-Maruyama, we recover $X_{t_{i+1}}$ using the inverse transport map:

$$X_{t_{i+1}} = Y_{t_{i+1}} + \frac{1}{\beta} s_\theta(\tilde{t}_{i+1}, Y_{t_{i+1}}, c_i)$$

where $\tilde{t}_{i+1} = t_{i+1} - \xi$ (slightly before the interval endpoint). We then update the context:

$$c_{i+1} = \Phi_\theta(X_{t_0}, X_{t_1}, \ldots, X_{t_{i+1}})$$

and continue to interval $i+1$ with $y_\text{current} = Y_{t_{i+1}}$.

**In code:**

```python
s_end_tensor = torch.full((n,), T_bridge - safe_s, device=self.device)
score_at_end = self.score_net.forward_with_context(s_end_tensor, y_current, context)
x_ti1 = y_to_x(y_current, score_at_end, self.beta)   # Y + (1/β) * score
trajectory[:, i + 1, :] = x_ti1
context = self.score_net.encode_trajectory(trajectory[:, :i + 2, :])
```

---

### 27. The ξ offset and why division-by-zero is a real concern

The score target in training is $\frac{Y_{t_{i+1}} - Y_t}{t_{i+1} - t}$. As $t \to t_{i+1}$, the denominator goes to zero. For a well-trained score network, the numerator $Y_{t_{i+1}} - Y_t$ also goes to zero (because $Y_t \to Y_{t_{i+1}}$ as the bridge closes), and the ratio converges to a finite limit. But numerically, both numerator and denominator can underflow to zero simultaneously and the ratio becomes $0/0$ — undefined.

The paper's solution: evaluate the score at $\tilde{t}_{i+1} = t_{i+1} - \xi$ for a small $\xi > 0$ (default $\xi = 0.01$ in our bridge-time coordinates). This keeps the denominator at least $\xi$ away from zero.

The score at $t_{i+1} - \xi$ is a good approximation to the limit $\lim_{t \to t_{i+1}} s^*(t, Y_t)$ because $\log h$ is continuous — the paper explicitly invokes this continuity to justify the approximation.

---

## Part VIII — Implementation Insights

### 28. Input normalization — why returns at scale 0.01 break training

Financial log returns have very small magnitude: typical daily log returns for a stock index are around ±0.5% to ±2%, so a typical standard deviation is around $0.007$ to $0.015$. This is roughly 2 orders of magnitude smaller than 1.

At sampling time, the model initializes with $X_0 \sim \mathcal{N}(0, I_d)$ — samples from a standard normal. This has standard deviation 1. The score network is trained on inputs that live near 0 with std ~0.01, but at sampling time it receives an initial point that could be ±3 or more.

This scale mismatch causes:
- Gradient norms during training that are orders of magnitude different from what the network experiences at sample time
- The score network's learned representations of "small" (training data) don't generalize to "normal" (sampling initialization)
- Sampling produces wildly out-of-distribution outputs or NaN

**Fix:** Normalize the training data to zero mean and unit standard deviation before fitting, then denormalize after sampling:

```python
# In fit():
self._train_mean = X.mean(dim=(0, 1), keepdim=True)   # per-feature mean
self._train_std  = X.std( dim=(0, 1), keepdim=True).clamp(min=1e-8)
X = (X - self._train_mean) / self._train_std

# In sample():
result = result * self._train_std.cpu().numpy() + self._train_mean.cpu().numpy()
```

This is the `normalize_input=True` parameter (on by default). For financial returns, this is not optional — it is necessary for training to work at all.

---

### 29. Bridge time normalization — the numerical stability fix

The loss function score target is $\frac{Y_{t_{i+1}} - Y_t}{t_{i+1} - t}$. When using actual calendar time (in the normalized $[0,1]$ range), the interval length for $T = 252$ time points is:

$$\Delta t = \frac{1}{252-1} \approx 0.004$$

So the denominator $t_{i+1} - t$ ranges from $\xi$ (minimum, at $t = t_{i+1}$) to $\Delta t \approx 0.004$ (maximum, at $t = t_i$). When $\xi = 0.001$ (in calendar time) and $\Delta t = 0.004$, the target magnitude is:

$$\left\|\frac{Y_{t_{i+1}} - Y_t}{t_{i+1} - t}\right\| \sim \frac{\text{step size}}{0.004} = 250 \times \text{step size}$$

This gives gradient norms in the thousands and makes training extremely unstable even with gradient clipping.

**Fix:** Normalize time *per interval* to $s \in [0, 1]$. Within each interval, the bridge runs from $s=0$ to $s=1$ regardless of the calendar-time length $\Delta t$. The denominator then ranges from $\xi$ to $1$, giving targets of order 1–2. Gradient norms are controllable.

This is why the code comment in `_compute_training_loss` says:
```
Each adjacent pair (x[i], x[i+1]) defines a Brownian bridge parametrised
in NORMALISED time s ∈ [0, T_bridge=1], matching the original paper repo.
```

This is a pure numerical trick — it does not change the mathematics (a monotone time reparametrization does not change the score function's direction, only its scale, which the network learns to absorb).

In the sampling code, the Euler-Maruyama steps also use bridge time $s \in [0, 1]$, with $\delta s = 1/N_\pi$. This consistency between training and sampling is essential.

---

### 30. The β·Δt condition in practice

The existence condition is $\beta \cdot \Delta t_i > 1$ for all intervals. In our bridge-time parametrization, all intervals have $\Delta t_\text{bridge} = 1$ by definition. So the condition becomes $\beta > 1$... but this refers to the *calendar* time $\Delta t$, not the bridge time.

In calendar time (with total time normalized to $T_\text{total} = 1$, $n = T-1$ intervals each of length $\Delta t = 1/(T-1)$):

$$\beta > \frac{1}{\Delta t} = T - 1$$

The `suggest_beta` function computes:
```python
dt = T_total / (n_time_points - 1)   # calendar Δt
return safety_factor / dt            # = safety_factor × (n_time_points - 1)
```

With `safety_factor=5` and `n_time_points=252`:
$$\beta = 5 \times 251 = 1255$$

The `validate_beta_condition` function in `transport_map.py` checks this at the start of fitting and raises an error if violated. The check is:

```python
beta_dt = self.beta * dt   # where dt = T / (n_time_points - 1)
if beta_dt <= 1.0:
    raise ValueError(...)
```

For the low-β warning threshold (default $\beta \cdot \Delta t < 3$), the InverseNet is activated as a fallback.

---

### 31. Low-β InverseNet mode

When $\beta \cdot \Delta t$ is close to 1 (just above the existence threshold), the large-β approximation $Y(x) \approx x - \frac{1}{\beta}s^*(t,x)$ becomes inaccurate. The true transport map requires iterative fixed-point solving.

As a practical solution for the low-β regime, the code includes an **InverseNet**: a small network that directly learns the residual correction $X_t - Y_t = \frac{1}{\beta} s^*(t, Y_t)$ given $Y_t$ as input:

```
InverseNet(t, Y_t) ≈ X_t - Y_t
```

In sampling, instead of $X = Y + \frac{1}{\beta} s_\theta(t, Y)$, the InverseNet predicts the correction directly:
```python
if self._is_low_beta and self.inverse_net is not None:
    x_ti1 = y_current + self.inverse_net(s_end_tensor, y_current)
```

The InverseNet is trained after each outer iteration using the current score network to compute target corrections. For typical financial data with large $\beta$, this mode is never activated — it is a safety net for edge cases.

---

## Part IX — Experiments

### 32. Heston model recovery

The Heston model is the benchmark stochastic volatility model in quantitative finance. It is a two-dimensional SDE:

$$dX_t = r X_t \, dt + \sqrt{v_t} \, X_t \, dW^X_t$$
$$dv_t = \kappa (\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW^v_t, \quad \text{Corr}(dW^X, dW^v) = \rho \, dt$$

Parameters and their meaning:
- $r$: the drift rate of the log price (related to the risk premium)
- $\kappa$: mean-reversion speed of variance — how quickly volatility returns to its long-run average after a shock
- $\theta$: long-run variance level — where volatility gravitates on average
- $\xi$: "vol of vol" — how much volatility itself fluctuates (if $\xi = 0$, volatility is constant)
- $\rho$: correlation between price moves and variance moves — negative $\rho$ is the leverage effect (price down = vol up)

The test protocol:
1. Generate 5000 Heston trajectories with random parameters from prescribed ranges
2. Fit SBBTS to these trajectories
3. Generate 5000 synthetic trajectories from the fitted model
4. Recover Heston parameters from the synthetic trajectories via maximum likelihood
5. Compare recovered vs. true parameters

**What SBBTS gets right that SBTS (Schrödinger Bridge only) gets wrong:**

The Schrödinger Bridge baseline fails on $\xi$ (vol of vol) and $\rho$ (price-vol correlation). The reason is structural: the SB framework fixes $\sigma_t = I_d$, so the quadratic variation of $X$ is deterministic. But $\xi$ encodes *how much the quadratic variation varies* (stochastic volatility) and $\rho$ encodes *how the quadratic variation correlates with price* (leverage). Both are features of $\sigma_t$. If you fix $\sigma_t = I_d$, you cannot recover $\xi$ and $\rho$ from the generated paths.

SBBTS, by allowing $\sigma_t$ to be stochastic and path-dependent, can generate paths where the quadratic variation varies (capturing $\xi$) and where it correlates with price movements (capturing $\rho$).

---

### 33. S&P 500 data augmentation

**The task:** Predict whether the next daily return of each S&P 500 stock will be positive or negative. This is a binary classification problem over 433 stocks.

**The challenge:** The training period (2010–2018) contains about 2263 daily observations. This sounds like a lot, but for a model trying to learn patterns over 252-day windows across 433 assets simultaneously, it is very limited — roughly 2000 windows of data for a problem with extremely low signal-to-noise ratio (financial returns are notoriously hard to predict).

**Why SBBTS helps:** By generating 200× more synthetic 252-day windows that faithfully reproduce the statistical properties of real market returns (correlations, volatility clustering, fat tails), we give the downstream model (TabICL) far more training scenarios. The model sees:
- More diverse market regimes (high and low volatility periods, trend and mean-reversion environments)
- More realizations of cross-asset correlation patterns
- More examples of fat-tailed events

**Why naive noise augmentation fails:** Adding Gaussian noise $\tilde{X} = X + \lambda \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma_X^2)$ creates samples near each real observation, but:
- The noise does not preserve temporal structure (the augmented sample doesn't have realistic vol clustering)
- The noise doesn't preserve cross-asset correlations
- The augmented samples are not plausible market scenarios — they are just jittered versions of real data

**Results (Table 1 from paper):**

| Method | Accuracy | ROC AUC | Sharpe |
|---|---|---|---|
| Zero-shot (TabICL, no fine-tuning) | 0.494 | 0.486 | −0.25 |
| Real data only | 0.521 | 0.497 | 1.61 |
| Real + Gaussian noise | 0.518 | 0.494 | 1.30 |
| **Real + SBBTS (200×)** | **0.532** | **0.521** | **2.11** |

The Sharpe ratio improvement from 1.61 to 2.11 represents a 31% gain in risk-adjusted return from the same trading strategy, using the same model, just with better training data. Gaussian noise augmentation actually hurts (1.30 vs 1.61 for real-only), confirming that the type of augmentation matters, not just the quantity.

**Statistical significance caveat (Table 5 from paper):**

The 95% bootstrap confidence intervals for Sharpe ratios overlap between real-only and SBBTS, so the improvement is not statistically significant at conventional levels with only 420 test observations. This is noted openly in the paper: establishing Sharpe ratio significance requires far more observations than a single train-test split provides. The result is directionally consistent across 5 seeds and economically meaningful.

**The dimensionality reduction pipeline:**

For 433 stocks with 252-day windows, directly fitting a single SBBTS model on $(N, 252, 433)$ data would be computationally infeasible. The paper's solution:

1. **PCA:** Project the 433-dimensional return space onto $m=16$ principal components (factors). These 16 factors capture the dominant correlation structure.
2. **K-means clustering:** Group the 16 factors into 3 clusters based on their statistical properties. Each cluster is a group of factors that "behave similarly."
3. **Per-cluster SBBTS:** Fit one SBBTS model per cluster (3 models, each on a small number of factors).
4. **Idiosyncratic residuals:** The part of each asset's return not captured by the 16 factors is modeled separately using a Gaussian mixture (heavy-tailed, independent across assets).
5. **Reconstruction:** Synthesize factor returns from the 3 SBBTS models, reconstruct asset returns via $\hat{X} = \hat{F} P^\top + \hat{R}$.

This pipeline is implemented in `sbbts/utils/dim_reduction.py` via `PCAKMeansReducer`.

---

## Part X — Code Map

### 34. File structure and code cross-reference

```
sbbts/
├── core/
│   ├── sbbts_solver.py       — Main SBBTS class (Algorithm 1, fit, sample, augment)
│   └── score_network.py      — ScoreNetwork class (drift head + encoder integration)
│
├── nn/
│   ├── encoder.py            — TrajectoryEncoder (causal transformer)
│   ├── drift_net.py          — DriftNet (FNN for time/state embedding + output head)
│   ├── inverse_net.py        — InverseNet (low-β fallback for Y → X recovery)
│   └── signature_encoder.py  — PathSignatureEncoder (alternative to transformer)
│
├── transport/
│   ├── transport_map.py      — x_to_y() and y_to_x() — the transport maps
│   ├── brownian_bridge.py    — Brownian bridge sampling (training step 3)
│   └── conditional_ot.py     — compute_conditional_transport, score target computation
│
├── utils/
│   ├── metrics.py            — compute_returns, var, ES, sharpe, compute_metrics, compute_tstr
│   ├── visualization.py      — All diagnostic plots (with logger= support)
│   ├── logger.py             — SBBTSLogger (diagnostics.log + training_epochs.log)
│   ├── sampling.py           — Euler-Maruyama, generate_brownian_motion, generate_gbm
│   ├── dim_reduction.py      — PCAKMeansReducer (high-d data handling)
│   └── early_stopping.py     — EarlyStopping (validation-based stopping)
│
└── benchmarks/
    ├── rough_volatility.py   — RoughHestonParams, simulate_rough_heston
    └── sp500.py              — Data loading utilities
```

**The central data flow, from training data to synthetic samples:**

```
Real price series
      ↓  compute_returns()      [sbbts/utils/metrics.py]
Log returns (N_DAYS,)
      ↓  sliding_window_view()  [numpy]
Windows: (N, T, d)
      ↓  SBBTS.fit()            [sbbts/core/sbbts_solver.py]
         ├── normalize_input: (N,T,d) → zero-mean, unit-std
         ├── init_score_network: ScoreNetwork on device
         ├── outer loop K times:
         │    ├── for each batch: _compute_training_loss()
         │    │    ├── encode_all_prefixes: context vectors for all intervals
         │    │    ├── x_to_y: batch endpoints → Y-space endpoints
         │    │    ├── sample Brownian bridge: intermediate Y_t points
         │    │    ├── compute score target: (Y_end - Y_t) / (1 - s)
         │    │    └── MSE(score_pred, target)
         │    └── Adam step + gradient clipping
         └── self._fitted = True
      ↓  SBBTS.sample(n)        [sbbts/core/sbbts_solver.py]
         ├── X_0 ~ N(0, I_d)
         ├── context = encode_trajectory(X_0)
         ├── for each interval:
         │    ├── Euler-Maruyama N_π steps in [0,1] bridge time
         │    └── y_to_x: Y_{end} → X_{i+1}
         ├── denormalize: × std + mean
         └── return trajectory (n, T, d)
Synthetic windows: (n, T, d)  ← same shape as real, statistically similar
```

**Key hyperparameter decisions:**

| Question | Answer | Code location |
|---|---|---|
| What β should I use? | `SBBTS.suggest_beta(T, safety_factor=5)` | `sbbts_solver.py::suggest_beta` |
| How many outer iterations? | K=5 for paper quality, K=2 for LITE | `n_steps` parameter |
| How big should the model be? | d_model=128, n_heads=16 (paper); d_model=32, n_heads=4 (LITE) | `d_model`, `n_heads` |
| Is my β valid? | Check at start of `fit()` | `sbbts_solver.py::_validate_beta_for_data` |
| How to handle large d? | PCAKMeansReducer → per-cluster SBBTS | `dim_reduction.py` |
| How to debug generation quality? | `full_diagnose(real, synth, logger=logger)` | `visualization.py` |
| What do the logs show? | Loss curves, gradient norms, vol ratios, TSTR score | `technical_logs/<run>/` |

---

*This document reflects my understanding of the paper as I implemented it. Some interpretations are my own — the paper is the authoritative source. Where the code diverges from the paper (e.g., bridge time normalization, which is not explicitly discussed in the paper but matches the original authors' repository), I have noted this explicitly.*

*The code is at: `github.com/JulianoLeperso/SBBTS`*
*The paper is at: `arXiv:2604.07159`*
