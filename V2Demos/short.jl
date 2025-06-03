# EchoKey Entropy Demo (Julia version)
#
# This demonstration implements key concepts from the EchoKey v2 paper:
# "A Universal Mathematical Programming Language for Complex Systems"
#
# The demo shows how equations from different fields can be imported,
# composed, and executed within a unified framework, following the
# mathematical libraries described in Section 3 of the paper.
#
# Run with: julia echokey_entropy_demo.jl
# Requires: Plots.jl (] add Plots)

using Random, Dates, Plots

# ---------- helper: seeded randomness ----------
Random.seed!(UInt32(Dates.unix2datetime(time()).instant.periods.value % UInt32))

# ---------- simulation parameters ----------
dt = 0.01
T = 50.0
t = 0:dt:T
n = length(t)

# ========================================================================
# IMPORTED EQUATION 1: LOGISTIC GROWTH (Biology)
# From Section 2.1: "Biology: dN/dt = rN(1 - N/K)"
# This represents the fundamental equation we're "importing" from ecology
# ========================================================================
r = rand(0.6:0.001:1.0)          # intrinsic growth rate
K = rand(8.0:0.01:12.0)          # carrying capacity
N = zeros(Float64, n)
N[1] = rand(0.1:0.001:1.0)       # initial population

# Parameters for LIBRARY 1: CYCLICITY (Section 3.1)
# "The Cyclicity library transforms any process into its periodic components"
# Mathematical form: C[f](t) = f(t) · (1 + ε∑An sin(ωnt + φn))
A = rand(0.2:0.001:0.5)      # cyclicity amplitude (An in paper)
ω = 2π / rand(5.0:0.01:15.0) # cyclicity period (ωn in paper)
ϕ = rand() * 2π              # phase shift (φn in paper)

# ========================================================================
# IMPORTED EQUATION 2: HARMONIC OSCILLATOR (Physics)
# While not explicitly the Schrödinger equation from Section 2.1,
# this represents importing a fundamental physics equation
# Following the paradigm: "Any equation from any field becomes a reusable component"
# ========================================================================
ω₀ = rand(0.8:0.001:1.2)         # angular frequency
x = zeros(Float64, n)             # position
v = zeros(Float64, n)             # velocity
x[1] = rand(-1.0:0.001:1.0)
v[1] = rand(-1.0:0.001:1.0)

# ========================================================================
# IMPORTED EQUATION 3: GEOMETRIC BROWNIAN MOTION (Finance)
# From Section 2.1: Black-Scholes equation
# "∂C/∂t + (1/2)σ²S²∂²C/∂S² + rS∂C/∂S - rC = 0"
# Here we implement the underlying asset dynamics: dS = μSdt + σSdW
# ========================================================================
μ = rand(-0.05:0.0001:0.05)  # drift coefficient
σ = rand(0.1:0.0001:0.3)     # volatility coefficient
C = zeros(Float64, n)             # asset price (or option value)
C[1] = rand(5.0:0.01:10.0)

# ========================================================================
# LIBRARY 5: SYNERGY - The Interaction Composer (Section 3.5)
# "The Synergy library captures emergent interactions between components"
# Mathematical form: S[E1, E2, ..., En] = ∑Ei + ∑κijEi⊗Ej + ...
# Here we implement a simple pairwise interaction
# ========================================================================
k₁ = rand(0.5:0.001:1.5)         # κ12: interaction coefficient between N and x
k₂ = rand(0.5:0.001:1.5)         # κ13: interaction coefficient with C
S = zeros(Float64, n)             # synergy state variable
S[1] = k₁ * N[1] * x[1] + k₂ * C[1]

# ========================================================================
# MAIN ECHOKEY EXECUTION ENGINE (Section 4.3)
# "The time evolution follows: dΨ/dt = ∑Fn[Cn(t)]·Gn(t) + S(Ψ) + O(t)"
# This loop implements the unified evolution of all composed equations
# ========================================================================
√dt = sqrt(dt)
for i in 2:n
    # --------------------------------------------------------------------
    # EQUATION 1 EVOLUTION: Logistic Growth
    # Base dynamics: dN/dt = rN(1 - N/K)
    # --------------------------------------------------------------------
    N[i] = N[i-1] + dt * r * N[i-1] * (1 - N[i-1] / K)

    # --------------------------------------------------------------------
    # APPLY LIBRARY 1: CYCLICITY
    # From Section 3.1.4: "C[E](t) = E(t)·(1 + ε∑An sin(ωnt + φn))"
    # This adds periodic modulation to the population dynamics
    # --------------------------------------------------------------------
    N[i] *= 1 + A * sin(ω * t[i] + ϕ)

    # --------------------------------------------------------------------
    # EQUATION 2 EVOLUTION: Harmonic Oscillator
    # Simple harmonic motion: d²x/dt² = -ω₀²x
    # Split into first-order system: dx/dt = v, dv/dt = -ω₀²x
    # --------------------------------------------------------------------
    v[i] = v[i-1] - dt * (ω₀^2) * x[i-1]
    x[i] = x[i-1] + dt * v[i]

    # --------------------------------------------------------------------
    # EQUATION 3 EVOLUTION: Geometric Brownian Motion
    # Implements stochastic dynamics: dC = μCdt + σCdW
    # This could represent option pricing dynamics from Black-Scholes
    # The √dt term comes from the Wiener process scaling
    # --------------------------------------------------------------------
    C[i] = C[i-1] + μ * C[i-1] * dt + σ * C[i-1] * √dt * randn()

    # --------------------------------------------------------------------
    # APPLY LIBRARY 5: SYNERGY
    # From Section 3.5.3: "S[E1,E2,...,En] = ∑Ei + ∑κijEi⊗Ej + ..."
    # This creates emergent behavior by coupling the three systems:
    # - Population (N) from biology
    # - Position (x) from physics  
    # - Price (C) from finance
    # The multiplication N[i]*x[i] represents the tensor product Ei⊗Ej
    # --------------------------------------------------------------------
    S[i] = k₁ * N[i] * x[i] + k₂ * C[i]
end

# ========================================================================
# VISUALIZATION
# Demonstrates the unified output of the composed system
# Each line represents an equation from a different field, now interacting
# ========================================================================
plot(t, N, label="Logistic N(t)")
plot!(t, x, label="Oscillator x(t)")
plot!(t, C, label="Option price C(t)")
plot!(t, S, label="Synergy S(t)", xlabel="t", ylabel="Value",
    title="EchoKey Entropy Demo (Julia)",
    legend=:topright)

# ========================================================================
# WHAT THIS DEMONSTRATES FROM THE PAPER:
#
# 1. EQUATION IMPORT (Section 2): We've imported equations from:
#    - Biology (logistic growth)
#    - Physics (harmonic oscillator)
#    - Finance (geometric Brownian motion)
#
# 2. LIBRARY APPLICATION (Section 3): We've applied:
#    - Cyclicity (Library 1): Periodic modulation of population
#    - Synergy (Library 5): Cross-domain coupling
#
# 3. COMPOSITION (Section 4.2): The equations are composed into a
#    unified system where biological, physical, and financial
#    dynamics interact through the synergy term
#
# 4. EXECUTION (Section 4.3): The Euler integration loop implements
#    the unified time evolution of the composed system
#
# 5. EMERGENT BEHAVIOR: The synergy variable S(t) shows complex
#    dynamics that emerge from the interaction of the three domains,
#    demonstrating how "mathematics becomes a programming language"
#
# MISSING LIBRARIES (for brevity):
# - Recursion (Library 2): Could add R[N](t) = N(f(N(t-1)))
# - Fractality (Library 3): Could add F[C](λt) = λ^H F[C](t)
# - Regression (Library 4): Could add mean reversion
# - Refraction (Library 6): Could transform across scales
# - Outliers (Library 7): Could add jump discontinuities
# ========================================================================