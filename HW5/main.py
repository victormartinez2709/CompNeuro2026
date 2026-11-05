import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

f = lambda x, eta: 1/(1 + np.exp(-np.clip(eta*x, -500, 500)))  # logistic

def peaks(y):  #indices of local maxima
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1

#Problem 1
eta, tau_E, w_EI, w_IE, Eb, Ib = 4.0, 5.0, 10.0, 10.0, 0.1, 0.1
finv = lambda y: np.log(y/(1 - y))/eta
B = eta**2*w_EI*w_IE*Eb*(1 - Eb)*Ib*(1 - Ib)/tau_E

def w_hopf(tau_I):
    return (1 + tau_E/tau_I)/(eta*Eb*(1 - Eb))

def simulate_wc(tau_I, eps, window=2000.0):
    w_EE = (1 + eps)*w_hopf(tau_I)
    p_E = finv(Eb) - w_EE*Eb + w_EI*Ib  #keeps equilibrium @ (0.1, 0.1)
    p_I = finv(Ib) - w_IE*Eb
    rhs = lambda t, y: [(-y[0] + f(w_EE*y[0] - w_EI*y[1] + p_E, eta))/tau_E,
                        (-y[1] + f(w_IE*y[0] + p_I, eta))/tau_I]
    T = max(4000.0, 12/(0.15*eps))  
    t_eval = np.linspace(T - window, T, int(round(window/0.1)) + 1)
    sol = solve_ivp(rhs, (0, T), [Eb + 0.001, Ib], t_eval=t_eval, rtol=1e-8, atol=1e-10, max_step=0.5)
    t, E, I = sol.t, sol.y[0], sol.y[1]
    pE, pI = peaks(E)[-100:], peaks(I)
    period = np.mean(np.diff(t[pE]))
    lags = []
    for p in pE[:-1]:  #time from each E peak to the next I peak
        later = pI[pI > p]
        if len(later) and t[later[0]] - t[p] < period:
            lags.append(t[later[0]] - t[p])
    return t, E, I, period, E.max() - E.min(), 360*np.mean(lags)/period

omega = np.sqrt(B/10 - 1/100)
print(f"wEE Hopf = {w_hopf(10):.4f}, pE = {finv(Eb) - w_hopf(10)*Eb + w_EI*Ib:.4f}, pI = {finv(Ib) - w_IE*Eb:.4f}")
print(f"B = {B:.4f}, omega = {omega:.4f} 1/ms, period = {2*np.pi/omega:.4f} ms, f = {1000*omega/(2*np.pi):.2f} Hz")
print(f"min tau_I = {1/B:.4f} ms, phase lag = {np.degrees(np.arctan(10*omega)):.2f} deg")

#(i) amplitude, period and lag versus eps
eps_vals = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
res = np.array([simulate_wc(10.0, e)[3:] for e in eps_vals])
slope, icpt = np.polyfit(np.log(eps_vals), np.log(res[:, 1]), 1)
print(f"{'eps':>6} {'period':>8} {'amplitude':>10} {'lag':>7}")
for e, (per, amp, lag) in zip(eps_vals, res):
    print(f"{e:>6.3f} {per:>8.2f} {amp:>10.5f} {lag:>7.2f}")
print(f"log-log slope = {slope:.4f}")

#(ii) frequency versus tau_I at eps = 0.02
taus = np.array([2, 3, 5, 7.5, 10, 15, 20])
f_sim = np.array([1000/simulate_wc(tau, 0.02, window=1000.0)[3] for tau in taus])
f_lin = 1000*np.sqrt(B/taus - 1/taus**2)/(2*np.pi)
for tau, fs, fl in zip(taus, f_sim, f_lin):
    print(f"tau_I = {tau:>4}: f sim = {fs:.2f} Hz, f linear = {fl:.2f} Hz")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].loglog(eps_vals, res[:, 1], "o", label="simulation")
ef = np.linspace(eps_vals.min(), eps_vals.max(), 300)
ax[0].loglog(ef, np.exp(icpt)*ef**slope, label=rf"fit slope $={slope:.3f}$")
ax[0].set_xlabel(r"$\epsilon$"); ax[0].set_ylabel(r"$\max E-\min E$")
ax[0].set_title(r"Amplitude, $\tau_I=10$ ms"); ax[0].legend()
ax[1].plot(taus, f_lin, "-", label="linear prediction")
ax[1].plot(taus, f_sim, "o", label="simulation")
ax[1].set_xlabel(r"$\tau_I$ (ms)"); ax[1].set_ylabel("frequency (Hz)")
ax[1].set_title(r"Frequency, $\epsilon=0.02$"); ax[1].legend()
fig.tight_layout(); fig.savefig("problem1_hopf_scaling.pdf")

# (iii) trace
t, E, I = simulate_wc(10.0, 0.1, window=500.0)[:3]
m = t >= t[-1] - 120
plt.figure(figsize=(7.5, 4.2))
plt.plot(t[m], E[m], label="$E(t)$"); plt.plot(t[m], I[m], label="$I(t)$")
plt.xlabel("time (ms)"); plt.ylabel("population activity")
plt.title(r"$\tau_I=10$ ms, $\epsilon=0.1$"); plt.legend(); plt.tight_layout()
plt.savefig("problem1_trace.pdf")

#Problem 2
beta, g, tau_a, I_R = 1.0, 2.0, 120.0, 1.3

def simulate_rivalry(I_L, w=0.8):
    def rhs(t, y):
        uL, uR, aL, aR = y
        return [-uL + f(w*uL - beta*uR - g*aL + I_L, 1000.0),
                -uR + f(w*uR - beta*uL - g*aR + I_R, 1000.0),
                (-aL + uL)/tau_a, (-aR + uR)/tau_a]
    t_eval = np.arange(0, 30*tau_a + 0.05, 0.1)
    sol = solve_ivp(rhs, (0, 30*tau_a), [1, 0, 0, 0], method="BDF", t_eval=t_eval,
                    rtol=1e-7, atol=1e-9, max_step=1.0)
    d = sol.y[0] - sol.y[1]
    dom = np.where(d > 0.5, 1, np.where(d < -0.5, -1, 0))  #1 = L dominant, -1 = R dominant
    hand, cur = [], 0
    for tj, dj in zip(sol.t, dom):
        if dj != 0 and dj != cur:
            hand.append((tj, dj)); cur = dj
    durs = [(hand[j][1], hand[j+1][0] - hand[j][0]) for j in range(len(hand) - 1)][3:]  #drop the first three
    T_L = np.mean([T for who, T in durs if who == 1])
    T_R = np.mean([T for who, T in durs if who == -1])
    return sol, T_L, T_R

def rivalry_theory(I_L):
    B_L, B_R = (I_L - beta)/g, (I_R - beta)/g
    return tau_a*np.log((1 - B_L)/B_R), tau_a*np.log((1 - B_R)/B_L)

sols = {}
for I_L in (1.30, 1.45, 1.60):
    sols[I_L], T_L, T_R = simulate_rivalry(I_L)
    th = rivalry_theory(I_L)
    print(f"I_L = {I_L:.2f}: T_L = {T_L:.1f}, T_R = {T_R:.1f};  theory T_L = {th[0]:.1f}, T_R = {th[1]:.1f}")

_, T_L, T_R = simulate_rivalry(1.3, w=0.3)
print(f"w = 0.3: T sim = {(T_L + T_R)/2:.1f}, theory = {tau_a*np.log((0.3 + 1.3)/(g - 0.3 - 1.3)):.1f}")

sol = sols[1.30]
t, B_eq = sol.t, (1.3 - beta)/g
m = (t >= 600) & (t <= 1500)
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(t[m], sol.y[0][m], label="$u_L$"); ax[0].plot(t[m], sol.y[1][m], label="$u_R$")
ax[0].set_ylabel("rate"); ax[0].legend()
ax[1].plot(t[m], sol.y[2][m], label="$a_L$"); ax[1].plot(t[m], sol.y[3][m], label="$a_R$")
ax[1].axhline(B_eq, linestyle="--", linewidth=1, label="$B$")
ax[1].axhline(1 - B_eq, linestyle=":", linewidth=1, label="$1-B$")
ax[1].set_xlabel("time (rate-time units)"); ax[1].set_ylabel("adaptation"); ax[1].legend()
fig.suptitle(r"$I_L=I_R=1.3$, $w=0.8$, $\tau_a=120$")
fig.tight_layout(); fig.savefig("problem2_rivalry_trace.pdf")

#Problem 3
dx = dt = 0.02
x = np.arange(-20, 20 + dx/2, dx)
N = len(x)
w = lambda z: 2*np.exp(-np.abs(z)) - np.exp(-np.abs(z)/2)
W = lambda z: np.sign(z)*2*(np.exp(-np.abs(z)/2) - np.exp(-np.abs(z)))  
L = 2**int(np.ceil(np.log2(2*N - 1)))  
w_hat = np.fft.rfft(w(x), L)

def conv(fx):
    full = np.fft.irfft(np.fft.rfft(fx, L)*w_hat, L)
    return dx*full[(N - 1)//2:(N - 1)//2 + N]

def widths(kappa):
    S = np.sqrt(1 - 2*kappa)
    return -np.log((1 + S)/2), -np.log((1 - S)/2)

def lam_wide(kappa):
    S = np.sqrt(1 - 2*kappa)
    return -S*(1 - S)/(1 + S*(1 - S)/2)

def half_width(u, kappa):
    c = N//2
    if u[c] < kappa:
        return 0.0
    j = c
    while j + 1 < N and u[j + 1] >= kappa:
        j += 1
    if j + 1 >= N:
        return x[-1]
    return x[j] + (kappa - u[j])*(x[j + 1] - x[j])/(u[j + 1] - u[j]) 

def simulate_field(kappa, a0, T=100.0):
    u = W(x + a0) - W(x - a0)
    times, hw = [], []
    for n in range(int(round(T/dt)) + 1):
        if n % 10 == 0:
            times.append(n*dt); hw.append(half_width(u, kappa))
        u += dt*(-u + conv(f(u - kappa, 100.0)))
    return np.array(times), np.array(hw)

kappa = 0.3
a_m, a_p = widths(kappa)
S = np.sqrt(1 - 2*kappa)
print(f"a- = {a_m:.4f}, a+ = {a_p:.4f}, U-(0) = {2*W(a_m):.4f}, U+(0) = {2*W(a_p):.4f}")
print(f"lambda_e(a-) = {S*(1 + S)/(1 - S*(1 + S)/2):.4f}, lambda_e(a+) = {lam_wide(kappa):.4f}")

plt.figure(figsize=(8, 5))
for lab, a0 in [("narrow, $a_--0.02$", a_m - 0.02), ("narrow, $a_-+0.02$", a_m + 0.02),
                ("wide, $a_+-0.05$", a_p - 0.05), ("wide, $a_++0.05$", a_p + 0.05)]:
    plt.plot(*simulate_field(kappa, a0), label=lab)
plt.axhline(a_m, linestyle="--", linewidth=1, label=r"$a_-$")
plt.axhline(a_p, linestyle=":", linewidth=1, label=r"$a_+$")
plt.xlabel("time"); plt.ylabel("active half-width")
plt.title(r"$\kappa=0.3$, $\Delta x=\Delta t=0.02$"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("problem3_halfwidth.pdf")

plt.figure(figsize=(7.5, 5))
for kap in (0.1, 0.3, 0.45):
    a_p = widths(kap)[1]
    tt, hw = simulate_field(kap, a_p + 0.05)
    err = np.abs(hw - a_p)
    fit = (tt >= 2) & (tt <= 50) & (err > 0.005) & (err < 0.05)  
    rate, c0 = np.polyfit(tt[fit], np.log(err[fit]), 1)
    print(f"kappa = {kap}: lambda_e(a+) = {lam_wide(kap):.4f}, fitted rate = {rate:.4f}")
    plt.plot(tt[err > 0], np.log(err[err > 0]), label=rf"$\kappa={kap:g}$")
    plt.plot(tt[fit], c0 + rate*tt[fit], "--")
plt.xlabel("time"); plt.ylabel(r"$\log|a(t)-a_+|$"); plt.title("Wide-bump relaxation")
plt.legend(); plt.tight_layout(); plt.savefig("problem3_relaxation.pdf")