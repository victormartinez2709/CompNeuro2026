import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_bvp
from scipy.stats import norm

#Problem 1
N, sig_e, dt, T = 200, 1.0, 0.02, 40.0
nruns = 20
steps = int(round(T/dt))
idx = np.arange(N)
d = np.minimum(idx, N-idx)  #periodic distance on the ring

def mexican_hat(sig_i):
    return np.exp(-d**2/(2*sig_e**2)) - (sig_e/sig_i)*np.exp(-d**2/(2*sig_i**2))

def run(sig_i, rng):
    w = rng.normal(0, 0.01, N)
    K = np.fft.fft(mexican_hat(sig_i))
    for _ in range(steps):
        w += dt*np.real(np.fft.ifft(K*np.fft.fft(w)))  #circular convolution with the FFT
        np.clip(w, -1, 1, out=w)
    return w

def period(w):
    s = np.where(w >= 0, 1, -1)
    pairs = np.sum(s != np.roll(s, 1)) // 2  #two sign changes = one righ left pair
    return N/pairs if pairs else np.inf

rng = np.random.default_rng(20261021)
finals = {}
print(f"{'sigma_i':>7} {'mean':>7} {'SD':>6} {'2pi/k*':>7} {'mode':>5} {'ring':>6}")
for sig_i in (2.0, 3.0):
    seeds = rng.integers(0, 2**32 - 1, size=nruns)
    runs = [run(sig_i, np.random.default_rng(int(s))) for s in seeds]
    per = np.array([period(w) for w in runs])
    finals[sig_i] = runs[0]
    ks = np.sqrt(4*np.log(sig_i/sig_e)/(sig_i**2 - sig_e**2))
    Kt = np.real(np.fft.fft(mexican_hat(sig_i)))
    m = np.arange(1, N//2+1)[np.argmax(Kt[1:N//2+1])]  #fastest mode that fits on the ring
    print(f"{sig_i:>7} {per.mean():>7.3f} {per.std(ddof=1):>6.3f} {2*np.pi/ks:>7.3f} {m:>5} {N/m:>6.3f}")

fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
for a, sig_i in zip(ax, (2.0, 3.0)):
    a.plot(idx, finals[sig_i])
    a.axhline(0, linestyle="--", linewidth=1)
    a.set_ylabel(r"$w_-(a)$")
    a.set_title(rf"$\sigma_i={sig_i:g}$")
ax[-1].set_xlabel("cortical position $a$ (sites)")
fig.suptitle(r"Ocular-dominance patterns, $N=200$, $\sigma_e=1$, $t=40$")
fig.tight_layout()
fig.savefig("problem1_stripes.pdf")

#Problem 2 ----------
mu, sig, I0 = 0.2, 0.5, 5.0
dt, Nt = 0.002, 10_000
thetas = np.arange(0.5, 4.01, 0.5)

def euler_maruyama_step(r1, r2, lam, rng):
    w_s = w_c = (1 + lam)/2
    noise1 = sig*np.sqrt(dt)*rng.standard_normal(len(r1))
    noise2 = sig*np.sqrt(dt)*rng.standard_normal(len(r1))
    dr1 = (-r1 + w_s*r1 - w_c*r2 + I0 + mu/2)*dt + noise1
    dr2 = (-r2 + w_s*r2 - w_c*r1 + I0 - mu/2)*dt + noise2
    return r1 + dr1, r2 + dr2

def simulate(lam, theta, rng):
    r1, r2 = np.full(Nt, I0), np.full(Nt, I0)
    decided = np.zeros(Nt, dtype=bool)
    correct = np.zeros(Nt, dtype=bool)
    decision_time = np.zeros(Nt)
    t = 0
    while not decided.all():
        undecided = ~decided
        r1[undecided], r2[undecided] = euler_maruyama_step(r1[undecided], r2[undecided], lam, rng)
        t += dt
        dd = r1 - r2
        just_decided = undecided & (np.abs(dd) >= theta)
        correct[just_decided] = dd[just_decided] > 0
        decision_time[just_decided] = t
        decided |= just_decided
    return correct.mean(), decision_time.mean()

def theory_perfect_integrator(theta):
    return 1/(1 + np.exp(-mu*theta/sig**2)), theta/mu*np.tanh(mu*theta/(2*sig**2))

def theory_unstable_integrator(lam, theta):
    g = lambda y: np.exp(-(lam*y**2 + 2*mu*y)/(2*sig**2))
    accuracy = quad(g, -theta, 0)[0]/quad(g, -theta, theta)[0]
    ode = lambda x, y: np.vstack((y[1], (-1 - (lam*x + mu)*y[1])/sig**2))  #sig^2 T'' + (lam x + mu) T' = -1
    bc = lambda ya, yb: np.array([ya[0], yb[0]])
    x = np.linspace(-theta, theta, 250)
    sol = solve_bvp(ode, bc, x, np.zeros((2, x.size)), tol=1e-6, max_nodes=5000)
    return accuracy, float(sol.sol(0)[0])

def theory(lam, theta):
    return theory_perfect_integrator(theta) if lam == 0 else theory_unstable_integrator(lam, theta)

rng = np.random.default_rng(20261022)
simulated = {lam: np.array([simulate(lam, th, np.random.default_rng(int(rng.integers(0, 2**32 - 1))))
                            for th in thetas]) for lam in (0.0, 0.2)}
predicted = {lam: np.array([theory(lam, th) for th in thetas]) for lam in (0.0, 0.2)}
saturation = norm.cdf(mu/(sig*np.sqrt(0.2)))

#Potential sketch
dv = np.linspace(-4, 4, 400)
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
for a, lam in zip(ax, (-0.2, 0.0, 0.2)):
    a.plot(dv, -0.5*lam*dv**2 - mu*dv)
    a.axhline(0, linestyle=":", linewidth=0.8)
    a.set_title(rf"$\lambda={lam:g}$")
    a.set_xlabel("$d$")
ax[0].set_ylabel("$U(d)$")
fig.suptitle(r"$U(d)=-\lambda d^2/2-\mu d$, $\mu=0.2$")
fig.tight_layout()
fig.savefig("problem2_potential.pdf")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for lam, mk in [(0.0, "o"), (0.2, "s")]:
    for k in (0, 1):
        ax[k].plot(thetas, predicted[lam][:, k], label=rf"theory, $\lambda={lam:g}$")
        ax[k].plot(thetas, simulated[lam][:, k], mk, label=rf"simulation, $\lambda={lam:g}$")
ax[0].axhline(saturation, linestyle="--", linewidth=1, label=r"$\lambda=0.2$ saturation")
ax[0].set_ylabel("accuracy")
ax[1].set_ylabel("mean decision time (rate-time units)")
for a in ax:
    a.set_xlabel(r"threshold $\theta$")
    a.legend(fontsize=8)
fig.suptitle(rf"$\mu={mu}$, $\sigma={sig}$, $I_0={I0}$, $dt={dt}$, {Nt} trials")
fig.tight_layout()
fig.savefig("problem2_decision.pdf")

# erfect integrator at the threshold giving the same theoretical accuracy
accuracy, mean_time = predicted[0.2][-1]
theta_same_accuracy = sig**2/mu*np.log(accuracy/(1 - accuracy))
accuracy_perfect, mean_time_perfect = simulate(0.0, theta_same_accuracy,
                                               np.random.default_rng(int(rng.integers(0, 2**32 - 1))))
print(f"saturation = {saturation:.4f}")
print(f"lambda = 0.2, theta = 4: theory acc = {accuracy:.4f}, T = {mean_time:.3f};  "
      f"sim acc = {simulated[0.2][-1, 0]:.4f}, T = {simulated[0.2][-1, 1]:.3f}")
print(f"perfect integrator, theta = {theta_same_accuracy:.3f}: theory T = {theory_perfect_integrator(theta_same_accuracy)[1]:.3f};  "
      f"sim acc = {accuracy_perfect:.4f}, T = {mean_time_perfect:.3f}")

#Problem 3
N = 500
als = np.arange(0.02, 0.620001, 0.04)
n_trials = 10
print(f"alpha for 1% unstable bits = {1/norm.ppf(0.99)**2:.4f}")
rng_seeds = np.random.default_rng(20261023)
frac_unstable, mbars = [], []

for a in als:
    rng = np.random.default_rng(int(rng_seeds.integers(0, 2**32 - 1)))
    P = int(round(a*N))
    xis = rng.choice([-1, 1], size=(P, N))
    W = xis.T @ xis  #integer N*W: same signs as the Hebb W, and h = 0 is exact so sgn(0) = +1 holds
    np.fill_diagonal(W, 0)
    H = xis @ W
    frac_unstable.append(np.mean(xis*H < 0))

    m = []
    for _ in range(n_trials):
        mu_ = int(rng.integers(P))
        s = xis[mu_].copy()
        flip = rng.choice(N, size=N//10, replace=False)
        s[flip] *= -1
        while True:  #asynchronous sweeps in random order until nothing changes
            changed = False
            for i in rng.permutation(N):
                new = 1 if W[i] @ s >= 0 else -1  # sgn(0) = +1
                if new != s[i]:
                    s[i] = new
                    changed = True
            if not changed:
                break
        m.append(xis[mu_] @ s / N)
    mbars.append(np.mean(m))
    print(f"alpha = {a:.2f}: unstable sim = {frac_unstable[-1]:.5f}, theory = {norm.cdf(-1/np.sqrt(a)):.5f}, <m> = {mbars[-1]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(als, norm.cdf(-1/np.sqrt(als)), label=r"$\Phi(-1/\sqrt{\alpha})$")
ax[0].plot(als, frac_unstable, "o", label="simulation")
ax[0].set_ylabel("fraction unstable")
ax[0].legend()
ax[1].plot(als, mbars, "o-")
ax[1].set_ylabel(r"mean recall overlap $\langle m\rangle$")
for x in ax:
    x.set_xlabel(r"$\alpha=P/N$")
fig.suptitle(rf"Hebbian Hopfield network, $N={N}$")
fig.tight_layout()
fig.savefig("problem3_capacity.pdf")

#Problem 4
tau, ratio = 20.0, 1.2  # tau_+ = tau_- (ms), A_-/A_+

def pairing_change(f, delta):  #Delta w / A_+ per period, T in ms
    T = 1000/f
    return (np.exp(-delta/tau) - ratio*np.exp(-(T - delta)/tau))/(1 - np.exp(-T/tau))

f = np.linspace(1, 50, 500)
f_switch = 1000/(20*(1 + np.log(ratio)))
print(f"pre-before-post switch frequency = {f_switch:.2f} Hz")

plt.figure(figsize=(7, 4.5))
plt.plot(f, pairing_change(f, 10.0), label="pre before post")
plt.plot(f, pairing_change(f, 1000/f - 10.0), label="post before pre")
plt.axhline(0, linestyle="--", linewidth=1)
plt.axvline(f_switch, linestyle=":", linewidth=1, label=f"switch = {f_switch:.2f} Hz")
plt.xlabel("pairing frequency $f$ (Hz)")
plt.ylabel(r"$\Delta w/A_+$ per period")
plt.title(r"$\tau_+=\tau_-=20$ ms, $A_-=1.2A_+$, 10 ms lag")
plt.legend()
plt.tight_layout()
plt.savefig("problem4_pairing.pdf")