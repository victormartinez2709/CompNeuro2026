import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


#problem 1

Istar = 1/(1 - np.exp(-1))

def G(phi, I, eps):
    return I + eps*(np.sin(2*np.pi*phi) - 2*np.pi*np.cos(2*np.pi*phi))/(1 + 4*np.pi**2) - Istar

#1a
e = np.linspace(0, 1, 200)
plt.fill_betweenx(e, Istar - e/np.sqrt(1 + 4*np.pi**2), Istar + e/np.sqrt(1 + 4*np.pi**2), alpha=0.3, color='red')
plt.xlim(1.2, 2)
plt.xlabel('I'); plt.ylabel(r'$\epsilon$')
plt.title('Candidate 1:1 locking region')
plt.savefig('problem1_tongue.pdf', bbox_inches='tight')
plt.close()

#1b
I, eps = 1.6, 0.5
for bracket in [[0, 0.5], [0.5, 1]]:
    phi = opt.root_scalar(G, bracket=bracket, args=(I, eps)).root
    Iphi = I + eps*np.sin(2*np.pi*phi)
    lam = np.exp(-1)*Iphi/(Iphi - 1)
    print(f"1b: phi = {phi:.4f}, lambda = {lam:.4f}")

#1c
def forced_lif(I, eps=0.5, T=50, dt=1e-4):
    t = np.arange(0, T, dt)
    forcing = (eps*np.sin(2*np.pi*t)).tolist()
    v, spikes = 0.0, []

    for k in range(len(t)):
        v += dt*(-v + I + forcing[k])

        if v >= 1:
            v = 0.0
            spikes.append((k + 1)*dt)

    return np.array(spikes)

eps = 0.5
T, t_start = 50, 10  #we can discard the first 10
I = np.linspace(0.9, 3, 100)
firing_nums = []
unforced_firing_nums = []

for I_val in I:
    spikes = forced_lif(I_val)
    firing_nums.append(np.sum(spikes >= t_start)/(T - t_start))
    unforced_firing_nums.append(0 if I_val <= 1 else 1/np.log(I_val/(I_val - 1)))

firing_nums = np.array(firing_nums)
Il = Istar - eps/np.sqrt(1 + 4*np.pi**2)
Ir = Istar + eps/np.sqrt(1 + 4*np.pi**2)
ones = I[firing_nums == 1]

print(f"1c: predicted 1:1 tongue = [{Il:.4f}, {Ir:.4f}]")
print(f"1c: measured 1:1 plateau = [{ones.min():.4f}, {ones.max():.4f}], grid spacing = {I[1] - I[0]:.4f}")

def find_plateau(p, q):
    lo, hi = 0.9, 3

    for _ in range(25):
        I_val = (lo + hi)/2
        spikes = forced_lif(I_val, T=400)
        spikes = spikes[spikes >= 100]

        if len(spikes) > q and np.max(np.abs(spikes[q:] - spikes[:-q] - p)) <= 3e-4:
            return I_val

        if len(spikes)/300 < q/p:
            lo = I_val
        else:
            hi = I_val

plt.figure(figsize=(8, 4))
plt.plot(I, firing_nums, label='Forced Firing Number')
plt.plot(I, unforced_firing_nums, label='Unforced Firing Number')
plt.axvline(x=Il, color='k', linestyle='--')
plt.axvline(x=Ir, color='k', linestyle='--')
plt.axvspan(Il, Ir, alpha=0.3, color='red', label='1:1 Locking Tongue')

for p, q in [(2, 1), (3, 2), (2, 3), (1, 2)]:
    I_val = find_plateau(p, q)
    print(f"1c: plateau {q}/{p} at I = {I_val:.4f}")
    plt.plot(I_val, q/p, 'ko')
    plt.annotate(f"{q}/{p}", (I_val, q/p), xytext=(4, -12), textcoords='offset points')

plt.xlabel('I'); plt.ylabel('Spikes per forcing period')
plt.title(r'$\epsilon = 0.5$, $dt = 10^{-4}$, $T = 50$')
plt.legend()
plt.savefig('problem1_firing_number.pdf', bbox_inches='tight')
plt.close()

spikes = forced_lif(1.6)
phases = spikes[spikes >= t_start] % 1
print(f"1c: phase at I = 1.6: {phases.mean():.4f}")


#problem 2

#2c
tau, vr, vth, I = 1, 0, 1, 1.5
D = tau*np.log((I*tau - vr)/(I*tau - vth))
dv, dt = 1e-3, 1e-5
phi = np.linspace(0, 0.99, 100)

#time of the next spike with a optional kick
def lif_spike(kick=None):
    v, t, kicked = float(vr), 0.0, False

    while v < vth:
        if kick is not None and not kicked and t >= kick:
            v += dv
            kicked = True

        v_old = v
        v += dt*(-v/tau + I)
        t += dt

    return t - dt*(v - vth)/(v - v_old) 

def R_theory(phi, tau, D, I, vr):
    return tau*np.exp(phi*D/tau)/(D*(I*tau - vr))

T0 = lif_spike()  # unperturbed Euler period
Rs = np.array([(T0 - lif_spike(T0*p))/(T0*dv) for p in phi])

print(f"2c LIF: period = {D:.4f} (Euler: {T0:.4f})")
print(f"2c LIF: simulated R from {Rs.min():.4f} to {Rs.max():.4f}")
print(f"2c LIF: theory    R from {R_theory(0, tau, D, I, vr):.4f} to {R_theory(0.99, tau, D, I, vr):.4f}")

plt.plot(phi, Rs, 'o', markersize=3, label='Simulated R(phi)')
plt.plot(phi, R_theory(phi, tau, D, I, vr), label='Theoretical R(phi)')
plt.xlabel('phi'); plt.ylabel('R(phi)')
plt.title(r'LIF: $\tau = 1$, $I = 1.5$, $v_r = 0$, $v_{th} = 1$, $\delta = 10^{-3}$')
plt.legend()
plt.savefig('problem2_lif_prc.pdf', bbox_inches='tight')
plt.close()

#2c
def f(v, w):
    return v - v**3/3 - w + 0.5, 0.08*(v + 0.7 - 0.8*w)

dv, dt = 1e-3, 1e-4
n_cycles = 5 

v, w, t = 0.0, 0.0, 0.0
crossings = []

while len(crossings) < 10:
    dvdt, dwdt = f(v, w)
    v_new, w_new = v + dt*dvdt, w + dt*dwdt
    t += dt

    if v < 0 <= v_new:
        crossings.append(t)

    v, w = v_new, w_new

D = crossings[-1] - crossings[-2]  # period
v0, w0 = v, w  # phase zero

def nth_crossing(kick=None):
    v, w, t = v0, w0, 0.0
    kicked = False
    count = 0

    while True:
        if kick is not None and not kicked and t >= kick:
            v += dv
            kicked = True

        dvdt, dwdt = f(v, w)
        v_new, w_new = v + dt*dvdt, w + dt*dwdt

        if v < 0 <= v_new:
            count += 1
            if count == n_cycles:
                return t + dt*(-v)/(v_new - v) 

        v, w = v_new, w_new
        t += dt

T0 = nth_crossing() 
Rs = np.array([(T0 - nth_crossing(D*p))/(D*dv) for p in phi])

print(f"2c FHN: period = {D:.4f}")
print(f"2c FHN: max R = {Rs.max():.4f} at phi = {phi[Rs.argmax()]:.2f}")
print(f"2c FHN: min R = {Rs.min():.4f} at phi = {phi[Rs.argmin()]:.2f}")

#zeros of R by l.i.
for i in range(len(phi) - 1):
    if Rs[i]*Rs[i+1] < 0:
        print(f"2c FHN: R changes sign at phi = {phi[i] - Rs[i]*(phi[i+1] - phi[i])/(Rs[i+1] - Rs[i]):.4f}")

plt.plot(phi, Rs, 'o', markersize=3, label='Simulated R(phi)')
plt.axhline(0, color='k', linewidth=0.5)
plt.xlabel('phi'); plt.ylabel('R(phi)')
plt.title(r'FitzHugh-Nagumo: $\delta = 10^{-3}$, shift in 5th crossing')
plt.legend()
plt.savefig('problem2_fhn_prc.pdf', bbox_inches='tight')
plt.close()


#problem 4

#4c
def sim_gap_junction_QIF(eps, phi0=0.4, L=200, dt=1e-4, T=50):
    D = 2*np.arctan(L)  #uncoupled period w the cutoff
    v1 = -float(L)
    v2 = float(np.tan((1 - phi0)*D - np.arctan(L)))  
    spikes1, spikes2 = [0.0], []  

    for k in range(round(T/dt)):
        v1, v2 = v1 + dt*(1 + v1**2 + eps*(v2 - v1)), v2 + dt*(1 + v2**2 + eps*(v1 - v2))

        if v1 >= L:
            v1 = -float(L)
            spikes1.append((k + 1)*dt)
        if v2 >= L:
            v2 = -float(L)
            spikes2.append((k + 1)*dt)

    spikes1, spikes2 = np.array(spikes1), np.array(spikes2)
    n = min(len(spikes1) - 1, len(spikes2))
    phik = (spikes2[:n] - spikes1[:n])/(spikes1[1:n+1] - spikes1[:n])

    return spikes1[:n], phik

def phi_theory(phi0, eps, t):
    return (1/np.pi)*np.arctan(np.tan(np.pi*phi0)*np.exp(-2*eps*t))

def decay_func(t, A, b):
    return A*np.exp(-b*t)

phi0 = 0.4
t = np.linspace(0, 50, 1000)

tk, phik = sim_gap_junction_QIF(0.02)
print(f"4c: first measured lag at eps = 0.02: {phik[0]:.4f}")
plt.plot(t, phi_theory(phi0, 0.02, t), 'k--', label='Theoretical Phase Difference')
plt.plot(tk, phik, 'ro', label='Simulated Phase Difference')
plt.xlabel('Time'); plt.ylabel('Phase Difference')
plt.title(r'$\epsilon = 0.02$, $L = 200$, $\phi_0 = 0.4$, $dt = 10^{-4}$')
plt.legend()

plt.savefig('problem4_lag.pdf', bbox_inches='tight')
plt.close()

for eps in [0.005, 0.02, 0.05]:
    tk, phik = sim_gap_junction_QIF(eps)
    (A, b), _ = opt.curve_fit(decay_func, tk, np.tan(np.pi*phik), p0=[np.tan(np.pi*phi0), 2*eps])
    print(f"4c: eps = {eps}: 2 eps = {2*eps:.4g}, fitted decay rate = {b:.4g}")

    line, = plt.plot(tk, np.tan(np.pi*phik), 'o', markersize=3, label=fr'$\epsilon = {eps}$')
    plt.plot(tk, decay_func(tk, A, b), color=line.get_color())

plt.xlabel('Time'); plt.ylabel(r'$\tan(\pi\phi_k)$')
plt.title(r'Fits $Ae^{-bt}$: $L = 200$, $dt = 10^{-4}$, $T = 50$')
plt.legend()

plt.savefig('problem4_decay.pdf', bbox_inches='tight')
plt.close()