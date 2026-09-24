import numpy as np
import matplotlib.pyplot as plt
from hh_sim import hh_sim

#PROBLEM 4

R = 1.0
tau_m = 10.0
tau_s = 2.0
u_rest = 0.0
u_th = 1.0


def peak_time(tm, ts):
    return (tm * ts / (ts - tm)) * np.log(ts / tm)


def transient_voltage(t, I0, tm=tau_m, ts=tau_s):
    return (
        R * I0 * ts / (ts - tm)
        * (np.exp(-t / ts) - np.exp(-t / tm))
    )


def threshold_amplitude(ts, tm=tau_m):
    r = np.asarray(ts, dtype=float) / tm

    #we can evaluate r^(1/(r-1)) in a numericaly safe way
    val = np.empty_like(r, dtype=float)

    close = np.isclose(r, 1.0)

    val[close] = np.e
    val[~close] = np.exp(
        np.log(r[~close]) / (r[~close] - 1)
    )

    answer = ((u_th - u_rest) / R) * val
    if answer.ndim == 0:
        return float(answer)

    return answer


def lif_euler(I0, tfinal=40.0, dt=0.001):
    """
    Threshold is used only as the spike criterion.
    """
    t = np.arange(0, tfinal + dt, dt)
    u = np.zeros(len(t))
    u[0] = u_rest

    for j in range(len(t) - 1):
        current = I0 * np.exp(-t[j] / tau_s)

        du = (
            -(u[j] - u_rest) + R * current
        ) / tau_m

        u[j + 1] = u[j] + dt * du

    return t, u


tp = peak_time(tau_m, tau_s)
Icrit = threshold_amplitude(tau_s)

print("Problem 4")
print(f"t_peak  = {tp:.4f} ms")
print(f"I0_crit = {Icrit:.4f}")
print()


#we pick one value below, one at, and one above threshold
scales = [0.75, 1.00, 1.25]

print(
    f"{'I0/Icrit':>10} "
    f"{'I0':>10} "
    f"{'tpeak num':>12} "
    f"{'upeak num':>12} "
    f"{'upeak exact':>14}"
)

voltage_runs = []

for s in scales:
    I0 = s * Icrit

    t, u = lif_euler(I0)

    imax = np.argmax(u)

    t_num = t[imax]
    u_num = u[imax]
    u_exact = transient_voltage(tp, I0)

    voltage_runs.append((s, I0, t, u))

    print(
        f"{s:10.2f} "
        f"{I0:10.4f} "
        f"{t_num:12.4f} "
        f"{u_num:12.4f} "
        f"{u_exact:14.4f}"
    )


#voltage plot:

plt.figure(figsize=(7.5, 4.8))

for s, I0, t, u in voltage_runs:
    plt.plot(
        t,
        u,
        label=rf"$I_0={s:.2f}I_0^{{crit}}$"
    )

plt.axhline(
    u_th,
    linestyle="--",
    label=r"$u_{th}=1$ mV"
)

plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title(
    r"LIF response to a decaying synaptic current"
    "\n"
    r"$\tau_m=10$ ms, $\tau_s=2$ ms"
)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig("problem4_voltage.pdf")
plt.close()


#critical amplitude versus tau_s

ts_values = np.logspace(
    np.log10(0.5),
    np.log10(100),
    300
)

critical_values = threshold_amplitude(ts_values)

plt.figure(figsize=(7.2, 4.8))

plt.loglog(
    ts_values,
    critical_values,
    label=r"$I_0^{crit}$"
)

plt.axhline(
    (u_th - u_rest) / R,
    linestyle="--",
    label="constant-current limit"
)

#short pulse approximation
short_limit = (
    (u_th - u_rest) / R
    * tau_m / ts_values
)

plt.loglog(
    ts_values,
    short_limit,
    linestyle=":",
    label=r"$I_0^{crit}\sim \tau_m/\tau_s$"
)

plt.xlabel(r"$\tau_s$ (ms)")
plt.ylabel(r"$I_0^{crit}$ (mA)")
plt.title(
    r"Critical synaptic amplitude, "
    r"$\tau_m=10$ ms, $R=1\,\Omega$"
)
plt.legend()
plt.grid(alpha=0.25, which="both")
plt.tight_layout()

plt.savefig("problem4_Icrit.pdf")
plt.close()


#PROBLEM 5

# We use the transient removal already built into hh_sim.
# skipfrac = 0.4 means the first 40% of the trace is ignored.
#
# Since hh_sim returns a positive rate only when there are
# at least two post-transient spikes, we use that as our
# numerical criterion for repetitive firing.

Tsim = 300.0
skip = 0.40


def hh_response(Id, gK=36.0):
    return hh_sim(
        Id=Id,
        gK=gK,
        tmax=Tsim,
        skipfrac=skip
    )


def repeats(Id, gK):
    _, _, spikes, _ = hh_response(Id, gK)
    return len(spikes) >= 2


def locate_rheobase(gK, left, right, tolerance=1e-4):
    """
    Bisection assuming:
        left  -> no repetitive firing
        right -> repetitive firing
    """

    while (right - left) > tolerance:

        test_current = (left + right) / 2

        if repeats(test_current, gK):
            right = test_current
        else:
            left = test_current

    return right


#5a

rheo36 = locate_rheobase(
    gK=36.0,
    left=6.0,
    right=6.5
)

print("\nProblem 5")
print(f"Rheobase, gK=36: {rheo36:.6f} uA/cm^2")
print(f"To three significant digits: {rheo36:.3g} uA/cm^2")


#currents slightly below and above rheobase
offset = 0.01

Iminus = rheo36 - offset
Iplus = rheo36 + offset

t1, V1, spk1, _ = hh_response(Iminus, 36.0)
t2, V2, spk2, _ = hh_response(Iplus, 36.0)

print(f"Below: {Iminus:.4f}, post-transient spikes = {len(spk1)}")
print(f"Above: {Iplus:.4f}, post-transient spikes = {len(spk2)}")


fig, ax = plt.subplots(
    2,
    1,
    figsize=(8, 6),
    sharex=True,
    sharey=True
)

ax[0].plot(t1, V1)
ax[1].plot(t2, V2)

ax[0].set_title(
    rf"Below rheobase: $I_d={Iminus:.4f}$"
)

ax[1].set_title(
    rf"Above rheobase: $I_d={Iplus:.4f}$"
)

for a in ax:
    a.set_xlim(0, Tsim)
    a.set_ylim(-20, 110)
    a.set_ylabel("V (mV)")
    a.grid(alpha=0.25)

ax[1].set_xlabel("Time (ms)")

fig.tight_layout()
fig.savefig("problem5_near_rheobase.pdf")
plt.close()

# 5b
currents36 = np.linspace(
    rheo36 + 0.001,
    2 * rheo36,
    50
)

rates36 = []

for Id in currents36:
    _, _, _, rate = hh_response(Id, 36.0)
    rates36.append(rate)

rates36 = np.asarray(rates36)

onset_rate = rates36[0]

print(f"\nOnset firing rate = {onset_rate:.3f} Hz")


plt.figure(figsize=(7.2, 4.8))

plt.plot(
    currents36,
    rates36,
    "o-",
    markersize=3,
    label=r"$g_K=36$ mS/cm$^2$"
)

plt.axvline(
    rheo36,
    linestyle="--",
    label=rf"rheobase = {rheo36:.3f}"
)

plt.xlabel(r"$I_d$ ($\mu$A/cm$^2$)")
plt.ylabel("Firing rate (Hz)")
plt.title(r"HH $f-I$ curve")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig("problem5_fI.pdf")
plt.close()

#5c

rheo30 = locate_rheobase(
    gK=30.0,
    left=2.0,
    right=3.0
)

print("\nRheobase comparison")
print("------------------------------")
print(" gK (mS/cm^2)    Rheobase")
print(f"      36          {rheo36:.3f}")
print(f"      30          {rheo30:.3f}")


#we can use the same upper current range

currents30 = np.linspace(
    rheo30 + 0.001,
    Imax,
    50
)

rates30 = []

for Id in currents30:
    _, _, _, rate = hh_response(Id, 30.0)
    rates30.append(rate)

rates30 = np.asarray(rates30)


plt.figure(figsize=(7.5, 5))

plt.plot(
    currents36,
    rates36,
    "o-",
    markersize=3,
    label=r"$g_K=36$ mS/cm$^2$"
)

plt.plot(
    currents30,
    rates30,
    "s-",
    markersize=3,
    label=r"$g_K=30$ mS/cm$^2$"
)

plt.axvline(
    rheo36,
    linestyle="--",
    label=rf"$I_{{rh}}={rheo36:.3f}$, $g_K=36$"
)

plt.axvline(
    rheo30,
    linestyle=":",
    label=rf"$I_{{rh}}={rheo30:.3f}$, $g_K=30$"
)

plt.xlabel(r"$I_d$ ($\mu$A/cm$^2$)")
plt.ylabel("Firing rate (Hz)")
plt.title(
    r"Effect of $g_K$ on the HH $f-I$ curve"
)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig("problem5_fI_comparison.pdf")
plt.close()