import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

#PROBLEM 1
def passive_peak_time(x, D=1.0):
    return 0.25 * (np.sqrt(1.0 + 4.0 * x**2 / D) - 1.0)


def front_speed(a, D=1.0):
    return np.sqrt(D) * (1.0 - 2.0 * a) / np.sqrt(a * (1.0 - a))

D = 1.0
x_values = np.array([0.1, 1.0, 10.0])
peak_values = passive_peak_time(x_values, D)

print("\nPROBLEM 1")
print("---------")
print("Passive peak times for D = 1")
for x, tp in zip(x_values, peak_values):
    print(f"x = {x:5.1f} : t_peak = {tp:.6f}")

passive_speed = 2.0 * np.sqrt(D)
a_cross = (2.0 - np.sqrt(2.0)) / 4.0

print(f"\nPassive far-field speed = {passive_speed:.6f}")
print(f"Active/passive crossing a = {a_cross:.6f}")
print("Front stalls at a = 0.5")


#Front speed figure
a_grid = np.linspace(0.01, 0.99, 1000)
c_grid = front_speed(a_grid, D)

plt.figure(figsize=(7.2, 4.8))

plt.plot(
    a_grid,
    c_grid,
    linewidth=2,
    label=r"active front $c(a)$"
)

plt.axhline(
    passive_speed,
    linestyle="--",
    linewidth=1.3,
    label=r"passive speed $2\sqrt{D}$"
)

plt.axvline(
    a_cross,
    linestyle=":",
    linewidth=1.4,
    label=rf"crossing $a={a_cross:.4f}$"
)

plt.axvline(
    0.5,
    linestyle="-.",
    linewidth=1.2,
    label=r"stall $a=1/2$"
)

plt.xlabel(r"threshold $a$")
plt.ylabel(r"speed $c$")
plt.title(r"Active front speed for $D=1$")
plt.ylim(-10, 10)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig("problem1_front_speed.pdf")
plt.close()

#PROBLEM 2

def membrane_tau(C, gL, ge=0.0, gi=0.0):
    return C / (gL + ge + gi)

C = 200.0
gL = 10.0

tau_rest = membrane_tau(C, gL)
tau_bomb = membrane_tau(C, gL, ge=20.0, gi=20.0)

print("\nPROBLEM 2")
print("---------")
print(f"Resting time constant     = {tau_rest:.4f} ms")
print(f"Bombarded time constant   = {tau_bomb:.4f} ms")
print(f"Current-response gain factor = {gL/(gL + 40):.4f}")


#2(c)

V0 = 16.13
EN = 0.0
gbar = 1.0        #nS. only fixes the vertical scale


def magnesium_block(V, Mg):
    return 1.0 / (
        1.0
        + (Mg / 3.57) * np.exp(-V / V0)
    )


def nmda_current(V, Mg):
    B = magnesium_block(V, Mg)
    return gbar * B * (V - EN)


def nmda_slope_factor(V, Mg=1.0):
    """
    dI/dV = gbar * B * slope_factor.
    Since gbar*B > 0, the zero of dI/dV is the zero of this.
    """
    B = magnesium_block(V, Mg)

    return (
        1.0
        + (V - EN) * (1.0 - B) / V0
    )


V_star = brentq(
    lambda V: nmda_slope_factor(V, 1.0),
    -90.0,
    0.0
)

print(f"NMDA zero-slope voltage for Mg = 1 mM: {V_star:.6f} mV")


V = np.linspace(-90.0, 0.0, 600)

plt.figure(figsize=(7.3, 4.8))

for Mg in [0.0, 0.1, 1.0]:
    plt.plot(
        V,
        nmda_current(V, Mg),
        linewidth=2,
        label=rf"$[\mathrm{{Mg}}^{{2+}}]={Mg:g}$ mM"
    )

plt.axvline(
    V_star,
    linestyle="--",
    linewidth=1.2,
    label=rf"zero slope at {V_star:.3f} mV"
)

plt.xlabel(r"$V$ (mV)")
plt.ylabel(r"$I_N$ (pA), $\bar g=1$ nS")
plt.title(
    r"Voltage-dependent NMDA current, "
    r"$E_N=0$ mV, $V_0=16.13$ mV"
)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig("problem2_nmda_current.pdf")
plt.close()

#PROBLEM 3

prior_mean = 5.0
b_prior = 2.0

a_prior = prior_mean * b_prior

K = 30.0
T_obs = 3.0

mle = K / T_obs

post_shape = a_prior + K
post_rate = b_prior + T_obs

post_mode = (post_shape - 1.0) / post_rate
post_mean = post_shape / post_rate
post_sd = np.sqrt(post_shape) / post_rate

extra_time = 15.0

print("\nPROBLEM 3")
print("---------")
print(f"Prior shape a = {a_prior:.4f}")
print(f"MLE = {mle:.4f} Hz")
print(f"Posterior mode = {post_mode:.4f} Hz")
print(f"Posterior mean = {post_mean:.4f} Hz")
print(f"Posterior SD = {post_sd:.4f} Hz")
print(f"Additional time to halve SD = {extra_time:.4f} s")

#PROBLEM 4

tau_seconds = 0.020
w_balanced = 0.5
target_mean = 10.0
target_var = 25.0

rate_difference = (
    target_mean
    / (w_balanced * tau_seconds)
)

rate_sum = (
    2.0 * target_var
    / (w_balanced**2 * tau_seconds)
)

nu_E = 0.5 * (rate_sum + rate_difference)
nu_I = 0.5 * (rate_sum - rate_difference)

kappa2_bal = (
    tau_seconds
    * w_balanced**2
    * (nu_E + nu_I)
    / 2.0
)

kappa3_bal = (
    tau_seconds
    * w_balanced**3
    * (nu_E - nu_I)
    / 3.0
)

balanced_skew = (
    kappa3_bal
    / kappa2_bal**1.5
)

print("\nPROBLEM 4")
print("---------")
print(f"nu_E = {nu_E:.4f} Hz")
print(f"nu_I = {nu_I:.4f} Hz")
print(f"Balanced kappa_2 = {kappa2_bal:.6f}")
print(f"Balanced kappa_3 = {kappa3_bal:.6f}")
print(f"Balanced skewness = {balanced_skew:.6f}")


#4(d)

def sample_skewness(x):
    mean = np.mean(x)
    variance = np.mean((x - mean)**2)

    return (
        np.mean((x - mean)**3)
        / variance**1.5
    )


def simulate_shot_noise(
    q,
    mean_voltage=10.0,
    tau=20.0,
    dt=0.05,
    total_time=200000.0,
    burn_time=2000.0,
    seed=42,
):
    """
    q = nu*tau.

    Time is in ms, so nu is events/ms.

    Between event bins the voltage is allowed to decay exactly:
        V_new = exp(-dt/tau)*V_old + w*N

    where N is the number of Poisson events in the bin.
    """

    w = mean_voltage / q
    nu = q / tau

    n_steps = int(round(total_time / dt)) + 1
    burn_index = int(round(burn_time / dt))

    rng = np.random.default_rng(seed)

    event_counts = rng.poisson(
        nu * dt,
        size=n_steps - 1
    )

    voltage = np.zeros(n_steps)

    decay = np.exp(-dt / tau)

    for j in range(n_steps - 1):
        voltage[j + 1] = (
            decay * voltage[j]
            + w * event_counts[j]
        )

    steady = voltage[burn_index:]

    measured_mean = np.mean(steady)
    measured_var = np.var(steady)
    measured_skew = sample_skewness(steady)

    theory_mean = mean_voltage
    theory_var = mean_voltage**2 / (2.0 * q)
    theory_skew = (
        2.0 * np.sqrt(2.0)
        / (3.0 * np.sqrt(q))
    )

    # Downsample only for plotting so the PDF is lighter.
    plot_samples = steady[::10].copy()

    return {
        "q": q,
        "w": w,
        "nu": nu,
        "mean_theory": theory_mean,
        "mean_sim": measured_mean,
        "var_theory": theory_var,
        "var_sim": measured_var,
        "skew_theory": theory_skew,
        "skew_sim": measured_skew,
        "samples": plot_samples,
    }


q_values = [2.0, 20.0, 200.0]
shot_results = []

for index, q in enumerate(q_values):

    result = simulate_shot_noise(
        q=q,
        seed=42 + index
    )

    shot_results.append(result)


print("\nShot-noise table")
print(
    f"{'nu*tau':>8} "
    f"{'w':>8} "
    f"{'mean th':>10} "
    f"{'mean sim':>10} "
    f"{'var th':>10} "
    f"{'var sim':>10} "
    f"{'skew th':>10} "
    f"{'skew sim':>10}"
)

for r in shot_results:
    print(
        f"{r['q']:8.0f} "
        f"{r['w']:8.4f} "
        f"{r['mean_theory']:10.4f} "
        f"{r['mean_sim']:10.4f} "
        f"{r['var_theory']:10.4f} "
        f"{r['var_sim']:10.4f} "
        f"{r['skew_theory']:10.4f} "
        f"{r['skew_sim']:10.4f}"
    )


# Histograms requested only for nu*tau = 2 and 200

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.0, 4.2)
)

for ax, index in zip(axes, [0, 2]):

    r = shot_results[index]

    samples = r["samples"]
    mean_th = r["mean_theory"]
    var_th = r["var_theory"]

    std_th = np.sqrt(var_th)

    x = np.linspace(
        mean_th - 4.0 * std_th,
        mean_th + 5.0 * std_th,
        500
    )

    gaussian = (
        np.exp(
            -(x - mean_th)**2
            / (2.0 * var_th)
        )
        / np.sqrt(2.0 * np.pi * var_th)
    )

    ax.hist(
        samples,
        bins=100,
        density=True,
        alpha=0.55,
        label="shot-noise simulation"
    )

    ax.plot(
        x,
        gaussian,
        linewidth=2,
        label="matched Gaussian"
    )

    ax.set_xlabel(r"$V$ (mV)")
    ax.set_ylabel("density")

    ax.set_title(
        rf"$\nu\tau={r['q']:.0f}$, "
        rf"$w={r['w']:.3g}$ mV"
    )

    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

fig.suptitle(
    r"Shot noise vs. diffusion approximation, "
    r"$\tau=20$ ms, $\langle V\rangle=10$ mV"
)

fig.tight_layout()

fig.savefig(
    "problem4_shot_noise_histograms.pdf"
)

plt.close(fig)


#PROBLEM 5

def below_reset_theory(
    sig2,
    mu=1.0,
    theta=1.0
):
    return (
        sig2
        / (mu * theta)
        * (
            1.0
            - np.exp(
                -mu * theta / sig2
            )
        )
    )


def stationary_density(
    u,
    sig2,
    mu=1.0,
    theta=1.0
):

    u = np.asarray(u)

    nu = mu / theta

    p = np.zeros_like(
        u,
        dtype=float
    )

    below = u < 0.0

    inside = (
        (u >= 0.0)
        & (u <= theta)
    )

    p[below] = (
        nu / mu
        * (
            1.0
            - np.exp(
                -mu * theta / sig2
            )
        )
        * np.exp(
            mu * u[below] / sig2
        )
    )

    p[inside] = (
        nu / mu
        * (
            1.0
            - np.exp(
                mu
                * (u[inside] - theta)
                / sig2
            )
        )
    )

    return p


def simulate_noisy_integrator(
    sig2,
    mu=1.0,
    theta=1.0,
    dt=2.5e-4,
    total_time=4000.0,
    burn_time=200.0,
    sample_every=10,
    seed=42,
):

    n_steps = int(round(total_time / dt))
    burn_steps = int(round(burn_time / dt))

    rng = np.random.default_rng(seed)

    #Pre generate noise so the simulation is reproducible
    noise = rng.standard_normal(n_steps)

    u = 0.0
    spike_count = 0

    below_count = 0
    observation_count = 0

    n_plot = (
        (n_steps - burn_steps)
        // sample_every
        + 2
    )

    samples = np.empty(n_plot)
    sample_index = 0

    noise_scale = np.sqrt(
        2.0 * sig2 * dt
    )

    for j in range(n_steps):

        proposed = (
            u
            + mu * dt
            + noise_scale * noise[j]
        )

        if proposed >= theta:

            if j >= burn_steps:
                spike_count += 1

            u = 0.0

        else:
            u = proposed

        if j >= burn_steps:

            observation_count += 1

            if u < 0.0:
                below_count += 1

            if (
                (j - burn_steps)
                % sample_every
                == 0
            ):
                samples[sample_index] = u
                sample_index += 1

    samples = samples[:sample_index]

    measured_rate = (
        spike_count
        / (total_time - burn_time)
    )

    fraction_below = (
        below_count
        / observation_count
    )

    return (
        samples,
        measured_rate,
        fraction_below
    )


sig2_values = [0.25, 1.0, 4.0]

integrator_results = []

print("\nPROBLEM 5")
print("---------")
print(
    f"{'sig2':>8} "
    f"{'rate th':>10} "
    f"{'rate sim':>10} "
    f"{'below th':>10} "
    f"{'below sim':>10}"
)

for index, sig2 in enumerate(sig2_values):

    samples, rate_sim, below_sim = (
        simulate_noisy_integrator(
            sig2=sig2,
            seed=42 + index
        )
    )

    rate_theory = 1.0

    below_theory = below_reset_theory(
        sig2
    )

    integrator_results.append(
        {
            "sig2": sig2,
            "samples": samples,
            "rate_theory": rate_theory,
            "rate_sim": rate_sim,
            "below_theory": below_theory,
            "below_sim": below_sim,
        }
    )

    print(
        f"{sig2:8.2f} "
        f"{rate_theory:10.4f} "
        f"{rate_sim:10.4f} "
        f"{below_theory:10.4f} "
        f"{below_sim:10.4f}"
    )


# Stationary-density comparison

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12.5, 3.8)
)

for ax, r in zip(
    axes,
    integrator_results
):

    sig2 = r["sig2"]
    samples = r["samples"]

    #we can use the actual sampled range so the histogram remains normalized
    x_min = samples.min()

    bins = np.linspace(
        x_min,
        1.0,
        140
    )

    ax.hist(
        samples,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Euler--Maruyama"
    )

    x = np.linspace(
        x_min,
        1.0,
        700
    )

    ax.plot(
        x,
        stationary_density(
            x,
            sig2=sig2
        ),
        linewidth=2,
        label="theory"
    )

    ax.axvline(
        0.0,
        linestyle=":",
        linewidth=1
    )

    ax.set_xlabel(r"$u$")
    ax.set_title(
        rf"$\sigma^2={sig2:g}$"
    )

    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

axes[0].set_ylabel(r"$p(u)$")

fig.suptitle(
    r"Stationary density of the noisy integrator, "
    r"$\mu=\theta=1$"
)

fig.tight_layout()

fig.savefig(
    "problem5_stationary_density.pdf"
)

plt.close(fig)


print("\nCreated figures:")
print("  problem1_front_speed.pdf")
print("  problem2_nmda_current.pdf")
print("  problem4_shot_noise_histograms.pdf")
print("  problem5_stationary_density.pdf")