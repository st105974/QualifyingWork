# dsmc_couette_parallel_convergence.py
from __future__ import annotations
import argparse, math, time as _time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange, set_num_threads, get_num_threads

KB = 1.380649e-23
PI = math.pi


@njit
def _rayleigh_speed(sig: float) -> float:
    r = np.random.random()
    if r < 1e-16:
        r = 1e-16
    return math.sqrt(-2.0 * sig * sig * math.log(r))


@njit(parallel=True, fastmath=True)
def streaming_reflect(y, vx, vy, dt, H, U1, U2, T_wall, m):
    sig = math.sqrt(KB * T_wall / m)
    n = y.shape[0]
    for i in prange(n):
        yi = y[i] + vy[i] * dt
        if yi < 0.0:
            yi = -yi
            vx[i] = np.random.normal(U1, sig)
            vy[i] = _rayleigh_speed(sig)
        elif yi > H:
            yi = 2.0 * H - yi
            vx[i] = np.random.normal(U2, sig)
            vy[i] = -_rayleigh_speed(sig)
        y[i] = yi



@njit
def bin_particles(y, dy, n_cells):
    n = y.size
    cell = np.empty(n, np.int32)
    counts = np.zeros(n_cells, np.int32)

    for i in range(n):
        c = int(y[i] // dy)
        # предохраняемся от случайно вылетевших за стенки частиц
        if c < 0:
            c = 0
        elif c >= n_cells:
            c = n_cells - 1
        cell[i] = c # записываем номер ячейки в которую попала i-я частица
        counts[c] += 1 # увеличиваем количество частиц в этой ячейке

    starts = np.empty(n_cells + 1, np.int32)
    starts[0] = 0
    for c in range(n_cells):
        starts[c + 1] = starts[c] + counts[c]

    idxs = np.empty(n, np.int32)
    cur = starts[:-1].copy()
    for i in range(n):
        c = cell[i]
        pos = cur[c]
        idxs[pos] = i
        cur[c] = pos + 1

    return idxs, starts

@njit
def _gmax_in_cell(idxs, s, e, vx, vy) -> float:
    cnt = e - s
    pairs = cnt * (cnt - 1) // 2
    samp = pairs if pairs < 200 else 200
    if samp < 10:
        samp = pairs
    gmax = 0.0
    for _ in range(samp):
        ia = idxs[s + np.random.randint(0, cnt)]
        ib = idxs[s + np.random.randint(0, cnt)]
        if ia == ib:
            continue
        dvx = vx[ia] - vx[ib]
        dvy = vy[ia] - vy[ib]
        g = math.hypot(dvx, dvy)
        if g > gmax:
            gmax = g
    return gmax * 1.1  # небольшой запас

@njit
def _post_HS(g):
    phi = 2.0 * PI * np.random.random()
    return g * math.cos(phi), g * math.sin(phi)

@njit
def _post_VSS(dvx, dvy, g, alpha):
    if g <= 0.0:
        return 0.0, 0.0

    # базис: n вдоль dv, o — ортогональный ей
    nx = dvx / g; ny = dvy / g
    ox = -ny;     oy = nx

    # закон VSS по дефлекционному углу
    r = np.random.random()
    cos_chi = 1.0 - r**(1.0/(alpha + 1.0))
    if cos_chi > 1.0: cos_chi = 1.0
    if cos_chi < -1.0: cos_chi = -1.0
    sin_chi = math.sqrt(max(0.0, 1.0 - cos_chi*cos_chi))

    # в 2D "азимут" — это просто сторона: плюс или минус по o
    s = 1.0 if np.random.random() < 0.5 else -1.0

    vrx = g * (cos_chi*nx + s*sin_chi*ox)
    vry = g * (cos_chi*ny + s*sin_chi*oy)
    return vrx, vry


@njit(parallel=True, fastmath=True)
def collisions(y, vx, vy, dy, dt, n_cells,
               sigma_ref, omega, alpha, g_ref,
               F_over_Vcell):
    idxs, starts = bin_particles(y, dy, n_cells)

    for c in prange(n_cells):
        s = starts[c]
        e = starts[c + 1]
        cnt = e - s
        if cnt < 2:
            continue

        gmax = _gmax_in_cell(idxs, s, e, vx, vy)
        if gmax <= 0.0:
            continue

        if omega == 1.0 and alpha == 0.0:
            sigma_max = sigma_ref
        else:
            sigma_max = sigma_ref * (gmax / g_ref) ** (2.0 * (1.0 - omega))

        pairs = cnt * (cnt - 1) // 2
        expected = pairs * sigma_max * gmax * F_over_Vcell * dt
        Npairs = int(expected)
        if np.random.random() < (expected - Npairs):
            Npairs += 1

        for _ in range(Npairs):
            ia = idxs[s + np.random.randint(0, cnt)]
            ib = idxs[s + np.random.randint(0, cnt)]
            if ia == ib:
                continue

            dvx = vx[ia] - vx[ib]
            dvy = vy[ia] - vy[ib]
            g = math.hypot(dvx, dvy)
            if g <= 0.0:
                continue

            if omega == 1.0 and alpha == 0.0:
                p = g / gmax
            else:
                p = (g / gmax) ** (2.0 * (1.0 - omega) + 1.0)

            if np.random.random() < p:
                if omega == 1.0 and alpha == 0.0:
                    vrx, vry = _post_HS(g)
                else:
                    vrx, vry = _post_VSS(dvx, dvy, g, alpha)
                vcmx = 0.5 * (vx[ia] + vx[ib])
                vcmy = 0.5 * (vy[ia] + vy[ib])
                vx[ia] = vcmx + 0.5 * vrx
                vy[ia] = vcmy + 0.5 * vry
                vx[ib] = vcmx - 0.5 * vrx
                vy[ib] = vcmy - 0.5 * vry


@njit(parallel=True)
def sample_snapshot(y, vx, vy, dy, n_cells):
    """
    Возвращает моментные суммы для ОДНОГО среза:
    sum_vx, sum_vy, sum_v2, count (все по ячейкам)
    """
    idxs, starts = bin_particles(y, dy, n_cells)
    sum_vx = np.zeros(n_cells)
    sum_vy = np.zeros(n_cells)
    sum_v2 = np.zeros(n_cells)
    count = np.zeros(n_cells)

    for c in prange(n_cells):
        s = starts[c]; e = starts[c+1]
        cnt = e - s
        if cnt == 0:
            continue
        svx = 0.0; svy = 0.0; sv2 = 0.0
        # для адекватной параллельности
        for k in range(s, e):
            i = idxs[k]
            vx_i = vx[i]; vy_i = vy[i]
            svx += vx_i
            svy += vy_i
            sv2 += vx_i*vx_i + vy_i*vy_i
        sum_vx[c] = svx
        sum_vy[c] = svy
        sum_v2[c] = sv2
        count[c] = cnt

    return sum_vx, sum_vy, sum_v2, count

# ============================
#  Driver
# ============================

def run_dsmc(
    N_PART, N_CELLS, H, TWALL, U1, U2, PRESSURE,
    SIGMA, omega, alpha, model,
    N_STEPS, SAVE_EVERY, DT_SAFETY, m,
    threads, tol, conv_window
):
    if threads is not None:
        set_num_threads(max(1, threads if threads > 0 else get_num_threads()))

    dy = H / N_CELLS
    n_real = PRESSURE / (KB * TWALL) # концентрация молекул
    mean_free_path = KB * TWALL / (math.sqrt(2.0) * PI * SIGMA * PRESSURE)
    vth = math.sqrt(2.0 * KB * TWALL / m)
    dt = DT_SAFETY * mean_free_path / vth
    F_over_Vcell = (n_real * H / N_PART) / dy  # F_over_Vcell показывает сколько реальных частиц приходится на 1 супермолекулу в данной ячейке

    y  = np.random.rand(N_PART) * H
    sig = math.sqrt(KB * TWALL / m)
    vx = np.random.normal(0.0, sig, N_PART)
    vy = np.random.normal(0.0, sig, N_PART)

    if model.upper() == "HS":
        omega_use, alpha_use = 1.0, 0.0
    else:
        omega_use, alpha_use = omega, alpha
    g_ref = math.sqrt(2.0 * KB * TWALL / m)

    # аккумуляторы для финального средне-временного профиля
    cum_vx = np.zeros(N_CELLS)
    cum_vy = np.zeros(N_CELLS)
    cum_v2 = np.zeros(N_CELLS)
    cum_cnt = np.zeros(N_CELLS)

    # диагностика стационара
    n_samples = N_STEPS // SAVE_EVERY
    times = np.zeros(n_samples)
    res_u = np.full(n_samples, np.nan)
    res_T = np.full(n_samples, np.nan)
    Tglob = np.zeros(n_samples)

    # временные ряды (скорость/температура во времени)
    u_center_ts = np.zeros(n_samples)
    u_avg_ts    = np.zeros(n_samples)
    T_center_ts = np.zeros(n_samples)

    prev_u = None
    prev_T = None

    lam = mean_free_path; tau_col = lam / (math.sqrt(2.0) * sig)
    if dy > lam:
        print(f"[warn] Δy={dy:.3e} > λ={lam:.3e} (увеличьте ncells)")
    if dt > 0.2 * tau_col:
        print(f"[warn] dt={dt:.3e} > 0.2τ≈{0.2*tau_col:.3e} (уменьшите dt_safety)")

    start = _time.time()
    s = 0
    for it in range(N_STEPS):
        streaming_reflect(y, vx, vy, dt, H, U1, U2, TWALL, m)
        collisions(y, vx, vy, dy, dt, N_CELLS, SIGMA,
                   omega_use, alpha_use, g_ref, F_over_Vcell)

        if (it + 1) % SAVE_EVERY == 0:
            svx, svy, sv2, cnt = sample_snapshot(y, vx, vy, dy, N_CELLS)

            with np.errstate(divide='ignore', invalid='ignore'): # строим массивы с локальными средними
                ux = np.where(cnt > 0, svx / cnt, 0.0)
                uy = np.where(cnt > 0, svy / cnt, 0.0)
                E2 = np.where(cnt > 0, sv2 / cnt, 0.0)
                Tcur = m * (E2 - ux*ux - uy*uy) / (2.0 * KB)

            # остатки
            if prev_u is not None:
                mask = cnt > 0
                if mask.any():
                    du = ux[mask] - prev_u[mask]
                    dT = Tcur[mask] - prev_T[mask]
                    denom_u = np.linalg.norm(prev_u[mask]) + 1e-30
                    denom_T = np.linalg.norm(prev_T[mask]) + 1e-30
                    res_u[s] = np.linalg.norm(du) / denom_u
                    res_T[s] = np.linalg.norm(dT) / denom_T

            prev_u = ux.copy()
            prev_T = Tcur.copy()

            # глобальные метрики
            wsum = cnt.sum() + 1e-30
            Tglob[s] = np.dot(Tcur, cnt) / wsum


            # накопление для средне-временных профилей
            cum_vx += svx
            cum_vy += svy
            cum_v2 += sv2
            cum_cnt += cnt

            # временные ряды: центр и доменное среднее
            c_mid = N_CELLS // 2
            u_center_ts[s] = ux[c_mid] if cnt[c_mid] > 0 else 0.0
            T_center_ts[s] = Tcur[c_mid] if cnt[c_mid] > 0 else 0.0
            u_avg_ts[s]    = np.dot(ux, cnt) / wsum

            times[s] = (it + 1) * dt
            s += 1

    elapsed = _time.time() - start
    print(f"[done] steps={N_STEPS}, samples={n_samples}, dt={dt:.2e}s, sim={elapsed:.2f}s, threads={get_num_threads()}")

    # средне-временные профили
    with np.errstate(divide='ignore', invalid='ignore'):
        ux_mean = np.where(cum_cnt > 0, cum_vx / cum_cnt, 0.0)
        uy_mean = np.where(cum_cnt > 0, cum_vy / cum_cnt, 0.0)
        E2_mean = np.where(cum_cnt > 0, cum_v2 / cum_cnt, 0.0)
        T_mean = m * (E2_mean - ux_mean*ux_mean - uy_mean*uy_mean) / (2.0 * KB)

    # простая детекция стационара
    conv_hit = False
    if n_samples >= conv_window + 1:
        tail_u = res_u[-conv_window:]
        tail_T = res_T[-conv_window:]
        if np.all(np.isfinite(tail_u)) and np.all(np.isfinite(tail_T)):
            if (tail_u < tol).all() and (tail_T < tol).all():
                conv_hit = True
    if conv_hit:
        t_when = times[-1]
        print(f"[converged] residuals stayed below tol={tol:g} for last {conv_window} samples (t≈{t_when:.3e} s)")
    else:
        print(f"[note] residuals last window: ru={res_u[-conv_window:]}  rT={res_T[-conv_window:]}  (tol={tol})")

    y_centers = (np.arange(N_CELLS) + 0.5) * dy
    return (y_centers, ux_mean, T_mean, lam, dt,
            times, res_u, res_T, Tglob,
            u_center_ts, u_avg_ts, T_center_ts)

def plot_profiles(yc, u_mean, T_mean, U1, U2, H, Twall, m, lam, dt):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    # скорость (аналитика no-slip)
    u_an = U1 + (U2 - U1) * (yc / H)
    ax[0].plot(yc, u_mean, 'o', label='DSMC (time-avg)')
    ax[0].plot(yc, u_an, '-', label='Analytic (no-slip)')
    ax[0].set(xlabel='y (m)', ylabel='u_x (m/s)',
              title=f'Velocity (Kn≈{lam/H:.3f}, dt={dt:.2e}s)')
    ax[0].grid(True); ax[0].legend()

    # температура (ориентир — классика вязкостного нагрева)
    mu_over_k = 4.0 * m / (15.0 * KB)
    A = 0.5 * mu_over_k * ((U2 - U1) / H) ** 2
    T_an = Twall + A * yc * (H - yc)
    ax[1].plot(yc, T_mean, 's', label='DSMC (time-avg)')
    ax[1].plot(yc, T_an, '-', label='Analytic viscous')
    ax[1].set(xlabel='y (m)', ylabel='T (K)',
              title=f'Temperature (Kn≈{lam/H:.3f})')
    ax[1].grid(True); ax[1].legend()
    plt.tight_layout(); plt.show()

def plot_convergence(times, res_u, res_T, Tglob, tol):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
    # остатки
    ax[0].semilogy(times, res_u, '-o', ms=3, label='||Δu||/||u||')
    ax[0].semilogy(times, res_T, '-s', ms=3, label='||ΔT||/||T||')
    ax[0].axhline(tol, ls='--', color='k', lw=1, label=f'tol={tol:g}')
    ax[0].set(xlabel='time (s)', ylabel='residual (relative)',
              title='Convergence Residuals')
    ax[0].grid(True, which='both', ls=':')
    ax[0].legend()

    # глобальные величины
    ax[1].plot(times, Tglob, '-', label='Global T (2D)')
    ax[1].set(xlabel='time (s)', ylabel='value',
              title='Global Diagnostics')
    ax[1].grid(True); ax[1].legend()
    plt.tight_layout(); plt.show()

def plot_time_series(times, u_center, u_avg, T_center, Tglob):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))

    # скорость во времени
    ax[0].plot(times, u_center, '-', label='u_center (y=H/2)')
    ax[0].plot(times, u_avg,   '-', label='u_avg (domain)')
    ax[0].set(xlabel='time (s)', ylabel='u_x (m/s)', title='Velocity vs time')
    ax[0].grid(True); ax[0].legend()

    # температура во времени
    ax[1].plot(times, T_center, '-', label='T_center (y=H/2)')
    ax[1].plot(times, Tglob,    '-', label='T_global (2D)')
    ax[1].set(xlabel='time (s)', ylabel='T (K)', title='Temperature vs time')
    ax[1].grid(True); ax[1].legend()

    plt.tight_layout(); plt.show()


def cli():
    p = argparse.ArgumentParser("DSMC Couette (parallel, HS/VSS) + convergence + time series")
    p.add_argument('--npart', type=int, default=120000)
    p.add_argument('--ncells', type=int, default=200)
    p.add_argument('--H', type=float, default=0.1)
    p.add_argument('--twall', type=float, default=300.0)
    p.add_argument('--u1', type=float, default=0.0)
    p.add_argument('--u2', type=float, default=200.0)
    p.add_argument('--pressure', type=float, default=9.0)
    p.add_argument('--sigma', type=float, default=1e-19)
    p.add_argument('--omega', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=0.0)
    p.add_argument('--model', choices=['HS','VSS'], default='HS')
    p.add_argument('--nsteps', type=int, default=120000)
    p.add_argument('--save_every', type=int, default=100)
    p.add_argument('--dt_safety', type=float, default=0.1)
    p.add_argument('--mass', type=float, default=4.65e-26)
    p.add_argument('--threads', type=int, default=0, help='0 = all; >0 set explicit; <0 keep default')
    # критерии стационара
    p.add_argument('--tol', type=float, default=1e-3, help='residual tolerance for convergence')
    p.add_argument('--conv_window', type=int, default=5, help='number of last samples to check residuals')
    args = p.parse_args()

    threads = None if args.threads < 0 else args.threads
    (yc, u_mean, T_mean, lam, dt,
     times, res_u, res_T, Tglob,
     u_center_ts, u_avg_ts, T_center_ts) = run_dsmc(
        N_PART=args.npart, N_CELLS=args.ncells, H=args.H,
        TWALL=args.twall, U1=args.u1, U2=args.u2,
        PRESSURE=args.pressure, SIGMA=args.sigma,
        omega=args.omega, alpha=args.alpha, model=args.model,
        N_STEPS=args.nsteps, SAVE_EVERY=args.save_every,
        DT_SAFETY=args.dt_safety, m=args.mass,
        threads=threads, tol=args.tol, conv_window=args.conv_window
    )

    # 1) усреднённые профили без «зелёных точек»
    plot_profiles(yc, u_mean, T_mean, args.u1, args.u2, args.H, args.twall, args.mass, lam, dt)
    # 2) сходимость
    plot_convergence(times, res_u, res_T, Tglob, args.tol)
    # 3) скорость/температура во времени
    plot_time_series(times, u_center_ts, u_avg_ts, T_center_ts, Tglob)

if __name__ == "__main__":
    cli()
