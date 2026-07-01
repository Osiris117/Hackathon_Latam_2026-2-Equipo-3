# -*- coding: utf-8 -*-
"""
qubo_solver.py
==============
Construcción de la matriz QUBO completa para el Falcon Reservoir Challenge
y su resolución por simulated annealing clásico.

Basado en qubo_formulation_notes.md (secciones 2-11).

Puede usarse como módulo:
    from qubo_solver import construir_qubo, resolver_qubo_sa, decodificar_solucion

O como script standalone:
    python qubo_solver.py
"""

import numpy as np
from typing import Tuple

# =========================================================
# CONSTANTES DE CODIFICACIÓN
# =========================================================
A_VEC = np.array([-2, -1, 0, 1, 2], dtype=float)  # coeficientes one-hot
L     = 5   # niveles benchmark oficial


# =========================================================
# BLOQUES AUXILIARES
# =========================================================
def _bloque_B() -> np.ndarray:
    """B upper-triangular: diagonal=aₖ², off-diag=2·aₖ·aₗ (QUBO notes §4,§10)"""
    B = np.zeros((L, L))
    for k in range(L):
        B[k, k] = A_VEC[k]**2
        for l in range(k+1, L):
            B[k, l] = 2 * A_VEC[k] * A_VEC[l]
    return B


def _bloque_C() -> np.ndarray:
    """C = −2·A_outer: término cruzado entre pasos consecutivos (QUBO notes §5)"""
    C = np.zeros((L, L))
    for k in range(L):
        for l in range(L):
            C[k, l] = -2 * A_VEC[k] * A_VEC[l]
    return C


# =========================================================
# 1. Q_onehot
# =========================================================
def q_onehot(T: int, lam: float) -> np.ndarray:
    """Penaliza (Σ_k x_{t,k} − 1)² (QUBO notes §3)"""
    n = L * T
    Q = np.zeros((n, n))
    for t in range(T):
        base = t * L
        for k in range(L):
            Q[base+k, base+k] += lam * (-1.0)
            for l in range(k+1, L):
                Q[base+k, base+l] += lam * 2.0
    return Q


# =========================================================
# 2. Q_dev
# =========================================================
def q_dev(T: int, delta_u: float, w2: float) -> np.ndarray:
    """C_dev = Σ u(t)² (QUBO notes §4)"""
    n    = L * T
    Q    = np.zeros((n, n))
    B    = _bloque_B()
    coef = w2 * delta_u**2
    for t in range(T):
        base = t * L
        Q[base:base+L, base:base+L] += coef * B
    return Q


# =========================================================
# 3. Q_smooth
# =========================================================
def q_smooth(T: int, delta_u: float, w3: float) -> np.ndarray:
    """C_smooth = Σ [u(t)−u(t-1)]² (QUBO notes §5)"""
    n    = L * T
    Q    = np.zeros((n, n))
    B    = _bloque_B()
    C    = _bloque_C()
    coef = w3 * delta_u**2
    for t in range(T):
        base = t * L
        mult = 2.0 if 0 < t < T-1 else 1.0
        Q[base:base+L, base:base+L] += coef * mult * B
        if t > 0:
            prev  = (t - 1) * L
            block = coef * C
            for k in range(L):
                for l in range(L):
                    r, c = prev + k, base + l
                    if r <= c:
                        Q[r, c] += block[k, l]
                    else:
                        Q[c, r] += block[k, l]
    return Q


# =========================================================
# 4. Q_crit
# =========================================================
def q_crit(T: int, delta_u: float, w1: float,
           Delta_S_obs: np.ndarray, S_inicial: float, S_min: float) -> np.ndarray:
    """
    Aproximación cuadrática de C_crit (QUBO notes §6).
    Nota: la eliminación del max(0,...) introduce mínimos espurios;
    el SA evalúa el SRS real para corregir esto.
    """
    n  = L * T
    Q  = np.zeros((n, n))
    A_t = np.zeros(T + 1)
    cs  = 0.0
    for t in range(T + 1):
        A_t[t] = S_min - S_inicial - cs
        if t < T:
            cs += Delta_S_obs[t]

    for tau in range(T):
        for k in range(L):
            idx_tk        = tau * L + k
            sum_A_future  = float(np.sum(A_t[tau+1:T+1]))
            diag_val      = w1 * (2 * delta_u * A_VEC[k] * sum_A_future
                                  + delta_u**2 * A_VEC[k]**2 * (T - tau))
            Q[idx_tk, idx_tk] += diag_val

            for sigma in range(tau, T):
                start_l = (k + 1) if sigma == tau else 0
                for l in range(start_l, L):
                    idx_sl = sigma * L + l
                    if idx_tk >= idx_sl:
                        continue
                    count   = T - max(tau, sigma)
                    off_val = 2 * w1 * delta_u**2 * A_VEC[k] * A_VEC[l] * count
                    Q[idx_tk, idx_sl] += off_val
    return Q


# =========================================================
# 5. Q_flujo  (R(t) ≥ 0)
# =========================================================
def q_flujo(T: int, R_obs: np.ndarray, delta_u: float, lam: float) -> np.ndarray:
    """Penaliza R(t)=Robs(t)+u(t)<0, solo diagonal (QUBO notes §7)"""
    n = L * T
    Q = np.zeros((n, n))
    for t in range(T):
        for k in range(L):
            if R_obs[t] + delta_u * A_VEC[k] < 0:
                Q[t*L + k, t*L + k] += lam
    return Q


# =========================================================
# 6. Q_balance  (|Σu| ≤ η·ΣRobs)
# =========================================================
def q_balance(T: int, delta_u: float, lam_bal: float) -> np.ndarray:
    """P_bal = λ·(Σu(t))² (QUBO notes §10)"""
    n    = L * T
    Q    = np.zeros((n, n))
    B    = _bloque_B()
    A    = np.outer(A_VEC, A_VEC)
    coef = lam_bal * delta_u**2
    for t in range(T):
        base = t * L
        Q[base:base+L, base:base+L] += coef * B
        for tau in range(t+1, T):
            base2 = tau * L
            for k in range(L):
                for l in range(L):
                    r, c = base + k, base2 + l
                    if r <= c:
                        Q[r, c] += 2 * coef * A[k, l]
                    else:
                        Q[c, r] += 2 * coef * A[k, l]
    return Q


# =========================================================
# FUNCIÓN PRINCIPAL: Construir Q total
# =========================================================
def construir_qubo(R_obs: np.ndarray,
                   Delta_S_obs: np.ndarray,
                   S_inicial: float,
                   S_max: float = 3_387_000.0,
                   eta: float = 0.10,
                   lam_onehot: float = None,
                   lam_R: float = None,
                   lam_bal: float = None) -> Tuple[np.ndarray, dict]:
    """
    Q = Q_dev + Q_smooth + Q_crit + Q_onehot + Q_flujo + Q_balance
    (QUBO notes §11)
    """
    T       = len(R_obs)
    S_min   = 0.25 * S_max
    delta_u = 0.25 * float(np.median(R_obs))
    u_max   = 2 * delta_u
    n       = L * T

    w1 = 1.0 / ((T + 1) * S_min**2)
    w2 = 0.1 / (T       * u_max**2)
    w3 = 0.1 / ((T - 1) * (2 * u_max)**2)

    scale_cost = w1 * S_min**2
    if lam_onehot is None:
        lam_onehot = max(10.0 * scale_cost, 1e-6)
    if lam_R is None:
        lam_R      = max(100.0 * scale_cost, 1e-6)
    if lam_bal is None:
        lam_bal    = max(5.0 * scale_cost, 1e-8)

    params = dict(T=T, L=L, S_min=S_min, S_max=S_max, delta_u=delta_u,
                  u_max=u_max, w1=w1, w2=w2, w3=w3,
                  lam_onehot=lam_onehot, lam_R=lam_R, lam_bal=lam_bal)

    print(f"  Construyendo QUBO {n}×{n} (T={T}, L={L})...")
    print(f"    Δu={delta_u:,.0f} TCM, umax={u_max:,.0f} TCM")
    print(f"    w1={w1:.2e}, w2={w2:.2e}, w3={w3:.2e}")
    print(f"    λ_onehot={lam_onehot:.2e}, λ_R={lam_R:.2e}, λ_bal={lam_bal:.2e}")

    Q = np.zeros((n, n))
    Q += q_dev(T, delta_u, w2);     print("    ✅ Q_dev")
    Q += q_smooth(T, delta_u, w3);  print("    ✅ Q_smooth")
    Q += q_crit(T, delta_u, w1, Delta_S_obs, S_inicial, S_min); print("    ✅ Q_crit")
    Q += q_onehot(T, lam_onehot);   print("    ✅ Q_onehot")
    Q += q_flujo(T, R_obs, delta_u, lam_R); print("    ✅ Q_flujo")
    Q += q_balance(T, delta_u, lam_bal);     print("    ✅ Q_balance")

    assert np.allclose(Q, np.triu(Q), atol=1e-10), "Q no es upper-triangular"
    print(f"    ✅ Q total ({n}×{n}) construida y verificada")
    return Q, params


# =========================================================
# DECODIFICACIÓN: x → u(t)
# =========================================================
def decodificar_solucion(x: np.ndarray, T: int, delta_u: float) -> np.ndarray:
    """u(t) = Δu · aᵀ · x_t  (QUBO notes §11)"""
    u_seq = np.zeros(T)
    for t in range(T):
        u_seq[t] = delta_u * float(A_VEC @ x[t*L:(t+1)*L])
    return u_seq


# =========================================================
# EVALUACIÓN SRS REAL (corrige la aproximación de max(0,...))
# =========================================================
def _srs_costo_onehot(x: np.ndarray, T: int, delta_u: float,
                       R_obs: np.ndarray, Delta_S_obs: np.ndarray,
                       S_inicial: float, S_max: float) -> float:
    """
    Costo SRS real = -SRS para una solución one-hot.
    Usa max(0,...) exacto. Penaliza infactibilidades con 1e12.
    """
    u_seq = decodificar_solucion(x, T, delta_u)
    S_min = 0.25 * S_max
    u_max = 2 * delta_u

    if abs(np.sum(u_seq)) > 0.10 * np.sum(R_obs):
        return 1e12

    S_opt    = np.zeros(T + 1)
    S_opt[0] = S_inicial
    for t in range(T):
        if R_obs[t] + u_seq[t] < 0:
            return 1e12
        S_opt[t+1] = S_opt[t] + Delta_S_obs[t] - u_seq[t]
        if S_opt[t+1] < 0 or S_opt[t+1] > S_max:
            return 1e12

    C_crit   = float(np.sum([max(0.0, S_min - S_opt[t])**2 for t in range(T+1)]))
    C_dev    = float(np.sum(u_seq**2))
    C_smooth = float(np.sum(np.diff(u_seq)**2))

    w1 = 1.0 / ((T+1) * S_min**2)
    w2 = 0.1  / (T    * u_max**2)
    w3 = 0.1  / ((T-1) * (2*u_max)**2)
    return w1*C_crit + w2*C_dev + w3*C_smooth


# =========================================================
# SOLVER: Simulated Annealing sobre SRS real
# =========================================================
def resolver_qubo_sa(Q: np.ndarray,
                     T: int,
                     R_obs: np.ndarray = None,
                     Delta_S_obs: np.ndarray = None,
                     S_inicial: float = 0.0,
                     S_max: float = 3_387_000.0,
                     n_restarts: int = 5,
                     n_iter: int = 50_000,
                     T_init: float = 1.0,
                     T_final: float = 1e-4,
                     seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    SA con moves one-hot.

    Si se pasan R_obs/Delta_S_obs, evalúa el SRS REAL (recomendado).
    Si no, usa xᵀQx (fallback educativo).
    """
    rng     = np.random.default_rng(seed)
    n       = L * T
    usa_srs = (R_obs is not None and Delta_S_obs is not None)
    delta_u = 0.25 * float(np.median(R_obs)) if usa_srs else 1.0
    Q_sym   = Q + Q.T - np.diag(np.diag(Q))

    def energia(x):
        if usa_srs:
            return _srs_costo_onehot(x, T, delta_u, R_obs, Delta_S_obs,
                                     S_inicial, S_max)
        return float(x @ Q_sym @ x)

    def inicializar_aleatorio():
        x = np.zeros(n, dtype=int)
        for t in range(T):
            x[t * L + int(rng.integers(0, L))] = 1
        return x

    def flip_onehot(x):
        x_new = x.copy()
        t     = int(rng.integers(0, T))
        cur   = int(np.argmax(x_new[t*L:(t+1)*L]))
        new_k = int(rng.integers(0, L))
        x_new[t*L + cur]   = 0
        x_new[t*L + new_k] = 1
        return x_new

    # Semilla factible: u=0 para todo t
    x_best = np.zeros(n, dtype=int)
    for t in range(T):
        x_best[t*L + 2] = 1
    E_best = energia(x_best)

    for restart in range(n_restarts):
        x = inicializar_aleatorio()
        E = energia(x)
        for step in range(n_iter):
            T_temp = T_init * (T_final / T_init) ** (step / n_iter)
            x_new  = flip_onehot(x)
            E_new  = energia(x_new)
            dE     = E_new - E
            if dE < 0 or rng.random() < np.exp(-dE / max(T_temp, 1e-300)):
                x, E = x_new, E_new
            if E < E_best:
                x_best, E_best = x.copy(), E

    return x_best, E_best


# =========================================================
# PIPELINE COMPLETO
# =========================================================
def optimizar_con_qubo(R_obs: np.ndarray,
                       Delta_S_obs: np.ndarray,
                       S_inicial: float,
                       S_max: float = 3_387_000.0,
                       n_restarts: int = 8,
                       n_iter: int = 80_000,
                       verbose: bool = True) -> np.ndarray:
    """
    Pipeline QUBO completo:
      1. Construye Q (educativo — matrices de QUBO notes)
      2. SA evaluando el SRS REAL (corrige aprox. max(0,...))
      3. Decodifica x_{t,k} → u(t)
    """
    T       = len(R_obs)
    delta_u = 0.25 * float(np.median(R_obs))

    if verbose:
        print("\n" + "═"*55)
        print("  ⚛️  SOLVER QUBO — Embalse Falcón")
        print("═"*55)

    Q, params = construir_qubo(R_obs, Delta_S_obs, S_inicial, S_max)

    if verbose:
        print(f"\n  SA con SRS real (n_iter={n_iter:,} × {n_restarts} restarts)...")

    x_best, E_best = resolver_qubo_sa(
        Q, T,
        R_obs=R_obs, Delta_S_obs=Delta_S_obs,
        S_inicial=S_inicial, S_max=S_max,
        n_restarts=n_restarts, n_iter=n_iter, seed=42
    )
    u_qubo = decodificar_solucion(x_best, T, delta_u)

    if verbose:
        print(f"  Costo SRS mínimo: {E_best:.6f}  →  SRS = {-E_best:.6f}")
        from collections import Counter
        dist = Counter(u_qubo.round(0))
        print("  Distribución de u(t):")
        for nivel, cnt in sorted(dist.items()):
            print(f"    u={nivel:>8.0f}: {cnt} semanas")

    return u_qubo


# =========================================================
# EJECUCIÓN STANDALONE
# =========================================================
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import pandas as pd
    from genetico_v3 import (preparar_ventana_semanal, run_genetico,
                              calcular_srs, auditar_restricciones)

    print("=" * 60)
    print("  QUBO SOLVER — Benchmark Oficial T=26")
    print("=" * 60)

    df_lib    = pd.read_excel("R_observ.xlsx")
    df_cambio = pd.read_csv(
        "DataSetExport-Discharge Total.Change-in-Storage@08461200"
        "-Instantaneous-TCM-20260622185956.csv", skiprows=1)
    df_total  = pd.read_csv(
        "DataSetExport-Total Storage.Web-Daily-ac-ft@08461200"
        "-Instantaneous-TCM-20260622185130.csv", skiprows=1)
    try:
        df_evap = pd.read_csv(
            "DataSetExport-Evaporation,accumltd.Daily Evaporation - mm@08461200"
            "-Instantaneous-mm-20260622185804.csv", skiprows=1)
    except FileNotFoundError:
        df_evap = pd.DataFrame(columns=['Timestamp (UTC-06:00)', 'Evaporacion_mm'])
    df_batimetria = pd.read_csv("tabla_elevacion_volumen_FINAL.csv")

    S_MAX = 3_387_000.0
    FECHA = "2024-01-01"
    R_obs, DS, S0 = preparar_ventana_semanal(
        df_lib, df_cambio, df_total, df_evap, df_batimetria, FECHA, 26)

    print("\nBaseline 2 — Genético:")
    u_gen, _, _ = run_genetico(R_obs, DS, S0, S_max=S_MAX,
                                n_pop=500, n_gen=200, verbose=False)

    print("\nSolución QUBO (SA + SRS real):")
    u_qubo = optimizar_con_qubo(R_obs, DS, S0, S_max=S_MAX,
                                 n_restarts=10, n_iter=100_000, verbose=True)

    T     = len(R_obs)
    S_min = 0.25 * S_MAX
    delta_u = 0.25 * np.median(R_obs)
    NIVELES = [-2*delta_u, -delta_u, 0.0, delta_u, 2*delta_u]

    u_hist = np.zeros(T)
    u_rule = np.zeros(T)
    S_r    = S0
    for t in range(T):
        u_rule[t] = -delta_u if S_r < S_min else 0.0
        S_r += DS[t] - u_rule[t]

    srs_hist, _ = calcular_srs(u_hist,  R_obs, DS, S0, S_MAX)
    srs_rule, _ = calcular_srs(u_rule,  R_obs, DS, S0, S_MAX)
    srs_gen,  _ = calcular_srs(u_gen,   R_obs, DS, S0, S_MAX)
    srs_qubo, _ = calcular_srs(u_qubo,  R_obs, DS, S0, S_MAX)

    print("\n" + "═"*65)
    print("  📊 TABLA BENCHMARK OFICIAL")
    print("═"*65)
    print(f"  {'Método':<35} {'SRS':>12}  {'ΔSRS':>12}")
    print(f"  {'-'*63}")
    for nombre, srs_val in [
        ('Baseline 0 — Histórico (u=0)',   srs_hist),
        ('Baseline 1 — Regla umbral',       srs_rule),
        ('Baseline 2 — Genético (DEAP)',    srs_gen),
        ('Cuántico   — QUBO + SA',          srs_qubo),
    ]:
        delta = srs_val - srs_hist
        marca = "—" if delta == 0 else (f"+{delta:.6f} ✅" if delta > 0 else f"{delta:.6f} ⚠️")
        print(f"  {nombre:<35} {srs_val:>12.6f}  {marca}")
    print("═"*65)

    auditar_restricciones(u_qubo, R_obs, DS, S0, S_MAX, NIVELES)
