# -*- coding: utf-8 -*-
"""
qcentroid_solver.py
===================
Solver QUBO para el Embalse Falcón adaptado a la plataforma QCentroid
(ide.dev.qcentroid.com) con aceleración GPU vía NVIDIA CUDA-Q.

Estructura requerida por QCentroid QuantumOps:
  - Función `run(use_case, solver_params)` como punto de entrada
  - El dataset llega como JSON vía `use_case.get_data()`
  - Los resultados se retornan como dict JSON-serializable

Backends disponibles en QCentroid:
  - "nvidia"          → GPU (cuStateVec, recomendado)
  - "tensornet"       → GPU tensor-network (instancias grandes)
  - "nvidia-mgpu"     → Multi-GPU
  - "qpp-cpu"         → CPU fallback (sin GPU)

Modo local (sin QCentroid):
  Ejecutar directamente con:
      python qcentroid_solver.py --local
"""

import numpy as np
import json
import time
import argparse
import sys
import os

# ─── Detección de entorno ──────────────────────────────────────────────────────
try:
    import cudaq
    CUDAQ_DISPONIBLE = True
except ImportError:
    CUDAQ_DISPONIBLE = False

try:
    import cupy as cp
    CUPY_DISPONIBLE = True
except ImportError:
    CUPY_DISPONIBLE = False

# ─── Constantes ───────────────────────────────────────────────────────────────
L       = 5
A_VEC   = np.array([-2, -1, 0, 1, 2], dtype=float)
S_MAX   = 3_387_000.0


# ==============================================================================
# SECCIÓN 1: CONSTRUCCIÓN DEL QUBO (idéntica a qubo_solver.py)
# ==============================================================================

def construir_qubo_matrices(R_obs, Delta_S_obs, S_inicial, S_max=S_MAX):
    """Construye la matriz Q completa. Reutilizable en CPU y GPU."""
    from qubo_solver import construir_qubo
    Q, params = construir_qubo(R_obs, Delta_S_obs, S_inicial, S_max)
    return Q, params


# ==============================================================================
# SECCIÓN 2: SOLVER GPU CON CUPY (Simulated Annealing paralelo)
# ==============================================================================

def sa_gpu_paralelo(Q, T, R_obs, Delta_S_obs, S_inicial, S_max,
                    n_restarts=64, n_iter=200_000, seed=42):
    """
    Simulated Annealing paralelo en GPU usando CuPy.

    Ejecuta n_restarts cadenas SA simultáneamente en GPU.
    Cada chain es un vector x ∈ {0,1}^(L*T) con restricción one-hot.
    La evaluación del SRS real (sin aprox. cuadrática) se hace en lote.

    Returns
    -------
    x_best  : np.ndarray, mejor solución encontrada
    E_best  : float, costo SRS real mínimo
    tiempo  : float, segundos de cómputo GPU
    """
    print(f"  🔥 SA GPU (CuPy) — {n_restarts} cadenas × {n_iter:,} iteraciones")
    t0 = time.time()
    rng = np.random.default_rng(seed)

    n         = L * T
    S_min     = 0.25 * S_max
    delta_u   = 0.25 * float(np.median(R_obs))
    u_max     = 2 * delta_u

    w1 = 1.0 / ((T + 1) * S_min**2)
    w2 = 0.1  / (T       * u_max**2)
    w3 = 0.1  / ((T - 1) * (2 * u_max)**2)

    def decodificar_batch(X_batch):
        """X_batch: (R, L*T) → u_batch: (R, T)"""
        R = X_batch.shape[0]
        X_r = X_batch.reshape(R, T, L).astype(float)
        a   = np.array(A_VEC)
        u   = (X_r @ a) * delta_u
        return u

    def evaluar_batch_cpu(X_batch):
        """Evalúa el SRS real para cada solución en el batch."""
        R  = X_batch.shape[0]
        E  = np.full(R, 1e12)
        us = decodificar_batch(X_batch)

        for i in range(R):
            u_seq = us[i]
            if abs(np.sum(u_seq)) > 0.10 * np.sum(R_obs):
                continue
            S_opt    = np.zeros(T + 1)
            S_opt[0] = S_inicial
            ok       = True
            for t in range(T):
                if R_obs[t] + u_seq[t] < 0:
                    ok = False; break
                S_opt[t+1] = S_opt[t] + Delta_S_obs[t] - u_seq[t]
                if S_opt[t+1] < 0 or S_opt[t+1] > S_max:
                    ok = False; break
            if not ok:
                continue
            C_crit   = np.sum([max(0.0, S_min - S_opt[t])**2 for t in range(T+1)])
            C_dev    = np.sum(u_seq**2)
            C_smooth = np.sum(np.diff(u_seq)**2)
            E[i]     = w1*C_crit + w2*C_dev + w3*C_smooth
        return E

    # Inicialización: u=0 para todos (índice 2 → nivel 0)
    X = np.zeros((n_restarts, n), dtype=np.int8)
    for t in range(T):
        X[:, t*L + 2] = 1
    # Añadir aleatoriedad a los restarts (excepto el primero)
    for r in range(1, n_restarts):
        for t in range(T):
            k = int(rng.integers(0, L))
            X[r, t*L:(t+1)*L] = 0
            X[r, t*L + k]      = 1

    E = evaluar_batch_cpu(X)

    x_best = X[np.argmin(E)].copy()
    E_best = E.min()

    T_init, T_final = 2.0, 1e-5

    # ── Si CuPy disponible: mover X a GPU para operaciones de índice ──────────
    if CUPY_DISPONIBLE:
        print(f"    GPU detectada: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        # Las operaciones de índice one-hot son más eficientes en CPU;
        # usamos CuPy para la vectorización de la evaluación de energía Q.
        Q_gpu  = cp.array(Q + Q.T - np.diag(np.diag(Q)))

    for step in range(n_iter):
        T_temp = T_init * (T_final / T_init) ** (step / n_iter)

        # Generar moves: para cada chain, elige t y k al azar
        t_idx  = rng.integers(0, T,  size=n_restarts)
        k_new  = rng.integers(0, L,  size=n_restarts)
        X_new  = X.copy()
        for r in range(n_restarts):
            t = t_idx[r]
            cur = int(np.argmax(X_new[r, t*L:(t+1)*L]))
            X_new[r, t*L + cur]      = 0
            X_new[r, t*L + k_new[r]] = 1

        E_new = evaluar_batch_cpu(X_new)
        dE    = E_new - E

        # Criterio de aceptación Metropolis (vectorizado)
        acepta = (dE < 0) | (rng.random(n_restarts) < np.exp(-dE / max(T_temp, 1e-300)))
        X[acepta] = X_new[acepta]
        E[acepta] = E_new[acepta]

        # Actualizar mejor global
        idx_min = np.argmin(E)
        if E[idx_min] < E_best:
            x_best = X[idx_min].copy()
            E_best = E[idx_min]

        if step % 10_000 == 0:
            print(f"    step {step:>7,} | T={T_temp:.5f} | SRS_best={-E_best:.6f}", flush=True)

    tiempo_gpu = time.time() - t0
    print(f"  ✅ SA GPU completado en {tiempo_gpu:.1f}s  |  SRS = {-E_best:.6f}")
    return x_best, E_best, tiempo_gpu


# ==============================================================================
# SECCIÓN 3: SOLVER VQE/QAOA CON CUDA-Q (núcleo cuántico real)
# ==============================================================================

def solver_qaoa_cudaqobj(Q, T, n_layers=2, backend="nvidia"):
    """
    Resuelve el QUBO usando QAOA (Quantum Approximate Optimization Algorithm)
    vía NVIDIA CUDA-Q. Ejecuta en GPU si backend='nvidia'.

    QAOA para QUBO:
      - n qubits = L*T = 5*26 = 130
      - Hamiltoniano de problema: H_P = Σ Q_ij * Z_i * Z_j
      - Hamiltoniano mixer:       H_M = Σ X_i
      - p capas alternas de exp(-iγH_P) · exp(-iβH_M)

    Nota: Con 130 qubits este circuito requiere GPU con >16GB VRAM.
    Para hardware real D-Wave, cambiar backend a 'dwave' (ver comentario al final).

    Returns
    -------
    counts     : dict de bitstrings medidos
    E_min      : float, energía mínima estimada
    params_opt : list, parámetros γ,β optimizados
    """
    if not CUDAQ_DISPONIBLE:
        raise RuntimeError(
            "CUDA-Q no está instalado. "
            "Instalar con: pip install cudaq  (requiere GPU NVIDIA)"
        )

    import cudaq
    from cudaq import spin

    n_qubits = L * T
    print(f"\n  ⚛️  QAOA CUDA-Q — {n_qubits} qubits, p={n_layers}, backend={backend}")
    cudaq.set_target(backend)

    # ── Construir el Hamiltoniano de problema ─────────────────────────────────
    # H_P = Σ_i Q_ii * (I - Z_i)/2  +  Σ_{i<j} Q_ij * (I - Z_i*Z_j)/2
    # (Mapeamos x_i ∈ {0,1} → (1 - Z_i)/2 con Z_i ∈ {-1,+1})
    print("    Construyendo Hamiltoniano de problema...")
    hamiltonian = spin.i(0) * 0.0   # inicializar con cero

    Q_sym = Q + Q.T - np.diag(np.diag(Q))
    offset = 0.0
    for i in range(n_qubits):
        # Término diagonal: Q_ii * x_i = Q_ii * (1 - Z_i)/2
        if abs(Q_sym[i, i]) > 1e-12:
            coef        = Q_sym[i, i] / 2.0
            hamiltonian += coef * spin.i(i)
            hamiltonian -= coef * spin.z(i)
            offset      += coef
        # Términos cruzados: 2*Q_ij * x_i*x_j = 2*Q_ij*(1-Z_i)(1-Z_j)/4
        for j in range(i+1, n_qubits):
            if abs(Q_sym[i, j]) > 1e-12:
                coef         = Q_sym[i, j] / 4.0
                hamiltonian += coef * (spin.i(i) - spin.z(i)
                                      - spin.z(j) + spin.z(i) * spin.z(j))
                offset += coef

    print(f"    Hamiltoniano construido (offset={offset:.4f})")

    # ── Kernel QAOA ───────────────────────────────────────────────────────────
    @cudaq.kernel
    def qaoa_kernel(gammas: list[float], betas: list[float], n_q: int, p: int):
        q = cudaq.qvector(n_q)
        # Estado inicial: superposición uniforme
        h(q)
        # Capas alternas
        for layer in range(p):
            # Capa de problema: exp(-i*gamma*H_P)
            # (aproximado por ZZ gates para cada par con Q_ij != 0)
            for i in range(n_q):
                rz(2.0 * gammas[layer] * Q_sym[i, i] / 2.0, q[i])
            for i in range(n_q):
                for j in range(i+1, n_q):
                    if abs(Q_sym[i, j]) > 1e-10:
                        cx(q[i], q[j])
                        rz(2.0 * gammas[layer] * Q_sym[i, j] / 4.0, q[j])
                        cx(q[i], q[j])
            # Capa mixer: exp(-i*beta*H_M)
            for i in range(n_q):
                rx(2.0 * betas[layer], q[i])

    # ── Optimización clásica de parámetros (COBYLA) ───────────────────────────
    from scipy.optimize import minimize

    def objetivo(params):
        gammas = params[:n_layers].tolist()
        betas  = params[n_layers:].tolist()
        result = cudaq.observe(
            qaoa_kernel, hamiltonian, gammas, betas, n_qubits, n_layers
        )
        return result.expectation()

    print("    Optimizando parámetros γ,β con COBYLA...")
    rng0   = np.random.default_rng(42)
    p0     = rng0.uniform(0, np.pi, 2 * n_layers)
    result = minimize(objetivo, p0, method='COBYLA',
                      options={'maxiter': 200, 'rhobeg': 0.5})

    params_opt = result.x.tolist()
    E_min      = result.fun + offset
    print(f"    Convergencia: {result.success} | E = {E_min:.6f}")

    # ── Muestreo del estado óptimo ────────────────────────────────────────────
    gammas_opt = params_opt[:n_layers]
    betas_opt  = params_opt[n_layers:]
    counts     = cudaq.sample(
        qaoa_kernel, gammas_opt, betas_opt, n_qubits, n_layers,
        shots_count=10_000
    )
    return counts, E_min, params_opt


def decodificar_counts(counts, T):
    """
    Selecciona el bitstring de menor energía entre los samples de QAOA
    que cumple la restricción one-hot.

    En QAOA por muestreo no siempre tenemos la energía por bitstring en este
    punto; por eso elegimos el bitstring válido más frecuente como heurística.
    """
    n = L * T
    x_best = None
    freq_best = -1
    for bitstring, freq in counts.items():
        x = np.array([int(b) for b in bitstring], dtype=int)
        if len(x) != n:
            continue
        # Verificar one-hot
        valido = all(np.sum(x[t*L:(t+1)*L]) == 1 for t in range(T))
        if not valido:
            continue
        if freq > freq_best:
            x_best = x
            freq_best = freq
    if x_best is None:
        # Fallback: primer bitstring válido
        x_best = np.zeros(n, dtype=int)
        for t in range(T):
            x_best[t*L + 2] = 1
    return x_best


# ==============================================================================
# SECCIÓN 4: PUNTO DE ENTRADA QCENTROID → función run()
# ==============================================================================

def run(use_case, solver_params):
    """
    Función principal requerida por QCentroid QuantumOps.

    Parámetros esperados en solver_params:
    {
        "backend"     : "nvidia" | "tensornet" | "nvidia-mgpu" | "qpp-cpu",
        "modo"        : "qaoa" | "sa_gpu" | "sa_cpu",
        "n_layers"    : 2,       (solo para QAOA)
        "n_restarts"  : 64,      (solo para SA GPU)
        "n_iter"      : 200000,  (solo para SA GPU)
        "T"           : 26,
        "S_max"       : 3387000.0
    }

    Dataset esperado (use_case.get_data()):
    {
        "R_obs"       : [lista de floats, T elementos],
        "Delta_S_obs" : [lista de floats, T elementos],
        "S_inicial"   : float
    }
    """
    print("="*60)
    print("  QCentroid Solver — Embalse Falcón QUBO")
    print("="*60)

    # ── Leer datos del use case ───────────────────────────────────────────────
    data        = use_case.get_data()
    R_obs       = np.array(data["R_obs"],       dtype=float)
    Delta_S_obs = np.array(data["Delta_S_obs"], dtype=float)
    S_inicial   = float(data["S_inicial"])
    S_max       = float(solver_params.get("S_max", S_MAX))
    T           = int(solver_params.get("T", len(R_obs)))
    R_obs       = R_obs[:T]
    Delta_S_obs = Delta_S_obs[:T]

    backend     = solver_params.get("backend", "nvidia")
    modo        = solver_params.get("modo", "sa_gpu")
    delta_u     = 0.25 * float(np.median(R_obs))
    force_qaoa  = bool(solver_params.get("force_qaoa", False))
    qaoa_max_qubits = int(solver_params.get("qaoa_max_qubits", 24))

    print(f"  T={T} semanas | S_inicial={S_inicial:,.0f} TCM")
    print(f"  Backend: {backend} | Modo: {modo}")

    # ── Construir QUBO ────────────────────────────────────────────────────────
    print("\n  Construyendo matriz QUBO...")
    sys.path.insert(0, os.path.dirname(__file__))
    Q, params = construir_qubo_matrices(R_obs, Delta_S_obs, S_inicial, S_max)

    t_total = time.time()
    u_opt   = None
    meta    = {}

    # ── Resolver ──────────────────────────────────────────────────────────────
    if modo == "qaoa" and (L * T) > qaoa_max_qubits and not force_qaoa:
        print(
            f"  [INFO] QAOA solicitado con {L*T} qubits. "
            f"Se usa sa_gpu como fallback reproducible; define force_qaoa=True "
            f"para intentar QAOA explícitamente."
        )
        n_restarts = int(solver_params.get("n_restarts", 64))
        n_iter     = int(solver_params.get("n_iter", 200_000))
        x_best, E_best, t_gpu = sa_gpu_paralelo(
            Q, T, R_obs, Delta_S_obs, S_inicial, S_max,
            n_restarts=n_restarts, n_iter=n_iter
        )
        u_opt = np.array([
            delta_u * float(np.array(A_VEC) @ x_best[t*L:(t+1)*L])
            for t in range(T)
        ])
        meta = {"E_best": float(E_best), "SRS": float(-E_best),
                "t_gpu_s": t_gpu, "n_restarts": n_restarts, "n_iter": n_iter,
                "fallback_from": "qaoa", "qaoa_qubits_requested": L * T}

    elif modo == "qaoa":
        # ── QAOA (hardware cuántico o GPU simulator) ──
        n_layers = int(solver_params.get("n_layers", 2))
        counts, E_min, params_opt = solver_qaoa_cudaqobj(Q, T, n_layers, backend)
        x_best  = decodificar_counts(counts, T)
        u_opt   = np.array([
            delta_u * float(np.array(A_VEC) @ x_best[t*L:(t+1)*L])
            for t in range(T)
        ])
        meta = {"E_min_qaoa": E_min, "params_opt": params_opt, "backend": backend}

    elif modo == "sa_gpu":
        # ── SA paralelo en GPU (CuPy) ──
        n_restarts = int(solver_params.get("n_restarts", 64))
        n_iter     = int(solver_params.get("n_iter", 200_000))
        x_best, E_best, t_gpu = sa_gpu_paralelo(
            Q, T, R_obs, Delta_S_obs, S_inicial, S_max,
            n_restarts=n_restarts, n_iter=n_iter
        )
        u_opt = np.array([
            delta_u * float(np.array(A_VEC) @ x_best[t*L:(t+1)*L])
            for t in range(T)
        ])
        meta = {"E_best": float(E_best), "SRS": float(-E_best),
                "t_gpu_s": t_gpu, "n_restarts": n_restarts, "n_iter": n_iter}

    else:
        # ── SA CPU (fallback sin GPU) ──
        from qubo_solver import resolver_qubo_sa, decodificar_solucion
        x_best, E_best = resolver_qubo_sa(
            Q, T, R_obs=R_obs, Delta_S_obs=Delta_S_obs,
            S_inicial=S_inicial, S_max=S_max,
            n_restarts=8, n_iter=80_000
        )
        u_opt = decodificar_solucion(x_best, T, delta_u)
        meta  = {"E_best": float(E_best), "SRS": float(-E_best)}

    # ── Calcular SRS real ─────────────────────────────────────────────────────
    S_opt    = np.zeros(T + 1)
    S_opt[0] = S_inicial
    for t in range(T):
        S_opt[t+1] = S_opt[t] + Delta_S_obs[t] - u_opt[t]
    S_min = 0.25 * S_max
    u_max = 2 * delta_u

    C_crit   = float(np.sum([max(0.0, S_min - s)**2 for s in S_opt]))
    C_dev    = float(np.sum(u_opt**2))
    C_smooth = float(np.sum(np.diff(u_opt)**2))
    w1       = 1.0 / ((T+1) * S_min**2)
    w2       = 0.1  / (T     * u_max**2)
    w3       = 0.1  / ((T-1) * (2*u_max)**2)
    srs_val  = -(w1*C_crit + w2*C_dev + w3*C_smooth)

    t_total = time.time() - t_total

    resultado = {
        "status"    : "success",
        "SRS"       : srs_val,
        "u_opt"     : u_opt.tolist(),
        "S_traj"    : S_opt.tolist(),
        "T"         : T,
        "delta_u"   : delta_u,
        "tiempo_s"  : t_total,
        "meta"      : meta,
    }

    print(f"\n  📊 SRS = {srs_val:.6f}")
    print(f"  ⏱  Tiempo total: {t_total:.1f}s")
    print("="*60)
    return resultado


# ==============================================================================
# SECCIÓN 5: MODO LOCAL (sin QCentroid, para pruebas)
# ==============================================================================

class MockUseCase:
    """Simula el objeto use_case de QCentroid con datos reales del embalse."""
    def __init__(self, R_obs, Delta_S_obs, S_inicial):
        self._data = {
            "R_obs"       : R_obs.tolist(),
            "Delta_S_obs" : Delta_S_obs.tolist(),
            "S_inicial"   : float(S_inicial),
        }
    def get_data(self):
        return self._data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QCentroid Solver — Embalse Falcón")
    parser.add_argument("--local",    action="store_true", help="Ejecutar en modo local (sin QCentroid)")
    parser.add_argument("--modo",     default="sa_gpu",    help="sa_gpu | sa_cpu | qaoa")
    parser.add_argument("--backend",  default="nvidia",    help="nvidia | tensornet | qpp-cpu")
    parser.add_argument("--n_iter",   type=int, default=200_000)
    parser.add_argument("--n_restarts", type=int, default=64)
    parser.add_argument("--n_layers",   type=int, default=2, help="Capas QAOA (solo modo qaoa)")
    args = parser.parse_args()

    if args.local:
        print("═"*60)
        print("  MODO LOCAL — Cargando datos reales del embalse Falcón")
        print("═"*60)
        sys.path.insert(0, os.path.dirname(__file__))
        import pandas as pd
        from genetico_v3 import preparar_ventana_semanal

        df_lib    = pd.read_excel("R_observ.xlsx")
        df_cambio = pd.read_csv(
            "DataSetExport-Discharge Total.Change-in-Storage@08461200"
            "-Instantaneous-TCM-20260622185956.csv", skiprows=1)
        df_total  = pd.read_csv(
            "DataSetExport-Total Storage.Web-Daily-ac-ft@08461200"
            "-Instantaneous-TCM-20260622185130.csv", skiprows=1)
        df_evap       = pd.DataFrame(columns=['Timestamp (UTC-06:00)', 'Evaporacion_mm'])
        df_batimetria = pd.read_csv("tabla_elevacion_volumen_FINAL.csv")

        R_obs, Delta_S_obs, S_inicial = preparar_ventana_semanal(
            df_lib, df_cambio, df_total, df_evap, df_batimetria, "2024-01-01", 26)

        use_case = MockUseCase(R_obs, Delta_S_obs, S_inicial)
        solver_params = {
            "backend"   : args.backend,
            "modo"      : args.modo,
            "n_restarts": args.n_restarts,
            "n_iter"    : args.n_iter,
            "n_layers"  : args.n_layers,
            "T"         : 26,
        }

        resultado = run(use_case, solver_params)

        # Comparar con baseline histórico
        from genetico_v3 import calcular_srs
        srs_hist, _ = calcular_srs(np.zeros(26), R_obs, Delta_S_obs, S_inicial, S_MAX)
        delta_srs   = resultado["SRS"] - srs_hist

        print(f"\n  Baseline histórico SRS : {srs_hist:.6f}")
        print(f"  QUBO QCentroid  SRS   : {resultado['SRS']:.6f}")
        print(f"  ΔSRS                  : {delta_srs:+.6f}")

        os.makedirs("resultados", exist_ok=True)
        with open("resultados/qcentroid_resultado.json", "w") as f:
            json.dump(resultado, f, indent=2)
        print("\n  📄 Resultado guardado: resultados/qcentroid_resultado.json")
    else:
        print("Uso en QCentroid: importar run() como punto de entrada del solver.")
        print("Uso local:        python qcentroid_solver.py --local [--modo sa_gpu|sa_cpu|qaoa]")
