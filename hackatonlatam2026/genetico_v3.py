# -*- coding: utf-8 -*-
"""
genetico_v3.py
==============
Algoritmo Genético (DEAP) para optimización de la operación del Embalse Falcón.

Puede usarse de dos formas:
  1. Como módulo importable:
       from genetico_v3 import preparar_ventana_semanal, mutar_nivel, run_genetico
  2. Como script standalone:
       python genetico_v3.py
"""

import os
import time
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy.interpolate import interp1d


# =========================================================
# FUNCIÓN AUXILIAR 1: Preparar ventana semanal
# =========================================================
def preparar_ventana_semanal(df_lib, df_cambio, df_total, df_evap, df_batimetria,
                              fecha_inicio_str, semanas):
    """
    Versión con Área Dinámica: Calcula la evaporación basándose en el volumen de la presa.

    Parámetros
    ----------
    df_lib            : DataFrame con columna 'Valor' (liberaciones observadas, m³/s)
    df_cambio         : DataFrame con columna 'Value (TCM)' (cambio de almacenamiento)
    df_total          : DataFrame con columna 'Value (TCM)' (almacenamiento total)
    df_evap           : DataFrame con columna 'Value (mm)' (evaporación acumulada diaria)
    df_batimetria     : DataFrame con columnas 'elevation_m' y 'volume_TCM'
    fecha_inicio_str  : str, fecha de inicio de la ventana (e.g. '2024-01-01')
    semanas           : int, número de semanas a extraer

    Retorna
    -------
    R_obs       : np.ndarray, liberaciones semanales (TCM)
    Delta_S_obs : np.ndarray, cambio semanal de almacenamiento (TCM)
    S_inicial   : float, almacenamiento al inicio de la ventana (TCM)
    """
    dataframes = [df_lib.copy(), df_cambio.copy(), df_total.copy(), df_evap.copy()]
    for i, df in enumerate(dataframes):
        df.columns = df.columns.str.strip()
        if 'Timestamp (UTC-06:00)' in df.columns:
            df.rename(columns={'Timestamp (UTC-06:00)': 'Fecha'}, inplace=True)
        if 'Value (mm)' in df.columns:
            df.rename(columns={'Value (mm)': 'Evaporacion_mm'}, inplace=True)
        if 'Fecha' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
            df = df.dropna(subset=['Fecha']).set_index('Fecha')
        dataframes[i] = df

    df_lib_p, df_cambio_p, df_total_p, df_evap_p = dataframes

    # Interpolador de área (Volumen TCM → Área km²)
    vol  = df_batimetria['volume_TCM'].values
    area = np.gradient(df_batimetria['volume_TCM'], df_batimetria['elevation_m'])
    f_area = interp1d(vol, area, kind='linear', fill_value="extrapolate")

    # Agrupación Semanal
    lib_semanal      = df_lib_p['Valor'].resample('W-SUN').mean() * 604.8
    cambio_semanal   = df_cambio_p['Value (TCM)'].resample('W-SUN').sum()
    total_semanal    = df_total_p['Value (TCM)'].resample('W-SUN').first()
    evap_semanal_mm  = df_evap_p['Evaporacion_mm'].resample('W-SUN').sum()

    # Evaporación Dinámica
    volumen_hist     = total_semanal.reindex(evap_semanal_mm.index).ffill()
    area_semanal     = f_area(volumen_hist)
    evap_semanal_tcm = evap_semanal_mm * area_semanal  # disponible si se necesita

    # Extracción de ventana
    fecha_inicio = pd.to_datetime(fecha_inicio_str)
    R_obs        = lib_semanal.loc[fecha_inicio:].head(semanas).values
    Delta_S_obs  = cambio_semanal.loc[fecha_inicio:].head(semanas).values
    S_inicial    = total_semanal.loc[:fecha_inicio].iloc[-1]

    return R_obs, Delta_S_obs, S_inicial


# =========================================================
# FUNCIÓN AUXILIAR 2: Mutación por niveles discretos
# =========================================================
def mutar_nivel(individuo, indpb, niveles_permitidos):
    """
    Operador de mutación: reemplaza cada gen con un nivel discreto aleatorio
    con probabilidad indpb.

    Parámetros
    ----------
    individuo         : lista DEAP Individual
    indpb             : float, probabilidad de mutar cada gen
    niveles_permitidos: list, valores discretos posibles para cada gen

    Retorna
    -------
    tuple(individuo,)  — formato requerido por DEAP
    """
    for i in range(len(individuo)):
        if random.random() < indpb:
            individuo[i] = random.choice(niveles_permitidos)
    return individuo,


# =========================================================
# FUNCIÓN PRINCIPAL: Ejecutar el Algoritmo Genético
# =========================================================
def run_genetico(R_obs, Delta_S_obs, S_inicial,
                 S_max=3_387_000.0,
                 n_pop=500, n_gen=150,
                 cxpb=0.7, mutpb=0.2, indpb=0.15,
                 verbose=True):
    """
    Ejecuta el Algoritmo Genético DEAP sobre una ventana temporal dada.

    Parámetros
    ----------
    R_obs       : np.ndarray, liberaciones observadas semanales (TCM)
    Delta_S_obs : np.ndarray, cambio semanal de almacenamiento (TCM)
    S_inicial   : float, almacenamiento inicial (TCM)
    S_max       : float, capacidad máxima de la presa (TCM)
    n_pop       : int, tamaño de la población
    n_gen       : int, número de generaciones
    cxpb        : float, probabilidad de cruce
    mutpb       : float, probabilidad de mutación por individuo
    indpb       : float, probabilidad de mutar cada gen
    verbose     : bool, mostrar estadísticas por generación

    Retorna
    -------
    mejor_secuencia : np.ndarray, vector óptimo u(t) de longitud T
    hof             : HallOfFame DEAP con el mejor individuo
    log             : Logbook con estadísticas de evolución
    """
    T            = len(R_obs)
    S_min        = 0.25 * S_max
    mediana_R    = np.median(R_obs)
    delta_u      = 0.25 * mediana_R
    u_max        = 2 * delta_u
    NIVELES      = [-2*delta_u, -delta_u, 0.0, delta_u, 2*delta_u]

    # ── DEAP: registrar tipos (seguro para reimportaciones) ──────────────
    if 'FitnessMin' in creator.__dict__:
        del creator.FitnessMin
    if 'Individual' in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_level", random.choice, NIVELES)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_level, n=T)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ── Función fitness (SRS) ─────────────────────────────────────────────
    def evaluar_srs(individuo):
        cromosoma_u = np.array(individuo)
        S_opt = np.zeros(T + 1)
        S_opt[0] = S_inicial

        if abs(np.sum(cromosoma_u)) > 0.10 * np.sum(R_obs):
            return (float('1e10'),)

        for t in range(T):
            S_opt[t+1] = S_opt[t] + Delta_S_obs[t] - cromosoma_u[t]
            if R_obs[t] + cromosoma_u[t] < 0:
                return (float('1e10'),)
            if S_opt[t+1] < 0 or S_opt[t+1] > S_max:
                return (float('1e10'),)

        C_crit  = np.sum([max(0, S_min - S_opt[t])**2 for t in range(T + 1)])
        C_dev   = np.sum(cromosoma_u**2)
        C_smooth = np.sum(np.diff(cromosoma_u)**2)

        w1 = 1   / ((T + 1) * (S_min**2))
        w2 = 0.1 / (T       * (u_max**2))
        w3 = 0.1 / ((T - 1) * ((2 * u_max)**2))
        return ((w1 * C_crit) + (w2 * C_dev) + (w3 * C_smooth),)

    toolbox.register("evaluate", evaluar_srs)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutar_nivel, indpb=indpb,
                     niveles_permitidos=NIVELES)
    toolbox.register("select", tools.selTournament, tournsize=4)

    # ── Población inicial (con línea base factible como semilla) ──────────
    pop = toolbox.population(n=n_pop)
    pop[0] = creator.Individual([0.0] * T)  # Línea base: sin ajustes

    hof   = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    if verbose:
        print(f"  Iniciando evolución: T={T} semanas, pop={n_pop}, gen={n_gen}")

    pop, log = algorithms.eaSimple(
        pop, toolbox,
        cxpb=cxpb, mutpb=mutpb, ngen=n_gen,
        stats=stats, halloffame=hof,
        verbose=verbose
    )

    mejor_secuencia = np.array(hof[0])

    if verbose:
        score = hof[0].fitness.values[0]
        print(f"\n  Mejor SRS: {score:.6f}")
        print(f"  Desviación acumulada: "
              f"{abs(np.sum(mejor_secuencia))/np.sum(R_obs)*100:.2f}% (límite 10%)")

    return mejor_secuencia, hof, log


# =========================================================
# FUNCIÓN DE MÉTRICAS OFICIALES: SRS y ΔSRS
# =========================================================
def calcular_srs(u_seq, R_obs, Delta_S_obs, S_inicial, S_max):
    """
    Calcula el Storage Resilience Score (SRS) oficial para una secuencia u(t).
    Fórmula exacta del PDF (§1, ec. 1-15).

    Parámetros
    ----------
    u_seq       : array-like, ajuste de liberación semanal u(t)
    R_obs       : array-like, liberación histórica observada (TCM)
    Delta_S_obs : array-like, cambio de almacenamiento semanal (TCM)
    S_inicial   : float, almacenamiento en t=0 (TCM)
    S_max       : float, capacidad total (TCM)

    Retorna
    -------
    srs    : float (más alto = mejor)
    S_opt  : np.ndarray, trayectoria de almacenamiento simulada
    """
    u_seq   = np.asarray(u_seq, dtype=float)
    T       = len(u_seq)
    S_min   = 0.25 * S_max
    delta_u = 0.25 * np.median(R_obs)
    u_max   = 2 * delta_u

    S_opt    = np.zeros(T + 1)
    S_opt[0] = S_inicial
    for t in range(T):
        S_opt[t + 1] = S_opt[t] + Delta_S_obs[t] - u_seq[t]

    C_crit   = float(np.sum([max(0.0, S_min - S_opt[t])**2 for t in range(T + 1)]))
    C_dev    = float(np.sum(u_seq**2))
    C_smooth = float(np.sum(np.diff(u_seq)**2))

    w1 = 1.0   / ((T + 1) * S_min**2)
    w2 = 0.1   / (T       * u_max**2)
    w3 = 0.1   / ((T - 1) * (2 * u_max)**2)

    srs = -(w1 * C_crit + w2 * C_dev + w3 * C_smooth)
    return srs, S_opt


def reportar_delta_srs(u_opt, R_obs, Delta_S_obs, S_inicial, S_max):
    """
    Compara 3 baselines vs. la solución optimizada y reporta ΔSRS.

    Baselines:
      0 — Replay histórico (u = 0)
      1 — Regla de umbral clásica (u = -Δu si S < Smin, 0 si S ≥ Smin)
      opt — Solución optimizada (genético / QUBO)
    """
    T       = len(R_obs)
    S_min   = 0.25 * S_max
    delta_u = 0.25 * np.median(R_obs)

    # ── Baseline 0: Replay histórico ────────────────────────
    u_hist             = np.zeros(T)
    srs_hist, S_hist   = calcular_srs(u_hist, R_obs, Delta_S_obs, S_inicial, S_max)

    # ── Baseline 1: Regla de umbral ─────────────────────────
    u_rule   = np.zeros(T)
    S_rule   = S_inicial
    for t in range(T):
        u_rule[t] = -delta_u if S_rule < S_min else 0.0
        S_rule   += Delta_S_obs[t] - u_rule[t]
    srs_rule, S_rule_traj = calcular_srs(u_rule, R_obs, Delta_S_obs, S_inicial, S_max)

    # ── Solución optimizada ──────────────────────────────────
    srs_opt, S_opt_traj = calcular_srs(u_opt,  R_obs, Delta_S_obs, S_inicial, S_max)

    print("\n" + "═"*60)
    print("  📊 BENCHMARK ΔSRS — REPORTE OFICIAL")
    print("═"*60)
    print(f"  {'Método':<30} {'SRS':>14}  {'ΔSRS vs Hist.':>14}")
    print(f"  {'-'*60}")
    print(f"  {'Baseline 0 — Histórico (u=0)':<30} {srs_hist:>14.6f}  {'—':>14}")
    print(f"  {'Baseline 1 — Regla umbral':<30} {srs_rule:>14.6f}  {srs_rule - srs_hist:>+14.6f}")
    print(f"  {'Optimizado (Genético)':<30} {srs_opt:>14.6f}  {srs_opt  - srs_hist:>+14.6f}")
    print(f"  {'-'*60}")
    delta_vs_hist = srs_opt - srs_hist
    delta_vs_rule = srs_opt - srs_rule
    print(f"\n  ΔSRS (opt vs Baseline-0 histórico): {delta_vs_hist:+.6f}")
    print(f"  ΔSRS (opt vs Baseline-1 umbral)   : {delta_vs_rule:+.6f}")
    if delta_vs_hist > 0:
        print("  ✅ El genético MEJORA el SRS respecto al histórico.")
    else:
        print("  ⚠️  El genético no supera al replay histórico.")
    if delta_vs_rule > 0:
        print("  ✅ El genético MEJORA el SRS respecto a la regla de umbral.")
    else:
        print("  ⚠️  El genético no supera la regla de umbral.")
    print("═"*60 + "\n")

    # Estadísticas de almacenamiento
    print("  Semanas bajo umbral crítico (25% NAMO):")
    print(f"    Histórico: {np.sum(S_hist  < S_min)} / {T+1} semanas")
    print(f"    Umbral:    {np.sum(S_rule_traj < S_min)} / {T+1} semanas")
    print(f"    Genético:  {np.sum(S_opt_traj  < S_min)} / {T+1} semanas")
    print()

    return {
        'srs_hist' : srs_hist,
        'srs_rule' : srs_rule,
        'srs_opt'  : srs_opt,
        'delta_vs_hist': delta_vs_hist,
        'delta_vs_rule': delta_vs_rule,
        'S_hist'   : S_hist,
        'S_rule'   : S_rule_traj,
        'S_opt'    : S_opt_traj,
    }


# =========================================================
# AUDITORÍA DE RESTRICCIONES (reutilizable)
# =========================================================
def auditar_restricciones(secuencia_u, R_obs, Delta_S_obs, S_inicial,
                           S_max, niveles_permitidos):
    """
    Audita la solución final contra todas las restricciones del benchmark.
    """
    print("\n" + "═"*55)
    print(" 🔎 AUDITORÍA DE RESTRICCIONES (COMPLIANCE CHECK)")
    print("═"*55)

    T     = len(secuencia_u)
    S_opt = np.zeros(T + 1)
    S_opt[0] = S_inicial
    fallos = 0

    niveles_validos = all(
        np.isclose(u, niveles_permitidos, atol=1e-5).any() for u in secuencia_u
    )
    print(f"[1] Niveles discretos válidos          : "
          f"{'✅ PASÓ' if niveles_validos else '❌ FALLÓ'}")
    if not niveles_validos:
        fallos += 1

    desviacion_total   = abs(np.sum(secuencia_u))
    volumen_historico  = np.sum(R_obs)
    limite_10          = 0.10 * volumen_historico
    balance_valido     = desviacion_total <= limite_10

    print(f"[2] Balance Acumulado (Límite 10%)     : "
          f"{'✅ PASÓ' if balance_valido else '❌ FALLÓ'}")
    print(f"    ├─ Desviación: {desviacion_total:,.1f} TCM")
    print(f"    └─ Límite 10%: {limite_10:,.1f} TCM")
    if not balance_valido:
        fallos += 1

    flujo_negativo = desborde = sequia = False
    for t in range(T):
        if R_obs[t] + secuencia_u[t] < 0:
            flujo_negativo = True
        S_opt[t+1] = S_opt[t] + Delta_S_obs[t] - secuencia_u[t]
        if S_opt[t+1] > S_max: desborde = True
        if S_opt[t+1] < 0:     sequia   = True

    print(f"[3] No-negatividad del flujo           : "
          f"{'❌ FALLÓ' if flujo_negativo else '✅ PASÓ'}")
    print(f"[4] Límite inferior del embalse (S≥0)  : "
          f"{'❌ FALLÓ' if sequia else '✅ PASÓ'}")
    print(f"[5] Límite superior (S≤S_max)          : "
          f"{'❌ FALLÓ' if desborde else '✅ PASÓ'}")
    if flujo_negativo: fallos += 1
    if sequia:         fallos += 1
    if desborde:       fallos += 1

    print("-" * 55)
    if fallos == 0:
        print("🚀 DICTAMEN: Solución 100% FACTIBLE.")
    else:
        print(f"⚠️  DICTAMEN: INVIABLE — {fallos} violaciones detectadas.")
    print("═"*55 + "\n")

    return {
        'factible': fallos == 0,
        'fallos': fallos,
        'niveles_validos': niveles_validos,
        'balance_valido': balance_valido,
        'flujo_negativo': flujo_negativo,
        'sequia': sequia,
        'desborde': desborde,
        'desviacion_total': desviacion_total,
        'limite_10': limite_10,
        'S_opt': S_opt,
    }


# =========================================================
# UTILIDADES DE DIAGNÓSTICO Y BENCHMARK EXACTO
# =========================================================
def calcular_parametros_oficiales(R_obs, S_max, T=None):
    """Calcula parámetros oficiales del benchmark para cualquier horizonte."""
    R_obs = np.asarray(R_obs, dtype=float)
    T = int(T or len(R_obs))
    S_min = 0.25 * S_max
    mediana_R = float(np.median(R_obs))
    delta_u = 0.25 * mediana_R
    u_max = 2 * delta_u
    w1 = 1.0 / ((T + 1) * S_min**2)
    w2 = 0.1 / (T * u_max**2)
    w3 = 0.1 / ((T - 1) * (2 * u_max)**2)
    niveles = np.array([-2, -1, 0, 1, 2], dtype=float) * delta_u
    return {
        'T': T,
        'S_min': S_min,
        'mediana_R': mediana_R,
        'delta_u': delta_u,
        'u_max': u_max,
        'w1': w1,
        'w2': w2,
        'w3': w3,
        'niveles': niveles,
    }


def visualizar_variables_calculadas(R_obs, Delta_S_obs, S_inicial, S_max,
                                    fecha_inicio=None):
    """Imprime un panel compacto con los parámetros derivados del benchmark."""
    p = calcular_parametros_oficiales(R_obs, S_max)
    print("\n" + "═"*65)
    print(" 📊 RESUMEN DE VARIABLES CALCULADAS")
    print("═"*65)
    print(f"Horizonte de planeación (T) : {p['T']} semanas")
    if fecha_inicio is not None:
        print(f"Fecha de inicio             : {fecha_inicio}")
    print("\n[ PARÁMETROS FÍSICOS ]")
    print(f"Capacidad máxima (S_max)    : {S_max:,.2f} TCM")
    print(f"Umbral crítico (S_min)      : {p['S_min']:,.2f} TCM")
    print(f"Volumen inicial (S_inicial) : {S_inicial:,.2f} TCM")
    print("\n[ ESPACIO DE DECISIÓN ]")
    print(f"Mediana R_obs               : {p['mediana_R']:,.2f} TCM/sem")
    print(f"Δu                          : {p['delta_u']:,.2f} TCM/sem")
    print(f"u_max                       : {p['u_max']:,.2f} TCM/sem")
    print("Niveles u(t)                : ["
          + ", ".join(f"{x:,.2f}" for x in p['niveles']) + "]")
    print("\n[ PESOS SRS OFICIALES ]")
    print(f"w1                          : {p['w1']:.12e}")
    print(f"w2                          : {p['w2']:.12e}")
    print(f"w3                          : {p['w3']:.12e}")
    print("\n[ PRIMERAS 5 SEMANAS ]")
    print(f"R_obs                       : {np.round(R_obs[:5], 2)}")
    print(f"ΔS_obs                      : {np.round(Delta_S_obs[:5], 2)}")
    print("═"*65 + "\n")
    return p


def resolver_benchmark_gurobi(R_obs, Delta_S_obs, S_inicial,
                              S_max=3_387_000.0,
                              eta=0.10,
                              time_limit=120,
                              mip_gap=0.01,
                              verbose=False):
    """
    Resuelve una referencia exacta MIQP con Gurobi si está instalado/licenciado.

    Es opcional: si Gurobi no está disponible, retorna None sin romper el pipeline.
    Mantiene la restricción oficial de balance |Σu| <= eta·ΣR_obs, no la variante
    más estricta Σ|u| <= eta·ΣR_obs del prototipo v7.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        print(f"  [INFO] Gurobi no disponible, se omite MIQP exacto: {exc}")
        return None

    R_obs = np.asarray(R_obs, dtype=float)
    Delta_S_obs = np.asarray(Delta_S_obs, dtype=float)
    T = len(R_obs)
    p = calcular_parametros_oficiales(R_obs, S_max, T)
    delta_u = p['delta_u']

    print("\nCalculando referencia exacta — Gurobi MIQP...")
    t0 = time.time()

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()
    model = gp.Model("falcon_release_miqp", env=env)
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap

    x = model.addVars(T, vtype=GRB.INTEGER, lb=-2, ub=2, name="x_level")
    u = model.addVars(T, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="u")
    S = model.addVars(T + 1, vtype=GRB.CONTINUOUS, lb=0, ub=S_max, name="S")
    deficit = model.addVars(T + 1, vtype=GRB.CONTINUOUS, lb=0, name="deficit")

    model.addConstr(S[0] == S_inicial, "initial_storage")
    model.addConstr(deficit[0] >= p['S_min'] - S[0], "deficit_0")
    for t in range(T):
        model.addConstr(u[t] == delta_u * x[t], f"release_adjustment_{t}")
        model.addConstr(S[t + 1] == S[t] + Delta_S_obs[t] - u[t],
                        f"storage_balance_{t}")
        model.addConstr(R_obs[t] + u[t] >= 0, f"nonnegative_release_{t}")
        model.addConstr(deficit[t + 1] >= p['S_min'] - S[t + 1],
                        f"deficit_{t + 1}")

    balance_limit = eta * float(np.sum(R_obs))
    model.addConstr(gp.quicksum(u[t] for t in range(T)) <= balance_limit,
                    "anti_hoarding_upper")
    model.addConstr(gp.quicksum(u[t] for t in range(T)) >= -balance_limit,
                    "anti_hoarding_lower")

    C_crit = gp.quicksum(deficit[t] * deficit[t] for t in range(T + 1))
    C_dev = gp.quicksum(u[t] * u[t] for t in range(T))
    C_smooth = gp.quicksum((u[t] - u[t - 1]) * (u[t] - u[t - 1])
                           for t in range(1, T))
    model.setObjective(p['w1'] * C_crit + p['w2'] * C_dev + p['w3'] * C_smooth,
                       GRB.MINIMIZE)
    model.optimize()

    if model.SolCount == 0:
        print(f"  [WARN] Gurobi no encontró solución factible. Estado: {model.Status}")
        return None

    u_seq = np.array([u[t].X for t in range(T)], dtype=float)
    srs, S_traj = calcular_srs(u_seq, R_obs, Delta_S_obs, S_inicial, S_max)
    elapsed = time.time() - t0
    print(f"  Tiempo Gurobi MIQP: {elapsed:.1f} s")
    print(f"  SRS Gurobi MIQP   : {srs:.6f}")
    return {
        'u': u_seq,
        'S': S_traj,
        'srs': srs,
        'objective': float(model.ObjVal),
        'status': int(model.Status),
        'gap': float(model.MIPGap) if hasattr(model, 'MIPGap') else None,
        'runtime_s': elapsed,
        'params': p,
    }


def graficar_convergencia(logbook, output_path='resultados/convergencia_genetico.png'):
    """Guarda la curva de convergencia del algoritmo genético."""
    if logbook is None:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    generaciones = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(generaciones, min_fitness, label="Mejor costo", lw=2.4, color="#38bdf8")
    ax.plot(generaciones, avg_fitness, label="Costo promedio", lw=1.4,
            color="#94a3b8", alpha=0.85)
    ax.set_title("Convergencia del Algoritmo Genético")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Costo SRS minimizado")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def graficar_comparativa_estrategias(R_obs, Delta_S_obs, S_inicial, S_max,
                                     estrategias,
                                     output_path='resultados/comparativa_estrategias.png'):
    """
    Grafica almacenamiento y decisiones u(t) para varias estrategias.

    estrategias: dict nombre -> secuencia u(t)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    T = len(R_obs)
    p = calcular_parametros_oficiales(R_obs, S_max, T)
    semanas_S = np.arange(T + 1)
    semanas_u = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=False,
        gridspec_kw={'height_ratios': [2.2, 1], 'hspace': 0.18}
    )

    colores = ["#cbd5e1", "#facc15", "#38bdf8", "#a78bfa", "#34d399", "#fb7185"]
    for idx, (nombre, u_seq) in enumerate(estrategias.items()):
        _, S_traj = calcular_srs(u_seq, R_obs, Delta_S_obs, S_inicial, S_max)
        color = colores[idx % len(colores)]
        ax1.plot(semanas_S, S_traj / 1000, label=nombre, lw=2.0, color=color)
        ax2.step(semanas_u, np.asarray(u_seq) / 1000, where='post',
                 label=nombre, lw=1.8, color=color)

    ax1.axhline(p['S_min'] / 1000, color="#ef4444", linestyle="--", lw=1.5,
                label="Umbral crítico 25%")
    ax1.axhspan(0, p['S_min'] / 1000, color="#ef4444", alpha=0.08)
    ax1.set_title("Comparativa de Estrategias — Almacenamiento")
    ax1.set_ylabel("Almacenamiento (miles TCM)")
    ax1.grid(True, linestyle="--", alpha=0.2)
    ax1.legend(fontsize=9)

    ax2.axhline(0, color="#64748b", lw=0.9)
    ax2.set_title("Decisiones de Ajuste u(t)")
    ax2.set_xlabel("Semana")
    ax2.set_ylabel("u(t) miles TCM/sem")
    ax2.grid(True, linestyle="--", alpha=0.2)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


# =========================================================
# EJECUCIÓN STANDALONE
# =========================================================
if __name__ == "__main__":
    print("Cargando datos...")
    df_lib        = pd.read_excel("R_observ.xlsx")
    df_cambio     = pd.read_csv(
        "DataSetExport-Discharge Total.Change-in-Storage@08461200"
        "-Instantaneous-TCM-20260622185956.csv", skiprows=1)
    df_total      = pd.read_csv(
        "DataSetExport-Total Storage.Web-Daily-ac-ft@08461200"
        "-Instantaneous-TCM-20260622185130.csv", skiprows=1)
    # Evaporación: creamos un DataFrame vacío si no está disponible
    try:
        df_evap = pd.read_csv(
            "DataSetExport-Evaporation,accumltd.Daily Evaporation - mm@08461200"
            "-Instantaneous-mm-20260622185804.csv", skiprows=1)
    except FileNotFoundError:
        print("  [INFO] Archivo de evaporación no encontrado, se omite.")
        df_evap = pd.DataFrame(columns=['Timestamp (UTC-06:00)', 'Evaporacion_mm'])
    df_batimetria = pd.read_csv("tabla_elevacion_volumen_FINAL.csv")

    # ── Instancia oficial Medium: T=26, L=5 ──────────────────────
    FECHA_INICIO = "2024-01-01"
    T_MEDIO      = 26       # Benchmark oficial (Medium instance)
    S_MAX        = 3_387_000.0

    print(f"\n{'='*60}")
    print(f"  BENCHMARK OFICIAL — Medium Instance (T={T_MEDIO}, L=5)")
    print(f"{'='*60}")

    R_obs, Delta_S_obs, S_inicial = preparar_ventana_semanal(
        df_lib, df_cambio, df_total, df_evap, df_batimetria,
        FECHA_INICIO, T_MEDIO
    )
    print(f"  Ventana: {FECHA_INICIO} → {T_MEDIO} semanas")
    print(f"  S_inicial: {S_inicial:,.0f} TCM")

    mejor_secuencia, hof, log = run_genetico(
        R_obs, Delta_S_obs, S_inicial, S_max=S_MAX,
        n_pop=500, n_gen=200, verbose=True
    )

    mediana_R = np.median(R_obs)
    delta_u   = 0.25 * mediana_R
    NIVELES   = [-2*delta_u, -delta_u, 0.0, delta_u, 2*delta_u]

    # ── Reporte ΔSRS oficial ───────────────────────────────────────
    resultados = reportar_delta_srs(
        mejor_secuencia, R_obs, Delta_S_obs, S_inicial, S_MAX
    )

    # ── Auditoría de restricciones ─────────────────────────────────
    auditar_restricciones(
        mejor_secuencia, R_obs, Delta_S_obs, S_inicial, S_MAX, NIVELES
    )

    # ── Instancia Large: T=52 (para análisis de escalabilidad) ────
    print(f"\n{'='*60}")
    print(f"  INSTANCIA LARGE (T=52) — Análisis de Escalabilidad")
    print(f"{'='*60}")
    R52, DS52, S52 = preparar_ventana_semanal(
        df_lib, df_cambio, df_total, df_evap, df_batimetria,
        FECHA_INICIO, 52
    )
    mejor_52, _, _ = run_genetico(
        R52, DS52, S52, S_max=S_MAX,
        n_pop=500, n_gen=150, verbose=False
    )
    reportar_delta_srs(mejor_52, R52, DS52, S52, S_MAX)
