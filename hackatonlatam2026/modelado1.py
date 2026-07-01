# -*- coding: utf-8 -*-
"""
modelado1.py
============
Modelo Híbrido RNA + QUBO para la operación del Embalse Falcón.

Flujo:
  1. Carga o entrena la RNA (si el modelo .pkl no existe, entrena automáticamente)
  2. Reconstruye el histórico del embalse desde FECHA_INICIO
  3. Simula SEMANAS_PREDICCION semanas futuras usando la RNA como política
  4. Imprime el resumen de la simulación
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Importamos las funciones del módulo de entrenamiento
from entrenar_rna import (
    cargar_datos,
    entrenar_rna,
    predecir_rna,
    FEATURES,
    NAMO,
    S_MAX,
)
from genetico_v3 import (preparar_ventana_semanal, run_genetico,
                          calcular_srs, reportar_delta_srs, auditar_restricciones,
                          visualizar_variables_calculadas,
                          resolver_benchmark_gurobi,
                          graficar_convergencia,
                          graficar_comparativa_estrategias)
from qubo_solver import optimizar_con_qubo

# ==========================================================
# PARÁMETROS GENERALES
# ==========================================================

FECHA_INICIO          = "2024-01-01"
SEMANAS_ENTRENAMIENTO = 26   # Instancia oficial Medium
SEMANAS_PREDICCION    = 26

V_MAX = S_MAX
S_MIN = 0.25 * V_MAX

# ==========================================================
# CARGA O ENTRENAMIENTO DE LA RNA
# ==========================================================

print("="*60)
print(" MODELO HÍBRIDO DE OPERACIÓN DEL EMBALSE")
print("="*60)

MODEL_PATH  = "modelo_rna_genetico.pkl"
SCALER_PATH = "scaler_rna.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print("\nCargando RNA desde archivo guardado...")
    rna    = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("  RNA cargada correctamente.")
else:
    print("\nNo se encontró modelo guardado. Entrenando RNA...")
    df_train       = cargar_datos(semanas=260)
    rna, scaler, _ = entrenar_rna(df_train, guardar=True, verbose=True)
    print("  RNA entrenada y guardada.")

# ==========================================================
# CARGA DE DATOS HISTÓRICOS (ventana de FECHA_INICIO)
# ==========================================================

print("\nLeyendo archivos históricos...")

df_lib    = pd.read_excel("R_observ.xlsx")
df_cambio = pd.read_csv(
    "DataSetExport-Discharge Total.Change-in-Storage@08461200"
    "-Instantaneous-TCM-20260622185956.csv", skiprows=1)
df_total  = pd.read_csv(
    "DataSetExport-Total Storage.Web-Daily-ac-ft@08461200"
    "-Instantaneous-TCM-20260622185130.csv", skiprows=1)

# Evaporación: opcional — se ignora si no existe
try:
    df_evap = pd.read_csv(
        "DataSetExport-Evaporation,accumltd.Daily Evaporation - mm@08461200"
        "-Instantaneous-mm-20260622185804.csv", skiprows=1)
except FileNotFoundError:
    df_evap = pd.DataFrame(columns=['Timestamp (UTC-06:00)', 'Evaporacion_mm'])
    print("  [INFO] Evaporación no disponible, se omite.")

df_batimetria = pd.read_csv("tabla_elevacion_volumen_FINAL.csv")

# ==========================================================
# VENTANA HISTÓRICA (usando la función de genetico_v3)
# ==========================================================

R_obs, Delta_S_obs, S_inicial = preparar_ventana_semanal(
    df_lib, df_cambio, df_total, df_evap, df_batimetria,
    FECHA_INICIO, SEMANAS_ENTRENAMIENTO
)

parametros_oficiales = visualizar_variables_calculadas(
    R_obs, Delta_S_obs, S_inicial, V_MAX, fecha_inicio=FECHA_INICIO
)

# ==========================================================
# RECONSTRUCCIÓN DEL ESTADO HISTÓRICO DEL EMBALSE
# ==========================================================

S_hist = np.zeros(SEMANAS_ENTRENAMIENTO)
S = S_inicial
for i in range(SEMANAS_ENTRENAMIENTO):
    S_hist[i] = S
    S = S + Delta_S_obs[i]

df_hist = pd.DataFrame({
    "Fecha"  : pd.date_range(start=FECHA_INICIO, periods=SEMANAS_ENTRENAMIENTO, freq="W-SUN"),
    "R_obs"  : R_obs,
    "Delta_S": Delta_S_obs,
    "S"      : S_hist,
})

print("\nHistórico reconstruido correctamente.")
print(df_hist[["Fecha", "S"]].tail(3).to_string(index=False))

# Estado inicial de la simulación
S_actual    = float(S_hist[-1])
fecha_actual = df_hist["Fecha"].iloc[-1]

print(f"\nÚltimo estado histórico")
print(f"------------------------")
print(f"Fecha : {fecha_actual.date()}")
print(f"S(t)  : {S_actual:,.0f} TCM  ({S_actual/NAMO*100:.1f}% NAMO)")

# ==========================================================
# FECHAS FUTURAS
# ==========================================================

fechas_futuras = pd.date_range(
    start   = fecha_actual + pd.Timedelta(days=7),
    periods = SEMANAS_PREDICCION,
    freq    = "W-SUN",
)

# ==========================================================
# TIEMPOS DE CÓMPUTO
# ==========================================================
import time
tiempos = {}

# ==========================================================
# BASELINE 2: OPTIMIZACIÓN GENÉTICA (comparador clásico)
# ==========================================================

print("\nCalculando Baseline 2 — Genético (T=26)...")
_t0 = time.time()
u_genetico, _, log_genetico = run_genetico(
    R_obs, Delta_S_obs, S_inicial, S_max=V_MAX,
    n_pop=500, n_gen=200, verbose=False
)
tiempos['Genético (DEAP)'] = time.time() - _t0
print(f"  ⏱ Tiempo Genético: {tiempos['Genético (DEAP)']:.1f} s")

# ==========================================================
# SOLUCIÓN QUBO (Simulated Annealing clásico)
# ==========================================================

print("\nCalculando solución QUBO (SA clásico)...")
_t0 = time.time()
u_qubo_hist = optimizar_con_qubo(
    R_obs, Delta_S_obs, S_inicial, S_max=V_MAX,
    n_restarts=8, n_iter=80_000, verbose=True
)
tiempos['QUBO + SA'] = time.time() - _t0
print(f"  ⏱ Tiempo QUBO+SA: {tiempos['QUBO + SA']:.1f} s")

# ==========================================================
# BENCHMARK EXACTO OPCIONAL: GUROBI MIQP
# ==========================================================

print("\nIntentando referencia exacta Gurobi MIQP (opcional)...")
resultado_gurobi = resolver_benchmark_gurobi(
    R_obs, Delta_S_obs, S_inicial, S_max=V_MAX,
    time_limit=60, mip_gap=0.02, verbose=False
)
if resultado_gurobi is not None:
    u_gurobi = resultado_gurobi['u']
    tiempos['Gurobi MIQP'] = resultado_gurobi['runtime_s']
else:
    u_gurobi = None

# ==========================================================
# CICLO DE PREDICCIÓN HÍBRIDA (RNA)
# ==========================================================

print(f"\nSimulando {SEMANAS_PREDICCION} semanas futuras con RNA...")
_t0 = time.time()

R_obs_hist = list(R_obs[-12:])
u_actual   = 0.0
hist_pred  = []

mediana_R = float(np.median(R_obs))
delta_u   = 0.25 * mediana_R
u_max     = 2 * delta_u
u_min     = -u_max

for fecha in fechas_futuras:

    # ── 1. PREDICCIÓN DE LA RNA ──────────────────────────────────
    u_rna = predecir_rna(
        rna=rna, scaler=scaler, S_actual=S_actual,
        Delta_S_obs=0.0, R_obs_hist=np.array(R_obs_hist),
        mes=fecha.month,
    )
    u_final = np.clip(u_rna, u_min, u_max)
    S_actual = np.clip(S_actual - u_final, 0, V_MAX)

    # Actualizar historial de liberaciones (para el siguiente paso)
    R_obs_hist.append(mediana_R)   # usar mediana como proxy futuro
    if len(R_obs_hist) > 12:
        R_obs_hist.pop(0)

    pct = S_actual / NAMO * 100
    if pct > 75:     zona = "🔵 Lleno"
    elif pct > 50:   zona = "🟢 Normal"
    elif pct > 25:   zona = "🟠 Crítico"
    elif pct > 12.5: zona = "🔴 Emergencia"
    else:            zona = "🆘 Mínimo"

    hist_pred.append({
        "Fecha": fecha, "S_TCM": S_actual, "pct_NAMO": pct,
        "u_RNA": u_rna, "u_final": u_final, "zona": zona,
    })

tiempos['RNA (política)'] = time.time() - _t0

# ==========================================================
# RESULTADOS
# ==========================================================

df_pred = pd.DataFrame(hist_pred)

print(f"\n{'='*65}")
print(f"  RESULTADO DE LA SIMULACIÓN RNA — {SEMANAS_PREDICCION} semanas")
print(f"{'='*65}")
print(f"  {'Fecha':<12} {'S(TCM)':>12} {'%NAMO':>7} {'u_RNA':>10}  Zona")
print(f"  {'-'*60}")
for _, r in df_pred.iterrows():
    print(f"  {str(r['Fecha'].date()):<12} {r['S_TCM']:>12,.0f} {r['pct_NAMO']:>6.1f}%"
          f" {r['u_RNA']:>10.0f}  {r['zona']}")

# ==========================================================
# ΔSRS BENCHMARK OFICIAL — Todas las soluciones
# ==========================================================

print(f"\nCalculando ΔSRS para todas las soluciones...")

u_rna_hist = np.array([hist_pred[i]['u_RNA'] for i in range(min(len(hist_pred), len(R_obs)))])
if len(u_rna_hist) < len(R_obs):
    u_rna_hist = np.concatenate([u_rna_hist, np.zeros(len(R_obs) - len(u_rna_hist))])

# Regla de umbral
u_rule = np.zeros(len(R_obs))
S_r    = S_inicial
for t in range(len(R_obs)):
    u_rule[t] = -delta_u if S_r < S_MIN else 0.0
    S_r += Delta_S_obs[t] - u_rule[t]

reportados = {
    'Baseline 0 — Histórico (u=0)':    np.zeros(len(R_obs)),
    'Baseline 1 — Regla umbral':         u_rule,
    'Baseline 2 — Genético (DEAP)':     u_genetico,
    'Cuántico   — QUBO + SA':             u_qubo_hist,
    'Política   — RNA (futura)':          u_rna_hist,
}
if u_gurobi is not None:
    reportados['Benchmark  — Gurobi MIQP'] = u_gurobi

resultados_srs = []
for nombre, u_seq in reportados.items():
    srs_val, _ = calcular_srs(u_seq, R_obs, Delta_S_obs, S_inicial, V_MAX)
    resultados_srs.append({'Metodo': nombre, 'SRS': srs_val})

srs_hist = next(r['SRS'] for r in resultados_srs if 'Hist' in r['Metodo'])
for r in resultados_srs:
    r['ΔSRS'] = r['SRS'] - srs_hist
    r['Tiempo_s'] = tiempos.get(r['Metodo'].split('—')[-1].strip(), None)

print(f"\n{'='*72}")
print(f"  📊 BENCHMARK ΔSRS OFICIAL — Embalse Falcón (T={len(R_obs)}, L=5)")
print(f"{'='*72}")
print(f"  {'Método':<40} {'SRS':>10}  {'ΔSRS':>12}  {'Tiempo':>7}")
print(f"  {'-'*70}")
for r in resultados_srs:
    delta  = r['ΔSRS']
    tiempo = f"{r['Tiempo_s']:.0f}s" if r['Tiempo_s'] else '  —'
    marca  = ("—" if delta == 0 else
              (f"+{delta:.6f} ✅" if delta > 0 else f"{delta:.6f} ⚠️"))
    print(f"  {r['Metodo']:<40} {r['SRS']:>10.6f}  {marca:>12}  {tiempo:>7}")
print(f"{'='*72}")

# ==========================================================
# GUARDAR RESULTADOS EN CARPETA resultados/
# ==========================================================

os.makedirs('resultados', exist_ok=True)

# ── 1. CSV con tabla ΔSRS ──────────────────────────────────────────
df_srs = pd.DataFrame(resultados_srs)
df_srs.to_csv('resultados/benchmark_delta_srs.csv', index=False)
print("\n  📄 Tabla DELTA_SRS guardada: resultados/benchmark_delta_srs.csv")

# ── 2. CSV con simulación RNA semana a semana ───────────────────────
df_pred.to_csv('resultados/simulacion_rna.csv', index=False)
print("  📄 Simulación RNA guardada:  resultados/simulacion_rna.csv")

# ── 3. Gráficas extra de diagnóstico ─────────────────────────────────
conv_path = graficar_convergencia(
    log_genetico, output_path='resultados/convergencia_genetico.png'
)
if conv_path:
    print(f"  📄 Convergencia genético guardada: {conv_path}")

estrategias_extra = {
    'Histórico': np.zeros(len(R_obs)),
    'Regla umbral': u_rule,
    'Genético': u_genetico,
    'QUBO + SA': u_qubo_hist,
}
if u_gurobi is not None:
    estrategias_extra['Gurobi MIQP'] = u_gurobi

extra_path = graficar_comparativa_estrategias(
    R_obs, Delta_S_obs, S_inicial, V_MAX, estrategias_extra,
    output_path='resultados/comparativa_estrategias.png'
)
print(f"  📄 Comparativa extra guardada: {extra_path}")

# ==========================================================
# GRÁFICA BENCHMARK — 2 paneles (sin proyección futura)
# [1] Almacenamiento S(t)   [2] Decisiones u(t)
# ==========================================================

_, S_gen_traj  = calcular_srs(u_genetico,  R_obs, Delta_S_obs, S_inicial, V_MAX)
_, S_qubo_traj = calcular_srs(u_qubo_hist, R_obs, Delta_S_obs, S_inicial, V_MAX)
_, S_rule_traj = calcular_srs(u_rule,      R_obs, Delta_S_obs, S_inicial, V_MAX)

fechas_opt  = pd.date_range(start=FECHA_INICIO, periods=len(R_obs)+1, freq='W-SUN')
S_hist_full = np.concatenate([[S_inicial], S_hist])
S_gen_full  = np.concatenate([[S_inicial], S_gen_traj[:-1]])
S_qubo_full = np.concatenate([[S_inicial], S_qubo_traj[:-1]])
S_rule_full = np.concatenate([[S_inicial], S_rule_traj[:-1]])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.06})
fig.patch.set_facecolor('#0f1117')
for ax in (ax1, ax2):
    ax.set_facecolor('#1a1d27')

# ── Panel 1: trayectorias de almacenamiento ─────────────────
ax1.fill_between(fechas_opt, S_hist_full / 1000, alpha=0.12, color='#aaaaaa')
ax1.plot(fechas_opt, S_hist_full / 1000,
         color='#cccccc', lw=2.0, label='Histórico observado', zorder=3)
ax1.plot(fechas_opt, S_rule_full / 1000,
         color='#facc15', lw=1.6, ls=':', label='Baseline 1 — Regla umbral', zorder=4)
ax1.plot(fechas_opt, S_gen_full / 1000,
         color='#38bdf8', lw=2.4, ls='--',
         label=f'Baseline 2 — Genético  (ΔSRS={resultados_srs[2]["ΔSRS"]:+.4f})', zorder=5)
ax1.plot(fechas_opt, S_qubo_full / 1000,
         color='#a78bfa', lw=2.6, ls='solid',
         label=f'QUBO + SA              (ΔSRS={resultados_srs[3]["ΔSRS"]:+.4f})', zorder=6)

ax1.axhline(NAMO  / 1000, color='#60a5fa', ls='--', lw=1.2, alpha=0.7,
            label=f'NAMO 100% ({NAMO/1000:,.0f}k TCM)')
ax1.axhline(S_MIN / 1000, color='#f87171', ls=':',  lw=2.0, alpha=0.95,
            label=f'Crítico 25% ({S_MIN/1000:,.0f}k TCM)')
ax1.axhspan(0, S_MIN / 1000, alpha=0.07, color='#f87171', zorder=0)
ax1.text(fechas_opt[1], S_MIN/1000 * 0.91, '⚠ Zona crítica (<25% NAMO)',
         color='#f87171', fontsize=8, alpha=0.85)

S_all = np.concatenate([S_hist_full, S_gen_full, S_qubo_full, S_rule_full])
ax1.set_ylim(max(0, S_all.min()/1000 * 0.88), S_all.max()/1000 * 1.10)
ax1.set_ylabel('Almacenamiento (miles TCM)', color='#aaaaaa', fontsize=11)
ax1.set_title('Benchmark Comparativo — Embalse Falcón  |  T=26 semanas, L=5 niveles',
              color='white', fontsize=13, fontweight='bold', pad=10)
ax1.legend(loc='lower right', fontsize=9, framealpha=0.35,
           facecolor='#1a1d27', labelcolor='white')
ax1.tick_params(colors='#aaaaaa')
ax1.grid(True, linestyle='--', alpha=0.15, color='white')

# ── Panel 2: decisiones u(t) semana a semana ────────────────
fechas_u = fechas_opt[:-1]
ancho     = 3.5   # días

ax2.bar(fechas_u - pd.Timedelta(days=ancho*1.5), u_rule    / 1000,
        width=ancho, color='#facc15', alpha=0.75, label='Regla umbral', align='center')
ax2.bar(fechas_u - pd.Timedelta(days=ancho*0.5), u_genetico / 1000,
        width=ancho, color='#38bdf8', alpha=0.85, label='Genético', align='center')
ax2.bar(fechas_u + pd.Timedelta(days=ancho*0.5), u_qubo_hist / 1000,
        width=ancho, color='#a78bfa', alpha=0.85, label='QUBO + SA', align='center')

ax2.axhline(0,            color='white',   lw=0.8, alpha=0.4)
ax2.axhline(-delta_u/1000, color='#f87171', ls=':', lw=1.0, alpha=0.5)
ax2.axhline( delta_u/1000, color='#60a5fa', ls=':', lw=1.0, alpha=0.5)
ax2.text(fechas_u[0], -delta_u/1000 * 1.35,
         '← u<0: reducir liberación (embalse se llena)',
         color='#f87171', fontsize=7.5, alpha=0.85)
ax2.text(fechas_u[0],  delta_u/1000 * 1.10,
         '→ u>0: aumentar liberación (embalse se vacía)',
         color='#60a5fa', fontsize=7.5, alpha=0.85)

ax2.set_ylabel('u(t)  (miles TCM/sem)', color='#aaaaaa', fontsize=10)
ax2.set_xlabel('Fecha', color='#aaaaaa', fontsize=11)
ax2.legend(loc='lower right', fontsize=8, framealpha=0.35,
           facecolor='#1a1d27', labelcolor='white')
ax2.tick_params(colors='#aaaaaa')
ax2.grid(True, linestyle='--', alpha=0.12, color='white')

for ax in (ax1, ax2):
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

plt.tight_layout()

graf_path = 'resultados/simulacion_hibrida.png'
plt.savefig(graf_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.savefig('simulacion_hibrida.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"  📄 Gráfica guardada: {graf_path}")

print(f"\n{'='*55}")
print("  ✅ Todos los resultados guardados en resultados/")
print(f"{'='*55}")
