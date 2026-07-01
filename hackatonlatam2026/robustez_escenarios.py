# -*- coding: utf-8 -*-
"""
robustez_escenarios.py
======================
Analisis de robustez para el Challenge A del Embalse Falcon.

Este modulo no reemplaza el pipeline principal. Toma la ventana oficial
T=26/L=5, genera escenarios hidrologicos perturbados y compara que tan
estable es cada politica frente a incertidumbre.

Salidas:
    resultados/resultados_robustez/resumen_robustez.csv
    resultados/resultados_robustez/detalle_escenarios.csv
    resultados/resultados_robustez/ranking_robustez.csv
    resultados/resultados_robustez/boxplot_srs_robustez.png
    resultados/resultados_robustez/heatmap_semanas_criticas.png
    resultados/resultados_robustez/ranking_robustez.png
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from genetico_v3 import calcular_srs, preparar_ventana_semanal
from qubo_solver import optimizar_con_qubo


S_MAX = 3_387_000.0
NAMO = 3_265_400.0
S_MIN = 0.25 * S_MAX
FECHA_INICIO = "2024-01-01"
T_OFICIAL = 26

PROJECT_DIR = Path(__file__).resolve().parent
EXTRA_DATA_DIR = Path(os.environ["FALCON_EXTRA_DATA_DIR"]) if os.environ.get("FALCON_EXTRA_DATA_DIR") else None
RESULTADOS_DIR = PROJECT_DIR / "resultados" / "resultados_robustez"


@dataclass(frozen=True)
class Escenario:
    nombre: str
    descripcion: str
    R_obs: np.ndarray
    Delta_S_obs: np.ndarray


def _leer_csv_con_fallback(nombre: str, skiprows: int = 1) -> pd.DataFrame:
    """Carga un CSV desde el proyecto; si no existe, usa la carpeta extra."""
    rutas = [PROJECT_DIR / nombre]
    if EXTRA_DATA_DIR is not None:
        rutas.append(EXTRA_DATA_DIR / nombre)
    for ruta in rutas:
        if ruta.exists():
            return pd.read_csv(ruta, skiprows=skiprows)
    raise FileNotFoundError(f"No se encontro {nombre} en las rutas configuradas.")


def _leer_datos_base():
    df_lib = pd.read_excel(PROJECT_DIR / "R_observ.xlsx")
    df_cambio = _leer_csv_con_fallback(
        "DataSetExport-Discharge Total.Change-in-Storage@08461200"
        "-Instantaneous-TCM-20260622185956.csv",
        skiprows=1,
    )
    df_total = _leer_csv_con_fallback(
        "DataSetExport-Total Storage.Web-Daily-ac-ft@08461200"
        "-Instantaneous-TCM-20260622185130.csv",
        skiprows=1,
    )

    evap_name = (
        "DataSetExport-Evaporation,accumltd.Daily Evaporation - mm@08461200"
        "-Instantaneous-mm-20260622185804.csv"
    )
    try:
        df_evap = _leer_csv_con_fallback(evap_name, skiprows=1)
        fuente_evap = "disponible"
    except FileNotFoundError:
        df_evap = pd.DataFrame(columns=["Timestamp (UTC-06:00)", "Evaporacion_mm"])
        fuente_evap = "no disponible"

    df_batimetria = pd.read_csv(PROJECT_DIR / "tabla_elevacion_volumen_FINAL.csv")

    R_obs, Delta_S_obs, S_inicial = preparar_ventana_semanal(
        df_lib,
        df_cambio,
        df_total,
        df_evap,
        df_batimetria,
        FECHA_INICIO,
        T_OFICIAL,
    )
    return R_obs.astype(float), Delta_S_obs.astype(float), float(S_inicial), fuente_evap


def politica_umbral(R_obs: np.ndarray, Delta_S_obs: np.ndarray, S_inicial: float) -> np.ndarray:
    delta_u = 0.25 * float(np.median(R_obs))
    u = np.zeros(len(R_obs))
    S = float(S_inicial)
    for t in range(len(R_obs)):
        u[t] = -delta_u if S < S_MIN else 0.0
        S += Delta_S_obs[t] - u[t]
    return u


def crear_escenarios(R_obs: np.ndarray, Delta_S_obs: np.ndarray, seed: int = 42) -> list[Escenario]:
    rng = np.random.default_rng(seed)
    T = len(R_obs)
    ruido_suave = rng.normal(0.0, 0.06, size=T)
    ruido_fuerte = rng.normal(0.0, 0.12, size=T)
    bloque = np.ones(T)
    bloque[T // 3 : T // 3 + 5] = 0.75
    shifted = np.roll(Delta_S_obs, 2)

    escenarios = [
        Escenario(
            "nominal",
            "Datos historicos sin perturbacion",
            R_obs.copy(),
            Delta_S_obs.copy(),
        ),
        Escenario(
            "sequia_10",
            "Liberaciones historicas 10% menores y cambios de almacenamiento mas secos",
            R_obs * 0.90,
            Delta_S_obs - 0.10 * np.maximum(R_obs, 0.0),
        ),
        Escenario(
            "sequia_20",
            "Liberaciones historicas 20% menores y mayor perdida neta",
            R_obs * 0.80,
            Delta_S_obs - 0.20 * np.maximum(R_obs, 0.0),
        ),
        Escenario(
            "ruido_suave",
            "Variacion semanal controlada de 6%",
            np.maximum(R_obs * (1.0 + ruido_suave), 0.0),
            Delta_S_obs * (1.0 + ruido_suave),
        ),
        Escenario(
            "ruido_fuerte",
            "Variacion semanal controlada de 12%",
            np.maximum(R_obs * (1.0 + ruido_fuerte), 0.0),
            Delta_S_obs * (1.0 + ruido_fuerte),
        ),
        Escenario(
            "bloque_critico",
            "Cinco semanas consecutivas con reduccion fuerte de liberaciones",
            R_obs * bloque,
            Delta_S_obs - (1.0 - bloque) * np.maximum(R_obs, 0.0),
        ),
        Escenario(
            "desfase_temporal",
            "Cambios de almacenamiento desplazados dos semanas",
            R_obs.copy(),
            shifted,
        ),
    ]
    return escenarios


def metricas_politica(
    metodo: str,
    escenario: Escenario,
    u: np.ndarray,
    S_inicial: float,
) -> dict:
    srs, S_traj = calcular_srs(
        u,
        escenario.R_obs,
        escenario.Delta_S_obs,
        S_inicial,
        S_MAX,
    )
    R_opt = escenario.R_obs + u
    balance_limite = 0.10 * float(np.sum(escenario.R_obs))
    viol_flujo = int(np.sum(R_opt < -1e-8))
    viol_inf = int(np.sum(S_traj < -1e-8))
    viol_sup = int(np.sum(S_traj > S_MAX + 1e-8))
    viol_balance = int(abs(float(np.sum(u))) > balance_limite + 1e-8)
    semanas_criticas = int(np.sum(S_traj < S_MIN))
    fallos = viol_flujo + viol_inf + viol_sup + viol_balance
    return {
        "escenario": escenario.nombre,
        "descripcion": escenario.descripcion,
        "metodo": metodo,
        "SRS": float(srs),
        "semanas_criticas": semanas_criticas,
        "S_min_traj": float(np.min(S_traj)),
        "S_final": float(S_traj[-1]),
        "viol_flujo": viol_flujo,
        "viol_embalse_min": viol_inf,
        "viol_embalse_max": viol_sup,
        "viol_balance": viol_balance,
        "fallos": int(fallos),
        "factible": fallos == 0,
        "sum_u": float(np.sum(u)),
        "balance_limite": balance_limite,
    }


def construir_politicas_nominales(
    R_obs: np.ndarray,
    Delta_S_obs: np.ndarray,
    S_inicial: float,
    n_restarts: int,
    n_iter: int,
) -> dict[str, np.ndarray]:
    print("Calculando politicas nominales...")
    politicas = {
        "Historico": np.zeros(len(R_obs)),
        "Regla umbral": politica_umbral(R_obs, Delta_S_obs, S_inicial),
    }
    t0 = time.time()
    politicas["QUBO+SA nominal"] = optimizar_con_qubo(
        R_obs,
        Delta_S_obs,
        S_inicial,
        S_max=S_MAX,
        n_restarts=n_restarts,
        n_iter=n_iter,
        verbose=False,
    )
    print(f"  QUBO nominal listo en {time.time() - t0:.1f} s")
    return politicas


def evaluar_robustez(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    R_obs, Delta_S_obs, S_inicial, fuente_evap = _leer_datos_base()
    print("=" * 72)
    print(" ANALISIS DE ROBUSTEZ - FALCON RESERVOIR")
    print("=" * 72)
    print(f"Ventana oficial: {FECHA_INICIO}, T={len(R_obs)}")
    print(f"Evaporacion: {fuente_evap}")
    print(f"Salidas: {RESULTADOS_DIR}")

    escenarios = crear_escenarios(R_obs, Delta_S_obs, seed=args.seed)
    politicas = construir_politicas_nominales(
        R_obs,
        Delta_S_obs,
        S_inicial,
        n_restarts=args.n_restarts,
        n_iter=args.n_iter,
    )

    rows = []
    for escenario in escenarios:
        print(f"\nEscenario: {escenario.nombre}")
        for metodo, u in politicas.items():
            rows.append(metricas_politica(metodo, escenario, u, S_inicial))

        if args.reoptimizar:
            t0 = time.time()
            u_reopt = optimizar_con_qubo(
                escenario.R_obs,
                escenario.Delta_S_obs,
                S_inicial,
                S_max=S_MAX,
                n_restarts=max(2, args.n_restarts // 2),
                n_iter=max(5_000, args.n_iter // 2),
                verbose=False,
            )
            met = metricas_politica("QUBO+SA reoptimizado", escenario, u_reopt, S_inicial)
            met["tiempo_reopt_s"] = time.time() - t0
            rows.append(met)
            print(f"  Reopt listo: SRS={met['SRS']:.6f}, t={met['tiempo_reopt_s']:.1f}s")

    detalle = pd.DataFrame(rows)
    resumen = (
        detalle.groupby("metodo")
        .agg(
            SRS_promedio=("SRS", "mean"),
            SRS_peor=("SRS", "min"),
            SRS_desv=("SRS", "std"),
            semanas_criticas_prom=("semanas_criticas", "mean"),
            fallos_totales=("fallos", "sum"),
            factibilidad_prom=("factible", "mean"),
        )
        .reset_index()
    )
    resumen["robustez_score"] = (
        resumen["SRS_peor"]
        - 0.01 * resumen["semanas_criticas_prom"]
        - 0.02 * resumen["fallos_totales"]
    )
    resumen = resumen.sort_values("robustez_score", ascending=False)

    detalle.to_csv(RESULTADOS_DIR / "detalle_escenarios.csv", index=False)
    resumen.to_csv(RESULTADOS_DIR / "resumen_robustez.csv", index=False)
    resumen.to_csv(RESULTADOS_DIR / "ranking_robustez.csv", index=False)
    _guardar_graficas(detalle, resumen)
    _guardar_reporte_md(detalle, resumen, fuente_evap)
    return detalle, resumen


def _guardar_graficas(detalle: pd.DataFrame, resumen: pd.DataFrame) -> None:
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(11, 6))
    metodos = list(detalle["metodo"].unique())
    data = [detalle.loc[detalle["metodo"] == m, "SRS"].values for m in metodos]
    ax.boxplot(data, tick_labels=metodos, showmeans=True)
    ax.set_title("Distribucion de SRS bajo escenarios de estres")
    ax.set_ylabel("SRS")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "boxplot_srs_robustez.png", dpi=150)
    plt.close(fig)

    pivot = detalle.pivot_table(
        index="metodo",
        columns="escenario",
        values="semanas_criticas",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title("Semanas bajo umbral critico por escenario")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, int(pivot.values[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="semanas criticas")
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "heatmap_semanas_criticas.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(resumen["metodo"], resumen["robustez_score"])
    ax.invert_yaxis()
    ax.set_title("Ranking de robustez")
    ax.set_xlabel("robustez_score")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "ranking_robustez.png", dpi=150)
    plt.close(fig)


def _guardar_reporte_md(detalle: pd.DataFrame, resumen: pd.DataFrame, fuente_evap: str) -> None:
    mejor = resumen.iloc[0]
    cols = [
        "metodo",
        "SRS_promedio",
        "SRS_peor",
        "SRS_desv",
        "semanas_criticas_prom",
        "fallos_totales",
        "factibilidad_prom",
        "robustez_score",
    ]
    tabla = resumen[cols].copy()
    for col in tabla.columns:
        if col != "metodo":
            tabla[col] = tabla[col].map(lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))
    markdown_rows = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in tabla.iterrows():
        markdown_rows.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    lineas = [
        "# Analisis de Robustez - Falcon Reservoir",
        "",
        "Este analisis evalua politicas bajo escenarios hidrologicos perturbados.",
        "La ruta principal del proyecto no se modifica; este modulo agrega evidencia de resiliencia.",
        "",
        f"- Ventana: `{FECHA_INICIO}`, `T={T_OFICIAL}`.",
        f"- Evaporacion: `{fuente_evap}`.",
        f"- Mejor metodo por robustez: `{mejor['metodo']}`.",
        f"- Peor SRS del mejor metodo: `{mejor['SRS_peor']:.6f}`.",
        "",
        "## Ranking",
        "",
        "\n".join(markdown_rows),
        "",
        "## Archivos generados",
        "",
        "- `detalle_escenarios.csv`",
        "- `resumen_robustez.csv`",
        "- `ranking_robustez.csv`",
        "- `boxplot_srs_robustez.png`",
        "- `heatmap_semanas_criticas.png`",
        "- `ranking_robustez.png`",
        "",
    ]
    (RESULTADOS_DIR / "REPORTE_ROBUSTEZ.md").write_text("\n".join(lineas), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Analisis de robustez del calendario Falcon.")
    parser.add_argument("--n-restarts", type=int, default=6, help="Restarts SA para QUBO nominal.")
    parser.add_argument("--n-iter", type=int, default=40_000, help="Iteraciones SA para QUBO nominal.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para escenarios con ruido.")
    parser.add_argument(
        "--sin-reoptimizar",
        action="store_true",
        help="No calcula QUBO reoptimizado por escenario.",
    )
    args = parser.parse_args()
    args.reoptimizar = not args.sin_reoptimizar
    return args


def main():
    detalle, resumen = evaluar_robustez(parse_args())
    print("\n" + "=" * 72)
    print(" RANKING DE ROBUSTEZ")
    print("=" * 72)
    print(resumen.to_string(index=False))
    print("\nArchivos guardados en:")
    print(f"  {RESULTADOS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
