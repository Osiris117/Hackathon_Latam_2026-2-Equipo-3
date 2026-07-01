# -*- coding: utf-8 -*-
"""
entrenar_rna.py
===============
Módulo de entrenamiento RNA + Genético para el Embalse Falcón.

Uso como módulo:
    from entrenar_rna import cargar_datos, entrenar_rna, predecir_rna, FEATURES, NAMO, S_MAX

Uso standalone:
    python3 entrenar_rna.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import joblib
import warnings
warnings.filterwarnings('ignore')

from genetico_v3 import run_genetico

# ── Constantes globales ───────────────────────────────────────────────
S_MAX = 3_387_000.0
NAMO  = 3_264_810.0

FEATURES = [
    'f_delta_s_norm',   # ΔS relativo al NAMO
    'f_pct_namo',       # % de llenado actual
    'f_tendencia_3w',   # Tendencia últimas 3 semanas
    'f_r_obs_norm',     # Liberaciones vs. promedio histórico
    'f_mes_sin',        # Estacionalidad (seno)
    'f_mes_cos',        # Estacionalidad (coseno)
    'f_dist_critico',   # Distancia al umbral crítico 25% NAMO
]

# =========================================================
# FUNCIÓN 1: Carga y agrupación semanal de datos
# =========================================================
def cargar_datos(
        ruta_lib     = "R_observ.xlsx",
        ruta_cambio  = "DataSetExport-Discharge Total.Change-in-Storage@08461200-Instantaneous-TCM-20260622185956.csv",
        ruta_total   = "DataSetExport-Total Storage.Web-Daily-ac-ft@08461200-Instantaneous-TCM-20260622185130.csv",
        semanas      = 260):
    """
    Lee los CSVs del IBWC y devuelve un DataFrame semanal listo para entrenar.

    Parámetros
    ----------
    ruta_lib    : str, ruta al Excel de liberaciones (R_observ.xlsx)
    ruta_cambio : str, ruta al CSV de Cambio-en-Almacenamiento
    ruta_total  : str, ruta al CSV de Almacenamiento Total
    semanas     : int, número de semanas más recientes a usar

    Retorna
    -------
    df_hist : DataFrame con columnas ['R_obs', 'Delta_S_obs', 'S_actual']
              indexado por fecha semanal (W-SUN)
    """
    def _limpiar(df):
        df.columns = df.columns.str.strip()
        for col in ['Timestamp (UTC-06:00)', 'Fecha', 'timestamp', 'date', 'Date']:
            if col in df.columns:
                df.rename(columns={col: 'Fecha'}, inplace=True)
                break
        if 'Fecha' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
            df = df.dropna(subset=['Fecha']).set_index('Fecha')
        return df

    df_lib_c    = _limpiar(pd.read_excel(ruta_lib))
    df_cambio_c = _limpiar(pd.read_csv(ruta_cambio, skiprows=1))
    df_total_c  = _limpiar(pd.read_csv(ruta_total,  skiprows=1))

    lib_w    = df_lib_c['Valor'].resample('W-SUN').mean() * 604.8
    cambio_w = df_cambio_c['Value (TCM)'].resample('W-SUN').sum()
    total_w  = df_total_c['Value (TCM)'].resample('W-SUN').first()

    df_hist = pd.DataFrame({
        'R_obs'      : lib_w,
        'Delta_S_obs': cambio_w,
        'S_actual'   : total_w,
    }).dropna().tail(semanas)

    return df_hist


# =========================================================
# FUNCIÓN 2: Construir features enriquecidas
# =========================================================
def construir_features(df_hist):
    """
    Agrega las 7 features de ingeniería al DataFrame histórico.

    Parámetros
    ----------
    df_hist : DataFrame con columnas ['R_obs', 'Delta_S_obs', 'S_actual',
                                       'Decision_Optima_u']

    Retorna
    -------
    df_feat : DataFrame con columnas FEATURES + 'Decision_Optima_u'
    """
    df = df_hist.copy()
    df['f_delta_s_norm'] = df['Delta_S_obs'] / NAMO
    df['f_pct_namo']     = df['S_actual']    / NAMO
    df['f_tendencia_3w'] = df['S_actual'].diff(3) / NAMO
    df['f_r_obs_norm']   = df['R_obs'] / df['R_obs'].rolling(12, min_periods=1).mean()
    df['f_mes_sin']      = np.sin(2 * np.pi * df.index.month / 12)
    df['f_mes_cos']      = np.cos(2 * np.pi * df.index.month / 12)
    df['f_dist_critico'] = (df['S_actual'] - 0.25 * NAMO) / NAMO
    return df.dropna(subset=FEATURES)


# =========================================================
# FUNCIÓN 3: Entrenar la RNA completa
# =========================================================
def entrenar_rna(df_hist=None, guardar=True, verbose=True):
    """
    Ejecuta el pipeline completo: Genético → features → RNA.

    Parámetros
    ----------
    df_hist  : DataFrame devuelto por cargar_datos(). Si es None, carga datos automáticamente.
    guardar  : bool, si True guarda modelo_rna_genetico.pkl y scaler_rna.pkl
    verbose  : bool, si True imprime el progreso

    Retorna
    -------
    rna    : MLPClassifier entrenado
    scaler : StandardScaler ajustado
    dist   : Counter con distribución de etiquetas del genético
    """
    if df_hist is None:
        if verbose: print("Cargando datos...")
        df_hist = cargar_datos()

    T         = len(df_hist)
    R_obs     = df_hist['R_obs'].values
    Delta_S   = df_hist['Delta_S_obs'].values
    S_inicial = df_hist['S_actual'].iloc[0]

    if verbose:
        print(f"  Ventana: {T} semanas "
              f"({df_hist.index[0].date()} → {df_hist.index[-1].date()})")

    # ── Paso 1: Genético ─────────────────────────────────────────────
    if verbose: print(f"\nIniciando Algoritmo Genético ({T} semanas)...")
    mejor_secuencia, hof, log = run_genetico(
        R_obs=R_obs, Delta_S_obs=Delta_S, S_inicial=S_inicial,
        S_max=S_MAX, n_pop=400, n_gen=120,
        cxpb=0.7, mutpb=0.2, indpb=0.15, verbose=False,
    )
    df_hist = df_hist.copy()
    df_hist['Decision_Optima_u'] = mejor_secuencia

    dist = Counter(mejor_secuencia.round(0))
    if verbose:
        print("  Distribución de etiquetas:")
        for nivel, cnt in sorted(dist.items()):
            print(f"    u={nivel:>8.0f}: {cnt:>3} veces")

    # ── Paso 2: Features ─────────────────────────────────────────────
    if verbose: print("\nConstruyendo features...")
    df_feat = construir_features(df_hist)
    X_raw   = df_feat[FEATURES].values
    y_cat   = df_feat['Decision_Optima_u'].round(0).astype(int).astype(str).values

    if verbose:
        print(f"  Muestras: {len(X_raw)}  |  Features: {len(FEATURES)}")

    # ── Paso 3: Normalización ─────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_raw)

    # ── Paso 4: Oversampling ──────────────────────────────────────────
    conteo_clases = Counter(y_cat)
    n_max         = max(conteo_clases.values())
    X_bal_list, y_bal_list = [X_train], [y_cat]
    for clase, n in conteo_clases.items():
        if n < n_max:
            idx   = np.where(y_cat == clase)[0]
            extra = np.random.default_rng(42).choice(idx, size=n_max - n, replace=True)
            X_bal_list.append(X_train[extra])
            y_bal_list.append(y_cat[extra])

    X_bal = np.vstack(X_bal_list)
    y_bal = np.concatenate(y_bal_list)

    if verbose:
        print(f"  Dataset balanceado: {len(y_bal)} muestras "
              f"({n_max}/clase × {len(conteo_clases)} clases)")

    # ── Paso 5: Entrenamiento RNA ─────────────────────────────────────
    if verbose: print("\nEntrenando RNA...")
    rna = MLPClassifier(
        hidden_layer_sizes  = (64, 32, 16),
        activation          = 'relu',
        solver              = 'adam',
        max_iter            = 2000,
        random_state        = 42,
        learning_rate_init  = 0.001,
        early_stopping      = True,
        validation_fraction = 0.15,
        n_iter_no_change    = 30,
    )
    rna.fit(X_bal, y_bal)

    # ── Paso 6: Evaluación ────────────────────────────────────────────
    if verbose:
        y_pred    = rna.predict(X_train)
        acc_train = rna.score(X_train, y_cat)
        print(f"\n  Épocas: {rna.n_iter_}  |  Precisión: {acc_train:.2%}")
        print(f"\n  Reporte por clase:\n"
              + classification_report(y_cat, y_pred, zero_division=0))

    # ── Paso 7: Guardar ───────────────────────────────────────────────
    if guardar:
        joblib.dump(rna,    'modelo_rna_genetico.pkl')
        joblib.dump(scaler, 'scaler_rna.pkl')
        if verbose:
            print("  Guardados: modelo_rna_genetico.pkl, scaler_rna.pkl")

    return rna, scaler, dist


# =========================================================
# FUNCIÓN 4: Predecir con la RNA (para usar en modelado1.py)
# =========================================================
def predecir_rna(rna, scaler, S_actual, Delta_S_obs, R_obs_hist, mes):
    """
    Genera la predicción de la RNA para un estado del embalse.

    Parámetros
    ----------
    rna        : MLPClassifier entrenado
    scaler     : StandardScaler ajustado
    S_actual   : float, almacenamiento actual (TCM)
    Delta_S_obs: float, cambio de almacenamiento esta semana (TCM)
    R_obs_hist : array-like, últimas 12 semanas de liberaciones (para normalizar)
    mes        : int, mes del año (1–12)

    Retorna
    -------
    u_decision : float, ajuste óptimo u(t) en TCM/semana
    """
    r_media = np.mean(R_obs_hist) if len(R_obs_hist) > 0 else 1.0
    r_norm  = (R_obs_hist[-1] / r_media) if r_media != 0 else 1.0

    x = np.array([[
        Delta_S_obs / NAMO,                      # f_delta_s_norm
        S_actual    / NAMO,                      # f_pct_namo
        0.0,                                     # f_tendencia_3w (sin histórico: 0)
        r_norm,                                  # f_r_obs_norm
        np.sin(2 * np.pi * mes / 12),            # f_mes_sin
        np.cos(2 * np.pi * mes / 12),            # f_mes_cos
        (S_actual - 0.25 * NAMO) / NAMO,         # f_dist_critico
    ]])
    x_scaled    = scaler.transform(x)
    pred        = rna.predict(x_scaled)
    return float(pred[0])


# =========================================================
# GRÁFICAS (standalone)
# =========================================================
def graficar_entrenamiento(rna, dist, y_cat, y_pred):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Entrenamiento RNA Mejorada — Embalse Falcón',
                 fontsize=13, fontweight='bold')

    axes[0].plot(rna.loss_curve_, color='teal', lw=2, label='Train loss')
    if hasattr(rna, 'validation_scores_'):
        axes[0].plot(rna.validation_scores_, color='orange', lw=2, ls='--', label='Val score')
    axes[0].set_title('Curva de Aprendizaje')
    axes[0].set_xlabel('Épocas')
    axes[0].set_ylabel('Pérdida / Score')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    niveles = sorted(dist.keys())
    colores = ['#f85149' if n < 0 else ('#3fb950' if n > 0 else '#e3b341') for n in niveles]
    axes[1].bar([str(int(n)) for n in niveles], [dist[n] for n in niveles],
                color=colores, edgecolor='white')
    axes[1].set_title('Etiquetas del Genético (u óptima)')
    axes[1].set_xlabel('Nivel u (TCM/semana)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)

    clases_ord = sorted(set(y_cat))
    cm   = confusion_matrix(y_cat, y_pred, labels=clases_ord)
    disp = ConfusionMatrixDisplay(cm, display_labels=[int(float(c)) for c in clases_ord])
    disp.plot(ax=axes[2], colorbar=False, cmap='Blues')
    axes[2].set_title('Matriz de Confusión')

    plt.tight_layout()
    plt.savefig('entrenamiento_rna_mejorada.png', dpi=130, bbox_inches='tight')
    print("  Gráfica guardada: entrenamiento_rna_mejorada.png")


# =========================================================
# EJECUCIÓN STANDALONE
# =========================================================
if __name__ == '__main__':
    print("="*60)
    print("  ENTRENAMIENTO RNA — Embalse Falcón")
    print("="*60)

    df_hist            = cargar_datos()
    rna, scaler, dist  = entrenar_rna(df_hist, guardar=True, verbose=True)

    # Reconstruir y_cat/y_pred para graficar
    df_feat  = construir_features(df_hist)
    # (dist ya tiene las etiquetas; reentrenar solo para obtener y_pred)
    import joblib as _jl
    _rna    = _jl.load('modelo_rna_genetico.pkl')
    _scaler = _jl.load('scaler_rna.pkl')
    X_raw   = df_feat[FEATURES].values
    y_cat   = df_feat['Decision_Optima_u'].round(0).astype(int).astype(str).values if 'Decision_Optima_u' in df_feat.columns else []
    y_pred  = _rna.predict(_scaler.transform(X_raw)) if len(y_cat) > 0 else []

    if len(y_cat) > 0:
        graficar_entrenamiento(_rna, dist, y_cat, y_pred)
