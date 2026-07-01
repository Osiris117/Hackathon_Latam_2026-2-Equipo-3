import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from scipy.interpolate import interp1d

# ==========================================================
# MODELO HÍBRIDO RNA + QUBO
# ==========================================================

print("="*60)
print(" MODELO HÍBRIDO DE OPERACIÓN DEL EMBALSE")
print("="*60)

# ==========================================================
# PARÁMETROS GENERALES
# ==========================================================

FECHA_INICIO = "2024-01-01"

SEMANAS_ENTRENAMIENTO = 27

SEMANAS_PREDICCION = 26

V_MAX = 3387000.0

S_MIN = 0.25 * V_MAX

# ==========================================================
# CARGA DE MODELOS
# ==========================================================

print("\nCargando RNA...")

rna = joblib.load("modelo_rna_genetico.pkl")

print("RNA cargada correctamente.")

# ==========================================================
# CARGA DE LOS DATOS HISTÓRICOS
# ==========================================================

print("\nLeyendo archivos históricos...")

df_lib = pd.read_excel("R_observ.xlsx")

df_cambio = pd.read_csv("Cambio_almacenamiento_historico.csv")

df_total = pd.read_csv("DataSetExport-Total Storage.csv")
df_evap = pd.read_csv(
    "DataSetExport-Evaporation,accumltd.Daily Evaporation - mm@08461200-Instantaneous-mm-20260622185804.csv",
    skiprows=1
)

df_batimetria = pd.read_csv(
    "tabla_elevacion_volumen_FINAL.csv"
)

# ==========================================================
# PREPARACIÓN DE LOS DATOS
# ==========================================================

def preparar_ventana_semanal(
        df_lib,
        df_cambio,
        df_total,
        df_evap,
        df_batimetria,
        fecha_inicio_str,
        semanas):

    dataframes = [
        df_lib.copy(),
        df_cambio.copy(),
        df_total.copy(),
        df_evap.copy()
    ]

    for i, df in enumerate(dataframes):

        df.columns = df.columns.str.strip()

        if "Timestamp (UTC-06:00)" in df.columns:
            df.rename(
                columns={
                    "Timestamp (UTC-06:00)": "Fecha"
                },
                inplace=True
            )

        if "Value (mm)" in df.columns:

            df.rename(
                columns={
                    "Value (mm)": "Evaporacion_mm"
                },
                inplace=True
            )
        if "Fecha" in df.columns:

            df["Fecha"] = pd.to_datetime(
                df["Fecha"],
                errors="coerce"
            )

            df = (
                df.dropna(subset=["Fecha"])
                  .set_index("Fecha")
            )

        dataframes[i] = df

    (
        df_lib_p,
        df_cambio_p,
        df_total_p,
        df_evap_p
    ) = dataframes

    # ======================================================
    # INTERPOLACIÓN BATIMÉTRICA
    # ======================================================

    volumen = df_batimetria["volume_TCM"].values

    area = np.gradient(
        df_batimetria["volume_TCM"],
        df_batimetria["elevation_m"]
    )

    f_area = interp1d(
        volumen,
        area,
        kind="linear",
        fill_value="extrapolate"
    )

    # ======================================================
    # AGRUPACIÓN SEMANAL
    # ======================================================

    lib_semanal = (
        df_lib_p["Valor"]
        .resample("W-SUN")
        .mean()
        * 604.8
    )

    cambio_semanal = (
        df_cambio_p["Value (TCM)"]
        .resample("W-SUN")
        .sum()
    )
    total_semanal = (
        df_total_p["Value (TCM)"]
        .resample("W-SUN")
        .first()
    )

    evap_semanal_mm = (
        df_evap_p["Evaporacion_mm"]
        .resample("W-SUN")
        .sum()
    )

    # ======================================================
    # EVAPORACIÓN DINÁMICA
    # ======================================================

    volumen_hist = (
        total_semanal
        .reindex(evap_semanal_mm.index)
        .ffill()
    )

    area_semanal = f_area(volumen_hist)

    evap_semanal_tcm = evap_semanal_mm * area_semanal

    # ======================================================
    # EXTRACCIÓN DE LA VENTANA HISTÓRICA
    # ======================================================

    fecha_inicio = pd.to_datetime(fecha_inicio_str)

    R_obs = (
        lib_semanal
        .loc[fecha_inicio:]
        .head(semanas)
        .values
    )

    Delta_S_obs = (
        cambio_semanal
        .loc[fecha_inicio:]
        .head(semanas)
        .values
    )

    S_inicial = (
        total_semanal
        .loc[:fecha_inicio]
        .iloc[-1]
    )

    return (
        R_obs,
        Delta_S_obs,
        S_inicial
    )
# ==========================================================
# OBTENER LA MISMA VENTANA USADA POR EL GENÉTICO
# ==========================================================

R_obs, Delta_S_obs, S_inicial = preparar_ventana_semanal(
    df_lib,
    df_cambio,
    df_total,
    df_evap,
    df_batimetria,
    FECHA_INICIO,
    SEMANAS_ENTRENAMIENTO
)

# ==========================================================
# RECONSTRUCCIÓN DEL HISTÓRICO DEL EMBALSE
# ==========================================================

S_hist = np.zeros(SEMANAS_ENTRENAMIENTO)

S = S_inicial

for i in range(SEMANAS_ENTRENAMIENTO):

    S_hist[i] = S

    S = S + Delta_S_obs[i]

# ==========================================================
# DATAFRAME HISTÓRICO
# ==========================================================

df_hist = pd.DataFrame({

    "Fecha": pd.date_range(
        start=pd.to_datetime(FECHA_INICIO),
        periods=SEMANAS_ENTRENAMIENTO,
        freq="W-SUN"
    ),

    "R_obs": R_obs,

    "Delta_S": Delta_S_obs,

    "S": S_hist

})

print("\nHistórico reconstruido correctamente.")

print(df_hist.head())
# ==========================================================
# ESTADO FINAL DEL HISTÓRICO
# ==========================================================

S_actual = S_hist[-1]

fecha_actual = df_hist["Fecha"].iloc[-1]

print("\nÚltimo estado histórico")
print("------------------------")
print(f"Fecha : {fecha_actual}")
print(f"S(t)  : {S_actual:,.2f} TCM")

# ==========================================================
# GENERAR FECHAS FUTURAS
# ==========================================================

fechas_futuras = pd.date_range(

    start=fecha_actual + pd.Timedelta(days=7),

    periods=SEMANAS_PREDICCION,

    freq="W-SUN"

)

# ==========================================================
# VARIABLES DEL SIMULADOR
# ==========================================================

u_actual = 0.0

hist_pred = []

print("\nIniciando simulación...")

# ==========================================================
# CICLO DE PREDICCIÓN
# ==========================================================

for fecha in fechas_futuras:

    # ---------------------------------------------
    # Estado observado por la RNA
    # ---------------------------------------------

    estado = pd.DataFrame({

        "Delta_S_obs":[0.0],

        "S_actual":[S_actual]

    })
        # =====================================================
    # 1. PREDICCIÓN DE LA RNA
    # =====================================================

    prediccion = rna.predict(estado)

    # La RNA devuelve el nivel aprendido durante el entrenamiento
    u_rna = float(prediccion[0])

    # =====================================================
    # 2. VECTOR DE ESTADO
    # =====================================================

    estado_vector = np.array([
        S_actual,
        u_rna
    ])

    # =====================================================
    # 3. TRANSFORMACIÓN LINEAL
    # (Será reemplazada por la transformación hacia QUBO)
    # =====================================================

    A = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    b = np.array([
        0.0,
        0.0
    ])

    y = A @ estado_vector + b

    # =====================================================
    # 4. MATRIZ QUBO (Placeholder)
    # =====================================================

    Q = np.zeros((3,3))

    # =====================================================
    # 5. DECISIÓN TEMPORAL
    # =====================================================

    accion = 1

    delta_u = 0.0
        # =====================================================
    # 6. ESPACIO DE DECISIONES
    # =====================================================

    # 0 -> Bajar
    # 1 -> Mantener
    # 2 -> Subir

    acciones = np.array([-1, 0, 1])

    # =====================================================
    # SOLVER TEMPORAL
    # (En el siguiente paso será sustituido por la QUBO)
    # =====================================================

    indice = 1

    accion = acciones[indice]

    # =====================================================
    # CAMBIO EN LA LIBERACIÓN
    # =====================================================

    delta_u = accion * 0.25 * np.median(R_obs)

    u_actual += delta_u

    # Límites físicos

    u_max = 2 * (0.25 * np.median(R_obs))

    u_min = -u_max

    u_actual = np.clip(
        u_actual,
        u_min,
        u_max
    )

    # =====================================================
    # ACTUALIZAR EL EMBALSE
    # =====================================================

    S_actual = S_actual - u_actual

    S_actual = np.clip(
        S_actual,
        0,
        V_MAX
    )