# Guia: ejecutar en QCentroid

Este documento explica como correr el solver del Falcon Reservoir Challenge en QCentroid. Hay dos rutas principales:

- **SA GPU (`modo="sa_gpu"`)**: ruta recomendada para la entrega. Es rapida, estable y reproduce el benchmark `T=26, L=5`.
- **QAOA (`modo="qaoa"`)**: ruta cuantica/variacional. Es util para demostrar formulacion cuantica, pero para `T=26` implica 130 qubits y puede ser muy costosa o fallar segun backend.

El archivo principal para QCentroid es:

```text
qcentroid_solver.py
```

Punto de entrada:

```python
run(use_case, solver_params)
```

El notebook `FalconChallenge_QCentroid.ipynb` ya incluye un selector por variable y corre por defecto `modo="sa_gpu"` al ejecutar todas las celdas.

Tambien existen dos notebooks de ejecucion directa por modo:

| Notebook | Modo fijo | Carpeta de salida |
|---|---|---|
| `FalconChallenge_QCentroid_sa_cpu.ipynb` | `sa_cpu` | `resultados/resultados_sa_cpu/` |
| `FalconChallenge_QCentroid_sa_gpu.ipynb` | `sa_gpu` | `resultados/resultados_sa_gpu/` |

Nota importante sobre QCentroid: se intento usar `ipywidgets` con botones/radios, pero en el entorno de QCentroid se imprimieron como texto (`RadioButtons(...)`) en vez de renderizarse como controles interactivos. Por eso la opcion confiable para la entrega es elegir el modo editando la variable `MODO_QCENTROID_ELEGIDO` en la celda de seleccion.

---

## 1. Archivos que debes subir a QCentroid

Sube la carpeta `hackatonlatam2026/` completa o, como minimo, estos archivos:

**Notebook**

```text
FalconChallenge_QCentroid.ipynb
FalconChallenge_QCentroid_sa_cpu.ipynb
FalconChallenge_QCentroid_sa_gpu.ipynb
```

**Scripts**

```text
qcentroid_solver.py
qubo_solver.py
genetico_v3.py
```

**Datos**

```text
R_observ.xlsx
DataSetExport-Discharge Total.Change-in-Storage@08461200-Instantaneous-TCM-20260622185956.csv
DataSetExport-Total Storage.Web-Daily-ac-ft@08461200-Instantaneous-TCM-20260622185130.csv
tabla_elevacion_volumen_FINAL.csv
```

---

## 2. Instalar dependencias

En una celda del notebook de QCentroid:

```python
!pip install cudaq cupy-cuda12x pandas numpy scikit-learn deap scipy joblib openpyxl matplotlib
```

Notas:

- Si el entorno usa CUDA 11, cambia `cupy-cuda12x` por `cupy-cuda11x`.
- QCentroid puede traer CUDA-Q preinstalado; si ya esta instalado, no pasa nada.
- Para `sa_cpu` no necesitas GPU ni CuPy.

---

## 3. Cargar datos reales

Ejecuta esta celda antes de elegir el modo:

```python
import sys
import pandas as pd

# Ajusta esta ruta si subiste el proyecto en otra ubicacion.
PROJECT_PATH = "."
sys.path.insert(0, PROJECT_PATH)

from genetico_v3 import preparar_ventana_semanal
from qcentroid_solver import run, MockUseCase

df_lib = pd.read_excel("R_observ.xlsx")
df_cambio = pd.read_csv(
    "DataSetExport-Discharge Total.Change-in-Storage@08461200-Instantaneous-TCM-20260622185956.csv",
    skiprows=1,
)
df_total = pd.read_csv(
    "DataSetExport-Total Storage.Web-Daily-ac-ft@08461200-Instantaneous-TCM-20260622185130.csv",
    skiprows=1,
)
df_evap = pd.DataFrame(columns=["Timestamp (UTC-06:00)", "Evaporacion_mm"])
df_batimetria = pd.read_csv("tabla_elevacion_volumen_FINAL.csv")

R_obs, Delta_S_obs, S_inicial = preparar_ventana_semanal(
    df_lib,
    df_cambio,
    df_total,
    df_evap,
    df_batimetria,
    "2024-01-01",
    26,
)

use_case = MockUseCase(R_obs, Delta_S_obs, S_inicial)
print("Datos listos:", len(R_obs), "semanas")
```

---

## 4. Selector de modo en QCentroid

En QCentroid usa el selector por variable. Es mas robusto que los botones porque no depende de que el entorno renderice `ipywidgets`.

En el notebook `FalconChallenge_QCentroid.ipynb`, ve a la celda **"Selector de modo QCentroid"** y cambia solo estas variables:

```python
MODO_QCENTROID_ELEGIDO = "sa_gpu"
FORZAR_QAOA_130_QUBITS = False
```

Opciones:

| Valor | Cuando usarlo | Nota al seleccionarlo |
|---|---|---|
| `"sa_gpu"` | Corrida oficial/recomendada | Rapido, estable y adecuado para `T=26`, `L=5`. No es QAOA puro; es annealing clasico/hibrido sobre QUBO. |
| `"qaoa"` | Demostracion cuantica/variacional | Para `T=26` requiere 130 qubits. Con `FORZAR_QAOA_130_QUBITS=False`, el solver protege la corrida y cae a `sa_gpu`; con `True`, intenta QAOA real y puede tardar o fallar segun backend. |
| `"sa_cpu"` | Fallback sin GPU | No requiere GPU ni CuPy, pero puede tardar mas y conviene usar menos iteraciones. |

Ejemplos:

```python
# Recomendado para entrega
MODO_QCENTROID_ELEGIDO = "sa_gpu"
FORZAR_QAOA_130_QUBITS = False

# Demo QAOA protegida: intenta ruta QAOA, pero evita romper con 130 qubits
MODO_QCENTROID_ELEGIDO = "qaoa"
FORZAR_QAOA_130_QUBITS = False

# QAOA forzado: solo si quieres probar explicitamente 130 qubits
MODO_QCENTROID_ELEGIDO = "qaoa"
FORZAR_QAOA_130_QUBITS = True

# Fallback sin GPU
MODO_QCENTROID_ELEGIDO = "sa_cpu"
FORZAR_QAOA_130_QUBITS = False
```

Si das **Restart kernel and run all cells**, el notebook correra con el valor que este escrito en esa celda. Si lo dejas como `"sa_gpu"`, ya corre la ruta recomendada sin preguntarte nada mas.

---

## 5. Opcion directa: SA GPU recomendado

Usa esto para la corrida principal de entrega:

```python
solver_params = {
    "backend": "nvidia",
    "modo": "sa_gpu",
    "n_restarts": 128,
    "n_iter": 500_000,
    "T": 26,
}

resultado = run(use_case, solver_params)
print(f"SRS = {resultado['SRS']:.6f}")
print(f"Tiempo = {resultado['tiempo_s']:.1f} s")
```

Pros:

- Estable para `T=26`, `L=5`.
- Usa GPU si CuPy esta disponible.
- Evalua SRS real y factibilidad.
- Es la mejor ruta para resultados reproducibles.

Contras:

- Es annealing clasico/hibrido, no una ejecucion QAOA pura.

---

## 6. Opcion directa: QAOA

### QAOA protegido

Esta opcion intenta QAOA, pero si `T=26` implica demasiados qubits y no se fuerza, el solver cae a `sa_gpu`.

```python
solver_params = {
    "backend": "nvidia",
    "modo": "qaoa",
    "n_layers": 2,
    "T": 26,
    "force_qaoa": False,
}

resultado = run(use_case, solver_params)
print(f"SRS = {resultado['SRS']:.6f}")
print(resultado["meta"])
```

### QAOA forzado

Usalo solo si el entorno tiene backend suficiente y quieres intentar la ruta cuantica completa:

```python
solver_params = {
    "backend": "nvidia",
    # "backend": "tensornet",
    "modo": "qaoa",
    "n_layers": 2,
    "T": 26,
    "force_qaoa": True,
    "qaoa_max_qubits": 130,
}

resultado = run(use_case, solver_params)
print(f"SRS = {resultado['SRS']:.6f}")
```

Pros:

- Es la ruta cuantica/variacional.
- Sirve para defender la formulacion Ising/QUBO en un marco QAOA.

Contras:

- `T=26`, `L=5` requiere 130 qubits.
- Puede tardar mucho o fallar por memoria/backend.
- La calidad puede variar con `n_layers`, optimizador y shots.

---

## 7. SA CPU fallback

```python
solver_params = {
    "modo": "sa_cpu",
    "T": 26,
}

resultado = run(use_case, solver_params)
print(f"SRS = {resultado['SRS']:.6f}")
```

Pros:

- No requiere GPU.
- Sirve para validar que el pipeline funciona.

Contras:

- Es mas lento.
- No aprovecha QCentroid GPU.

---

## 8. Como solver registrado en QuantumOps

Si tienes acceso a `qcentroid_sdk`:

```python
from qcentroid_sdk import QCentroidClient

client = QCentroidClient(api_key="TU_API_KEY")

dataset_id = client.upload_dataset({
    "R_obs": R_obs.tolist(),
    "Delta_S_obs": Delta_S_obs.tolist(),
    "S_inicial": float(S_inicial),
})

job = client.run_job(
    solver_file="qcentroid_solver.py",
    dataset_id=dataset_id,
    solver_params={
        "backend": "nvidia",
        "modo": "sa_gpu",
        "n_restarts": 128,
        "n_iter": 500_000,
        "T": 26,
    },
)

resultado = job.wait_for_result()
print(resultado)
```

---

## 9. Pruebas locales

Desde la carpeta `hackatonlatam2026/`:

```bash
# Fallback CPU
python qcentroid_solver.py --local --modo sa_cpu

# GPU con CuPy/NVIDIA
python qcentroid_solver.py --local --modo sa_gpu --n_restarts 64 --n_iter 200000

# QAOA protegido: para T=26 hara fallback si no se fuerza desde solver_params
python qcentroid_solver.py --local --modo qaoa --backend nvidia --n_layers 2
```

---

## 10. Recomendacion para entrega

Para benchmark oficial y demo estable:

```text
usar modo="sa_gpu"
```

Para narrativa cuantica:

```text
explicar formulacion QUBO/Ising y mostrar QAOA como ruta disponible,
pero reportar claramente si se uso fallback por limite de qubits.
```

Para intentar QAOA real:

```text
usar force_qaoa=True solo en un entorno GPU/tensornet adecuado.
```
