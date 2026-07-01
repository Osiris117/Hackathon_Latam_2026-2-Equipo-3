# Resilient Release Scheduling for the International Falcon Reservoir

**Guided Quantum Computing Challenge - Hackathon Latinoamerica 2026**  
Challenge A | SDG 6.4, 6.5, 13.1 | Rio Bravo / Rio Grande Basin

Este repositorio implementa un pipeline clasico-cuantico/hibrido para optimizar calendarios semanales de liberacion de agua del Embalse Internacional Falcon. El objetivo es mejorar el **Storage Resilience Score (SRS)** durante periodos de bajo almacenamiento usando datos reales de IBWC.

---

## Enlaces de entrega

- [Diapositivas del Equipo 3](https://docs.google.com/presentation/d/1MqzQAkY8zrc-ki2wj6XYoOeNK-eEwUsQtrqIDfzMslA/edit?pli=1&slide=id.p#slide=id.p)
- Dashboard local: `react-main/frontend/`
- Reporte final: `hackatonlatam2026/reporte_resultados_falcon.pdf`

---

## 1. Resumen rapido

La decision semanal es un ajuste `u(t)` sobre la liberacion historica:

```text
Ropt(t) = Robs(t) + u(t)
Sopt(t+1) = Sopt(t) + Delta_S_obs(t) - u(t)
```

Instancia oficial:

```text
T = 26 semanas
L = 5 niveles
u(t) in {-2*Delta_u, -Delta_u, 0, Delta_u, 2*Delta_u}
QUBO = 26 * 5 = 130 variables binarias
Espacio = 5^26 ~= 1.49e18 calendarios posibles
```

Metricas:

```text
SRS = -(w1*Ccrit + w2*Cdev + w3*Csmooth)
Delta SRS = SRSopt - SRShist
```

Restricciones oficiales:

```text
R(t) >= 0
|u(t)| <= u_max
0 <= Sopt(t) <= Smax
|sum(u(t))| <= 0.10 * sum(Robs(t))
```

---

## 2. Estructura del repositorio

```text
Hackathon_Latam_2026-2/
├── README.md
├── react-main/
│   └── frontend/
├── hackatonlatam2026/
│   ├── FalconChallenge_V6.md
│   ├── FalconChallenge_QCentroid.ipynb
│   ├── FalconChallenge_QCentroid_sa_cpu.ipynb
│   ├── FalconChallenge_QCentroid_sa_gpu.ipynb
│   ├── QCENTROID_GUIDE.md
│   ├── CONTEXTO_TECNICO.md
│   ├── genetico_v3.py
│   ├── entrenar_rna.py
│   ├── modelado1.py
│   ├── robustez_escenarios.py
│   ├── qubo_solver.py
│   ├── qcentroid_solver.py
│   ├── qubo_formulation_notes.md
│   ├── reporte_resultados_falcon.tex
│   ├── resultados/
│   └── datos IBWC / modelos .pkl
├── Versiones/
│   └── notebooks toy QAOA
└── Logica/
    └── copias de notas/enunciado
```

Si eres nuevo en el repo, lee primero:

```text
hackatonlatam2026/CONTEXTO_TECNICO.md
hackatonlatam2026/QCENTROID_GUIDE.md
```

---

## 3. Archivos principales

| Archivo | Rol |
|---|---|
| `react-main/frontend/` | Dashboard web React para presentar benchmark, QCentroid, robustez y reportes. |
| `ENTREGA_RUBRICA.md` | Checklist de entrega mapeado contra la rubrica oficial y nota de uso de herramientas de apoyo. |
| `hackatonlatam2026/modelado1.py` | Pipeline completo: datos, RNA, genetico, QUBO+SA, Gurobi opcional, resultados y graficas. |
| `hackatonlatam2026/robustez_escenarios.py` | Analisis de robustez bajo escenarios hidrologicos perturbados; guarda en `resultados/resultados_robustez/`. |
| `hackatonlatam2026/genetico_v3.py` | Baseline clasico DEAP, SRS oficial, auditoria, Gurobi MIQP opcional. |
| `hackatonlatam2026/qubo_solver.py` | Construccion QUBO 130x130 y solver SA con evaluacion SRS real. |
| `hackatonlatam2026/qcentroid_solver.py` | Solver listo para QCentroid: `sa_gpu`, `sa_cpu`, `qaoa`. |
| `hackatonlatam2026/entrenar_rna.py` | Entrena RNA que imita politica del genetico. |
| `Versiones/reservoir_falcon_benchmark.ipynb` | Notebook de validacion QAOA reducida y benchmark de escalabilidad. |
| `hackatonlatam2026/FalconChallenge_QCentroid_sa_cpu.ipynb` | Notebook QCentroid fijado a `sa_cpu`; guarda en `resultados/resultados_sa_cpu/`. |
| `hackatonlatam2026/FalconChallenge_QCentroid_sa_gpu.ipynb` | Notebook QCentroid fijado a `sa_gpu`; guarda en `resultados/resultados_sa_gpu/`. |
| `hackatonlatam2026/QCENTROID_GUIDE.md` | Guia de ejecucion en QCentroid con selector de modo por variable. |
| `hackatonlatam2026/CONTEXTO_TECNICO.md` | Bitacora tecnica del pipeline, decisiones de implementacion y continuidad de desarrollo. |
| `hackatonlatam2026/reporte_resultados_falcon.tex` | Reporte LaTeX final con resultados nominales, QCentroid, robustez, figuras y trazabilidad. |

---

## 4. Datos

Datos principales usados:

```text
R_observ.xlsx
DataSetExport-Discharge Total.Change-in-Storage@08461200-Instantaneous-TCM-20260622185956.csv
DataSetExport-Total Storage.Web-Daily-ac-ft@08461200-Instantaneous-TCM-20260622185130.csv
DataSetExport-Reservoir Elevation.Web-Daily-m@08461200-Aggregate-m-20260629164752.csv
tabla_elevacion_volumen_FINAL.csv
```

Fuentes:

- Estacion `08461200` - International Falcon Reservoir.
- Estacion `08461300` - Rio Grande Below Falcon Dam.
- Portal IBWC: <https://waterdata.ibwc.gov>

---

## 5. Instalacion local

Desde la carpeta del proyecto:

```bash
cd "/Volumes/Osiris/Hack Cuantico/Hackathon_Latam_2026-2/hackatonlatam2026"
python -m pip install pandas numpy scikit-learn deap scipy joblib matplotlib openpyxl
```

Opcional para benchmark exacto:

```bash
python -m pip install gurobipy
```

Nota: Gurobi puede requerir licencia. Si `gurobipy` no esta disponible, el pipeline debe continuar sin Gurobi.

---

## 6. Ejecucion local

### Dashboard web

```bash
cd "/Volumes/Osiris/Hack Cuantico/Hackathon_Latam_2026-2/react-main/frontend"
npm install
npm run dev
```

Abrir:

```text
http://127.0.0.1:5173/
```

La pagina visualiza resultados ya generados desde `public/data/` y `public/reports/`. No ejecuta optimizadores; sirve como capa de presentacion para benchmark, QCentroid, robustez y reporte.

### Pipeline completo

```bash
cd "/Volumes/Osiris/Hack Cuantico/Hackathon_Latam_2026-2/hackatonlatam2026"
python modelado1.py
```

Esto:

1. Carga la RNA guardada o la entrena si no existe.
2. Reconstruye la ventana oficial `T=26` desde `2024-01-01`.
3. Corre baseline genetico DEAP.
4. Corre QUBO + Simulated Annealing.
5. Intenta Gurobi MIQP si esta disponible.
6. Calcula tabla `SRS` y `Delta SRS`.
7. Genera CSVs y graficas.

Salidas esperadas:

```text
hackatonlatam2026/resultados/benchmark_delta_srs.csv
hackatonlatam2026/resultados/simulacion_rna.csv
hackatonlatam2026/resultados/simulacion_hibrida.png
hackatonlatam2026/resultados/convergencia_genetico.png
hackatonlatam2026/resultados/comparativa_estrategias.png
```

### Solo QUBO

```bash
python qubo_solver.py
```

### Reentrenar RNA

```bash
python entrenar_rna.py
```

### Analisis de robustez

Este modulo agrega evidencia de resiliencia sin modificar la ruta principal. Evalua politicas bajo escenarios hidrologicos perturbados:

- nominal
- sequia 10%
- sequia 20%
- ruido semanal suave
- ruido semanal fuerte
- bloque critico de cinco semanas
- desfase temporal de cambios de almacenamiento

Comando recomendado:

```bash
python robustez_escenarios.py
```

Comando rapido de validacion:

```bash
python robustez_escenarios.py --n-restarts 2 --n-iter 1000 --sin-reoptimizar
```

Salidas:

```text
hackatonlatam2026/resultados/resultados_robustez/REPORTE_ROBUSTEZ.md
hackatonlatam2026/resultados/resultados_robustez/detalle_escenarios.csv
hackatonlatam2026/resultados/resultados_robustez/resumen_robustez.csv
hackatonlatam2026/resultados/resultados_robustez/ranking_robustez.csv
hackatonlatam2026/resultados/resultados_robustez/boxplot_srs_robustez.png
hackatonlatam2026/resultados/resultados_robustez/heatmap_semanas_criticas.png
hackatonlatam2026/resultados/resultados_robustez/ranking_robustez.png
```

Resultado de la corrida registrada: `QUBO+SA reoptimizado` obtuvo el mejor `robustez_score`, seguido por `QUBO+SA nominal`; ambos fueron factibles en todos los escenarios evaluados.

---

## 7. Ejecucion en QCentroid

Lee la guia completa:

```text
hackatonlatam2026/QCENTROID_GUIDE.md
```

Resumen:

El notebook `hackatonlatam2026/FalconChallenge_QCentroid.ipynb` usa un selector por variable en la celda **"Selector de modo QCentroid"**:

```python
MODO_QCENTROID_ELEGIDO = "sa_gpu"
FORZAR_QAOA_130_QUBITS = False
```

QCentroid no renderizo correctamente los controles `ipywidgets` de botones/radios durante la prueba; los mostro como texto. Por eso el flujo recomendado es editar `MODO_QCENTROID_ELEGIDO` antes de ejecutar la celda:

| Valor | Recomendacion |
|---|---|
| `"sa_gpu"` | Recomendado para resultados oficiales: rapido, estable y adecuado para `T=26`, `L=5`. |
| `"qaoa"` | Demostrativo/experimental: para `T=26` implica 130 qubits; con `FORZAR_QAOA_130_QUBITS=False` hay fallback a `sa_gpu`. |
| `"sa_cpu"` | Fallback sin GPU: mas lento, pero robusto si no hay CUDA/CuPy. |

Si usas **Restart kernel and run all cells**, se ejecutara el modo que este escrito en esa celda. Si no cambias nada, corre `sa_gpu`, que es la ruta recomendada.

Tambien hay dos notebooks separados para corridas limpias por modo:

| Notebook | Modo fijo | Carpeta de salida |
|---|---|---|
| `hackatonlatam2026/FalconChallenge_QCentroid_sa_cpu.ipynb` | `sa_cpu` | `hackatonlatam2026/resultados/resultados_sa_cpu/` |
| `hackatonlatam2026/FalconChallenge_QCentroid_sa_gpu.ipynb` | `sa_gpu` | `hackatonlatam2026/resultados/resultados_sa_gpu/` |

### Modo recomendado: SA GPU

```python
solver_params = {
    "backend": "nvidia",
    "modo": "sa_gpu",
    "n_restarts": 128,
    "n_iter": 500_000,
    "T": 26,
}
resultado = run(use_case, solver_params)
```

Pros:

- Rapido y estable.
- Adecuado para `T=26`, `L=5`.
- Recomendado para benchmark oficial.

### Modo QAOA

```python
solver_params = {
    "backend": "nvidia",
    "modo": "qaoa",
    "n_layers": 2,
    "T": 26,
    "force_qaoa": False,
}
resultado = run(use_case, solver_params)
```

Nota: para `T=26`, QAOA implica 130 qubits. Por seguridad, `qcentroid_solver.py` hace fallback automatico a `sa_gpu` si no se define `force_qaoa=True`.

Para forzar QAOA:

```python
solver_params = {
    "backend": "nvidia",
    "modo": "qaoa",
    "n_layers": 2,
    "T": 26,
    "force_qaoa": True,
    "qaoa_max_qubits": 130,
}
```

---

## 8. Resultados benchmark conocidos

Resultados previos reportados para `T=26`, `L=5`:

| Metodo | SRS | Delta SRS vs historico |
|---|---:|---:|
| Historico `u=0` | -0.1563 | 0 |
| Regla de umbral | -0.1500 | +0.0063 |
| Genetico DEAP | -0.1454 | +0.0109 |
| QUBO + SA | -0.1454 | +0.0109 |

Interpretacion:

```text
QUBO + SA iguala la calidad del mejor baseline clasico conocido.
```

Los numeros finales deben recalcularse despues de cada cambio con:

```bash
python modelado1.py
```

---

## 9. Como explicar la solucion

Mensaje recomendado:

```text
Usamos datos reales IBWC para construir una instancia semanal T=26/L=5.
Formulamos el problema como QUBO one-hot de 130 variables binarias.
Comparamos historico, regla de umbral, genetico DEAP, QUBO+SA y Gurobi opcional.
El solver QUBO usa una aproximacion cuadratica de Ccrit, pero la evaluacion final
se hace con SRS real y auditoria de restricciones oficiales.
Adicionalmente, evaluamos robustez bajo escenarios de estres hidrologico para
mostrar estabilidad de la solucion frente a incertidumbre.
```

No conviene decir:

```text
"QAOA de 130 qubits ya supera a los clasicos"
```

Si no se corrio y valido explicitamente.

---

## 10. Reporte LaTeX

Reporte final:

```text
hackatonlatam2026/reporte_resultados_falcon.tex
hackatonlatam2026/reporte_resultados_falcon.pdf
```

Checklist de rubrica:

```text
ENTREGA_RUBRICA.md
```

El reporte ya integra:

1. Tabla nominal de SRS y Delta SRS.
2. Corridas QCentroid `sa_cpu`, `sa_gpu` y `qaoa` protegido.
3. Validacion QAOA reducida de 18 qubits y benchmark de escalabilidad.
4. Analisis de robustez bajo siete escenarios.
5. Figuras generadas por el pipeline.
6. Limitaciones, uso de herramientas de apoyo y checklist de reproduccion.

Para recompilar:

```bash
pdflatex reporte_resultados_falcon.tex
```

Si usas Overleaf, sube:

```text
reporte_resultados_falcon.tex
resultados/simulacion_hibrida.png
resultados/comparativa_estrategias.png
resultados/convergencia_genetico.png
```

---

## 11. Notas sobre notebooks de `Versiones`

La carpeta `Versiones/` contiene notebooks QAOA de juguete:

```text
T=3 o T=6
L=3
datos sinteticos
```

Sirven como demostracion QAOA pequena, pero no reemplazan el pipeline oficial `T=26`, `L=5` con datos reales.

---
