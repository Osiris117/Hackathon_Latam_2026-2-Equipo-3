# Bitacora Tecnica - Falcon Reservoir Challenge A

Este documento resume el estado del proyecto para que cualquier integrante del equipo pueda retomar el trabajo sin perder contexto tecnico.

## Objetivo del proyecto

Estamos resolviendo el **Challenge A: Resilient Release Scheduling for the International Falcon Reservoir** del hackathon. El objetivo es encontrar una politica semanal de liberacion de agua para el Embalse Falcon que mejore el **Storage Resilience Score (SRS)** durante periodos de bajo almacenamiento.

La variable de decision es el ajuste semanal sobre la liberacion historica:

```text
Ropt(t) = Robs(t) + u(t)
Sopt(t+1) = Sopt(t) + Delta_S_obs(t) - u(t)
```

La instancia oficial benchmark es:

```text
T = 26 semanas
L = 5 niveles
u(t) in {-2*Delta_u, -Delta_u, 0, Delta_u, 2*Delta_u}
Variables QUBO = 26 * 5 = 130 bits
Espacio de busqueda = 5^26 ~= 1.49e18 schedules
```

## Metrica oficial

El score oficial es:

```text
SRS = -(w1*Ccrit + w2*Cdev + w3*Csmooth)
```

Donde:

```text
Ccrit   = sum_t max(0, Smin - Sopt(t))^2
Cdev    = sum_t u(t)^2
Csmooth = sum_t (u(t) - u(t-1))^2
Smin    = 0.25 * Smax
Delta_u = 0.25 * median(Robs semanal)
u_max   = 2 * Delta_u
```

Restricciones oficiales:

```text
1. R(t) >= 0
2. |u(t)| <= u_max
3. 0 <= Sopt(t) <= Smax
4. |sum_t u(t)| <= 0.10 * sum_t Robs(t)
```

Nota importante: la restriccion 4 oficial es sobre `abs(sum(u))`, no sobre `sum(abs(u))`.

## Carpeta principal

Ruta del proyecto:

```text
/Volumes/Osiris/Hack Cuantico/Hackathon_Latam_2026-2/hackatonlatam2026
```

Archivos principales:

```text
modelado1.py
genetico_v3.py
robustez_escenarios.py
qubo_solver.py
qcentroid_solver.py
entrenar_rna.py
FalconChallenge_QCentroid.ipynb
qubo_formulation_notes.md
FalconChallenge_V6.md
QCENTROID_GUIDE.md
```

Tambien hay datos reales IBWC:

```text
R_observ.xlsx
DataSetExport-Discharge.Best Available@08461300-Instantaneous-m^3 s-20260622190542.csv
DataSetExport-Discharge Total.Change-in-Storage@08461200-Instantaneous-TCM-20260622185956.csv
DataSetExport-Total Storage.Web-Daily-ac-ft@08461200-Instantaneous-TCM-20260622185130.csv
DataSetExport-Reservoir Elevation.Web-Daily-m@08461200-Aggregate-m-20260629164752.csv
tabla_elevacion_volumen_FINAL.csv
```

Fallback opcional revisado durante desarrollo:

```text
FALCON_EXTRA_DATA_DIR=/ruta/a/Hackathon_Latam_2026-4/hackatonlatam2026
```

Contenido util detectado:

- Archivo de evaporacion diaria:
  `DataSetExport-Evaporation,accumltd.Daily Evaporation - mm@08461200-Instantaneous-mm-20260622185804.csv`
- Versiones alternativas/simplificadas de almacenamiento total y cambio de almacenamiento.
- Misma batimetria y datos base IBWC.

`robustez_escenarios.py` busca primero los archivos en la carpeta principal del proyecto y, si falta algun CSV compatible, puede usar la ruta opcional definida en `FALCON_EXTRA_DATA_DIR`. En la corrida registrada, la evaporacion aparecio como `disponible`.

## Rol de cada codigo

### `genetico_v3.py`

Es el modulo clasico principal. Contiene:

- `preparar_ventana_semanal(...)`: resamplea datos a semanas.
- `run_genetico(...)`: algoritmo genetico DEAP para optimizar `u(t)`.
- `calcular_srs(...)`: calcula SRS oficial.
- `reportar_delta_srs(...)`: compara historico, regla umbral y solucion optimizada.
- `auditar_restricciones(...)`: verifica factibilidad.

Mejoras integradas desde `genetico_v7_vd.py`:

- `calcular_parametros_oficiales(...)`
- `visualizar_variables_calculadas(...)`
- `resolver_benchmark_gurobi(...)`
- `graficar_convergencia(...)`
- `graficar_comparativa_estrategias(...)`

La integracion de Gurobi es **opcional**: si `gurobipy` no esta instalado/licenciado, el pipeline lo omite sin romper.

### `qubo_solver.py`

Construye la matriz QUBO oficial/aproximada:

```text
Q = Q_dev + Q_smooth + Q_crit + Q_onehot + Q_flujo + Q_balance
```

Detalles importantes:

- Usa one-hot con 5 niveles por semana.
- `Q_crit` aproxima el termino `max(0, ...)` quitando temporalmente el max.
- El simulated annealing evalua el **SRS real** para corregir esa aproximacion.
- El solver usa movimientos one-hot, por lo que evita soluciones binarias invalidas.

Conclusion tecnica: es mejor describirlo como **QUBO hibrido con SA y evaluacion exacta del SRS/factibilidad**, no como QAOA puro.

### `entrenar_rna.py`

Entrena una `MLPClassifier` que imita decisiones del genetico:

- Carga datos semanales.
- Corre genetico para generar etiquetas `Decision_Optima_u`.
- Construye features hidrologicas y temporales.
- Guarda:

```text
modelo_rna_genetico.pkl
scaler_rna.pkl
```

La RNA es util como capa de politica/aprendizaje, pero no es el benchmark oficial principal.

### `modelado1.py`

Pipeline integrador:

1. Carga o entrena la RNA.
2. Prepara ventana oficial `FECHA_INICIO = "2024-01-01"`, `T=26`.
3. Imprime parametros oficiales.
4. Corre baseline genetico.
5. Corre QUBO + SA.
6. Intenta correr Gurobi MIQP si esta disponible.
7. Simula politica futura con RNA.
8. Calcula tabla `Delta SRS`.
9. Guarda resultados y graficas.

Salidas esperadas:

```text
resultados/benchmark_delta_srs.csv
resultados/simulacion_rna.csv
resultados/simulacion_hibrida.png
resultados/convergencia_genetico.png
resultados/comparativa_estrategias.png
```

### `qcentroid_solver.py`

Archivo pensado para QCentroid/QuantumOps. Punto de entrada:

```python
run(use_case, solver_params)
```

Modos:

```text
sa_cpu
sa_gpu
qaoa
```

Selector en notebook QCentroid:

- El notebook `FalconChallenge_QCentroid.ipynb` usa selector por variable, no botones.
- Variable principal: `MODO_QCENTROID_ELEGIDO = "sa_gpu" | "qaoa" | "sa_cpu"`.
- Variable de proteccion QAOA: `FORZAR_QAOA_130_QUBITS = False | True`.
- Motivo: en QCentroid, los widgets `ipywidgets` se imprimieron como texto (`RadioButtons(...)`) y no se renderizaron como controles interactivos. La ruta confiable es editar la variable en la celda de seleccion.
- Si el usuario corre **Restart kernel and run all cells**, se ejecuta el valor escrito en esa celda. Por default debe quedar `"sa_gpu"`.

Este archivo fue modificado. Cambios realizados:

- Si se pide `modo="qaoa"` con demasiados qubits, cae automaticamente a `sa_gpu` salvo que `force_qaoa=True`.
- Se agrego `qaoa_max_qubits`, por default `24`.
- La decodificacion de QAOA ahora elige el bitstring one-hot valido mas frecuente.

Motivo: QAOA para 130 qubits es poco realista en simulacion directa; el fallback protege la demo y evita fallos durante entrega.

Para intentar QAOA de todos modos:

```python
solver_params = {
    "modo": "qaoa",
    "backend": "nvidia",
    "force_qaoa": True,
    "qaoa_max_qubits": 130,
    "n_layers": 2,
    "T": 26,
}
```

### `robustez_escenarios.py`

Modulo adicional de resiliencia. No reemplaza `modelado1.py` ni modifica el pipeline principal.

Objetivo:

- Tomar la ventana oficial `T=26`, `L=5`.
- Calcular politicas base: Historico, Regla umbral, `QUBO+SA nominal`.
- Generar escenarios hidrologicos perturbados:
  - nominal
  - sequia 10%
  - sequia 20%
  - ruido semanal suave
  - ruido semanal fuerte
  - bloque critico de cinco semanas
  - desfase temporal de cambios de almacenamiento
- Evaluar SRS, semanas criticas, factibilidad y fallos por metodo.
- Opcionalmente reoptimizar QUBO+SA dentro de cada escenario.

Comando recomendado:

```bash
python robustez_escenarios.py
```

Comando rapido:

```bash
python robustez_escenarios.py --n-restarts 2 --n-iter 1000 --sin-reoptimizar
```

Salidas:

```text
resultados/resultados_robustez/REPORTE_ROBUSTEZ.md
resultados/resultados_robustez/detalle_escenarios.csv
resultados/resultados_robustez/resumen_robustez.csv
resultados/resultados_robustez/ranking_robustez.csv
resultados/resultados_robustez/boxplot_srs_robustez.png
resultados/resultados_robustez/heatmap_semanas_criticas.png
resultados/resultados_robustez/ranking_robustez.png
```

Resultado de corrida registrada:

```text
QUBO+SA reoptimizado  robustez_score = -0.456030
QUBO+SA nominal       robustez_score = -0.457947
Regla umbral          robustez_score = -0.468279
Historico             robustez_score = -0.488136
```

Interpretacion: la historia de entrega puede decir que no solo se optimizo el caso nominal, sino que se evaluo la estabilidad de las politicas bajo incertidumbre hidrologica. En esa evaluacion, QUBO+SA fue factible en todos los escenarios y lidero el ranking de robustez.

## Codigo nuevo de companeros

Archivo anexado:

```text
genetico_v7_vd.py
```

Contenido valioso:

- Panel de variables calculadas.
- Benchmark exacto con Gurobi MIQP.
- Graficas de convergencia y comparativa.

Partes que NO se copiaron tal cual:

- `!pip install` dentro de `.py`.
- Carga de datos al importar.
- Dependencia obligatoria de Gurobi.
- Restriccion `sum(abs(u)) <= 0.10*sum(Robs)`, porque no es la restriccion oficial.

Se integro lo util como funciones opcionales dentro de `genetico_v3.py`.

## Notebooks de `Versiones`

Ruta:

```text
/Volumes/Osiris/Hack Cuantico/Hackathon_Latam_2026-2/Versiones
```

Contiene notebooks QAOA toy:

```text
reservoir_qubo_qaoa_interesting_toy_t4_gpu_fixed_sampler_v2_fixed.ipynb
reservoir_qubo_qaoa_gpu_complex_t6.ipynb
reservoir_qubo_qaoa_gpu_complex_t6_transpiled_fixed.ipynb
```

Diagnostico:

- Son ejemplos de juguete `T=3` o `T=6`, `L=3`.
- No usan datos reales.
- No reemplazan el pipeline oficial.
- Sirven para una demo/apendice QAOA pequena.
- El notebook `T=6` original fallo con `AerError: unknown instruction: QAOA`.
- El `T=6_transpiled_fixed` agrega un fix con transpiler/pass manager, pero no tiene una ejecucion QAOA final claramente guardada.

## Resultados reportados antes de las mejoras

README reportaba para Medium `T=26, L=5`:

```text
Historico:        SRS = -0.1563
Regla umbral:     SRS = -0.1500
Genetico DEAP:    SRS = -0.1454
QUBO + SA:        SRS = -0.1454
```

Interpretacion:

- QUBO + SA iguala al mejor baseline clasico.
- Delta SRS vs historico aprox `+0.0109`.

## Dependencias

Dependencias minimas:

```bash
pip install pandas numpy scikit-learn deap scipy joblib matplotlib openpyxl
```

Dependencia opcional:

```bash
pip install gurobipy
```

Gurobi puede requerir licencia valida. Si no esta disponible, el pipeline debe seguir funcionando.

Para QAOA/Qiskit toy:

```bash
pip install qiskit qiskit-aer qiskit-optimization qiskit-algorithms
```

Para QCentroid/CUDA-Q puede requerirse el entorno propio de QCentroid.

## Comando principal recomendado

Desde la carpeta:

```bash
cd "/Volumes/Osiris/Hack Cuantico/Hackathon_Latam_2026-2/hackatonlatam2026"
python modelado1.py
```

Si se quiere probar solo QUBO:

```bash
python qubo_solver.py
```

Si se quiere probar QCentroid local:

```bash
python qcentroid_solver.py --local --modo sa_cpu
```

## Advertencias tecnicas

1. `Q_crit` en QUBO es una aproximacion cuadratica sin `max(0, ...)`.
2. El SA de `qubo_solver.py` evalua SRS real y restricciones, por eso la solucion final se reporta con metrica oficial.
3. QAOA de 130 qubits no debe ser promesa principal.
4. La solucion fuerte para entrega es:

```text
Historico vs Regla umbral vs Genetico vs QUBO+SA vs Gurobi opcional
```

5. Gurobi debe presentarse como benchmark exacto/clasico opcional, no como parte cuantica.

## Siguiente trabajo recomendado

Estado registrado al 1 de julio de 2026:

```text
SRS historico:              -0.156312
SRS regla umbral:           -0.149991
SRS genetico DEAP:          -0.145371
SRS QUBO+SA:                -0.145371
Delta SRS QUBO+SA:          +0.010941
QCentroid sa_cpu:           -0.145371
QCentroid sa_gpu:           -0.145371
QCentroid qaoa protegido:   -0.145371 con fallback desde qaoa a sa_gpu
Validacion QAOA reducida:   T=6, L=3, 18 qubits; QAOA alcanza minimo QUBO exacto
Mejor robustez:             QUBO+SA reoptimizado, score -0.456030
```

Artefactos finales:

```text
ENTREGA_RUBRICA.md
hackatonlatam2026/reporte_resultados_falcon.tex
hackatonlatam2026/reporte_resultados_falcon.pdf
react-main/frontend/public/reports/reporte_resultados_falcon.pdf
react-main/frontend/public/data/qcentroid_sa_cpu_resultado.json
react-main/frontend/public/data/qcentroid_sa_gpu_resultado.json
react-main/frontend/public/data/qcentroid_qaoa_resultado.json
hackatonlatam2026/resultados/resultados_benchmark_qaoa/
react-main/frontend/public/data/benchmark_qaoa/
```

Trabajo recomendado si se generan nuevas corridas:

1. Reemplazar los CSV/JSON en `resultados/`.
2. Recompilar `reporte_resultados_falcon.tex`.
3. Copiar el PDF/LaTeX y JSON nuevos a `react-main/frontend/public/`.
4. Ejecutar `npm run build` en `react-main/frontend`.
5. Mantener esta narrativa final:

```text
Usamos datos reales IBWC, construimos un benchmark oficial T=26/L=5,
comparamos baselines clasicos con una formulacion QUBO hibrida,
y validamos factibilidad contra las cuatro restricciones oficiales.
```

## Nota para continuidad de desarrollo

```text
Para continuar el desarrollo, revisar primero
hackatonlatam2026/CONTEXTO_TECNICO.md y despues revisar modelado1.py,
genetico_v3.py, qubo_solver.py y qcentroid_solver.py.

No reemplazar el pipeline principal con los notebooks toy de Versiones.
La solucion fuerte es el benchmark oficial T=26/L=5 con datos reales:
Historico, Regla umbral, Genetico DEAP, QUBO+SA y Gurobi opcional.

Si se modifica QCentroid, conservar el fallback que evita intentar QAOA
de 130 qubits salvo force_qaoa=True.
```
