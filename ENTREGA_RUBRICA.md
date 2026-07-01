# Guia de cumplimiento de rubrica

Este documento mapea la entrega contra la rubrica indicada. No reemplaza las diapositivas; sirve como checklist tecnico para que la presentacion y la defensa oral apunten a los criterios correctos.

## 1. Problem Formulation - 25%

**Que debe verse:** problema claro, especifico, conectado con literatura/ODS, impacto social.

**Evidencia del repo:**

- `README.md`: descripcion del Challenge A, datos IBWC, SRS y ODS 6.4, 6.5, 13.1.
- `hackatonlatam2026/reporte_resultados_falcon.tex`: secciones `Contexto del reto`, `Datos e instancia benchmark`, `Metrica oficial`.
- `hackatonlatam2026/FalconChallenge_V6.md`: documento base del reto.

**Mensaje clave para defensa:**

El problema no es "predecir agua"; es optimizar una politica semanal de liberacion `u(t)` para conservar resiliencia del almacenamiento sin violar restricciones fisicas ni balance acumulado.

**ODS/impacto:**

- ODS 6.4: eficiencia y sostenibilidad en uso de agua.
- ODS 6.5: gestion integrada de recursos hidricos transfronterizos.
- ODS 13.1: resiliencia ante riesgos climaticos/sequia.

## 2. Baseline - 20%

**Que debe verse:** metricas validas y comparadores claros.

**Evidencia del repo:**

- Metrica oficial: `SRS = -(w1*Ccrit + w2*Cdev + w3*Csmooth)`.
- Baseline 0: historico `u(t)=0`.
- Baseline 1: regla de umbral.
- Baseline 2: genetico DEAP.
- Tabla final en `resultados/benchmark_delta_srs.csv`.

**Resultados clave:**

| Metodo | SRS | Delta SRS vs historico |
|---|---:|---:|
| Historico | -0.156312 | 0.000000 |
| Regla umbral | -0.149991 | +0.006321 |
| Genetico DEAP | -0.145371 | +0.010941 |
| QUBO+SA | -0.145371 | +0.010941 |

**Mensaje clave para defensa:**

QUBO+SA iguala al mejor baseline clasico conocido y supera tanto al historico como a la regla de umbral.

## 3. Quantum Implementation - 25%

**Que debe verse:** codigo correcto, limpio, documentado y enfoque cuantico justificado.

**Evidencia del repo:**

- `hackatonlatam2026/qubo_solver.py`: construccion QUBO 130x130.
- `hackatonlatam2026/qcentroid_solver.py`: modos `sa_cpu`, `sa_gpu`, `qaoa`.
- `hackatonlatam2026/FalconChallenge_QCentroid.ipynb`: notebook QCentroid.
- `Versiones/reservoir_falcon_benchmark.ipynb`: validacion QAOA reducida.

**Justificacion cuantica/hibrida:**

- La politica se codifica como variables binarias one-hot.
- La instancia oficial tiene `T=26`, `L=5`, por lo tanto `130` variables binarias.
- El espacio de busqueda es `5^26 ~= 1.49e18`.
- La forma QUBO/Ising es natural para QAOA, annealing cuantico o solvers hibridos.

**Punto importante:**

No afirmar que QAOA de 130 qubits resolvio oficialmente el problema. El resultado oficial estable es QUBO+SA. QAOA se valida en una instancia real reducida de 18 qubits y queda como ruta experimental para hardware/simuladores adecuados.

## 4. Benchmarking - 20%

**Que debe verse:** escalabilidad y comparacion directa contra baseline.

**Evidencia del repo:**

- `resultados/benchmark_delta_srs.csv`: comparacion nominal.
- `resultados/resultados_benchmark_qaoa/`: validacion QAOA reducida y escalabilidad.
- `resultados/resultados_robustez/`: escenarios de estres.
- Dashboard React: graficas de benchmark, robustez, QAOA y escalabilidad.

**Escalabilidad:**

| Instancia | T | L | Qubits | Politicas |
|---|---:|---:|---:|---:|
| Validacion QAOA reducida | 6 | 3 | 18 | 729 |
| Medium oficial | 26 | 5 | 130 | 1.49e18 |

**Robustez:**

QUBO+SA reoptimizado obtuvo el mejor `robustez_score = -0.456030`; QUBO+SA nominal fue segundo. Esto muestra que la politica no solo funciona en el caso nominal.

## 5. Quality of Presentation - 10%

**Que debe verse:** comprension del reto, explicacion de resultados, ventajas y desventajas.

**Ventajas a explicar:**

- Usa datos reales IBWC.
- Reporta SRS oficial y restricciones.
- Compara contra baselines claros.
- QUBO+SA es reproducible y escalable frente a busqueda exhaustiva.
- Se agrega robustez bajo escenarios perturbados.

**Desventajas/limitaciones a decir abiertamente:**

- Modelo de embalse simplificado.
- QAOA completo de 130 qubits no es realista en simulacion directa.
- El solver QUBO usa aproximacion cuadratica para `Ccrit`, por eso se reevalua con SRS real.
- No sustituye reglas operativas oficiales del embalse.

## Nota requerida: uso de herramientas de apoyo

La rubrica pide explicar el uso de herramientas de apoyo. Propuesta breve para
incluir en la entrega:

```text
AI-assisted tools were used as support for code review, documentation drafting,
debugging guidance, and organization of the final report/dashboard. The team
defined the problem formulation, selected the hydrological data, implemented
and ran the optimization notebooks/scripts, verified the numerical results, and
is responsible for the final technical interpretation.
```

Version breve en espanol para defensa oral:

```text
Usamos herramientas de apoyo para revisar codigo, ordenar documentacion y acelerar
la redaccion tecnica. Los datos, corridas, validacion numerica, interpretacion y
decision final de la solucion son responsabilidad del equipo.
```

## Fuentes base para citar

- UN SDG 6: https://sdgs.un.org/goals/goal6
- UN SDG 13: https://sdgs.un.org/goals/goal13
- Farhi, Goldstone, Gutmann. A Quantum Approximate Optimization Algorithm. https://arxiv.org/abs/1411.4028
- Lucas. Ising formulations of many NP problems. https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2014.00005/full
- IBWC Water Data: https://waterdata.ibwc.gov
