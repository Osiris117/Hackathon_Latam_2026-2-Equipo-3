# 🌊 Falcon Challenge — Documento Oficial
## Guided Challenge A: Resilient Release Scheduling for the International Falcon Reservoir

> **Fuente:** `FalconChallenge_V6.pdf` — Guided Quantum Computing Challenges for Transboundary Water Systems  
> **Fecha:** 24 de junio, 2026  
> **Contexto:** Cuenca del Río Bravo / Río Grande — Sistema binacional México–EE.UU.

---

## 🎯 ODS Aplicados

| ODS | Meta | Conexión |
|-----|------|----------|
| **SDG 6 — Agua limpia** | **6.4** — Eficiencia en uso de agua para 2030 | Gestión de disponibilidad limitada bajo escasez |
| **SDG 6 — Agua limpia** | **6.5** — Gestión integrada de recursos hídricos | Falcón es parte del sistema binacional México–EE.UU. |
| **SDG 13 — Acción climática** | **13.1** — Resiliencia ante riesgos climáticos | Sequías, períodos de bajo caudal, mayor evaporación |

**Pregunta central del reto:**
> ¿Puede la optimización basada en datos identificar calendarios de liberación que mejoren la resiliencia del embalse durante períodos de bajo almacenamiento?

---

## 📐 Sección 1 — La Métrica: Storage Resilience Score (SRS)

La métrica oficial del reto es el **SRS**, definido como:

```
SRS = -(w₁·Ccrit + w₂·Cdev + w₃·Csmooth)
```

**Queremos maximizar el SRS** (equivale a minimizar los tres costos).

### Componentes del SRS

| Término | Fórmula | Penaliza | Peso |
|---------|---------|----------|------|
| `Ccrit` | Σ [max(0, Smin − Sopt(t))]² para t=0..T | Caer bajo el almacenamiento mínimo crítico | **w₁ grande** |
| `Cdev` | Σ u(t)² para t=0..T-1 | Grandes desviaciones respecto a la operación histórica | w₂ pequeño |
| `Csmooth` | Σ [u(t) − u(t−1)]² para t=1..T-1 | Cambios bruscos semana a semana | w₃ pequeño |

### Ponderaciones oficiales

```
w₁ = 1 / [(T+1) · Sscale²]     donde Sscale = Smin
w₂ = 0.1 / [T · umax²]
w₃ = 0.1 / [(T−1) · (2·umax)²]
```

---

## ⚙️ Sección 2 — El Modelo del Embalse

### Variables

| Variable | Símbolo | Descripción |
|----------|---------|-------------|
| Almacenamiento optimizado | `Sopt(t)` | Nivel del embalse en la semana t bajo nuestra política |
| Liberación observada histórica | `Robs(t)` | Descarga histórica real — dato, no variable |
| Ajuste de liberación | `u(t)` | **Lo que controlamos** — nuestro ajuste semanal |
| Cambio natural de almacenamiento | `ΔSobs(t)` | Lluvia, evaporación, afluentes — dato, no controlable |

### Ecuación de dinámica del embalse

```
Sopt(t+1) = Sopt(t) + ΔSobs(t) − u(t)
```

- Si `u(t) = 0` → reproduce la operación histórica exactamente
- Si `u(t) > 0` → se libera **más** que históricamente (embalse baja más)
- Si `u(t) < 0` → se libera **menos** (embalse conserva agua)

La liberación resultante es: `R(t) = Robs(t) + u(t)`

### Los 5 niveles discretos de ajuste (benchmark oficial L=5)

```
u(t) ∈ {−2Δu, −Δu, 0, +Δu, +2Δu}

Δu   = 0.25 × mediana(Robs_semanal)
umax = 2 × Δu
```

---

## 🚧 Sección 2 — Restricciones (Feasibility)

El reto exige que toda solución cumpla **4 restricciones duras**:

| # | Restricción | Fórmula | Significado |
|---|-------------|---------|-------------|
| 1 | No-negatividad del flujo | `R(t) ≥ 0` | No se puede liberar agua negativa |
| 2 | Límite de ajuste | `|u(t)| ≤ umax` | El ajuste no puede ser mayor que 2·Δu |
| 3 | Capacidad física del embalse | `0 ≤ Sopt(t) ≤ Smax` | No desborde ni vaciado total |
| 4 | Balance acumulado (global) | `|Σ u(t)| ≤ η · Σ Robs(t)` con η = 0.10 | La solución no puede simplemente retener agua sin redistribuir |

> **Restricción 4 — Clave conceptual:** Evita que el optimizador "gane" simplemente liberando menos agua en todo el horizonte. Obliga a *redistribuir temporalmente* las liberaciones, no a reducirlas sistemáticamente.

---

## 📊 Sección 3 — Baselines (Comparadores)

### Baseline 0 — Replay Histórico

```
uhist(t) = 0  →  Rhist(t) = Robs(t)
```

Sin ninguna optimización. Reproduce la operación real. Su score: **SRShist**

### Baseline 1 — Regla de Umbral Clásica (obligatorio)

```
urule(t) = -Δu   si  Srule(t) < Smin
urule(t) =  0    si  Srule(t) >= Smin
```

Reduce liberaciones solo cuando el embalse está bajo el umbral crítico. Simple, reproducible, determinista.

### Baseline 2 — Optimización Clásica (opcional, recomendado)

Ejemplos válidos:
- Programación dinámica
- Simulated annealing
- Optimización entera mixta (MIP)
- **Algoritmo genético / evolutivo** ← implementado con DEAP en `genetico_v3.py`

> Las comparaciones contra el baseline opcional deben reportarse **separadas** del benchmark oficial.

---

## ⚛️ Sección 4 — Implementación Cuántica

### Enfoques recomendados

| Enfoque | Descripción |
|---------|-------------|
| **QAOA** | Quantum Approximate Optimization Algorithm |
| **Quantum Annealing** | D-Wave o similar |
| **Híbrido cuántico-clásico** | Clásico para parámetros, cuántico para búsqueda combinatoria |

### Lo que deben hacer los participantes

1. **Discretizar** los posibles ajustes de liberación
2. **Codificar** los ajustes como variables binarias
3. **Justificar** por qué el problema tiene estructura apropiada para optimización cuántica
4. **Implementación reproducible** — autocontenida y benchmarkeada

### Lo que el reto NO define (responsabilidad del equipo)

Los equipos son responsables de:
- Estrategia de discretización para instancias grandes
- Derivar la formulación **QUBO** o Ising
- Seleccionar el optimizador cuántico o híbrido
- Aplicar restricciones
- Ajustar parámetros
- Analizar escalabilidad

---

## 🏆 Sección 5 — Solución Benchmarkeada

```
ΔSRS = SRSopt − SRSbaseline
```

Los equipos deben reportar:
- **ΔSRS** respecto al baseline seleccionado
- **Tiempo de cómputo** del método clásico vs. cuántico/híbrido
- **Comportamiento de escalabilidad** al aumentar T y L

La liberación optimizada: `Ropt(t) = Robs(t) + uopt(t)`

---

## 📏 Sección 6 — Instancias de Escalabilidad

| Instancia | Semanas (T) | Niveles (L) | Uso |
|-----------|-------------|-------------|-----|
| **Small** | 12 | 3 | Validación y debugging |
| **Medium** (oficial) | **26** | **5** | **Benchmark oficial** |
| **Large** | 52 | 5 ó 7 | Análisis de escalabilidad |

### Explosión combinatoria: `N_schedules = L^T`

| Instancia | Cálculo | Horarios posibles |
|-----------|---------|-------------------|
| Small | 3^12 | 531,441 |
| Medium | **5^26** | **~1.49 × 10^18** |
| Large | 5^52 | ~2.22 × 10^36 |

> Esta es la justificación del uso cuántico: la instancia medium ya supera el número de operaciones posibles para cualquier búsqueda exhaustiva clásica.

---

## ⚠️ Sección 7 — Limitaciones Cuánticas y Alcance

> El hardware cuántico actual **no se espera que supere** a los métodos clásicos maduros para instancias grandes realistas.

**Propósito del reto:** No demostrar ventaja cuántica, sino evaluar si un problema simplificado de programación de liberaciones puede ser codificado, benchmarkeado y escalado con métodos cuánticos o híbridos.

**Advertencia importante:** El modelo del embalse es **intencionalmente simplificado**. No representa la política operacional oficial del Embalse Falcón, ni la complejidad legal, hidrológica, agrícola, municipal o política del sistema binacional completo.

---

## 💾 Sección 8 — Fuentes de Datos (IBWC)

Portal: `https://waterdata.ibwc.gov`

### Estación 08461200 — International Falcon Reservoir

| Dataset | Variable |
|---------|----------|
| `Total Storage.Web-Daily-tcm@08461200` | Almacenamiento total diario (TCM) |
| `Reservoir Elevation.Web-Daily-m@08461200` | Elevación diaria (metros) |
| `Evaporation,accumltd.Daily Evaporation - mm@08461200` | Evaporación diaria |
| `Discharge.Total.Change-in-Storage@08461200` | **ΔSobs(t)** — Cambio de almacenamiento |
| `Percentage.Conservation-Web-Telemetry@08461200` | % de conservación |

### Estación 08461300 — Rio Grande Below Falcon Dam

| Dataset | Variable |
|---------|----------|
| `Discharge.Best Available@08461300` | **Robs(t)** — Liberación histórica observada |

> Si el gasto viene en m³/s, debe convertirse a volumen por paso de tiempo (× 604,800 s/semana para TCM).

---

## 📦 Sección 9 — Dataset del Hackathon

**Carpeta oficial:**
```
https://cicesemx0-my.sharepoint.com/:f:/g/personal/fadomin_cicese_mx/
IgDamhhYBeWfSb73hdPj-EmXAWlRxnqXUyBlI0u9GUiJvrM?e=YDLUL0
```

Contiene las series de tiempo necesarias para reproducir el baseline histórico, calcular el SRS y comparar métodos.

---

## 📚 Referencias

| # | Referencia |
|---|-----------|
| [1] | Giuliani et al. (2021) — *Water Resources Research*, vol. 57. DOI: 10.1029/2021WR029927 |
| [2] | Sun et al. (2018) — *Water*, vol. 10, no. 11. DOI: 10.3390/w10111540 |
| [3] | Zhao & Zhao (2014) — *Mathematical Problems in Engineering*, Article ID 853186. DOI: 10.1155/2014/853186 |

---

## 🗺️ Flujo General del Reto

```
DATOS IBWC
  ├─ Robs(t)   ← Descarga histórica (08461300)
  └─ ΔSobs(t)  ← Cambio de almacenamiento (08461200)
          │
          ▼
  MODELO: Sopt(t+1) = Sopt(t) + ΔSobs(t) − u(t)
          │
          ▼
  OPTIMIZADOR (elige u(t) ∈ {-2Δu, -Δu, 0, +Δu, +2Δu})
  ├─ Baseline 0: u=0             → SRShist
  ├─ Baseline 1: Regla umbral    → SRSrule
  ├─ Baseline 2: Genético DEAP   → SRSclassical
  └─ Solución cuántica: QAOA     → SRSopt
          │
          ▼
  MÉTRICA: ΔSRS = SRSopt − SRSbaseline
```

---

## 🔗 Nuestros Archivos de Implementación

| Archivo | Rol en el challenge |
|---------|-------------------|
| `genetico_v3.py` | **Baseline 2** — Algoritmo Genético DEAP que maximiza el SRS |
| `entrenar_rna.py` | **Imitation Learning** — RNA que imita la política del genético |
| `modelado1.py` | **Pipeline híbrido** — RNA exploratoria + QUBO/SA conectado a evaluación SRS |
| `tabla_elevacion_volumen_FINAL.csv` | Batimetría h→V para el modelo de almacenamiento |
