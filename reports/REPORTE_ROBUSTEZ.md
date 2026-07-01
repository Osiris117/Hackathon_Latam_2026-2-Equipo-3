# Analisis de Robustez - Falcon Reservoir

Este analisis evalua politicas bajo escenarios hidrologicos perturbados.
La ruta principal del proyecto no se modifica; este modulo agrega evidencia de resiliencia.

- Ventana: `2024-01-01`, `T=26`.
- Evaporacion: `disponible`.
- Mejor metodo por robustez: `QUBO+SA reoptimizado`.
- Peor SRS del mejor metodo: `-0.217459`.

## Ranking

| metodo | SRS_promedio | SRS_peor | SRS_desv | semanas_criticas_prom | fallos_totales | factibilidad_prom | robustez_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QUBO+SA reoptimizado | -0.162869 | -0.217459 | 0.027348 | 23.857143 | 0 | 1.000000 | -0.456030 |
| QUBO+SA nominal | -0.163411 | -0.219375 | 0.028021 | 23.857143 | 0 | 1.000000 | -0.457947 |
| Regla umbral | -0.167510 | -0.219708 | 0.026749 | 22.857143 | 1 | 0.857143 | -0.468279 |
| Historico | -0.176004 | -0.239564 | 0.031138 | 24.857143 | 0 | 1.000000 | -0.488136 |

## Archivos generados

- `detalle_escenarios.csv`
- `resumen_robustez.csv`
- `ranking_robustez.csv`
- `boxplot_srs_robustez.png`
- `heatmap_semanas_criticas.png`
- `ranking_robustez.png`
