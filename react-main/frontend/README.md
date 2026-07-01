# Falcon Reservoir Dashboard

Dashboard React/Vite para presentar resultados del Challenge A.

Sitio desplegado:

```text
https://osiris117.github.io/Hackathon_Latam_2026-2-Equipo-3/
```

## Ejecutar

```bash
cd "/Volumes/Osiris/Hack Cuantico/Hackathon_Latam_2026-2/react-main/frontend"
npm install
npm run dev
```

Abrir:

```text
http://127.0.0.1:5173/
```

## Compilar

```bash
npm run build
```

## Datos usados por la pagina

La pagina lee artefactos estaticos desde:

```text
public/data/
public/reports/
```

Archivos actuales:

```text
public/data/benchmark_delta_srs.csv
public/data/resumen_robustez.csv
public/data/detalle_escenarios.csv
public/data/qcentroid_qaoa_resultado.json
public/data/simulacion_hibrida.png
public/data/boxplot_srs_robustez.png
public/data/heatmap_semanas_criticas.png
public/data/ranking_robustez.png
public/data/qcentroid_sa_cpu_resultado.json
public/data/qcentroid_sa_gpu_resultado.json
public/data/qcentroid_qaoa_resultado.json
public/data/benchmark_qaoa/qaoa_validation_18q.png
public/data/benchmark_qaoa/scaling_search_runtime.png
public/data/benchmark_qaoa/cost_breakdown_srs.png
public/data/benchmark_qaoa/resumen_benchmark_qaoa.csv
public/reports/reporte_resultados_falcon.pdf
public/reports/reporte_resultados_falcon.tex
```

Los resultados finales de `sa_cpu`, `sa_gpu`, `qaoa` protegido, validacion QAOA reducida, escalabilidad y el reporte LaTeX ya estan conectados al dashboard.

## Enfoque

La pagina no ejecuta optimizadores. Solo visualiza resultados ya generados por los notebooks y scripts del proyecto. Esto mantiene separado el pipeline tecnico de la capa de presentacion.
