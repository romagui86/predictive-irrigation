# Riego predictivo

Prueba de concepto con datos sintéticos para estimar humedad de suelo y predecir estrés hídrico a
24 horas. El sistema combina un filtro de Kalman escalar y modelos de clasificación con dos
políticas: **P0**, calendario fijo, y **P1**, recomendación basada en la probabilidad de estrés
calculada con el contrafactual de no regar hoy.

Los resultados solo describen el simulador. La validación con sensores de campo sigue pendiente.

## Requisitos e instalación

Requiere Python 3.10 o superior. Desde PowerShell, en la carpeta raíz:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Ejecutar el pipeline

```powershell
python -m unittest discover -s tests -v
python -m src.pipeline
```

La ejecución por defecto produce 200 temporadas independientes de 120 días y 8 zonas: 140
temporadas para entrenamiento, 30 para validación y 30 para test. El split se realiza por
`sim_id`; las filas de una temporada nunca se reparten entre conjuntos. Se ajustan los modelos y
los parámetros de P0/P1 en entrenamiento y validación. Las métricas finales se calculan en test.

La dosis de riego es 10 mm. El costo es un supuesto de referencia: 0.05 unidades monetarias por
mm aplicado más 100 por unidad de pérdida relativa de rendimiento. Se puede ejecutar una muestra
pequeña para revisar el código:

```powershell
python -c "from src.pipeline import run; run(n=20)"
```

Esta ejecución rápida sobrescribe las salidas y no reemplaza el experimento de 200 temporadas.

## Estructura

```text
api/                 API HTTP WSGI
data/raw/            observaciones sintéticas y verdad latente para evaluación
data/processed/      features observables y target del modelo
models/              modelos serializados y metadata de P1
notebooks/           análisis separado por fase
results/             métricas, calibración, ablaciones, robustez y gráficas
src/                 simulador, filtro, features y pipeline
tests/               pruebas del pipeline y de la API
```

Los datos raw contienen campos latentes (`Dr`, `theta_real`, `Ks`, parámetros del suelo, drenaje y
target). Se guardan para auditar la simulación y evaluar, pero **no son features**. La tabla
procesada conserva identificadores, observaciones, features permitidas y `estres_en_t+1`; excluye
estados latentes y resultados posteriores a la decisión, como el drenaje.

## Notebooks

Abre la carpeta `notebooks/` en Jupyter o VS Code después de ejecutar el pipeline. Para configurar
un kernel local de Jupyter se instala `ipykernel` con `requirements.txt`.

- `01_clima.ipynb`: lluvia, temperatura, pronóstico y ET0.
- `02_suelo_sensor.ipynb`: balance hídrico y lecturas de sensores.
- `03_estimacion_features.ipynb`: filtro de Kalman, RMSE y features.
- `04_modelos.ipynb`: modelos predictivos y métricas.
- `05_politicas.ipynb`: comparación P0/P1 y frontera agua-rendimiento.
- `06_ablaciones_robustez.ipynb`: ablaciones, calibración y sensibilidad.

## API

Primero entrena y guarda el modelo con `python -m src.pipeline`. Luego, en una terminal aparte,
inicia el servidor:

```bash
python -m api.app
```

La API local queda en `http://127.0.0.1:8000`. `GET /health` comprueba el servicio y
`POST /predict` recibe las siete features: `theta_est_t`, `delta_theta`, `dist_umbral`, `etc_est`,
`lluvia_pron_24h`, `riego_t` y `dias_desde_ultimo_riego`. Para tomar la decisión P1, el servicio
fija `riego_t=0` y devuelve la probabilidad de estrés a 24 h, la decisión y la dosis sugerida.

Ejemplo en Bash:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "theta_est_t": 0.22,
      "delta_theta": -0.003,
      "dist_umbral": -0.01,
      "etc_est": 4.2,
      "lluvia_pron_24h": 0.0,
      "riego_t": 0.0,
      "dias_desde_ultimo_riego": 4
    }
  }'
echo
```

El umbral de decisión que devuelve la API se guarda en `models/metadata.json` y se selecciona en
validación. Los modelos son específicos de esta simulación y requieren validación antes de usarse
para manejo agrícola real.

## Archivos de resultados

- `results/metricas_prediccion.csv`: PR-AUC, F1, precisión, recall y Brier en test.
- `results/metricas_politicas.csv`: agua, drenaje, pérdida y costo para P0/P1 en test.
- `results/frontera_validacion.csv` y `results/frontera_pareto.png`: barrido de parámetros de P0/P1
  en validación y puntos seleccionados evaluados en test.
- `results/ablaciones.csv`, `results/calibracion.csv` y `results/robustez.csv`: análisis adicionales.
# predictive-irrigation
