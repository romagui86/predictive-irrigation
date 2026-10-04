# Proyecto: riego predictivo con datos sintéticos

## 1. Alcance actual

Este proyecto es una prueba de concepto de ciencia de datos para predecir estrés hídrico en el
siguiente día y evaluar una recomendación de riego. Trabaja con un cultivo, zonas independientes y
datos sintéticos. El resultado no acredita eficacia agronómica en campo; aún se requiere validar
con sensores reales.

La comparación de manejo queda acotada a dos políticas:

- **P0 — calendario fijo:** aplica una dosis cada cierto número de días y no consulta sensores.
- **P1 — recomendación predictiva:** aplica la misma dosis cuando la probabilidad de estrés a 24 h,
  bajo el supuesto de no regar hoy, supera un umbral elegido en validación.

La antigua política reactiva con sensor crudo y la reactiva mejorada no forman parte del alcance
implementado.

## 2. Pregunta y criterio de evaluación

La pregunta es si una política predictiva reduce el costo frente a un calendario fijo, considerando
el compromiso entre agua aplicada y pérdida potencial de rendimiento. El proyecto también reporta
honestamente cuando la predicción no mejora la decisión frente a una regla sencilla.

El costo de referencia es un supuesto, no un precio observado:

```text
costo = 0.05 × agua aplicada + 100 × pérdida relativa de rendimiento
```

Las unidades de agua y drenaje son mm promedio por zona y temporada. Los valores económicos deben
ajustarse para cada contexto agrícola antes de interpretar recomendaciones.

## 3. Arquitectura del pipeline

1. El simulador crea clima, parámetros heterogéneos del suelo, lecturas con ruido/fallos y acciones
   de comportamiento variadas.
2. Un filtro de Kalman escalar estima la humedad a partir del balance nominal y las observaciones.
3. Se generan siete variables predictoras disponibles al momento de decidir.
4. Se entrenan modelos binarios para estimar estrés en el siguiente día.
5. P0 y P1 se simulan en lazo cerrado sobre las mismas semillas climáticas.
6. Se comparan costo, agua, drenaje y pérdida; se guardan métricas y visualizaciones.

La decisión se toma al comienzo del día. Los pronósticos y sensores disponibles entonces son las
entradas. El target se deriva de la verdad latente del simulador al inicio del día siguiente.

## 4. Simulador y datos

La corrida principal genera **200 temporadas × 8 zonas × 120 días = 192 000 filas**. Cada temporada
usa un clima seco o húmedo; todas las zonas dentro de una temporada comparten su clima, pero tienen
parámetros de suelo propios.

El simulador incluye:

- Lluvia mediante una cadena de Markov seco/húmedo, cantidades Gamma y dos regímenes climáticos.
- Temperatura estacional con ruido AR(1) y ET0 de Hargreaves con radiación FAO-56.
- Balance diario de agotamiento del suelo, estrés, eficiencia de riego y drenaje profundo.
- Variación entre zonas en capacidad de campo, punto de marchitez, profundidad radicular y fracción
  de agotamiento permisible.
- Sensor con sesgo, deriva, ruido gaussiano, datos faltantes y episodios de lectura pegada.
- Pronóstico de lluvia y temperatura imperfecto.
- Acciones variadas durante la generación de entrenamiento: calendario, reglas reactivas y acciones
  aleatorias.

### Archivos de datos

- `data/raw/dataset_sintetico.csv`: observaciones, acciones y verdad latente (`Dr`, humedad real,
  estrés, drenaje y variables del suelo). Sirve para auditoría y evaluación; no se entrega completo
  al modelo.
- `data/processed/features_sinteticas.csv`: IDs, observaciones permitidas, features y
  `estres_en_t+1`. Excluye estados latentes y resultados posteriores a la decisión.

## 5. Estimación y features

El filtro de Kalman escalar predice humedad con el balance nominal, la acción pasada, lluvia
observada y demanda estimada. Si hay lectura, actualiza el estado; si falta, conserva la predicción
y aumenta la incertidumbre.

Las siete features de P1 son:

1. `theta_est_t`: humedad estimada hoy.
2. `delta_theta`: cambio respecto al día anterior.
3. `dist_umbral`: distancia a un umbral nominal de humedad.
4. `etc_est`: demanda del cultivo estimada desde temperatura pronosticada.
5. `lluvia_pron_24h`: lluvia pronosticada.
6. `riego_t`: acción de riego.
7. `dias_desde_ultimo_riego`: historia de manejo.

Para la decisión contrafactual P1 se establece `riego_t = 0`. Las features no incluyen verdad
latente. La humedad cruda aparece en el baseline y en la ablación, pero no reemplaza la estimación
de estado en el modelo completo.

## 6. Modelos

- **B0 Persistencia:** clasifica estrés con un umbral nominal aplicado a la lectura actual.
- **B1 Regresión logística:** referencia interpretable con las siete features.
- **M Boosting:** `HistGradientBoostingClassifier`, modelo principal y motor de P1.

Los modelos completos se guardan con `joblib` en `models/`. `models/metadata.json` registra la
lista de features, la dosis y el umbral P1 seleccionado usando validación.

## 7. Split y evaluación

Las 200 simulaciones se dividen por `sim_id`: 140 para entrenamiento, 30 para validación y 30 para
test. Nunca se mezclan filas de una misma temporada entre los conjuntos. Los parámetros del
calendario y el umbral predictivo se ajustan en validación; la evaluación final se reporta en test.

### Métricas predictivas

PR-AUC es la métrica principal por el desbalance del target. También se calculan F1, precisión,
recall y Brier score. La curva de calibración ayuda a comprobar si las probabilidades son útiles para
la decisión.

### Métricas de política

- Agua aplicada por zona y temporada.
- Drenaje profundo.
- Pérdida relativa de rendimiento aproximada con factores de sensibilidad por etapa.
- Costo con los precios de referencia declarados.
- Barrido agua-rendimiento con parámetros candidatos evaluados en validación y políticas elegidas
  evaluadas en test.

Las comparaciones usan las mismas semillas de simulación, de modo que las políticas enfrentan las
mismas temporadas y realizaciones de clima.

### Ablaciones y robustez

Las ablaciones comparan humedad cruda, retiro del pronóstico, retiro de la demanda estimada y el
modelo completo. La robustez evalúa P1 con parámetros nominales de suelo mal estimados, sensor más
ruidoso, pronóstico sin acierto y cambio de entrenamiento húmedo a evaluación seca. Estos escenarios
no reajustan el umbral.

## 8. Resultados de la corrida de referencia

En el test de 30 temporadas, el modelo principal obtuvo PR-AUC aproximada de **0.896** y Brier de
**0.058**. El promedio de las políticas fue:

| Política | Agua (mm/zona/temporada) | Drenaje (mm/zona/temporada) | Pérdida relativa | Costo |
|---|---:|---:|---:|---:|
| P0 — calendario | 200.0 | 558.5 | 0.551 | 65.08 |
| P1 — predictiva | 332.7 | 606.2 | 0.017 | 18.35 |

Con los precios de referencia, P1 redujo la pérdida y el costo frente a P0, a costa de aplicar más
agua y producir algo más de drenaje. Esta conclusión depende de los supuestos económicos y del
simulador; no se interpreta como evidencia de rendimiento real en campo.

## 9. API y notebooks

La API WSGI de `api/app.py` publica `GET /health` y `POST /predict`. El endpoint recibe las siete
features, fuerza `riego_t = 0` y devuelve probabilidad de estrés a 24 h, decisión de riego y dosis.
Antes de iniciarla se deben generar los modelos con `python -m src.pipeline`.

Las notebooks están divididas por fase:

1. `01_clima.ipynb`: generación y visualización del clima.
2. `02_suelo_sensor.ipynb`: balance hídrico y observación del sensor.
3. `03_estimacion_features.ipynb`: filtro de Kalman y construcción de features.
4. `04_modelos.ipynb`: split, entrenamiento y métricas.
5. `05_politicas.ipynb`: evaluación P0/P1.
6. `06_ablaciones_robustez.ipynb`: calibración, ablaciones y escenarios de robustez.

## 10. Ejecución

Desde PowerShell, en la raíz del proyecto:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.pipeline
python -m api.app
```

La API queda en `http://127.0.0.1:8000`. Las métricas, tablas y figuras se guardan en `results/`.

## 11. Estructura actual

```text
api/                 API WSGI de inferencia
data/raw/            observaciones y verdad sintética
data/processed/      features admisibles y target
models/              clasificadores y metadata P1
notebooks/           seis notebooks por fase
results/             métricas, tablas y visualizaciones
src/                 simulador, Kalman, features y pipeline
tests/               pruebas del pipeline y de la API
```

## 12. Limitaciones

El simulador simplifica la respuesta del cultivo y usa una sola dosis fija. Las zonas son
independientes; no se representa la red hidráulica ni la variación espacial. El target se conoce
porque la verdad es sintética y no existiría directamente con sensores reales. Las probabilidades y
políticas requieren calibración, precios locales y validación agronómica antes de una aplicación
real.
