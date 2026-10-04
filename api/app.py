"""API WSGI mínima para consultar riesgo y recomendación de la política P1."""
# Librerías HTTP/JSON y lectura del pipeline serializado.
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.features import FEATURES


# Construcción de la aplicación y rutas disponibles para el cliente.
def create_app(model, threshold=.2, dose_mm=10):
    """Define health check y predicción contrafactual sin depender de un framework."""
    def application(environ, start_response):
        method=environ.get("REQUEST_METHOD","GET")
        path=environ.get("PATH_INFO","/")
        if method=="GET" and path=="/health":
            return respond(start_response,"200 OK",{"status":"ok"})
        if method!="POST" or path!="/predict":
            return respond(start_response,"404 Not Found",{"error":"Ruta no encontrada"})
        # Validar que la solicitud contenga todas las variables numéricas del modelo.
        try:
            size=int(environ.get("CONTENT_LENGTH") or 0)
            payload=json.loads(environ["wsgi.input"].read(size))
            values=payload["features"]
            row={name:float(values[name]) for name in FEATURES}
            if not np.isfinite(list(row.values())).all(): raise ValueError("Las features deben ser finitas")
        except (KeyError,TypeError,ValueError,json.JSONDecodeError) as error:
            return respond(start_response,"400 Bad Request",{"error":str(error)})
        # P1 consulta el riesgo bajo el contrafactual de no regar hoy.
        row["riego_t"]=0.0
        # Convertir la probabilidad estimada en la acción fija de la política P1.
        probability=float(model.predict_proba(pd.DataFrame([row],columns=FEATURES))[0,1])
        irrigate=probability>threshold
        return respond(start_response,"200 OK",{"probabilidad_estres_24h":probability,
            "regar":irrigate,"riego_recomendado_mm":dose_mm if irrigate else 0,
            "umbral":threshold})
    return application


def respond(start_response, status, body):
    """Codifica la respuesta común como JSON WSGI."""
    data=json.dumps(body,ensure_ascii=False).encode("utf-8")
    start_response(status,[("Content-Type","application/json; charset=utf-8"),
                           ("Content-Length",str(len(data)))])
    return [data]


def load_app(model_dir="models"):
    """Carga el clasificador persistido y el umbral elegido en validación."""
    folder=Path(model_dir)
    model=joblib.load(folder/"M_boosting.joblib")
    metadata=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    return create_app(model,metadata["decision_threshold"],metadata["dose_mm"])


_cached_app=None


def application(environ, start_response):
    """Punto de entrada WSGI; carga el modelo una sola vez por proceso."""
    global _cached_app
    try:
        if _cached_app is None: _cached_app=load_app(os.environ.get("RIEGO_MODEL_DIR","models"))
    except FileNotFoundError:
        return respond(start_response,"503 Service Unavailable",{"error":"Ejecute primero el pipeline para entrenar el modelo"})
    return _cached_app(environ,start_response)


if __name__=="__main__":
    # Servidor local para desarrollo; los valores se pueden cambiar por variables de entorno.
    from wsgiref.simple_server import make_server
    host=os.environ.get("RIEGO_API_HOST","127.0.0.1")
    port=int(os.environ.get("RIEGO_API_PORT","8000"))
    print(f"API disponible en http://{host}:{port}")
    make_server(host,port,application).serve_forever()
