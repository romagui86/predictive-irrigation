# Herramientas para construir solicitudes WSGI de prueba sin abrir un puerto.
import io
import json
import unittest
import numpy as np
import pandas as pd
from wsgiref.util import setup_testing_defaults
from api.app import create_app
from src.features import FEATURES


class FakeModel:
    # Sustituye el modelo serializado para probar la API sin depender del entrenamiento.
    def predict_proba(self, frame):
        self.input=frame
        return np.array([[.1,.9]])


class ApiTests(unittest.TestCase):
    # La solicitud P1 evalúa el contrafactual forzando la acción de hoy a cero.
    def test_predict_returns_action_and_uses_no_irrigation_counterfactual(self):
        model=FakeModel()
        app=create_app(model,threshold=.5,dose_mm=10)
        payload={"features":{name:1 for name in FEATURES}}
        body=json.dumps(payload).encode()
        env={}; setup_testing_defaults(env)
        env.update({"REQUEST_METHOD":"POST","PATH_INFO":"/predict",
                    "CONTENT_LENGTH":str(len(body)),"wsgi.input":io.BytesIO(body)})
        result={}
        response=b"".join(app(env,lambda status,headers:result.update(status=status,headers=headers)))
        self.assertEqual(result["status"],"200 OK")
        self.assertEqual(json.loads(response)["riego_recomendado_mm"],10)
        self.assertEqual(model.input.loc[0,"riego_t"],0)

    # Un registro incompleto debe fallar con un mensaje HTTP claro.
    def test_predict_rejects_missing_features(self):
        app=create_app(FakeModel())
        body=json.dumps({"features":{}}).encode()
        env={}; setup_testing_defaults(env)
        env.update({"REQUEST_METHOD":"POST","PATH_INFO":"/predict",
                    "CONTENT_LENGTH":str(len(body)),"wsgi.input":io.BytesIO(body)})
        result={}
        response=b"".join(app(env,lambda status,headers:result.update(status=status)))
        self.assertEqual(result["status"],"400 Bad Request")
        self.assertIn("error",json.loads(response))


if __name__=="__main__": unittest.main()
