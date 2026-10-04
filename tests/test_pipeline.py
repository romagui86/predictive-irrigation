# Herramientas de prueba y componentes del pipeline que se verifican.
import unittest
import numpy as np
from src.features import make_features
from src.simulator import climate, simulate
from src.pipeline import PROCESSED_COLUMNS, split_ids


class PipelineTests(unittest.TestCase):
    # El split debe separar temporadas completas para bloquear fuga de clima.
    def test_split_keeps_simulations_disjoint(self):
        train, val, test = split_ids(200)
        self.assertEqual((len(train),len(val),len(test)),(140,30,30))
        self.assertFalse(set(train)&set(val) or set(train)&set(test) or set(val)&set(test))

    # Verificar dimensiones y rangos básicos del generador físico sintético.
    def test_climate_and_season_shapes(self):
        w=climate(1)
        self.assertEqual(len(w),120)
        self.assertTrue(w.et0.between(0,15).all())
        obs,truth,_=simulate(2,return_truth=True)
        self.assertEqual(len(obs),8*120)
        self.assertTrue(set(truth["estres_en_t+1"].unique()) <= {0,1})

    # Una política recibe pronósticos, no lluvia ni temperatura ya observadas.
    def test_policy_receives_forecasts_not_realized_weather(self):
        received=[]
        def policy(z,t,theta_obs,forecast,*args):
            received.append(set(forecast.index))
            return 0.0
        simulate(3,policy=policy)
        self.assertTrue(received)
        self.assertTrue(all(x=={"rain_forecast","temp_forecast"} for x in received))

    # El filtro tolera faltantes, reduce RMSE y mantiene verdad fuera de las features.
    def test_filter_handles_missing_sensor_without_latent_inputs(self):
        obs,truth,_=simulate(4,return_truth=True)
        obs=obs.iloc[:120].copy()
        truth=truth.iloc[:120]
        obs.loc[obs.index[4],"theta_obs"]=np.nan
        features=make_features(obs)
        self.assertTrue(np.isfinite(features.theta_est_t).all())
        raw_rmse=np.sqrt(np.nanmean((obs.theta_obs-truth.theta_real)**2))
        estimated_rmse=np.sqrt(np.mean((features.theta_est_t-truth.theta_real)**2))
        self.assertLess(estimated_rmse,raw_rmse)
        self.assertTrue({"theta_est_t","dist_umbral","etc_est"}.issubset(features.columns))
        self.assertFalse({"Dr","Ks","theta_real"}.intersection(features.columns))
        self.assertNotIn("drainage",PROCESSED_COLUMNS)


if __name__=="__main__": unittest.main()
