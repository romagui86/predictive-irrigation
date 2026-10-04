"""Simulación sencilla de clima, suelo, sensores y acciones diversas."""
import numpy as np
import pandas as pd

# Constantes del ciclo fenológico único que se estudia en v1.
DAYS = 120
ZONES = 8
KC = np.array([0.7, 1.0, 1.15, 0.8])
KY = np.array([0.2, 0.6, 1.0, 0.5])


def climate(seed, regime="mixed", forecast_skill=0.7):
    """Genera una temporada climática y un pronóstico imperfecto reproducibles."""
    # Estados húmedo/seco con memoria y cantidades de lluvia Gamma.
    rng = np.random.default_rng(seed)
    wet = np.zeros(DAYS, dtype=bool)
    p_wet, p_stay = (0.12, 0.45) if regime == "dry" else (0.30, 0.65)
    wet[0] = rng.random() < p_wet
    for t in range(1, DAYS):
        wet[t] = rng.random() < (p_stay if wet[t-1] else p_wet)
    rain = np.where(wet, rng.gamma(2.0, 7.0, DAYS), 0.0)
    # Temperatura con variación estacional y ruido autorregresivo AR(1).
    ar = np.zeros(DAYS)
    noise = rng.normal(0, 2.0, DAYS)
    for t in range(1, DAYS):
        ar[t] = .85 * ar[t-1] + noise[t]
    temp = 23 + 5*np.sin(2*np.pi*np.arange(DAYS)/DAYS) + ar
    trange = np.maximum(4, rng.normal(12, 2, DAYS) - wet*3)
    tmax, tmin = temp + trange/2, temp - trange/2
    # Radiación extraterrestre FAO-56 y ET0 de Hargreaves en mm/día.
    lat = np.deg2rad(-12.0)
    doy = np.arange(1, DAYS+1)
    dr = 1 + .033*np.cos(2*np.pi*doy/365)
    delta = .409*np.sin(2*np.pi*doy/365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat)*np.tan(delta), -1, 1))
    ra = (24*60/np.pi)*.0820*dr*(ws*np.sin(lat)*np.sin(delta)+np.cos(lat)*np.cos(delta)*np.sin(ws))
    et0 = np.maximum(0, .0023*(temp+17.8)*np.sqrt(trange)*ra*.408)
    # El acierto controla detección de lluvia y el pronóstico conserva error de cantidad.
    detect = rng.random(DAYS) < forecast_skill
    forecast = np.where(wet & detect, rain*rng.uniform(.55, 1.1, DAYS), 0.0)
    false_alarm = (~wet) & (rng.random(DAYS) < (1-forecast_skill)*.25)
    forecast[false_alarm] = rng.gamma(1.5, 3.0, false_alarm.sum())
    return pd.DataFrame({"t": np.arange(DAYS), "rain": rain, "rain_forecast": forecast,
                         "temp_forecast": temp+rng.normal(0, 2, DAYS), "et0": et0,
                         "temp": temp})


def stage(t):
    """Asigna cada día a una de las cuatro etapas de 30 días."""
    return min(3, t // 30)


def forecast_et0(t, temp, trange=12.0):
    """Estima ET0 con temperatura pronosticada y rango térmico nominal."""
    # Recalcular Ra para el día del ciclo usando la latitud fija del caso sintético.
    lat=np.deg2rad(-12.0); doy=t+1
    dr=1+.033*np.cos(2*np.pi*doy/365)
    delta=.409*np.sin(2*np.pi*doy/365-1.39)
    ws=np.arccos(np.clip(-np.tan(lat)*np.tan(delta),-1,1))
    ra=(24*60/np.pi)*.0820*dr*(ws*np.sin(lat)*np.sin(delta)+np.cos(lat)*np.cos(delta)*np.sin(ws))
    return max(0,.0023*(temp+17.8)*np.sqrt(trange)*ra*.408)


def simulate(seed, regime="mixed", policy=None, forecast_skill=.7, sensor_sigma=.012,
             soil_error=0.0, return_truth=False):
    """Simula una temporada de suelo, sensores y manejo para ocho zonas.

    Sin callback de política se muestrea manejo diverso para formar entrenamiento.
    Con callback, solo se entrega información observable y nominal al decisor.
    """
    # Semilla separada para parámetros de suelo y mediciones; clima comparte seed.
    rng = np.random.default_rng(seed + 100_003)
    weather = climate(seed, regime, forecast_skill)
    soil = pd.DataFrame({"zone_id": np.arange(ZONES),
        "fc": rng.normal(.30, .012, ZONES).clip(.26,.34),
        "wp": rng.normal(.12, .009, ZONES).clip(.09,.15),
        "zr": rng.uniform(.45,.75,ZONES), "p": rng.uniform(.38,.55,ZONES)})
    rows, truth = [], []
    # Evolucionar cada zona de forma independiente bajo el mismo clima.
    for z, s in soil.iterrows():
        taw = 1000*(s.fc-s.wp)*s.zr
        raw = s.p*taw
        dr = rng.uniform(.2,.55)*raw
        bias, drift = rng.normal(0,.008), 0.0
        stuck = None
        last_irrig = -120
        behavior = rng.integers(0,4)
        interval = int(rng.integers(5,13))
        threshold = rng.uniform(.35,.65)*raw
        for t, w in weather.iterrows():
            # La humedad real se conserva como verdad; la medición incluye sesgo y fallos.
            theta = s.fc-dr/(1000*s.zr)
            noisy = theta+bias+drift+rng.normal(0,sensor_sigma)
            if stuck is not None and t < stuck[1]: noisy=stuck[0]
            elif rng.random()<.018: stuck=(noisy,t+int(rng.integers(3,8)))
            if rng.random()<.03: noisy=np.nan
            obs_rain = max(0, (weather.rain.iloc[t-1] if t else 0)+rng.normal(0,.7))
            kc=KC[stage(t)]; etc=kc*w.et0
            etc_forecast=kc*forecast_et0(t,w.temp_forecast)
            # Visible model receives only nominal parameters, never the actual soil parameters.
            fc_nom=.30+soil_error; zr_nom=.60+soil_error
            theta_threshold=fc_nom-((.45*1000*(fc_nom-.12)*zr_nom)/(1000*zr_nom))
            # En entrenamiento se alternan calendario, reacción y acción aleatoria.
            if policy is None:
                if behavior==0: action=10.0 if t%interval==0 else 0.0
                elif behavior==1: action=10.0 if np.isfinite(noisy) and noisy < theta_threshold+rng.uniform(-.012,.012) else 0.0
                elif behavior==2: action=10.0 if np.isfinite(noisy) and noisy < theta_threshold+.015 and w.rain_forecast<5 else 0.0
                else: action=10.0 if rng.random()<.22 else 0.0
            else:
                forecast=w[["rain_forecast","temp_forecast"]]
                action=float(policy(z,t,noisy,forecast,last_irrig,fc_nom,zr_nom,
                                    (rows[-1]["riego_t"] if t else 0.0),obs_rain,
                                    (rows[-1]["etc_est"] if t else etc_forecast)))
            # Balance de agua: riego efectivo, lluvia, ETa y drenaje profundo.
            dr_next=max(0, dr-.9*action-w.rain*.85+etc*(1 if dr<=raw else max(0,(taw-dr)/(taw-raw))))
            drainage=max(0, .9*action+w.rain*.85+dr-etc-dr_next)
            stress_next=int(dr_next>raw)
            rows.append({"sim_id":seed,"zona_id":z,"t":t,"etapa":stage(t),"theta_obs":noisy,
                "lluvia_obs":obs_rain,"lluvia_pron_24h":w.rain_forecast,"temp_pron":w.temp_forecast,
                "riego_t":action,"etc_est":etc_forecast,"theta_fc_nom":fc_nom,
                "zr_nom":zr_nom,"drainage":drainage,"last_irrig":last_irrig})
            truth.append({"Dr":dr,"theta_real":theta,"estres_real":int(dr>raw),
                "estres_en_t+1":stress_next,"Ks":1 if dr<=raw else max(0,(taw-dr)/(taw-raw)),
                "ETa":etc*(1 if dr<=raw else max(0,(taw-dr)/(taw-raw))),"ETc":etc,"RAW":raw})
            dr=dr_next; drift+=rng.normal(0,.00035)
            if action>0: last_irrig=t
    # Devolver observaciones y verdad por separado para impedir fuga al modelo.
    obs=pd.DataFrame(rows); latent=pd.DataFrame(truth)
    if return_truth: return obs, latent, weather
    return obs
