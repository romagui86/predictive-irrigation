"""Filtro de Kalman escalar y features disponibles al decidir en el día t."""
# Dependencias para operaciones numéricas y tablas por zona.
import numpy as np
import pandas as pd

# Features que están disponibles antes de ejecutar la decisión del día.
FEATURES = ["theta_est_t", "delta_theta", "dist_umbral", "etc_est",
            "lluvia_pron_24h", "riego_t", "dias_desde_ultimo_riego"]


def estimate_group(group, q=.000025, r=.012**2):
    """Aplica el observador de Kalman a una zona, procesando días en orden.

    La predicción usa acciones y mediciones de lluvia pasadas; la actualización
    solo usa la lectura de humedad del día, si está disponible.
    """
    # Orden temporal y estado inicial nominal del filtro.
    g=group.sort_values("t").copy()
    estimates=[]; variances=[]; previous=None; p=.0004
    prev_action=prev_rain=prev_etc=0.0
    last_irrig=-120

    # Predecir humedad y covarianza; omitir actualización si falta el sensor.
    for row in g.itertuples():
        fc=float(row.theta_fc_nom); zr=float(row.zr_nom)
        if previous is None:
            prior=fc-.035
        else:
            prior=min(fc, previous+(0.9*prev_action+prev_rain-prev_etc)/(1000*zr))
        p_prior=p+q
        measurement=row.theta_obs
        if np.isfinite(measurement):
            gain=p_prior/(p_prior+r)
            estimate=prior+gain*(measurement-prior); p=(1-gain)*p_prior
        else: estimate=prior; p=p_prior
        estimates.append(estimate); variances.append(p)
        previous=estimate; prev_action=row.riego_t; prev_rain=row.lluvia_obs; prev_etc=row.etc_est
        if row.riego_t>0: last_irrig=row.t

    # Convertir la trayectoria estimada en features de tendencia y umbral.
    g["theta_est_t"]=estimates; g["var_theta"]=variances
    g["delta_theta"]=g.theta_est_t.diff().fillna(0)
    threshold=g.theta_fc_nom-.45*(g.theta_fc_nom-.12)
    g["dist_umbral"]=g.theta_est_t-threshold
    g["dias_desde_ultimo_riego"]=(g.t-g.last_irrig).clip(lower=0)
    return g


def make_features(frame):
    """Calcula features de cada zona por separado para no mezclar estados."""
    # Cada grupo es una serie independiente dentro de una simulación.
    groups=[estimate_group(group) for _,group in frame.groupby(["sim_id","zona_id"],sort=False)]
    return pd.concat(groups,ignore_index=True)
