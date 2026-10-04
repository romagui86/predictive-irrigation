"""Construye datos, ajusta modelos/políticas y produce evaluaciones reproducibles."""
# Dependencias: datos, serialización de modelos, métricas y figuras estáticas.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import FEATURES, make_features
from .simulator import ZONES, KY, forecast_et0, simulate

# Esquema publicable: variables previas a la decisión, target e IDs; sin estados futuros.
PROCESSED_COLUMNS=list(dict.fromkeys(["sim_id","zona_id","t","etapa","theta_obs",
    "lluvia_obs","lluvia_pron_24h","temp_pron","theta_fc_nom",*FEATURES,
    "delta_theta_raw","dist_umbral_raw","estres_en_t+1"]))


def split_ids(n=200, seed=2026):
    """Divide simulaciones completas en train, validación y test."""
    # Un mismo clima/temporada nunca puede filtrarse entre particiones.
    ids=np.random.default_rng(seed).permutation(n)
    a=int(.70*n); b=int(.85*n)
    return ids[:a],ids[a:b],ids[b:]


def build_data(n=200, raw_path=None):
    """Simula temporadas, crea features y guarda opcionalmente la verdad raw."""
    # Se conserva la verdad para auditoría, pero se procesa solo la parte observable.
    chunks=[]; raw_chunks=[]
    for sim_id in range(n):
        regime="dry" if sim_id%2 else "wet"
        obs,truth,_=simulate(sim_id,regime=regime,return_truth=True)
        raw_chunks.append(pd.concat([obs,truth],axis=1))
        data=make_features(obs)
        data["estres_en_t+1"]=truth["estres_en_t+1"].to_numpy()
        chunks.append(data)
    processed=pd.concat(chunks,ignore_index=True)
    if raw_path is not None:
        Path(raw_path).parent.mkdir(parents=True,exist_ok=True)
        pd.concat(raw_chunks,ignore_index=True).to_csv(raw_path,index=False)
    return processed


def fit_models(train):
    """Ajusta los dos clasificadores comparables sobre el mismo set de features."""
    # La imputación queda dentro del pipeline para ajustarse solo con train.
    x=train[FEATURES]; y=train["estres_en_t+1"]
    models={
        "B1_logistica":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                                     LogisticRegression(max_iter=500,class_weight="balanced")),
        "M_boosting":make_pipeline(SimpleImputer(strategy="median"),
                 HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,
                                                 learning_rate=.08,l2_regularization=1.0))}
    for model in models.values(): model.fit(x,y)
    return models


def predictive_metrics(data, models):
    """Calcula métricas de clasificación en el conjunto que recibe la función."""
    # B0 es persistencia por umbral; los demás modelos entregan probabilidades.
    y=data["estres_en_t+1"].astype(int).to_numpy()
    out=[]
    # Persistencia usa solo la lectura contemporánea, con umbral nominal conocido.
    p0=(data.theta_obs < data.theta_fc_nom-.45*(data.theta_fc_nom-.12)).fillna(False).astype(float)
    predictions={"B0_persistencia":p0.to_numpy()}
    predictions.update({name:model.predict_proba(data[FEATURES])[:,1] for name,model in models.items()})
    for name,p in predictions.items():
        pred=p>=.5
        out.append({"modelo":name,"pr_auc":average_precision_score(y,p),
                    "f1":f1_score(y,pred,zero_division=0),
                    "precision":precision_score(y,pred,zero_division=0),
                    "recall":recall_score(y,pred,zero_division=0),
                    "brier":brier_score_loss(y,p)})
    return pd.DataFrame(out),predictions


def policy_factory(kind, model=None, param=10):
    """Construye la política P0 o la recomendación predictiva P1."""
    # Mantener un filtro simple por zona permite decidir en línea sin verdad latente.
    states={}
    def decide(z,t,theta_obs,w,last_irrig,fc,zr,prev_action,prev_rain,prev_etc):
        state=states.setdefault(z,{"est":fc-.035,"p":.0004,"last":-120})
        old_est=state["est"]
        prior=min(fc,state["est"]+(0.9*prev_action+prev_rain-prev_etc)/(1000*zr))
        pp=state["p"]+.000025
        if np.isfinite(theta_obs):
            gain=pp/(pp+.012**2); est=prior+gain*(theta_obs-prior); state["p"]=(1-gain)*pp
        else: est=prior; state["p"]=pp
        state["est"]=est
        days=t-state["last"]
        threshold=fc-.45*(fc-.12)
        # P0 ignora las señales; P1 pregunta al modelo por el escenario sin riego.
        if kind=="P0": action=10.0 if t%int(param)==0 else 0.0
        elif kind=="P1":
            etc_est=(.7,1.0,1.15,.8)[min(3,t//30)]*forecast_et0(t,w.temp_forecast)
            feats=pd.DataFrame([[est,est-old_est,est-threshold,etc_est,w.rain_forecast,0,days]],columns=FEATURES)
            action=10.0 if model.predict_proba(feats)[0,1]>param else 0.0
        else:
            raise ValueError("La política debe ser P0 o P1")
        if action>0: state["last"]=t
        return action
    return decide


def season_metrics(seed, kind, param, model=None, regime="dry", **sim_options):
    """Evalúa una política en una temporada y resume agua, pérdida y costo."""
    # La misma semilla permite que políticas distintas compartan realizaciones.
    policy=policy_factory(kind,model,param)
    obs,truth,weather=simulate(seed,regime,policy=policy,return_truth=True,**sim_options)
    sim=obs.sim_id.iloc[0]
    # Aggregate FAO-33 stage losses across each zone, then average zones.
    loss=[]
    for z in range(ZONES):
        idx=obs.zona_id.to_numpy()==z
        t=obs.t.to_numpy()[idx]
        eta=truth.ETa.to_numpy()[idx]; etc=truth.ETc.to_numpy()[idx]
        loss.append(sum(KY[stage]*(1-eta[t//30==stage].sum()/max(etc[t//30==stage].sum(),1e-8))
                        for stage in range(4)))
    water=obs.riego_t.sum()/ZONES; drainage=obs.drainage.sum()/ZONES
    mean_loss=float(np.clip(np.mean(loss),0,1))
    return {"sim_id":sim,"policy":kind,"param":param,"agua_mm":float(water),
            "drenaje_mm":float(drainage),"perdida_rendimiento":mean_loss,
            "costo":.05*water+100*mean_loss}


def tune_policies(val_ids, model):
    """Selecciona calendario y umbral P1 por costo medio en simulaciones de validación."""
    grids={"P0":[6,8,10,12,15],"P1":[.2,.35,.5,.65,.8]}
    best={}; sweep=[]
    # Same validation subset and common seeds for every candidate keep tuning comparable.
    tuning_ids=list(val_ids)[:min(10,len(val_ids))]
    for kind,params in grids.items():
        results=[]
        for p in params:
            for sid in tuning_ids:
                row=season_metrics(int(sid),kind,p,model if kind=="P1" else None)
                row["policy"]=kind; results.append(row); sweep.append(row)
        tab=pd.DataFrame(results).groupby("param").mean(numeric_only=True)
        best[kind]=float(tab.costo.idxmin())
    return best,pd.DataFrame(sweep)


def run(output="results", n=200, data_dir="data", model_dir="models"):
    """Ejecuta todas las fases y persiste datasets, modelos, métricas y visuales."""
    # Preparar datos reproducibles, manteniendo verdad raw separada de features.
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    data_root=Path(data_dir)
    data=build_data(n,raw_path=data_root/"raw"/"dataset_sintetico.csv")
    data_root.joinpath("processed").mkdir(parents=True,exist_ok=True)
    data=data.sort_values(["sim_id","zona_id","t"]).copy()
    data["delta_theta_raw"]=data.groupby(["sim_id","zona_id"]).theta_obs.diff().fillna(0)
    data["dist_umbral_raw"]=data.theta_obs-(data.theta_fc_nom-.45*(data.theta_fc_nom-.12))
    # Separar por temporada antes de entrenar o ajustar umbrales.
    train_ids,val_ids,test_ids=split_ids(n)
    train=data[data.sim_id.isin(train_ids)]
    test=data[data.sim_id.isin(test_ids)]
    # Entrenar y guardar clasificadores; el umbral de P1 se completa tras validación.
    models=fit_models(train)
    model_root=Path(model_dir); model_root.mkdir(parents=True,exist_ok=True)
    for name,model in models.items(): joblib.dump(model,model_root/f"{name}.joblib")
    data[PROCESSED_COLUMNS].to_csv(data_root/"processed"/"features_sinteticas.csv",index=False)
    # Test se reserva para la evaluación final de predicción.
    metrics,preds=predictive_metrics(test,models)
    metrics.to_csv(out/"metricas_prediccion.csv",index=False)
    # Fit calibration and decision models without touching test outcomes.
    best_model=models["M_boosting"]
    params,sweep=tune_policies(val_ids,best_model)
    (model_root/"metadata.json").write_text(json.dumps({"policy":"P1","features":FEATURES,
        "decision_threshold":params["P1"],"dose_mm":10,"model":"M_boosting",
        "target":"estres_en_t+1"},indent=2),encoding="utf-8")
    sweep.to_csv(out/"frontera_validacion.csv",index=False)
    pd.DataFrame([season_metrics(int(s),k,p,best_model if k=="P1" else None)
                  for k,p in params.items() for s in test_ids]).to_csv(out/"metricas_politicas.csv",index=False)
    y=test["estres_en_t+1"].astype(int).to_numpy()
    # Ablaciones retiran una fuente de información cada vez para medir su aporte.
    ablations={"A1_humedad_cruda":["theta_obs","delta_theta_raw","dist_umbral_raw",
                 "etc_est","lluvia_pron_24h","riego_t","dias_desde_ultimo_riego"],
               "A2_sin_pronostico":[f for f in FEATURES if f!="lluvia_pron_24h"],
               "A3_sin_demanda":[f for f in FEATURES if f!="etc_est"]}
    ablation_rows=[]
    for name, columns in ablations.items():
        ab_model=make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingClassifier(
            max_iter=120,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1.0))
        ab_model.fit(train[columns],train["estres_en_t+1"])
        score=ab_model.predict_proba(test[columns])[:,1]
        ablation_rows.append({"experimento":name,"pr_auc":average_precision_score(y,score),
                              "brier":brier_score_loss(y,score)})
    ablation_rows.append({"experimento":"A4_completo",
                          "pr_auc":average_precision_score(y,preds["M_boosting"]),
                          "brier":brier_score_loss(y,preds["M_boosting"])})
    pd.DataFrame(ablation_rows).to_csv(out/"ablaciones.csv",index=False)
    # Agrupar probabilidades para observar calibración del modelo principal.
    p=preds["M_boosting"]
    bins=pd.cut(p,np.linspace(0,1,11),include_lowest=True)
    calibration=pd.DataFrame({"p":p,"y":y,"bin":bins}).groupby("bin",observed=False).agg(probabilidad=("p","mean"),frecuencia=("y","mean"),n=("y","size")).reset_index()
    calibration.to_csv(out/"calibracion.csv",index=False)
    fig,ax=plt.subplots(figsize=(5,5)); ax.plot([0,1],[0,1],"k--"); ax.plot(calibration.probabilidad,calibration.frecuencia,"o-"); ax.set(xlabel="Probabilidad predicha",ylabel="Frecuencia observada",title="Calibración en test"); fig.tight_layout(); fig.savefig(out/"calibracion.png",dpi=140); plt.close(fig)
    policies=pd.read_csv(out/"metricas_politicas.csv")
    fig,ax=plt.subplots(figsize=(7,5))
    for name,g in sweep.groupby("policy"):
        ax.scatter(g.agua_mm,g.perdida_rendimiento,label=f"{name} validación",alpha=.65)
    for name,g in policies.groupby("policy"):
        ax.scatter(g.agua_mm,g.perdida_rendimiento,marker="x",s=38,label=f"{name} test")
    ax.set(xlabel="Agua aplicada media (mm/zona/temporada)",ylabel="Pérdida relativa de rendimiento",title="Frontera de políticas y selección (validación/test)"); ax.legend(ncol=2,fontsize=8); fig.tight_layout(); fig.savefig(out/"frontera_pareto.png",dpi=140); plt.close(fig)
    # Robustez fija políticas ya seleccionadas y cambia una condición por escenario.
    robust=[]
    for scenario,options in {"suelo_mal_estimado":{"soil_error":.025},
                             "sensor_mas_ruidoso":{"sensor_sigma":.025},
                             "sin_pronostico":{"forecast_skill":0.0}}.items():
        for sid in list(test_ids)[:5]:
            row=season_metrics(int(sid),"P1",params["P1"],best_model,**options)
            row["escenario"]=scenario; robust.append(row)
    if n>=10:
        wet_train=train[train.sim_id%2==0]
        shift_model=make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingClassifier(
            max_iter=120,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1.0))
        shift_model.fit(wet_train[FEATURES],wet_train["estres_en_t+1"])
        dry_test=[int(s) for s in test_ids if s%2==1][:5]
        for sid in dry_test:
            row=season_metrics(sid,"P1",params["P1"],shift_model,regime="dry")
            row["escenario"]="entrena_humedo_evalua_seco"; robust.append(row)
    pd.DataFrame(robust).to_csv(out/"robustez.csv",index=False)
    (out/"resumen.json").write_text(json.dumps({"train_sims":len(train_ids),"validation_sims":len(val_ids),"test_sims":len(test_ids),"politicas":params,"limitacion":"Resultados sintéticos; falta validación con sensores de campo."},indent=2),encoding="utf-8")
    metrics.to_string(index=False)
    return metrics, policies.groupby("policy")[["agua_mm","drenaje_mm","perdida_rendimiento","costo"]].mean()


def stage(t): return min(3,t//30)

if __name__=="__main__":
    m,p=run(); print("\nMétricas predictivas\n",m.to_string(index=False)); print("\nPolíticas promedio en test\n",p.to_string())
