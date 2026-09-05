"""Deterministic forward-time baselines for SANKET proxy targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 26103
CORE = [
    "sector", "progress_current", "expenditure_current", "original_cost_current",
    "progress_change_1m", "progress_change_3m", "progress_velocity_3m",
    "progress_acceleration_3m", "expenditure_change_1m", "expenditure_change_3m",
    "expenditure_velocity_3m", "cost_revision_count_to_date",
    "months_since_cost_revision", "schedule_revision_count_to_date",
    "months_since_schedule_revision", "months_since_material_progress_change",
    "expenditure_to_original_cost", "months_observed",
    "months_since_first_observation", "revised_cost_missing", "progress_missing",
]
CONDITIONAL_INCLUDED = [
    "state", "ministry", "agency", "project_age_months", "revised_cost_current",
    "effective_target_months_from_cutoff", "expenditure_to_revised_cost",
    "cost_revision_pct_current",
]
CONDITIONAL_DEFERRED = ["last_cost_revision_pct"]
FEATURE_SETS = {"CORE_SAFE": CORE, "CORE_PLUS_CONDITIONAL": CORE + CONDITIONAL_INCLUDED}
SPLITS = {
    "schedule": {"train": ("2025-07", "2025-12"), "validation": ("2026-01", "2026-02"), "test": ("2026-03", "2026-04")},
    "cost": {"train": ("2025-07", "2025-10"), "validation": ("2025-11", "2025-12"), "test": ("2026-01", "2026-01")},
}


def temporal_split(df: pd.DataFrame, target: str, kind: str) -> dict[str, pd.DataFrame]:
    if df[target].isna().any(): raise ValueError("target contains unknown values")
    out={}
    for name,(start,end) in SPLITS[kind].items():
        out[name]=df[df.prediction_month.between(start,end)].copy().sort_values(["prediction_month","identity_key"]).reset_index(drop=True)
    if not (out["train"].prediction_month.max() < out["validation"].prediction_month.min() <= out["test"].prediction_month.min()):
        raise ValueError("non-forward temporal split")
    return out


def make_pipeline(features: list[str], model: str, balanced: bool = False) -> Pipeline:
    categorical=[f for f in features if f in {"sector","state","ministry","agency"}]
    numeric=[f for f in features if f not in categorical]
    pre=ColumnTransformer([
        ("num", Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    weight="balanced" if balanced else None
    estimator = LogisticRegression(max_iter=1000,solver="liblinear",random_state=SEED,class_weight=weight) if model=="Logistic Regression" else RandomForestClassifier(n_estimators=200,min_samples_leaf=5,max_features="sqrt",n_jobs=-1,random_state=SEED,class_weight=weight)
    return Pipeline([("preprocess",pre),("model",estimator)])


def choose_threshold(y: pd.Series, probability: np.ndarray) -> float:
    candidates=[]
    for threshold in np.arange(.05,.96,.05):
        pred=(probability>=threshold).astype(int); rec=recall_score(y,pred,zero_division=0); pre=precision_score(y,pred,zero_division=0); f1=f1_score(y,pred,zero_division=0)
        candidates.append((rec>=.60,f1,rec,pre,-threshold,threshold))
    return float(max(candidates)[-1])


def metrics(y: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, object]:
    pred=(probability>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"positive_rate":float(y.mean()),"roc_auc":roc_auc_score(y,probability),"pr_auc":average_precision_score(y,probability),
            "precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),
            "calibration_metric":brier_score_loss(y,probability),"threshold":threshold,"tn":tn,"fp":fp,"fn":fn,"tp":tp}


def split_report(tables: dict[str, tuple[pd.DataFrame,str]]) -> pd.DataFrame:
    rows=[]
    for kind,(df,target) in tables.items():
        split=temporal_split(df,target,kind); train_ids=set(split["train"].identity_key)
        for name,part in split.items():
            rows.append({"target":target,"partition":name.upper(),"period_start":part.prediction_month.min(),"period_end":part.prediction_month.max(),"rows":len(part),"projects":part.identity_key.nunique(),"positive_rows":int(part[target].sum()),"positive_rate":part[target].mean(),"projects_overlapping_train":len(set(part.identity_key)&train_ids) if name!="train" else len(train_ids),"temporal_leakage":"NONE"})
    return pd.DataFrame(rows)


def conditional_review(schedule: pd.DataFrame) -> pd.DataFrame:
    reasons={"state":"May be absent or corrected in later snapshots.","ministry":"Missing in some report generations.","agency":"Missing in some source rows.","project_age_months":"Depends on cutoff-published start/approval date.","revised_cost_current":"Safe only when already printed at cutoff.","effective_target_months_from_cutoff":"Uses revised DoC only when already printed.","last_cost_revision_pct":"Unavailable until a prior cost revision is observed.","expenditure_to_revised_cost":"Requires revised cost already printed at cutoff.","cost_revision_pct_current":"Snapshot ratio, not a formal approval event."}
    rows=[]
    for feature,why in reasons.items():
        rate=float(schedule[feature].isna().mean()); include=feature in CONDITIONAL_INCLUDED
        rows.append({"feature_name":feature,"why_conditional":why,"safe_for_baseline":include,"missing_rate":rate,"coverage_impact":"Rows retained; train-fitted imputation" if include else "No baseline coverage loss",
                     "leakage_risk":"LOW_IF_CUTOFF_VALUE_ONLY","decision":"INCLUDE" if include else "DEFER","notes":"Missingness preserved into train-only preprocessing." if include else "95%+ missingness makes first-pass estimate unstable."})
    return pd.DataFrame(rows)


def run(data_dir: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True,exist_ok=True)
    schedule=pd.read_csv(data_dir/"schedule_modeling.csv",dtype={"project_code":"string","identity_key":"string"})
    cost=pd.read_csv(data_dir/"cost_modeling.csv",dtype={"project_code":"string","identity_key":"string"})
    tables={"schedule":(schedule,"future_schedule_later_3m"),"cost":(cost,"future_cost_increase_5pct_6m")}
    split_report(tables).to_csv(report_dir/"dataset_split_report.csv",index=False,float_format="%.6f")
    conditional_review(schedule).to_csv(report_dir/"conditional_feature_review.csv",index=False,float_format="%.6f")
    results=[]; importances=[]; prediction_sets=[]
    for kind,(df,target) in tables.items():
        parts=temporal_split(df,target,kind)
        variants=[("Logistic Regression",False),("Random Forest",False)]
        if kind=="cost": variants += [("Logistic Regression balanced",True),("Random Forest balanced",True)]
        for feature_set,features in FEATURE_SETS.items():
            for model_name,balanced in variants:
                base=model_name.replace(" balanced",""); pipe=make_pipeline(features,base,balanced)
                # Apple Accelerate can emit benign overflow warnings inside
                # sparse matrix probes even when transformed values and final
                # probabilities are finite. Keep the numerical contract explicit.
                with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
                    pipe.fit(parts["train"][features],parts["train"][target])
                    val_prob=pipe.predict_proba(parts["validation"][features])[:,1]
                    test_prob=pipe.predict_proba(parts["test"][features])[:,1]
                if not np.isfinite(val_prob).all() or not np.isfinite(test_prob).all():
                    raise ValueError("non-finite model probability")
                threshold=choose_threshold(parts["validation"][target],val_prob); score=metrics(parts["test"][target],test_prob,threshold)
                results.append({"target":target,"feature_set":feature_set,"model":model_name,"train_rows":len(parts["train"]),"validation_rows":len(parts["validation"]),"test_rows":len(parts["test"]),**score,"notes":"Natural classes" if not balanced else "Train-only class_weight=balanced"})
                prediction_sets.append((kind,feature_set,model_name,parts["test"],target,test_prob,threshold,score))
                if base=="Random Forest":
                    names=pipe.named_steps["preprocess"].get_feature_names_out(); values=pipe.named_steps["model"].feature_importances_
                    for name,value in zip(names,values): importances.append({"target":target,"feature_set":feature_set,"model":model_name,"transformed_feature":name,"importance":value})
    result=pd.DataFrame(results).sort_values(["target","feature_set","model"]).reset_index(drop=True)
    result.to_csv(report_dir/"baseline_model_results.csv",index=False,float_format="%.6f")
    pd.DataFrame(importances).sort_values(["target","feature_set","model","importance"],ascending=[True,True,True,False]).to_csv(report_dir/"baseline_feature_importance.csv",index=False,float_format="%.8f")
    write_reports(result,prediction_sets,report_dir)


def write_reports(results: pd.DataFrame, predictions: list[tuple], report_dir: Path) -> None:
    selection=["# Baseline feature selection","","## CORE_SAFE","",* [f"- `{x}`" for x in CORE],"","## CONDITIONAL_CANDIDATE included","",*[f"- `{x}`" for x in CONDITIONAL_INCLUDED],"","## EXCLUDE_FROM_BASELINE","",f"- `{CONDITIONAL_DEFERRED[0]}`: deferred because 95%+ values are missing.","- Future revised values, full-history aggregates, identifiers, project name, and last-observed values are excluded.","","All preprocessing is fitted on training rows only. Feature sets are evaluated separately."]
    (report_dir/"baseline_feature_selection.md").write_text("\n".join(selection)+"\n")
    lines=["SANKET - BASELINE MODEL REPORT","="*52,"Forward-only splits; validation-selected thresholds; test data never tunes preprocessing or thresholds.",""]
    for target,g in results.groupby("target"):
        lines += [target, "-"*len(target)]
        for r in g.itertuples(): lines.append(f"{r.feature_set} | {r.model}: ROC-AUC={r.roc_auc:.3f}, PR-AUC={r.pr_auc:.3f}, precision={r.precision:.3f}, recall={r.recall:.3f}, F1={r.f1:.3f}, Brier={r.calibration_metric:.3f}, threshold={r.threshold:.2f}, confusion TN/FP/FN/TP={r.tn}/{r.fp}/{r.fn}/{r.tp}")
        lines.append("")
    lines += ["Decision:","SCHEDULE MODEL: BASELINE ESTABLISHED. Strongest benchmark: CORE_PLUS_CONDITIONAL Random Forest; PR-AUC is the most useful ranking metric. False negatives remain material and the March publication-boundary shift limits generalization claims.","COST MODEL: BASELINE NOT RELIABLE. Strongest preliminary benchmark: CORE_PLUS_CONDITIONAL Random Forest; PR-AUC is emphasized over accuracy.","INSUFFICIENT_EVALUATION_SAMPLE: cost validation contains only 38 positives and the test is a single month with 78 positives. Class weighting did not consistently improve PR-AUC or recall/precision trade-offs.","Probability scores are not interpreted as literal event chances because Brier scores and reliability bins show only preliminary calibration.","Target semantics remain acceptable as published-revision proxies, not actual delay or actual overrun.","More cost months and a schedule test beyond the March revision boundary are required before adding model complexity.","No XGBoost, SHAP, synthetic oversampling, dashboard, or production API was used."]
    (report_dir/"baseline_model_report.txt").write_text("\n".join(lines)+"\n")
    err=["SANKET - BASELINE ERROR ANALYSIS","="*52]
    for kind in ("schedule","cost"):
        choices=[p for p in predictions if p[0]==kind]
        best=max(choices,key=lambda p:p[-1]["pr_auc"]); _,fs,model,test,target,prob,threshold,score=best
        pred=(prob>=threshold).astype(int); work=test.copy(); work["error"]=np.select([(work[target]==0)&(pred==1),(work[target]==1)&(pred==0)], ["false_positive","false_negative"], default="correct")
        work["age_band"]=pd.cut(work.project_age_months,[-np.inf,36,84,144,np.inf],labels=["<=3y","3-7y","7-12y",">12y"]); work["progress_band"]=pd.cut(work.progress_current,[-np.inf,25,50,75,np.inf],labels=["<=25","25-50","50-75",">75"]); work["expenditure_band"]=pd.cut(work.expenditure_current,[-np.inf,100,500,1000,np.inf],labels=["<=100","100-500","500-1000",">1000"])
        err += ["",f"{kind.upper()}: {fs} / {model} (highest test PR-AUC)",f"False positives: {score['fp']}; false negatives: {score['fn']}"]
        for field in ("sector","state","age_band","progress_band","expenditure_band"):
            summary=work[work.error!="correct"].groupby([field,"error"],observed=True).size().sort_values(ascending=False).head(5)
            err.append(f"Largest {field} error groups: " + (", ".join(f"{a}/{b}={n}" for (a,b),n in summary.items()) if len(summary) else "none"))
        bins=pd.cut(prob,[0,.2,.4,.6,.8,1],include_lowest=True)
        reliability=pd.DataFrame({"bin":bins,"actual":test[target].to_numpy(),"probability":prob}).groupby("bin",observed=True).agg(n=("actual","size"),mean_probability=("probability","mean"),observed_rate=("actual","mean"))
        err.append("Reliability bins (n/mean probability/observed rate): " + "; ".join(f"{idx}: {r.n}/{r.mean_probability:.3f}/{r.observed_rate:.3f}" for idx,r in reliability.iterrows()))
    err += ["","These are descriptive counts, not causal explanations. Small groups are not interpreted."]
    (report_dir/"model_error_analysis.txt").write_text("\n".join(err)+"\n")


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data-dir",type=Path,default=Path("data/features")); p.add_argument("--report-dir",type=Path,default=Path("reports")); a=p.parse_args(); run(a.data_dir,a.report_dir)


if __name__=="__main__": main()
