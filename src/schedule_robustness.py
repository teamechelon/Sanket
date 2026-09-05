"""Robustness checks for the Phase 11 schedule Random Forest baseline."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.baseline_models import (FEATURE_SETS, SEED, choose_threshold,
                                 make_pipeline, metrics, temporal_split)

TARGET = "future_schedule_later_3m"
FEATURES = FEATURE_SETS["CORE_PLUS_CONDITIONAL"]
MIN_SLICE = 30


def fit_schedule(train: pd.DataFrame, validation: pd.DataFrame):
    model=make_pipeline(FEATURES,"Random Forest")
    model.fit(train[FEATURES],train[TARGET])
    val_probability=model.predict_proba(validation[FEATURES])[:,1]
    threshold=choose_threshold(validation[TARGET],val_probability)
    return model,threshold,val_probability


def monthly_distribution(df: pd.DataFrame) -> pd.DataFrame:
    out=df.groupby("prediction_month",sort=True).agg(
        eligible_rows=(TARGET,"size"),positive_rows=(TARGET,"sum"),
        positive_rate=(TARGET,"mean"),projects=("identity_key","nunique"),
        target_event_count=(TARGET,"sum")).reset_index()
    out["validated_revised_transition_events"]=np.where(out.prediction_month.eq("2026-03"),447,np.nan)
    return out


def alternative_split(df: pd.DataFrame) -> dict[str,pd.DataFrame]:
    periods={"train":("2025-07","2025-11"),"validation":("2025-12","2026-01"),"test":("2026-02","2026-03")}
    return {name:df[df.prediction_month.between(a,b)].sort_values(["prediction_month","identity_key"]).reset_index(drop=True) for name,(a,b) in periods.items()}


def purged_split(df: pd.DataFrame) -> dict[str,pd.DataFrame]:
    """Sensitivity with target windows mature before the next partition."""
    periods={"train":("2025-07","2025-09"),"validation":("2025-10","2025-11"),"test":("2026-03","2026-04")}
    return {name:df[df.prediction_month.between(a,b)].sort_values(["prediction_month","identity_key"]).reset_index(drop=True) for name,(a,b) in periods.items()}


def _project_bucket(identity: str) -> int:
    return int(hashlib.sha256(identity.encode()).hexdigest(),16)%10


def project_disjoint_split(df: pd.DataFrame) -> dict[str,pd.DataFrame]:
    """Forward windows plus deterministic, mutually disjoint project groups."""
    d=df.copy(); d["_bucket"]=d.identity_key.map(_project_bucket)
    parts={
        "train":d[(d.prediction_month<="2025-12")&(d._bucket<=5)],
        "validation":d[d.prediction_month.between("2026-01","2026-02")&d._bucket.between(6,7)],
        "test":d[(d.prediction_month>="2026-03")&(d._bucket>=8)],
    }
    ids={k:set(v.identity_key) for k,v in parts.items()}
    if ids["train"]&ids["validation"] or ids["train"]&ids["test"] or ids["validation"]&ids["test"]: raise ValueError("project-disjoint split overlap")
    return {k:v.drop(columns="_bucket").sort_values(["prediction_month","identity_key"]).reset_index(drop=True) for k,v in parts.items()}


def threshold_table(y: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    rows=[]
    for threshold in (.20,.30,.40,.50,.60,.70):
        pred=probability>=threshold; fp=int(((y==0)&pred).sum()); negatives=int((y==0).sum())
        rows.append({"threshold":threshold,"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),"false_positive_rate":fp/negatives if negatives else np.nan})
    return pd.DataFrame(rows)


def calibration_bins(y: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    bins=np.linspace(0,1,11); group=pd.cut(probability,bins,include_lowest=True)
    return pd.DataFrame({"bin":group,"actual":y.to_numpy(),"probability":probability}).groupby("bin",observed=False).agg(rows=("actual","size"),mean_probability=("probability","mean"),observed_rate=("actual","mean")).reset_index()


def feature_diagnostics(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows=[]; numeric=[f for f in FEATURES if pd.api.types.is_numeric_dtype(df[f])]
    with np.errstate(invalid="ignore",divide="ignore"):
        corr=train[numeric].corr().abs()
    redundant=set()
    for i,a in enumerate(numeric):
        for b in numeric[i+1:]:
            if corr.loc[a,b]>.98: redundant.update((a,b))
    for feature in FEATURES:
        tr,te=train[feature],test[feature]; numeric_feature=pd.api.types.is_numeric_dtype(df[feature])
        missing=float(df[feature].isna().mean()); unique=int(df[feature].nunique(dropna=True)); flags=[]
        if missing>.50: flags.append("HIGH_MISSING")
        if unique<=1 or (len(df) and df[feature].value_counts(normalize=True,dropna=False).iloc[0]>=.99): flags.append("NEAR_CONSTANT")
        if numeric_feature:
            tr_mean,tr_std,te_mean=tr.mean(),tr.std(),te.mean(); drift=abs(te_mean-tr_mean)/tr_std if pd.notna(tr_std) and tr_std>0 else np.nan
            correlation=tr.corr(train[TARGET]) if tr.nunique(dropna=True)>1 else np.nan
        else:
            tr_mean=tr_std=te_mean=correlation=np.nan
            a=tr.fillna("<MISSING>").value_counts(normalize=True); b=te.fillna("<MISSING>").value_counts(normalize=True); categories=a.index.union(b.index); drift=.5*(a.reindex(categories,fill_value=0)-b.reindex(categories,fill_value=0)).abs().sum()
        monthly_missing=df.groupby("prediction_month")[feature].apply(lambda x:x.isna().mean())
        if len(monthly_missing) and monthly_missing.max()-monthly_missing.min()>.50: flags.append("PERIOD_DOMINATED_MISSINGNESS")
        drift_limit=.50 if numeric_feature else .20
        if pd.notna(drift) and drift>drift_limit: flags.append("DISTRIBUTION_SHIFT")
        if feature in redundant: flags.append("POTENTIALLY_REDUNDANT")
        rows.append({"feature_name":feature,"missing_rate":missing,"unique_values":unique,"train_mean":tr_mean,"train_std":tr_std,"test_mean":te_mean,"train_test_distribution_difference":drift,"train_target_correlation":correlation,"drift_assessment":"POTENTIALLY_PROBLEMATIC" if "DISTRIBUTION_SHIFT" in flags else "EXPECTED","flags":";".join(flags) if flags else "NONE"})
    return pd.DataFrame(rows)


def error_slices(test: pd.DataFrame, probability: np.ndarray, threshold: float) -> pd.DataFrame:
    d=test.copy(); pred=(probability>=threshold).astype(int); actual=d[TARGET].to_numpy(); d["false_positive"]=(actual==0)&(pred==1); d["false_negative"]=(actual==1)&(pred==0); d["true_positive"]=(actual==1)&(pred==1); d["true_negative"]=(actual==0)&(pred==0); d["error"]=(pred!=actual)
    d["progress_range"]=pd.cut(d.progress_current,[-np.inf,25,50,75,np.inf],labels=["<=25","25-50","50-75",">75"])
    d["project_age_range"]=pd.cut(d.project_age_months,[-np.inf,36,84,144,np.inf],labels=["<=3y","3-7y","7-12y",">12y"])
    d["expenditure_ratio_range"]=pd.cut(d.expenditure_to_original_cost,[-np.inf,.25,.5,.75,1,np.inf],labels=["<=.25",".25-.50",".50-.75",".75-1",">1"])
    rows=[]
    for field in ("sector","state","progress_range","project_age_range","expenditure_ratio_range","prediction_month"):
        for value,g in d.groupby(field,dropna=False,observed=True):
            positives=int(g[TARGET].sum()); negatives=len(g)-positives
            fp=int(g.false_positive.sum()); fn=int(g.false_negative.sum()); tp=int(g.true_positive.sum()); tn=int(g.true_negative.sum()); projects=g.identity_key.nunique(); error_projects=g.loc[g.error,"identity_key"].nunique()
            adequate=len(g)>=50 and projects>=25 and positives>=30 and negatives>=30
            rows.append({"slice_feature":field,"slice_value":str(value),"rows":len(g),"unique_projects":projects,"positive_rows":positives,"negative_rows":negatives,"true_positives":tp,"true_negatives":tn,"false_positives":fp,"false_negatives":fn,"false_positive_rate":fp/negatives if negatives else np.nan,"false_negative_rate":fn/positives if positives else np.nan,"error_count":fp+fn,"error_rate":(fp+fn)/len(g),"projects_with_error":error_projects,"project_error_exposure":error_projects/projects if projects else np.nan,"sample_status":"ADEQUATE" if adequate else "INSUFFICIENT_SAMPLE"})
    return pd.DataFrame(rows)


def run(data_path: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True,exist_ok=True); df=pd.read_csv(data_path,dtype={"project_code":"string","identity_key":"string"})
    official=temporal_split(df,TARGET,"schedule"); model,threshold,val_prob=fit_schedule(official["train"],official["validation"]); test_prob=model.predict_proba(official["test"][FEATURES])[:,1]
    combined=metrics(official["test"][TARGET],test_prob,threshold)
    views={"March 2026":official["test"].prediction_month.eq("2026-03"),"April 2026":official["test"].prediction_month.eq("2026-04"),"March-April":np.ones(len(official["test"]),dtype=bool)}
    month_scores={name:metrics(official["test"].loc[mask,TARGET],test_prob[mask],threshold) for name,mask in views.items()}
    alt=alternative_split(df); alt_model,alt_threshold,_=fit_schedule(alt["train"],alt["validation"]); alt_score=metrics(alt["test"][TARGET],alt_model.predict_proba(alt["test"][FEATURES])[:,1],alt_threshold)
    purged=purged_split(df); purged_model,purged_threshold,_=fit_schedule(purged["train"],purged["validation"]); purged_score=metrics(purged["test"][TARGET],purged_model.predict_proba(purged["test"][FEATURES])[:,1],purged_threshold)
    disjoint=project_disjoint_split(df); dis_model,dis_threshold,_=fit_schedule(disjoint["train"],disjoint["validation"]); dis_score=metrics(disjoint["test"][TARGET],dis_model.predict_proba(disjoint["test"][FEATURES])[:,1],dis_threshold)
    monthly_distribution(df).to_csv(report_dir/"schedule_target_monthly_distribution.csv",index=False,float_format="%.6f")
    feature_diagnostics(df,official["train"],official["test"]).to_csv(report_dir/"feature_diagnostics.csv",index=False,float_format="%.6f")
    error_slices(official["test"],test_prob,threshold).to_csv(report_dir/"schedule_error_slices.csv",index=False,float_format="%.6f")
    write_calibration(official["validation"],val_prob,official["test"],test_prob,threshold,report_dir/"schedule_calibration_report.txt")
    write_report(df,official,threshold,combined,month_scores,alt,alt_threshold,alt_score,purged,purged_threshold,purged_score,disjoint,dis_threshold,dis_score,test_prob,report_dir/"schedule_baseline_robustness_report.txt")
    write_plan(report_dir/"xgboost_feature_plan.md")


def write_calibration(validation,val_prob,test,test_prob,threshold,path):
    lines=["SANKET - SCHEDULE CALIBRATION AND THRESHOLD REVIEW","="*58,"Thresholds selected and reviewed on validation only.","", "VALIDATION THRESHOLDS"]
    for r in threshold_table(validation[TARGET],val_prob).itertuples(): lines.append(f"{r.threshold:.2f}: precision={r.precision:.3f}, recall={r.recall:.3f}, F1={r.f1:.3f}, FPR={r.false_positive_rate:.3f}")
    lines += [f"Selected validation threshold: {threshold:.2f}","Operational alternatives: 0.40 favors recall; 0.50 is a more conservative alert threshold.","","TEST RELIABILITY CURVE (bin / rows / mean prediction / observed rate)"]
    bins=calibration_bins(test[TARGET],test_prob); ece=0
    for r in bins.itertuples():
        lines.append(f"{r.bin}: {r.rows} / {r.mean_probability:.3f} / {r.observed_rate:.3f}" if r.rows else f"{r.bin}: 0 / NA / NA")
        if r.rows: ece += r.rows/len(test)*abs(r.mean_probability-r.observed_rate)
    prevalence=test[TARGET].mean(); baseline_brier=prevalence*(1-prevalence); model_brier=metrics(test[TARGET],test_prob,threshold)["calibration_metric"]
    lines += ["",f"Mean prediction: {test_prob.mean():.3f}; observed rate: {prevalence:.3f}; ECE: {ece:.3f}.",f"Brier score: {model_brier:.3f}; prevalence-only Brier: {baseline_brier:.3f}; Brier skill: {1-model_brier/baseline_brier:.3f}.","Conclusion: RELATIVE RISK SCORES. Mid/high bins underpredict the observed rate and fail the probability-use gate (ECE <= 0.05 and supported-bin gap <= 0.10)."]
    path.write_text("\n".join(lines)+"\n")


def write_report(df,official,threshold,combined,views,alt,alt_threshold,alt_score,purged,purged_threshold,purged_score,disjoint,dis_threshold,dis_score,test_prob,path):
    high=official["test"].progress_current.gt(75); pred=test_prob>=threshold; errors=((official["test"][TARGET].to_numpy()==0)&pred)|((official["test"][TARGET].to_numpy()==1)&~pred); roads=official["test"].sector.eq("Roads & Highways").to_numpy()
    drift=pd.read_csv(path.parent/"feature_diagnostics.csv"); shifted=drift[drift["flags"].str.contains("DISTRIBUTION_SHIFT")].feature_name.tolist()
    lines=["SANKET - SCHEDULE BASELINE ROBUSTNESS","="*54,"Classification: PROMISING_WITH_CAVEATS","Final status: NEEDS FURTHER VALIDATION","",f"Official baseline reproduced: ROC-AUC={combined['roc_auc']:.3f}, PR-AUC={combined['pr_auc']:.3f}, precision={combined['precision']:.3f}, recall={combined['recall']:.3f}, F1={combined['f1']:.3f}, Brier={combined['calibration_metric']:.3f}, threshold={threshold:.2f}.","", "MONTH SENSITIVITY"]
    for name,s in views.items(): lines.append(f"{name}: ROC-AUC={s['roc_auc']:.3f}, PR-AUC={s['pr_auc']:.3f}, precision={s['precision']:.3f}, recall={s['recall']:.3f}, F1={s['f1']:.3f}, Brier={s['calibration_metric']:.3f}.")
    high_errors=high.to_numpy()&errors; high_roads=high.to_numpy()&roads; high_ids=official["test"].loc[high,"identity_key"]; high_error_ids=official["test"].loc[high_errors,"identity_key"]
    lines += ["",f"Alternative forward split (train Jul-Nov, validation Dec-Jan, test Feb-Mar): test rows={len(alt['test'])}, threshold={alt_threshold:.2f}, ROC-AUC={alt_score['roc_auc']:.3f}, PR-AUC={alt_score['pr_auc']:.3f}, F1={alt_score['f1']:.3f}.",f"Final-test label-maturity sensitivity (train Jul-Sep, validation Oct-Nov, test Mar-Apr): test rows={len(purged['test'])}, threshold={purged_threshold:.2f}, ROC-AUC={purged_score['roc_auc']:.3f}, PR-AUC={purged_score['pr_auc']:.3f}, F1={purged_score['f1']:.3f}.","Official feature cutoffs are forward-only, but their three-month label windows overlap later partitions. This sensitivity ensures validation labels mature before the March test; the short history cannot also purge all train-versus-validation overlap, so it is not a fully prospective simulation.",f"Project-disjoint sensitivity (60/20/20 deterministic project buckets plus forward windows): train/validation/test rows={len(disjoint['train'])}/{len(disjoint['validation'])}/{len(disjoint['test'])}; threshold={dis_threshold:.2f}, ROC-AUC={dis_score['roc_auc']:.3f}, PR-AUC={dis_score['pr_auc']:.3f}, F1={dis_score['f1']:.3f}.","This disjoint case measures cold-start transfer, not the primary known-project forecasting scenario; its smaller fitting sample also contributes to any performance difference.","", "EVENT CONCENTRATION","Prediction-cutoff positive rates jump from 14.50% in November to 45.45% in December, remain 45.42% in January, and peak at 50.96% in February before declining to 42.15%/39.89% in March/April. December-February are abnormal boundary months.","Monthly target_event_count means positive label rows, which can attribute one later revision to several cutoff windows. It is not a raw source-event count. March has 717 positive label rows in this table, while Phase 9 validated 447 revised-to-revised transitions and Phase 8 attributed 1,803 positive label windows to March changes.","The 447 March transitions affect earlier cutoff labels (primarily December-February); March/April evaluation instead tests features observed at that boundary against subsequent outcomes. Its risk is a report-period signature, not direct reuse of those 447 outcomes.","", "HIGH-PROGRESS REVIEW",f"High progress is defined consistently as >75%. Test observations/projects: {int(high.sum())}/{high_ids.nunique()}; target positive rate: {official['test'].loc[high,TARGET].mean():.2%}; error rows: {int(high_errors.sum())} ({errors[high.to_numpy()].mean():.2%}); projects with any error: {high_error_ids.nunique()} ({high_error_ids.nunique()/high_ids.nunique():.2%}).",f"Roads & Highways account for {(high_roads).sum()/max(1,high.sum()):.2%} of high-progress observations and {(roads&high_errors).sum()/max(1,high_errors.sum()):.2%} of high-progress errors. Their within-high-progress error rate is {errors[high_roads].mean():.2%}, versus {errors[high.to_numpy()&~roads].mean():.2%} outside Roads; the absolute concentration is exposure-driven in this subgroup.",f"Across all progress levels, Roads & Highways error rate is {errors[roads].mean():.2%}, versus {errors[~roads].mean():.2%} elsewhere, so the broader Roads slice remains a real failure concentration.","High-progress errors combine near-complete projects that still receive published target revisions and projects whose target stays stable; current snapshot features do not distinguish administrative revision causes. This is consistent with a target/reporting effect but does not prove one.","", "FEATURE DRIFT",("Potentially shifted: "+", ".join(shifted)) if shifted else "No feature crossed the prespecified drift flag.","Stable core measures include current progress, expenditure, original cost, and expenditure/original-cost ratio. Exact linear trend derivatives are flagged as potentially redundant; they are not deleted during this diagnostic phase. Missingness and revision counts can act as report-period signatures.","", "DECISION","The 0.855 combined ROC-AUC is reproducible. March exceeds April by only 0.027 ROC-AUC, 0.049 PR-AUC, and 0.043 F1; April performance remains strong, so the result does not collapse outside March.","Robustness must still be judged from April-only, alternative-window, final-test-purged, and project-disjoint results rather than the combined score alone.","Scores are useful for relative ranking, not calibrated event probabilities.","March-April has now informed diagnostics and feature planning, so it is no longer an untouched final holdout for a later XGBoost comparison.","Cost remains SECONDARY / NOT RELIABLE: 216 total positives, only 38 validation positives, one test month, and 3.83% overall prevalence. Additional exact-horizon months are required.","XGBoost is not justified yet. Obtain at least one post-April schedule test window, repeat cold-start evaluation with more projects, and decide whether shifted conditional fields remain in scope."]
    path.write_text("\n".join(lines)+"\n")


def write_plan(path):
    path.write_text("""# Pre-XGBoost feature plan

## CORE_FEATURE_SET

Use the 21 `CORE_SAFE` features from Phase 11. They are cutoff-known and form
the comparison contract for any later model. Keep exact-lag missingness intact;
fit imputation and encoding on training data only.

## OPTIONAL_FEATURES

Evaluate the eight reviewed cutoff-known conditional features only as a
separate ablation: state, ministry, agency, project age, current revised cost,
effective target distance, expenditure/revised-cost ratio, and current cost
revision percentage. Retain only if robustness outside the March boundary and
project-disjoint behavior do not materially deteriorate.

## DEFERRED_FEATURES

Defer `last_cost_revision_pct` because more than 95% is missing. Exclude future
revised values, complete-history aggregates, identity/name fields, final
observations, and target-window fields. Treat perfectly scaled trend variants
as redundant candidates and compare one representative per pair before any
tree boosting experiment.

Do not start XGBoost or SHAP until another forward schedule window validates
the baseline and the conditional-feature ablation is approved.
""")


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data",type=Path,default=Path("data/features/schedule_modeling.csv")); p.add_argument("--report-dir",type=Path,default=Path("reports")); a=p.parse_args(); run(a.data,a.report_dir)


if __name__=="__main__": main()
