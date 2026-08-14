#!/usr/bin/env python3
"""
Study 1: Factorial process-chain test for OWGE -> EAC -> DQU integration.

Prospective confirmatory mechanism study. It tests whether weighted observation/
retention, endogenous nonlinear composition, and uncertainty-aware decision
commitment behave as separable/interacting bottlenecks in matched latent worlds.

A1-A6 are later pooled with Study 2 B1-B4 under ONE global Holm correction.
"""

import argparse, json, math, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

CONFIRMATORY_SEEDS = [95001 + 101*i for i in range(60)]
SMOKE_SEEDS = [991001, 991103, 991211]

HYPOTHESES = [
    ("A1_observation_ablation", "Full pipeline outperforms the no-OWGE ablation on compound decision utility"),
    ("A2_eac_ablation", "Full pipeline outperforms the no-EAC ablation on compound decision utility"),
    ("A3_dqu_ablation", "Full pipeline outperforms the no-DQU ablation on compound decision utility"),
    ("A4_full_vs_best_two_stage", "Full pipeline outperforms the best single-stage ablation within seed"),
    ("A5_owge_eac_interaction", "The EAC gain is larger when useful peripheral evidence was retained"),
    ("A6_adaptive_thought_budget", "DQU-style adaptive thought allocation outperforms always-think on net utility"),
]

def cfg(preset):
    if preset == "smoke":
        return {"seeds": SMOKE_SEEDS, "episodes": 500}
    return {"seeds": CONFIRMATORY_SEEDS, "episodes": 5000}

def logistic(x):
    return 1.0/(1.0+np.exp(-np.clip(x,-40,40)))

def paired_test(name,desc,t,c):
    t=np.asarray(t,float); c=np.asarray(c,float); d=t-c
    n=len(d); md=float(d.mean()); sd=float(d.std(ddof=1))
    se=sd/math.sqrt(n) if sd>0 else 0.0
    if se==0:
        ts=np.inf if md>0 else (-np.inf if md<0 else 0.0)
        p=0.0 if md>0 else (1.0 if md<0 else .5)
        lo=hi=md; dz=np.inf if md>0 else (-np.inf if md<0 else 0.0)
    else:
        ts=md/se; p=float(stats.t.sf(ts,n-1))
        crit=float(stats.t.ppf(.975,n-1)); lo=md-crit*se; hi=md+crit*se; dz=md/sd
    nz=d[d!=0]
    sign_p=float(stats.binomtest(int((nz>0).sum()),len(nz),.5,alternative="greater").pvalue) if len(nz) else 1.0
    try:
        wil=float(stats.wilcoxon(d,alternative="greater",zero_method="wilcox").pvalue)
    except Exception:
        wil=float("nan")
    return dict(hypothesis=name,description=desc,n=n,treatment_mean=float(t.mean()),
                control_mean=float(c.mean()),mean_diff=md,ci_low=float(lo),ci_high=float(hi),
                t=float(ts),p_one_sided=p,cohen_dz=float(dz),
                wins=int((d>0).sum()),ties=int((d==0).sum()),losses=int((d<0).sum()),
                wilcoxon_p=wil,sign_test_p=sign_p)

def simulate_condition(rng, n, obs_on, eac_on, dqu_on, pars):
    c=rng.normal(size=n)
    p=rng.normal(size=n)
    noise=rng.normal(size=(n,4))

    true_logit = pars["wc"]*c + pars["wp"]*p + pars["wi"]*c*p
    p_true = logistic(true_logit)
    p_cat = 0.005 + 0.105*logistic(-1.25*p + 0.55*np.abs(noise[:,0]))

    p_seen=p if obs_on else np.zeros_like(p)
    logit_est = pars["wc_hat"]*c + pars["wp_hat"]*p_seen
    if eac_on:
        logit_est += pars["wi_hat"]*c*p_seen
    belief=logistic(logit_est)

    gain, ordinary_loss, catastrophe_loss, safe_u = 12.0, -8.0, 60.0, 3.0
    true_risky = p_true*gain + (1-p_true)*ordinary_loss - p_cat*catastrophe_loss

    if obs_on:
        pcat_hat = 0.005 + 0.105*logistic(-1.25*p_seen + 0.55*np.abs(noise[:,0]))
    else:
        pcat_hat = np.full(n, 0.060)

    if dqu_on:
        est_risky = belief*gain + (1-belief)*ordinary_loss - pcat_hat*catastrophe_loss
        choose_risky = (pcat_hat <= 0.085) & (est_risky > safe_u)
    else:
        est_risky = belief*gain + (1-belief)*ordinary_loss
        choose_risky = est_risky > safe_u

    chosen_u=np.where(choose_risky,true_risky,safe_u)
    oracle=np.maximum(true_risky,safe_u)
    regret=oracle-chosen_u
    obs_belief=logistic(pars["wc_hat"]*c + pars["wp_hat"]*p_seen)

    return {
        "utility":float(np.mean(chosen_u)),
        "regret":float(np.mean(regret)),
        "obs_brier":float(np.mean((obs_belief-p_true)**2)),
        "final_brier":float(np.mean((belief-p_true)**2)),
        "cat_exposure":float(np.mean(np.where(choose_risky,p_cat,0.0))),
        "risky_rate":float(np.mean(choose_risky)),
    }

def metareasoning(rng,n,pars):
    c=rng.normal(size=n); p=rng.normal(size=n); z=rng.normal(size=n)
    true_p=logistic(pars["wc"]*c+pars["wp"]*p+pars["wi"]*c*p)
    pcat=0.005+0.105*logistic(-1.25*p+0.55*np.abs(z))
    base=logistic(pars["wc_hat"]*c+pars["wp_hat"]*p)
    thought=logistic(pars["wc_hat"]*c+pars["wp_hat"]*p+pars["wi_hat"]*c*p)

    gain,loss,cat_loss,safe=12.,-8.,60.,3.
    true_risky=true_p*gain+(1-true_p)*loss-pcat*cat_loss

    def dqu_value(b):
        est=b*gain+(1-b)*loss-pcat*cat_loss
        risky=(pcat<=.085)&(est>safe)
        return np.where(risky,true_risky,safe)

    u0=dqu_value(base); u1=dqu_value(thought)
    thought_cost=.22
    always=u1-thought_cost
    never=u0

    interaction_signal=np.abs(pars["wi_hat"]*c*p)
    uncertainty=1.0-2.0*np.abs(base-.5)
    proxy=interaction_signal*np.clip(uncertainty,0,1)
    think=proxy>.26
    adaptive=np.where(think,u1-thought_cost,u0)
    return dict(adaptive=float(adaptive.mean()),always=float(always.mean()),
                never=float(never.mean()),think_rate=float(think.mean()))

def run_seed(seed,c):
    rng=np.random.default_rng(seed)
    pars={"wc":rng.normal(1.05,.08),"wp":rng.normal(1.25,.10),"wi":rng.normal(1.10,.10)}
    pars["wc_hat"]=pars["wc"]+rng.normal(0,.08)
    pars["wp_hat"]=pars["wp"]+rng.normal(0,.10)
    pars["wi_hat"]=pars["wi"]+rng.normal(0,.10)

    conds={}
    combos=[("000",0,0,0),("001",0,0,1),("010",0,1,0),("011",0,1,1),
            ("100",1,0,0),("101",1,0,1),("110",1,1,0),("111",1,1,1)]
    for j,(name,o,e,d) in enumerate(combos):
        crng=np.random.default_rng(seed+1000+j*17)
        conds[name]=simulate_condition(crng,c["episodes"],o,e,d,pars)

    full=conds["111"]["utility"]; no_o=conds["011"]["utility"]
    no_e=conds["101"]["utility"]; no_d=conds["110"]["utility"]
    best_ablation=max(no_o,no_e,no_d)
    e_gain_o_on=conds["111"]["utility"]-conds["101"]["utility"]
    e_gain_o_off=conds["011"]["utility"]-conds["001"]["utility"]
    mr=metareasoning(np.random.default_rng(seed+70001),c["episodes"],pars)

    prim={
        "A1_observation_ablation":(full,no_o),
        "A2_eac_ablation":(full,no_e),
        "A3_dqu_ablation":(full,no_d),
        "A4_full_vs_best_two_stage":(full,best_ablation),
        "A5_owge_eac_interaction":(e_gain_o_on,e_gain_o_off),
        "A6_adaptive_thought_budget":(mr["adaptive"],mr["always"]),
    }
    diag={"seed":seed,"metareasoning_think_rate":mr["think_rate"],
          "metareasoning_never_utility":mr["never"]}
    for name,m in conds.items():
        for k,v in m.items():
            diag[f"cond_{name}_{k}"]=v
    return prim,diag

def make_plots(out,tests,diag):
    p=Path(out)/"plots"; p.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,5.8))
    y=np.arange(len(tests)); lo=tests.mean_diff-tests.ci_low; hi=tests.ci_high-tests.mean_diff
    ax.errorbar(tests.mean_diff,y,xerr=np.vstack([lo,hi]),fmt="o",capsize=4)
    ax.axvline(0,linewidth=1); ax.set_yticks(y); ax.set_yticklabels(tests.hypothesis)
    ax.set_xlabel("Treatment - control"); ax.set_title("Study 1 primary effects with 95% confidence intervals")
    fig.tight_layout(); fig.savefig(p/"study1_primary_effects.png",dpi=180); plt.close(fig)

    keys=["cond_000_utility","cond_001_utility","cond_010_utility","cond_011_utility",
          "cond_100_utility","cond_101_utility","cond_110_utility","cond_111_utility"]
    vals=[diag[k].mean() for k in keys]
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.bar([k.split("_")[1] for k in keys],vals)
    ax.set_xlabel("OWGE / EAC / DQU enabled bits"); ax.set_ylabel("Mean compound utility")
    ax.set_title("2 x 2 x 2 process-chain conditions")
    fig.tight_layout(); fig.savefig(p/"study1_factorial_means.png",dpi=180); plt.close(fig)

def run(preset,output):
    c=cfg(preset); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    rows=[]; diags=[]
    for i,s in enumerate(c["seeds"],1):
        prim,d=run_seed(s,c); diags.append(d)
        for h,(t,co) in prim.items():
            rows.append(dict(seed=s,hypothesis=h,treatment=t,control=co,difference=t-co))
        print(f"[{i}/{len(c['seeds'])}] Study 1 seed {s}",flush=True)
    seedm=pd.DataFrame(rows); diag=pd.DataFrame(diags); tests=[]
    for h,d in HYPOTHESES:
        g=seedm[seedm.hypothesis==h].sort_values("seed")
        tests.append(paired_test(h,d,g.treatment,g.control))
    tests=pd.DataFrame(tests)
    seedm.to_csv(out/"study1_primary_seed_metrics.csv",index=False)
    diag.to_csv(out/"study1_diagnostics.csv",index=False)
    tests.to_csv(out/"study1_unadjusted_tests.csv",index=False)
    (out/"study1_run_config.json").write_text(json.dumps({
        "study":"Integrated Process Chain Study 1","preset":preset,"seeds":c["seeds"],
        "episodes_per_seed":c["episodes"],"python":sys.version,"platform":platform.platform(),
        "numpy":np.__version__,"pandas":pd.__version__},indent=2),encoding="utf-8")
    make_plots(out,tests,diag)
    print("Study 1 results:",out)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--preset",choices=["smoke","confirmatory"],default="smoke")
    ap.add_argument("--output",required=True); a=ap.parse_args(); run(a.preset,a.output)
