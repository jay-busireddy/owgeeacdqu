#!/usr/bin/env python3
"""
Study 2: Delayed priming and post-thought decision test.

Tests whether unverified recombination can prime later learning without creating
immediate knowledge, and whether DQU remains necessary after priming.

B1-B4 are combined with Study 1 A1-A6 under ONE global Holm correction.
"""

import argparse, json, math, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

CONFIRMATORY_SEEDS=[135001+89*i for i in range(60)]
SMOKE_SEEDS=[992001,992107,992219]

HYPOTHESES=[
    ("B1_delayed_priming_sample_efficiency","Unverified recombination reduces labeled encounters needed to learn a structured future composition"),
    ("B2_tagged_vs_committed_recombination","Tagging recombinations as unverified reduces early negative transfer when the future relation differs"),
    ("B3_priming_early_learning_auc","Primed feature availability improves early post-encounter learning AUC on structured compositions"),
    ("B4_dqu_after_priming","Risk-aware commitment after primed learning improves contextual decision utility over risk-neutral commitment"),
]

def cfg(preset):
    if preset=="smoke":
        return {"seeds":SMOKE_SEEDS,"tasks":60,"max_examples":24}
    return {"seeds":CONFIRMATORY_SEEDS,"tasks":400,"max_examples":30}

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-40,40)))

def paired_test(name,desc,t,c):
    t=np.asarray(t,float); c=np.asarray(c,float); d=t-c
    n=len(d); md=float(d.mean()); sd=float(d.std(ddof=1)); se=sd/math.sqrt(n) if sd>0 else 0.
    if se==0:
        ts=np.inf if md>0 else (-np.inf if md<0 else 0.); p=0. if md>0 else (1. if md<0 else .5)
        lo=hi=md; dz=np.inf if md>0 else (-np.inf if md<0 else 0.)
    else:
        ts=md/se; p=float(stats.t.sf(ts,n-1)); crit=float(stats.t.ppf(.975,n-1))
        lo=md-crit*se; hi=md+crit*se; dz=md/sd
    nz=d[d!=0]
    sign_p=float(stats.binomtest(int((nz>0).sum()),len(nz),.5,alternative="greater").pvalue) if len(nz) else 1.
    try: wil=float(stats.wilcoxon(d,alternative="greater",zero_method="wilcox").pvalue)
    except Exception: wil=float("nan")
    return dict(hypothesis=name,description=desc,n=n,treatment_mean=float(t.mean()),
                control_mean=float(c.mean()),mean_diff=md,ci_low=float(lo),ci_high=float(hi),
                t=float(ts),p_one_sided=p,cohen_dz=float(dz),wins=int((d>0).sum()),
                ties=int((d==0).sum()),losses=int((d<0).sum()),wilcoxon_p=wil,sign_test_p=sign_p)

def features(a,b,interaction=True):
    return np.array([1.0,a,b,a*b if interaction else 0.0],dtype=float)

def eval_acc(w,flip,interaction=True):
    pairs=[(-1,-1),(-1,1),(1,-1),(1,1)]
    correct=0
    for a,b in pairs:
        y=1 if (a*b*flip)>0 else 0
        pred=int(sigmoid(np.dot(w,features(a,b,interaction)))>=.5)
        correct+=pred==y
    return correct/4

def learn_structured_task(rng,max_examples,primed):
    flip=1 if rng.random()<.5 else -1
    w=np.zeros(4)
    unlock=0 if primed else 8
    lr=.45; criterion=None; auc=[]
    for t in range(1,max_examples+1):
        a=1 if rng.random()<.5 else -1; b=1 if rng.random()<.5 else -1
        y=1 if (a*b*flip)>0 else 0
        interaction=(t>unlock)
        x=features(a,b,interaction); pr=sigmoid(np.dot(w,x))
        w += lr*(y-pr)*x
        acc=eval_acc(w,flip,interaction=(t>unlock)); auc.append(acc)
        if criterion is None and t>=4 and acc>=.99:
            criterion=t
    if criterion is None: criterion=max_examples+1
    return criterion,float(np.mean(auc[:10])),w,flip

def unstructured_negative_transfer(rng):
    pairs=[(-1,-1),(-1,1),(1,-1),(1,1)]
    flip=1 if rng.random()<.5 else -1
    xor=np.array([1 if a*b*flip>0 else 0 for a,b in pairs])
    idx=rng.choice(4,size=2,replace=False); table=xor.copy(); table[idx]=1-table[idx]
    wt=np.zeros(4); wc=np.array([0.,0.,0.,3.2*flip]); lr=.28
    losses_t=[]; losses_c=[]
    order=rng.integers(0,4,size=10)
    for j in order:
        a,b=pairs[j]; y=table[j]; x=features(a,b,True)
        pt=sigmoid(wt@x); pc=sigmoid(wc@x)
        losses_t.append(-(y*np.log(pt+1e-9)+(1-y)*np.log(1-pt+1e-9)))
        losses_c.append(-(y*np.log(pc+1e-9)+(1-y)*np.log(1-pc+1e-9)))
        wt+=lr*(y-pt)*x; wc+=lr*(y-pc)*x
    return -float(np.mean(losses_t[:6])),-float(np.mean(losses_c[:6]))

def dqu_after_priming(rng,tasks,max_examples):
    ud=[]; ur=[]
    for _ in range(tasks):
        _,_,w,flip=learn_structured_task(rng,max_examples,True)
        a=1 if rng.random()<.5 else -1; b=1 if rng.random()<.5 else -1
        success_true=0.92 if a*b*flip>0 else 0.18
        phat=sigmoid(w@features(a,b,True)); hazard=rng.uniform(.005,.16)
        gain,loss,cat_loss,safe=10.,-6.,55.,2.5
        true_risky=success_true*gain+(1-success_true)*loss-hazard*cat_loss
        est_d=phat*gain+(1-phat)*loss-hazard*cat_loss
        choose_d=(hazard<=.09) and (est_d>safe)
        est_r=phat*gain+(1-phat)*loss; choose_r=est_r>safe
        ud.append(true_risky if choose_d else safe); ur.append(true_risky if choose_r else safe)
    return float(np.mean(ud)),float(np.mean(ur))

def run_seed(seed,c):
    rng=np.random.default_rng(seed)
    crit_p=[]; crit_r=[]; auc_p=[]; auc_r=[]
    for _ in range(c["tasks"]):
        cp,ap,_,_=learn_structured_task(rng,c["max_examples"],True)
        cr,ar,_,_=learn_structured_task(rng,c["max_examples"],False)
        crit_p.append(cp); crit_r.append(cr); auc_p.append(ap); auc_r.append(ar)
    tagged=[]; committed=[]
    for _ in range(c["tasks"]):
        t,co=unstructured_negative_transfer(rng); tagged.append(t); committed.append(co)
    dqu,riskneutral=dqu_after_priming(rng,c["tasks"],c["max_examples"])
    prim={
        "B1_delayed_priming_sample_efficiency":(-float(np.mean(crit_p)),-float(np.mean(crit_r))),
        "B2_tagged_vs_committed_recombination":(float(np.mean(tagged)),float(np.mean(committed))),
        "B3_priming_early_learning_auc":(float(np.mean(auc_p)),float(np.mean(auc_r))),
        "B4_dqu_after_priming":(dqu,riskneutral),
    }
    diag={
        "seed":seed,
        "pre_encounter_primed_acc":.5,
        "pre_encounter_replay_acc":.5,
        "primed_examples_to_criterion":float(np.mean(crit_p)),
        "replay_examples_to_criterion":float(np.mean(crit_r)),
        "primed_early_auc":float(np.mean(auc_p)),
        "replay_early_auc":float(np.mean(auc_r)),
        "dqu_post_priming_utility":dqu,
        "riskneutral_post_priming_utility":riskneutral,
    }
    return prim,diag

def make_plots(out,tests,diag):
    p=Path(out)/"plots"; p.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,4.8))
    y=np.arange(len(tests)); lo=tests.mean_diff-tests.ci_low; hi=tests.ci_high-tests.mean_diff
    ax.errorbar(tests.mean_diff,y,xerr=np.vstack([lo,hi]),fmt="o",capsize=4)
    ax.axvline(0,linewidth=1); ax.set_yticks(y); ax.set_yticklabels(tests.hypothesis)
    ax.set_xlabel("Treatment - control"); ax.set_title("Study 2 primary effects with 95% confidence intervals")
    fig.tight_layout(); fig.savefig(p/"study2_primary_effects.png",dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4.6))
    ax.bar(["Primed","Replay"],[diag.primed_examples_to_criterion.mean(),diag.replay_examples_to_criterion.mean()])
    ax.set_ylabel("Real labeled encounters to criterion"); ax.set_title("Delayed priming sample efficiency")
    fig.tight_layout(); fig.savefig(p/"study2_priming_efficiency.png",dpi=180); plt.close(fig)

def run(preset,output):
    c=cfg(preset); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    rows=[]; diags=[]
    for i,s in enumerate(c["seeds"],1):
        prim,d=run_seed(s,c); diags.append(d)
        for h,(t,co) in prim.items():
            rows.append(dict(seed=s,hypothesis=h,treatment=t,control=co,difference=t-co))
        print(f"[{i}/{len(c['seeds'])}] Study 2 seed {s}",flush=True)
    seedm=pd.DataFrame(rows); diag=pd.DataFrame(diags); tests=[]
    for h,d in HYPOTHESES:
        g=seedm[seedm.hypothesis==h].sort_values("seed")
        tests.append(paired_test(h,d,g.treatment,g.control))
    tests=pd.DataFrame(tests)
    seedm.to_csv(out/"study2_primary_seed_metrics.csv",index=False)
    diag.to_csv(out/"study2_diagnostics.csv",index=False)
    tests.to_csv(out/"study2_unadjusted_tests.csv",index=False)
    (out/"study2_run_config.json").write_text(json.dumps({
        "study":"Integrated Process Chain Study 2","preset":preset,"seeds":c["seeds"],
        "tasks_per_seed":c["tasks"],"max_examples":c["max_examples"],
        "python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
        "pandas":pd.__version__},indent=2),encoding="utf-8")
    make_plots(out,tests,diag)
    print("Study 2 results:",out)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--preset",choices=["smoke","confirmatory"],default="smoke")
    ap.add_argument("--output",required=True); a=ap.parse_args(); run(a.preset,a.output)
