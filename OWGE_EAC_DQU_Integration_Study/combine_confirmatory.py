#!/usr/bin/env python3
"""Apply ONE global Holm correction across Study 1 A1-A6 and Study 2 B1-B4."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def holm(p):
    p=np.asarray(p,float); m=len(p); order=np.argsort(p); out=np.empty(m); run=0.
    for rank,idx in enumerate(order):
        run=max(run,(m-rank)*p[idx]); out[idx]=min(1.,run)
    return out

ap=argparse.ArgumentParser()
ap.add_argument("--study1",required=True); ap.add_argument("--study2",required=True); ap.add_argument("--output",required=True)
a=ap.parse_args()
s1=pd.read_csv(Path(a.study1)/"study1_unadjusted_tests.csv")
s2=pd.read_csv(Path(a.study2)/"study2_unadjusted_tests.csv")
alltests=pd.concat([s1,s2],ignore_index=True)
alltests["holm_global_p"]=holm(alltests["p_one_sided"].to_numpy())
alltests["decision"]=np.where(alltests["holm_global_p"]<.05,"REJECT_H0","FAIL_TO_REJECT_H0")
out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
alltests.to_csv(out/"global_hypothesis_tests.csv",index=False)
lines=["OWGE-EAC-DQU INTEGRATION CONFIRMATORY SUMMARY","="*74,
       "Multiplicity: ONE Holm family across all 10 prespecified primaries.",""]
for _,r in alltests.iterrows():
    lines.append(f"{r.hypothesis}: treatment={r.treatment_mean:.6f}, control={r.control_mean:.6f}, "
                 f"diff={r.mean_diff:.6f}, 95%CI=[{r.ci_low:.6f},{r.ci_high:.6f}], "
                 f"raw p={r.p_one_sided:.3e}, global Holm p={r.holm_global_p:.3e}, {r.decision}")
(out/"SUMMARY.txt").write_text("\n".join(lines),encoding="utf-8")
print(alltests[["hypothesis","mean_diff","holm_global_p","decision"]].to_string(index=False))
