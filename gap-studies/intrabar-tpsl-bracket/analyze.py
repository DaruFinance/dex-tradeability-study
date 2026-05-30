"""Combine real-vs-null summaries, build the bracketing tables + figures.
Reads summary_{real,null}_{day,hour}.csv and evals_*.csv produced by run_bracket.py."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("./_gaps/intrabar-tpsl-bracket")
FIG = OUT/"figs"; FIG.mkdir(exist_ok=True)
RULES = ["sl_first","split","tp_first"]
RLAB  = {"sl_first":"SL-first (conservative)","split":"split (E[outcome])","tp_first":"TP-first (optimistic)"}

def load_summaries():
    frames=[]
    for tf in ["day","hour"]:
        for tag in ["real","null"]:
            f=OUT/f"summary_{tag}_{tf}.csv"
            if f.exists(): frames.append(pd.read_csv(f, keep_default_na=False, na_values=[""]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def main():
    S=load_summaries()
    if S.empty: print("no summaries yet"); return
    S=S.sort_values(["tf","rule","tag"])
    S.to_csv(OUT/"master_summary.csv", index=False)
    print(S.to_string(index=False))

    # ---- combined real-vs-null table per tf,rule ----
    rows=[]
    for tf in S.tf.unique():
        for r in RULES:
            rr=S[(S.tf==tf)&(S.rule==r)]
            real=rr[rr.tag=="real"]; null=rr[rr.tag=="null"]
            if real.empty or null.empty: continue
            rows.append({
                "tf":tf,"rule":r,
                "real_med_pf":real.median_oos_pf.iloc[0],"null_med_pf":null.median_oos_pf.iloc[0],
                "real_mean_net":real.mean_net_per_trade.iloc[0],"null_mean_net":null.mean_net_per_trade.iloc[0],
                "real_pct_pf_gt1":real.pct_pf_gt1.iloc[0],"null_pct_pf_gt1":null.pct_pf_gt1.iloc[0],
                "span_frac":real.span_frac.iloc[0],
            })
    T=pd.DataFrame(rows); T.to_csv(OUT/"real_vs_null_table.csv", index=False)
    print("\n=== REAL vs NULL ==="); print(T.to_string(index=False))

    # ---- clip quantification: how much SL-first clips vs the optimistic/expected bound ----
    clip=[]
    for tf in S.tf.unique():
        for tag in ["real","null"]:
            d={x.rule:x for _,x in S[(S.tf==tf)&(S.tag==tag)].iterrows()}
            if "sl_first" in d and "tp_first" in d:
                clip.append({"tf":tf,"tag":tag,
                    "net_sl":d["sl_first"].mean_net_per_trade,
                    "net_split":d.get("split",d["sl_first"]).mean_net_per_trade,
                    "net_tp":d["tp_first"].mean_net_per_trade,
                    "clip_bp_sl_to_tp":1e4*(d["tp_first"].mean_net_per_trade-d["sl_first"].mean_net_per_trade),
                    "pf_sl":d["sl_first"].median_oos_pf,"pf_tp":d["tp_first"].median_oos_pf})
    C=pd.DataFrame(clip); C.to_csv(OUT/"clip_quantification.csv", index=False)
    print("\n=== CLIP (SL-first -> TP-first) ==="); print(C.to_string(index=False))

    # ---- FIG 1: median OOS PF by rule, real vs null, grouped by tf ----
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,tf in zip(axes, [t for t in ["day","hour"] if t in S.tf.unique()]):
        x=np.arange(len(RULES)); w=0.36
        rv=[S[(S.tf==tf)&(S.rule==r)&(S.tag=="real")].median_oos_pf.iloc[0] for r in RULES]
        nv=[S[(S.tf==tf)&(S.rule==r)&(S.tag=="null")].median_oos_pf.iloc[0] for r in RULES]
        ax.bar(x-w/2, rv, w, label="real", color="#1f77b4")
        ax.bar(x+w/2, nv, w, label="null (bar-shuffle)", color="#aaaaaa")
        ax.axhline(1.0, ls="--", c="k", lw=.8, label="PF=1 (breakeven)")
        ax.set_xticks(x); ax.set_xticklabels([RLAB[r] for r in RULES], rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{tf} bars"); ax.set_ylabel("median OOS profit factor (all evals)")
        ax.legend(fontsize=7)
    fig.suptitle("Median OOS PF by intrabar TP/SL resolution rule, real vs null")
    fig.tight_layout(); fig.savefig(FIG/"fig1_medpf_by_rule.pdf"); plt.close(fig)

    # ---- FIG 2: mean net/trade (bp) by rule, real vs null ----
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,tf in zip(axes, [t for t in ["day","hour"] if t in S.tf.unique()]):
        x=np.arange(len(RULES)); w=0.36
        rv=[1e4*S[(S.tf==tf)&(S.rule==r)&(S.tag=="real")].mean_net_per_trade.iloc[0] for r in RULES]
        nv=[1e4*S[(S.tf==tf)&(S.rule==r)&(S.tag=="null")].mean_net_per_trade.iloc[0] for r in RULES]
        ax.bar(x-w/2, rv, w, label="real", color="#2ca02c")
        ax.bar(x+w/2, nv, w, label="null", color="#aaaaaa")
        ax.axhline(0, ls="--", c="k", lw=.8)
        ax.set_xticks(x); ax.set_xticklabels([RLAB[r] for r in RULES], rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{tf} bars"); ax.set_ylabel("mean net return per trade (bp)")
        ax.legend(fontsize=7)
    fig.suptitle("Mean net return/trade by resolution rule, real vs null (net of ~164bp RT cost)")
    fig.tight_layout(); fig.savefig(FIG/"fig2_net_by_rule.pdf"); plt.close(fig)

    # ---- FIG 3: per-family OOS PF under SL-first vs TP-first (real, the strongest tf) ----
    tf_use = "hour" if (OUT/"evals_real_hour_tp_first.csv").exists() else "day"
    try:
        a=pd.read_csv(OUT/f"evals_real_{tf_use}_sl_first.csv")
        b=pd.read_csv(OUT/f"evals_real_{tf_use}_tp_first.csv")
        fa=a.groupby("family").oos_pf.median(); fb=b.groupby("family").oos_pf.median()
        fams=fa.index.tolist(); x=np.arange(len(fams)); w=0.4
        fig,ax=plt.subplots(figsize=(12,4.5))
        ax.bar(x-w/2, fa.values, w, label="SL-first", color="#d62728")
        ax.bar(x+w/2, [fb.get(f,np.nan) for f in fams], w, label="TP-first (optimistic)", color="#1f77b4")
        ax.axhline(1.0, ls="--", c="k", lw=.8, label="PF=1")
        ax.set_xticks(x); ax.set_xticklabels(fams, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("median OOS PF"); ax.set_title(f"Per-family median OOS PF ({tf_use}, real): does any family cross PF=1 under the optimistic bound?")
        ax.legend()
        fig.tight_layout(); fig.savefig(FIG/"fig3_perfamily_bounds.pdf"); plt.close(fig)
        # family table: any cross into edge under optimistic?
        ft=pd.DataFrame({"family":fams,"med_pf_sl":fa.values,"med_pf_tp":[fb.get(f,np.nan) for f in fams]})
        ft["crosses_pf1_only_under_tp"]=(ft.med_pf_sl<=1.0)&(ft.med_pf_tp>1.0)
        ft["pf1_under_both"]=(ft.med_pf_sl>1.0)&(ft.med_pf_tp>1.0)
        ft.to_csv(OUT/f"family_bounds_{tf_use}.csv", index=False)
        print(f"\n=== per-family bounds ({tf_use}) ==="); print(ft.to_string(index=False))
    except FileNotFoundError as e:
        print("family fig skipped:", e)

    # ---- family-bounds tables for EVERY tf that has evals (so day+hour both saved) ----
    for tf in ["day","hour"]:
        fa_f=OUT/f"evals_real_{tf}_sl_first.csv"; fb_f=OUT/f"evals_real_{tf}_tp_first.csv"
        if fa_f.exists() and fb_f.exists():
            a=pd.read_csv(fa_f); b=pd.read_csv(fb_f)
            fa=a.groupby("family").oos_pf.median(); fb=b.groupby("family").oos_pf.median()
            fams=fa.index.tolist()
            ft=pd.DataFrame({"family":fams,"med_pf_sl":fa.values,"med_pf_tp":[fb.get(f,np.nan) for f in fams]})
            # also mean net/trade per family under each rule
            na=a.groupby("family").avg_per_trade.mean(); nb=b.groupby("family").avg_per_trade.mean()
            ft["mean_net_sl_bp"]=[1e4*na.get(f,np.nan) for f in fams]
            ft["mean_net_tp_bp"]=[1e4*nb.get(f,np.nan) for f in fams]
            ft["crosses_pf1_only_under_tp"]=(ft.med_pf_sl<=1.0)&(ft.med_pf_tp>1.0)
            ft["pf1_under_both"]=(ft.med_pf_sl>1.0)&(ft.med_pf_tp>1.0)
            ft.to_csv(OUT/f"family_bounds_{tf}.csv", index=False)
    print("\nfigures + tables written to", OUT)

if __name__=="__main__":
    main()
