from pathlib import Path
import ast, argparse, hashlib, json, math, os, time
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.font_manager import fontManager

ROOT = Path(os.environ.get("LAI_NPP_WORK", Path(__file__).resolve().parent / "work")).resolve()
PREP = Path(os.environ.get("LAI_NPP_PREPARED", ROOT / "prepared")).resolve()
CALIBRATION = Path(os.environ.get("LAI_NPP_CALIBRATION", ROOT)).resolve()
OUT = ROOT / "overlap_validation"
METHODS=["direct","global_affine","pixel_offset","pixel_shrunk_slope"]

def settings(var):
    if var=="LAI":
        return list(range(2001,2016)),[set(range(2001,2006)),set(range(2006,2011)),set(range(2011,2016))],range(1,13),"LAI_GIMMS_AVHRR","LAI_MODIS",10.0
    return list(range(2001,2019)),[set(range(2001,2007)),set(range(2007,2013)),set(range(2013,2019))],[0],"NPP_GLASS","NPP_MODIS",5.0

def load(product,stamp):
    with rasterio.open(PREP/product/f"{stamp}.tif") as src:
        a=src.read(1).astype("float32")
        a[a==src.nodata]=np.nan
    return a

def main(var):
    start_time=time.time()
    chosen = dict(line.split("=", 1) for line in (CALIBRATION / "chosen_methods.txt").read_text(encoding="utf-8").splitlines())[var]
    for subdirectory in ("metrics", "pairs", "figures", "qa"):
        (OUT / subdirectory).mkdir(parents=True, exist_ok=True)
    if (OUT / "metrics" / f"{var}_散点全量指标.json").exists():
        raise FileExistsError(f"Validation outputs already exist for {var}: {OUT}")
    with rasterio.open(PREP/"china_fraction.tif") as src:
        fraction=src.read(1); transform=src.transform; shape=src.shape
    land=fraction>0
    lat=transform.f+(np.arange(shape[0])+0.5)*transform.e
    weight=(np.cos(np.deg2rad(lat))[:,None]*fraction).astype("float64")
    legacy_path=Path(__file__).resolve().with_name("02_calibrate_and_stitch.py")
    legacy=legacy_path.read_text(encoding="utf-8-sig")
    tree=ast.parse(legacy)
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("fit","predict","score")]
    assert len(selected)==3
    namespace={"np":np,"SHAPE":shape,"WEIGHT":weight,"LAND":land}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(legacy_path),"exec"),namespace)
    fit,predict,score=[namespace[n] for n in ("fit","predict","score")]
    old=pd.read_csv(CALIBRATION/f"{var}_blocked_validation.csv")
    old=old[old.method==chosen].set_index(["month","year"])
    expected_total=int(old.pixels.sum())
    years,folds,parts,source,reference,maximum=settings(var)
    rng=np.random.default_rng(20261017+(0 if var=="LAI" else 1))
    n_display=min(150000,expected_total)
    selected_indices=np.sort(rng.choice(expected_total,n_display,replace=False))
    sample={k:[] for k in ("reference","prediction","effective_weight","row","column","year","month","fold")}
    moments={k:0.0 for k in ("w","x","y","xx","yy","xy","sse","sae","error")}
    hist_limit=12.0 if var=="LAI" else 6.0
    bins=1024
    histogram=np.zeros((bins,bins),dtype="float64")
    offset=0; date_rows=[]
    extrema={"minimum_reference":np.inf,"maximum_reference":-np.inf,"minimum_prediction":np.inf,"maximum_prediction":-np.inf}
    for part in parts:
        stamps=[f"{y}_{part:02d}" if var=="LAI" else str(y) for y in years]
        x=np.stack([load(source,s) for s in stamps])
        y=np.stack([load(reference,s) for s in stamps])
        for fold,test_years in enumerate(folds,1):
            train_years=set(years)-test_years
            assert not train_years.intersection(test_years)
            model=fit(x,y,years,train_years)
            for i,year in enumerate(years):
                if year not in test_years: continue
                preds={m:predict(x[i],model,m,maximum) for m in METHODS}
                common=np.isfinite(y[i])&np.logical_and.reduce([np.isfinite(z) for z in preds.values()])&land
                flat_ids=np.flatnonzero(common)
                n=len(flat_ids)
                expected=old.loc[(part,year)]
                assert n==int(expected.pixels),(var,part,year,n,int(expected.pixels))
                pred=preds[chosen]
                rec=score(np.where(common,pred,np.nan),np.where(common,y[i],np.nan))
                deltas={f"old_diff_{metric}":float(rec[metric]-expected[metric]) for metric in ("rmse","mae","bias")}
                assert max(abs(v) for v in deltas.values())<2e-8,(var,part,year,deltas)
                date_rows.append({"variable":var,"month":part,"year":year,"fold":fold,**rec,**deltas})
                xv=y[i][common].astype("float64")
                yv=pred[common].astype("float64")
                we=weight[common]*(n/float(weight[common].sum()))
                moments["w"]+=n
                for key,vals in (("x",xv),("y",yv),("xx",xv*xv),("yy",yv*yv),("xy",xv*yv)):
                    moments[key]+=float(np.dot(we,vals))
                moments["sse"]+=n*rec["rmse"]**2
                moments["sae"]+=n*rec["mae"]
                moments["error"]+=n*rec["bias"]
                for key,val in (("minimum_reference",xv.min()),("maximum_reference",xv.max()),("minimum_prediction",yv.min()),("maximum_prediction",yv.max())):
                    extrema[key]=(min if key.startswith("minimum") else max)(extrema[key],float(val))
                assert xv.min()>=0 and yv.min()>=0
                assert xv.max()<=hist_limit and yv.max()<=hist_limit
                h,_,_=np.histogram2d(xv,yv,bins=bins,range=((0,hist_limit),(0,hist_limit)))
                histogram+=h
                first,last=np.searchsorted(selected_indices,[offset,offset+n])
                local=selected_indices[first:last]-offset
                sample["reference"].append(xv[local].astype("float32"))
                sample["prediction"].append(yv[local].astype("float32"))
                sample["effective_weight"].append(we[local])
                rr,cc=np.unravel_index(flat_ids[local],shape)
                sample["row"].append(rr.astype("uint16")); sample["column"].append(cc.astype("uint16"))
                for key,val,dtype in (("year",year,"uint16"),("month",part,"uint8"),("fold",fold,"uint8")):
                    sample[key].append(np.full(len(local),val,dtype=dtype))
                offset+=n
        print(json.dumps({"variable":var,"part":part,"completed_pairs":offset,"expected_pairs":expected_total}),flush=True)
    assert offset==expected_total
    sample={k:np.concatenate(v) for k,v in sample.items()}
    assert len(sample["reference"])==n_display
    assert int(histogram.sum())==expected_total
    w=moments["w"]; mx=moments["x"]/w; my=moments["y"]/w
    sxx=moments["xx"]-moments["x"]**2/w
    syy=moments["yy"]-moments["y"]**2/w
    sxy=moments["xy"]-moments["x"]*moments["y"]/w
    metrics={
        "variable":var,"domain":"China: fraction>0; per-date common support of all four legacy candidate methods",
        "reference":"local MODIS reference, not independent field truth",
        "method":chosen,"temporal_scale":"monthly" if var=="LAI" else "annual",
        "period":[min(years),max(years)],"heldout_year_blocks":[sorted(f) for f in folds],
        "pair_count":expected_total,"date_count":len(date_rows),"displayed_pairs":n_display,
        "R2":1-moments["sse"]/sxx,"R2_definition":"1-SSE/SST using effective weights, pooled held-out pairs",
        "pearson_r_squared":sxy*sxy/(sxx*syy),
        "RMSE":math.sqrt(moments["sse"]/w),"MAE":moments["sae"]/w,"Bias":moments["error"]/w,
        "regression_slope":sxy/sxx,"regression_intercept":my-sxy/sxx*mx,
        "weighting":"Per-date cosine(latitude)*China fraction normalized to sum=date pair count; exactly matches legacy summary",
        "density_colours":"Counts of ALL held-out pairs per 2D bin; sample only used for visual points",
        "sample_seed":20261017+(0 if var=="LAI" else 1),"sample_method":"uniform without replacement from all pair indices",
        "legacy_source_sha256":hashlib.sha256(legacy_path.read_bytes()).hexdigest(),**extrema,
    }
    old_summary=pd.read_csv(CALIBRATION/f"{var}_method_summary.csv").set_index("method").loc[chosen]
    metrics["legacy_summary_differences"]={key:float(metrics[key]-old_summary[key.lower()]) for key in ("RMSE","MAE","Bias")}
    assert max(abs(v) for v in metrics["legacy_summary_differences"].values())<2e-8
    pd.DataFrame(date_rows).to_csv(OUT/"metrics"/f"{var}_逐日期留出验证复核.csv",index=False,encoding="utf-8-sig")
    (OUT/"metrics"/f"{var}_散点全量指标.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    np.savez_compressed(OUT/"pairs"/f"{var}_散点显示配对_150000.npz",**sample)
    np.savez_compressed(OUT/"pairs"/f"{var}_全量配对密度.npz",counts=histogram,range=np.array([0,hist_limit]),moments=np.array([moments[k] for k in moments]),moment_names=np.array(list(moments)))
    make_plot(var,sample,histogram,hist_limit,metrics)
    checks={"variable":var,"all_legacy_date_counts_match":True,"all_legacy_date_metrics_match_tolerance":2e-8,
            "no_heldout_year_used_for_training":True,"no_full_overlap_fit_predictions":True,
            "pair_count":offset,"sample_size":n_display,"scope":"China fraction grid only",
            "max_old_date_metric_difference":max(abs(r[k]) for r in date_rows for k in ("old_diff_rmse","old_diff_mae","old_diff_bias")),
            "elapsed_seconds":time.time()-start_time}
    (OUT/"qa"/f"{var}_复现核验.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:metrics[k] for k in ("variable","pair_count","R2","RMSE","MAE","Bias")}),flush=True)

def make_plot(var,sample,hist,hist_limit,metrics):
    for f in ("times.ttf","timesbd.ttf","timesi.ttf","timesbi.ttf"):
        path = Path(r"C:\Windows\Fonts") / f
        if path.is_file():
            fontManager.addfont(str(path))
    plt.rcParams.update({"font.family":"Times New Roman","font.size":15,"font.weight":"bold",
                         "axes.labelweight":"bold","axes.titleweight":"bold","axes.unicode_minus":False,
                         "mathtext.fontset":"stix","svg.fonttype":"none"})
    xv=sample["reference"]; yv=sample["prediction"]
    nb=hist.shape[0]
    ix=np.clip((xv/hist_limit*nb).astype(int),0,nb-1)
    iy=np.clip((yv/hist_limit*nb).astype(int),0,nb-1)
    density=hist[ix,iy]
    order=np.argsort(density)
    m=max(metrics["maximum_reference"],metrics["maximum_prediction"])
    magnitude=10**math.floor(math.log10(m))
    step=magnitude/5
    upper=math.ceil(m/step)*step
    fig=plt.figure(figsize=(8,8),dpi=300,facecolor="white")
    ax=fig.add_axes([0.155,0.195,0.66,0.66])
    points=ax.scatter(xv[order],yv[order],c=density[order],s=0.9,cmap="viridis",
                      norm=LogNorm(vmin=1,vmax=max(2,hist.max())),linewidths=0,
                      alpha=0.88,rasterized=True,zorder=2)
    ax.plot([0,upper],[0,upper],"--",color="#687780",lw=1.35,zorder=3)
    xx=np.array([0,upper])
    ax.plot(xx,metrics["regression_intercept"]+metrics["regression_slope"]*xx,
            color="#B54A50",lw=1.55,zorder=4)
    ax.set_xlim(0,upper); ax.set_ylim(0,upper); ax.set_aspect("equal",adjustable="box")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.tick_params(axis="both",direction="out",length=5,width=1.0,labelsize=15)
    for sp in ax.spines.values(): sp.set_color("#495A64"); sp.set_linewidth(1.0)
    ax.grid(alpha=0.18,color="#B9C4CA",lw=0.7,zorder=0)
    units=r"(m$^2$ m$^{-2}$)" if var=="LAI" else r"(kg C m$^{-2}$ yr$^{-1}$)"
    early="GIMMS" if var=="LAI" else "GLASS"
    ax.set_xlabel(f"MODIS {var} "+units,fontsize=17,labelpad=10)
    ax.set_ylabel(f"Calibrated {early} {var} "+units,fontsize=17,labelpad=11)
    decimals=3 if var=="LAI" else 4
    stat=(f"r² = {metrics['pearson_r_squared']:.3f}\n"
          f"RMSE = {metrics['RMSE']:.{decimals}f}\n"
          f"MAE = {metrics['MAE']:.{decimals}f}\n"
          f"Bias = {metrics['Bias']:+.4f}\n"
          f"Pairs = {metrics['pair_count']:,}")
    ax.text(0.035,0.965,stat,transform=ax.transAxes,ha="left",va="top",fontsize=14.4,
            linespacing=1.42,bbox=dict(boxstyle="round,pad=0.35",fc="white",ec="none",alpha=0.91),zorder=6)
    ax.legend(handles=[Line2D([0],[0],color="#687780",ls="--",lw=1.35,label="1:1 line"),
                       Line2D([0],[0],color="#B54A50",lw=1.55,label="Weighted fit")],
              loc="lower right",frameon=False,fontsize=12.5,handlelength=2.2)
    cbax=fig.add_axes([0.845,0.195,0.024,0.66])
    cb=fig.colorbar(points,cax=cbax)
    cb.set_label("Pairs per bin",fontsize=13,labelpad=8)
    cb.ax.tick_params(labelsize=12,length=3)
    cb.outline.set_linewidth(0.7)
    fig.text(0.155,0.956,f"{var} stitching validation",ha="left",va="center",fontsize=22,fontweight="bold")
    scale="monthly" if var=="LAI" else "annual"
    period=f"{metrics['period'][0]}–{metrics['period'][1]}"
    fig.text(0.155,0.908,f"China | {period} | {scale} | 3-fold temporal holdout",
             ha="left",va="center",fontsize=11.9,fontweight="normal",color="#36454E")
    fig.text(0.155,0.055,"Metrics: all held-out pairs, area weighted; Bias = calibrated − MODIS.",
             ha="left",fontsize=10.4,fontweight="normal",color="#46555F")
    fig.text(0.155,0.027,f"Displayed: {len(xv):,} randomly sampled pairs; colour: full-data density.",
             ha="left",fontsize=10.4,fontweight="normal",color="#46555F")
    for ext in ("png","svg"):
        fig.savefig(OUT/"figures"/f"{var}_中国区域_拼接留出散点精度图.{ext}",dpi=300,facecolor="white")
    plt.close(fig)

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--variable",choices=["LAI","NPP"],required=True)
    main(parser.parse_args().variable)

