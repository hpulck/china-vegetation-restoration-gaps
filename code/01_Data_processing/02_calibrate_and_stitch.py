"""Blocked validation, constrained calibration, final annual rasters and figures."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import linregress

ROOT = Path(os.environ.get("LAI_NPP_WORK", Path(__file__).resolve().parent / "work")).resolve()
PREP = Path(os.environ.get("LAI_NPP_PREPARED", ROOT / "prepared")).resolve()
FINAL = ROOT / "final_annual"
FIG = ROOT / "figures"
ND = -9999.0
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10})

with rasterio.open(PREP / "china_fraction.tif") as _s:
    PROFILE = _s.profile.copy()
    FRACTION = _s.read(1)
    TRANSFORM = _s.transform
    BOUNDS = _s.bounds
    SHAPE = _s.shape
LAND = FRACTION > 0
LAT = TRANSFORM.f + (np.arange(SHAPE[0]) + .5) * TRANSFORM.e
WEIGHT = np.cos(np.deg2rad(LAT))[:, None] * FRACTION
WEIGHT = WEIGHT.astype("float64")


def load(product: str, stamp: str) -> np.ndarray:
    with rasterio.open(PREP / product / f"{stamp}.tif") as src:
        arr = src.read(1).astype("float32")
        arr[arr == src.nodata] = np.nan
        return arr


def save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = PROFILE.copy()
    profile.update(dtype="float32", nodata=ND, compress="deflate", predictor=3, tiled=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(arr), arr, ND).astype("float32"), 1)


def weighted_mean(a: np.ndarray, mask: np.ndarray) -> float:
    ok = np.isfinite(a) & mask
    return float(np.average(a[ok], weights=WEIGHT[ok])) if ok.any() else np.nan


def fit(x: np.ndarray, y: np.ndarray, years: list[int], train_years: set[int]) -> dict:
    select = np.array([z in train_years for z in years])[:, None, None]
    valid = np.isfinite(x) & np.isfinite(y) & select & LAND
    count = valid.sum(axis=0)
    xx = np.where(valid, x, 0.)
    yy = np.where(valid, y, 0.)
    mx = np.divide(xx.sum(axis=0), count, out=np.zeros(SHAPE, "float32"), where=count > 0)
    my = np.divide(yy.sum(axis=0), count, out=np.zeros(SHAPE, "float32"), where=count > 0)
    dx = np.where(valid, x-mx, 0.)
    dy = np.where(valid, y-my, 0.)
    cov = (dx*dy).sum(axis=0)
    var = (dx*dx).sum(axis=0)
    braw = np.divide(cov, var, out=np.full(SHAPE, np.nan, "float32"), where=var > 1e-7)

    # Slope is based on temporal anomalies, so large spatial LAI/NPP gradients do not dominate it.
    w = WEIGHT * (count >= 5)
    denom = float(np.sum(w*var))
    bg = float(np.clip(np.sum(w*cov)/denom, .1, 3.)) if denom > 0 else 1.
    allw = WEIGHT[None, :, :] * valid
    # Replace NaNs before weighted sum; multiplication by zero does not remove NaNs.
    global_x = float((allw*np.nan_to_num(x, nan=0)).sum()/allw.sum())
    global_y = float((allw*np.nan_to_num(y, nan=0)).sum()/allw.sum())
    ag = global_y-bg*global_x
    good = count >= 6
    b = np.where(good & np.isfinite(braw),
                 (count*np.clip(np.nan_to_num(braw, nan=bg), .1, 3.) + 8*bg)/(count+8), bg)
    b = np.clip(b, .1, 3.).astype("float32")
    a = np.where(good, my-b*mx, ag).astype("float32")
    offset = np.where(good, my-mx, global_y-global_x).astype("float32")
    return {"global_slope": bg, "global_intercept": ag, "slope": b,
            "intercept": a, "offset": offset, "train_count": count,
            "raw_negative_slope": (braw < 0) & (count >= 6)}


def predict(x: np.ndarray, model: dict, method: str, maximum: float) -> np.ndarray:
    if method == "direct":
        out = x.copy()
    elif method == "global_affine":
        out = model["global_slope"]*x + model["global_intercept"]
    elif method == "pixel_offset":
        out = x + model["offset"]
    elif method == "pixel_shrunk_slope":
        out = model["slope"]*x + model["intercept"]
    else:
        raise ValueError(method)
    out = out.astype("float32")
    out[(out < 0) | (out > maximum) | ~np.isfinite(x) | ~LAND] = np.nan
    return out


def score(pred: np.ndarray, ref: np.ndarray) -> dict:
    ok = np.isfinite(pred) & np.isfinite(ref) & LAND
    delta = pred[ok]-ref[ok]
    weight = WEIGHT[ok]
    return {"pixels": int(ok.sum()), "mae": float(np.average(abs(delta), weights=weight)),
            "rmse": float(np.sqrt(np.average(delta**2, weights=weight))),
            "bias": float(np.average(delta, weights=weight)),
            "mean_reference": float(np.average(ref[ok], weights=weight))}


def annual_months(months: list[np.ndarray]) -> np.ndarray:
    data = np.stack(months)
    n = np.isfinite(data).sum(axis=0)
    out = np.divide(np.nansum(data, axis=0), n,
                    out=np.full(SHAPE, np.nan, "float32"), where=n > 0)
    out[n < 10] = np.nan
    return out


def save_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def cv_variable(variable: str) -> tuple[str, list[dict], dict]:
    if variable == "LAI":
        years = list(range(2001, 2016))
        folds = [set(range(2001, 2006)), set(range(2006, 2011)), set(range(2011, 2016))]
        slices = list(range(1, 13))
        source, reference, maximum = "LAI_GIMMS_AVHRR", "LAI_MODIS", 10
    else:
        years = list(range(2001, 2019))
        folds = [set(range(2001, 2007)), set(range(2007, 2013)), set(range(2013, 2019))]
        slices = [0]
        source, reference, maximum = "NPP_GLASS", "NPP_MODIS", 5
    methods = ["direct", "global_affine", "pixel_offset", "pixel_shrunk_slope"]
    rows = []
    full = {}
    for part in slices:
        stamps = [f"{y}_{part:02d}" if variable == "LAI" else str(y) for y in years]
        x = np.stack([load(source, z) for z in stamps])
        y = np.stack([load(reference, z) for z in stamps])
        for fold, test in enumerate(folds, 1):
            model = fit(x, y, years, set(years)-test)
            for i, year in enumerate(years):
                if year not in test:
                    continue
                preds = {m: predict(x[i], model, m, maximum) for m in methods}
                # Exact same evaluation support for all candidate methods.
                common = np.isfinite(y[i]) & np.logical_and.reduce([np.isfinite(z) for z in preds.values()])
                for method, pred in preds.items():
                    rec = score(np.where(common, pred, np.nan), np.where(common, y[i], np.nan))
                    rows.append({"variable": variable, "month": part, "year": year,
                                 "fold": fold, "method": method, **rec})
        full[part] = fit(x, y, years, set(years))
        print("CV", variable, part, flush=True)
    tab = pd.DataFrame(rows)
    # Compare means of datewise spatial MSE, weighted by evaluated pixel count.
    summary = tab.groupby("method").apply(lambda z: pd.Series({
        "rmse": float(np.sqrt(np.average(z.rmse**2, weights=z.pixels))),
        "mae": float(np.average(z.mae, weights=z.pixels)),
        "bias": float(np.average(z.bias, weights=z.pixels)),
        "valid_pixels_per_date": float(z.pixels.mean())}), include_groups=False)
    summary.to_csv(ROOT/f"{variable}_method_summary.csv", encoding="utf-8-sig")
    # Direct is retained as baseline, selection is among calibrated candidates.
    chosen = str(summary.drop(index="direct").rmse.idxmin())
    print("CHOSEN", variable, chosen, summary.to_string(), flush=True)
    save_rows(ROOT/f"{variable}_blocked_validation.csv", rows)
    return chosen, rows, full


def build_lai(method: str, models: dict) -> None:
    (ROOT/"final_monthly_LAI").mkdir(exist_ok=True)
    for year in range(1982, 2025):
        months = []
        for month in range(1, 13):
            stamp = f"{year}_{month:02d}"
            if year <= 2000:
                x = load("LAI_GIMMS_AVHRR", stamp)
                arr = predict(x, models[month], method, 10)
            else:
                arr = load("LAI_MODIS", stamp)
            save(ROOT/"final_monthly_LAI"/f"{stamp}.tif", arr)
            months.append(arr)
        save(FINAL/"LAI"/f"{year}.tif", annual_months(months))
        print("FINAL LAI", year, flush=True)


def build_npp(method: str, model: dict) -> None:
    for year in range(1982, 2025):
        if year <= 2000:
            x = load("NPP_GLASS", str(year))
            arr = predict(x, model, method, 5)
        else:
            arr = load("NPP_MODIS", str(year))
        save(FINAL/"NPP"/f"{year}.tif", arr)
        print("FINAL NPP", year, flush=True)


def outputs(lai_choice: str, npp_choice: str, lai_models: dict, npp_model: dict) -> None:
    rows = []
    # Fixed footprints: both products available in at least 80% of overlap years.
    for var, early, late, period in [("LAI", "LAI_GIMMS_AVHRR", "LAI_MODIS", range(2001, 2016)),
                                     ("NPP", "NPP_GLASS", "NPP_MODIS", range(2001, 2019))]:
        if var == "LAI":
            ea = [annual_months([load(early, f"{y}_{m:02d}") for m in range(1,13)]) for y in period]
            la = [annual_months([load(late, f"{y}_{m:02d}") for m in range(1,13)]) for y in period]
        else:
            ea = [load(early, str(y)) for y in period]
            la = [load(late, str(y)) for y in period]
        both = np.isfinite(ea).sum(axis=0) >= int(np.ceil(len(ea)*.8))
        both &= np.isfinite(la).sum(axis=0) >= int(np.ceil(len(la)*.8))
        both &= LAND
        save(ROOT/f"{var}_fixed_support.tif", both.astype("float32"))
        for year in range(1982, 2025):
            final = load_final(var, year)
            valid = np.isfinite(final) & both
            rows.append({"variable": var, "year": year, "method": "final", "mean": weighted_mean(final, both),
                         "valid_pixels": int(valid.sum()), "fixed_pixels": int(both.sum()),
                         "fixed_coverage": float(valid.sum()/both.sum())})
            if year <= 2000:
                if var == "LAI":
                    raw = annual_months([load(early, f"{year}_{m:02d}") for m in range(1,13)])
                else:
                    raw = load(early, str(year))
                rows.append({"variable": var, "year": year, "method": "direct_splice", "mean": weighted_mean(raw,both),
                             "valid_pixels": int((np.isfinite(raw)&both).sum()), "fixed_pixels": int(both.sum()),
                             "fixed_coverage": float((np.isfinite(raw)&both).sum()/both.sum())})
            else:
                rows.append({"variable": var, "year": year, "method": "direct_splice", "mean": weighted_mean(final,both),
                             "valid_pixels": int(valid.sum()), "fixed_pixels": int(both.sum()),
                             "fixed_coverage": float(valid.sum()/both.sum())})
        print("TREND", var, "fixed pixels", both.sum(), flush=True)
    save_rows(ROOT/"annual_china_statistics.csv", rows)
    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True, constrained_layout=True)
    tab = pd.DataFrame(rows)
    for ax, var, unit in zip(axs, ["LAI", "NPP"], ["LAI", "kg C/(m²·yr)"]):
        for method, label, color in [("direct_splice", "原值直接拼接", "#1b827e"),
                                     ("final", "重新校准拼接", "#b95049")]:
            z = tab[(tab.variable==var)&(tab.method==method)].sort_values("year")
            ax.plot(z.year,z["mean"],"o-",ms=3,lw=1.5,label=label,color=color)
        ax.axvline(2000.5,ls="--",color="#777",lw=1)
        ax.set_ylabel(f"{var}：固定中国有效区域均值 ({unit})")
        ax.grid(axis="y",alpha=.25); ax.legend(frameon=False)
    axs[-1].set_xlabel("年份")
    fig.suptitle("1982–2024 年中国研究区：新旧拼接年度趋势",fontsize=17)
    fig.savefig(FIG/"年度趋势对比.png",dpi=180); plt.close(fig)

    fig, axs = plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
    for i,var in enumerate(["LAI","NPP"]):
        for j,year in enumerate([1982,2000,2001,2024]):
            a = load_final(var,year)
            ax=axs[i,j]
            im=ax.imshow(a,extent=[BOUNDS.left,BOUNDS.right,BOUNDS.bottom,BOUNDS.top],
                         origin="upper",vmin=0,vmax=5 if var=="LAI" else 1.2,cmap="viridis")
            ax.set_title(f"{var} {year}")
            ax.set_xlabel("经度"); ax.set_ylabel("纬度")
            fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle("重建产品：接缝前后与首末年空间分布",fontsize=17)
    fig.savefig(FIG/"年度空间对照.png",dpi=170);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(14,5.5),constrained_layout=True)
    for ax,var in zip(axs,["LAI","NPP"]):
        z=tab[(tab.variable==var)&(tab.method=="final")]
        ax.plot(z.year,z.fixed_coverage*100,"o-",ms=3)
        ax.set_title(var);ax.set_ylabel("固定区域有效覆盖率 (%)")
        ax.set_xlabel("年份");ax.grid(alpha=.25)
    fig.suptitle("最终产品逐年有效像元覆盖",fontsize=17)
    fig.savefig(FIG/"有效像元覆盖率.png",dpi=180);plt.close(fig)

    for var, model in [("LAI",lai_models[7]),("NPP",npp_model)]:
        save(ROOT/f"{var}_slope_example.tif",model["slope"])
        save(ROOT/f"{var}_training_count_example.tif",model["train_count"].astype("float32"))
    with (ROOT/"chosen_methods.txt").open("w",encoding="utf-8") as f:
        f.write(f"LAI={lai_choice}\nNPP={npp_choice}\n")


def load_final(var: str, year: int) -> np.ndarray:
    with rasterio.open(FINAL/var/f"{year}.tif") as src:
        a=src.read(1)
        a[a==src.nodata]=np.nan
        return a


def accuracy_figure() -> None:
    fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    names=["direct","global_affine","pixel_offset","pixel_shrunk_slope"]
    labels=["直接拼接","整体线性","逐像元偏差","约束逐像元斜率"]
    for ax,var in zip(axs,["LAI","NPP"]):
        t=pd.read_csv(ROOT/f"{var}_method_summary.csv",index_col=0)
        ax.bar(labels,[t.loc[n,"rmse"] for n in names],color=["#148578","#748ca3","#d59b52","#bc4d45"])
        ax.set_title(f"{var}：连续年份留出验证")
        ax.set_ylabel("面积加权 RMSE")
        ax.tick_params(axis="x",rotation=16)
    fig.suptitle("四种接合方法的独立验证误差",fontsize=17)
    fig.savefig(FIG/"方法精度对比.png",dpi=180);plt.close(fig)


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=False)
    FIG.mkdir(exist_ok=True)
    lai_choice, _, lai_models = cv_variable("LAI")
    npp_choice, _, npp_models = cv_variable("NPP")
    build_lai(lai_choice,lai_models)
    build_npp(npp_choice,npp_models[0])
    outputs(lai_choice,npp_choice,lai_models,npp_models[0])
    accuracy_figure()
    print("FINISHED",flush=True)


if __name__=="__main__":
    main()
