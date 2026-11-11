"""Align original LPJ-GUESS annual LAI/NPP to satellite publication grid and pair by valid support."""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

MODEL = Path(os.environ.get("LAI_NPP_LPJ_ROOT", r"H:\2. 小论文\4. 中国植被恢复生产力\13. 重新论文修改\1. LPJ模型输出\1. 无人为扰动\china_nat_national"))
ROOT = Path(os.environ.get("LAI_NPP_WORK", Path(__file__).resolve().parent / "work")).resolve()
SAT = Path(os.environ.get("LAI_NPP_STITCHED", ROOT)).resolve()
PREP = Path(os.environ.get("LAI_NPP_PREPARED", ROOT / "prepared")).resolve()
OUT = ROOT / "LPJ_satellite_alignment_mapmax_20260917"
NODATA = -9999.0
YEARS = np.arange(1982, 2025)


def load_chunk(folder: Path, name: str, n: int) -> np.ndarray:
    x = np.loadtxt(folder / f"{name}.out", skiprows=1, dtype="float32")
    if x.shape[0] != n * 44:
        raise ValueError(f"{folder.name}/{name}: bad row count")
    x = x.reshape(n, 44, -1)
    if not np.all(x[:, :, 2] == np.arange(1982, 2026)[None, :]):
        raise ValueError(f"{folder.name}/{name}: bad years")
    return x


def save(path: Path, arr: np.ndarray, ref) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = ref.profile.copy()
    meta.update(driver="GTiff", dtype="float32", count=1, nodata=NODATA,
                compress="deflate", predictor=3, tiled=True)
    with rasterio.open(path, "w", **meta) as ds:
        ds.write(np.where(np.isfinite(arr), arr, NODATA).astype("float32"), 1)


def read(path: Path):
    with rasterio.open(path) as ds:
        x = ds.read(1).astype("float32")
        x[x == ds.nodata] = np.nan
        return x


def fill_to_support(paired, standalone, support):
    """Fill missing LPJ cells inside the satellite footprint."""
    model = paired.copy()
    holes = support & ~np.isfinite(model)
    available = holes & np.isfinite(standalone)
    model[available] = standalone[available]
    holes &= ~available
    if holes.any():
        source = np.argwhere(np.isfinite(standalone))
        distance, neighbors = cKDTree(source).query(np.argwhere(holes), k=4)
        points = source[neighbors]
        weight = 1.0 / np.square(distance)
        model[holes] = np.sum(standalone[points[..., 0], points[..., 1]] * weight, axis=1) / weight.sum(axis=1)
    model[~support] = np.nan
    return model


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    index = pd.read_csv(MODEL / "input" / "chunk_index_500.csv")
    if len(index) != 183 or index.n_cells.sum() != 91255:
        raise ValueError("Unexpected model inventory")
    src_ref = MODEL / "statistics_recheck_20260917" / "rasters" / "NPP_1982.tif"
    sat_ref = SAT / "final_annual" / "NPP" / "1982.tif"
    with rasterio.open(src_ref) as sr, rasterio.open(sat_ref) as tr:
        src_transform, target_transform = sr.transform, tr.transform
        sh, sw = sr.shape
        th, tw = tr.shape
        if sr.crs != tr.crs or sr.crs.to_epsg() != 4326 or (sh, sw) != (353, 615) or (th, tw) != (355, 616):
            raise ValueError("Grid shape or CRS changed")
        if not (np.isclose(src_transform.a, .1) and np.isclose(target_transform.a, .1)
                and np.isclose(src_transform.c-target_transform.c, .05)
                and np.isclose(target_transform.f-src_transform.f, .05)):
            raise ValueError("Expected exact half-cell offset changed")

    # Keep original monthly LAI so each annual LPJ value can use exactly the
    # months that contributed to the satellite annual mean at that pixel.
    source_mlai = np.full((43, 12, 91255), np.nan, dtype="float32")
    source_npp = np.full((43, sh, sw), np.nan, dtype="float32")
    row_all = np.empty(91255, dtype="int16")
    col_all = np.empty(91255, dtype="int16")
    occupied = np.zeros((sh, sw), bool)

    for _, rec in index.iterrows():
        folder = MODEL / "output_dynamic_ndep_globfirm" / f"chunk_{int(rec['chunk']):03d}"
        n = int(rec["n_cells"])
        mlai = load_chunk(folder, "mlai", n)
        anpp = load_chunk(folder, "anpp", n)
        if not np.array_equal(mlai[:, :, :2], anpp[:, :, :2]):
            raise ValueError(f"{folder.name}: coordinate mismatch")
        lon = mlai[:, 0, 0]
        lat = mlai[:, 0, 1]
        col = np.rint((lon - (src_transform.c+.05))/.1).astype(int)
        row = np.rint(((src_transform.f-.05) - lat)/.1).astype(int)
        if not np.all((row>=0)&(row<sh)&(col>=0)&(col<sw)):
            raise ValueError("Source coordinate out of grid")
        if occupied[row,col].any():
            raise ValueError("Duplicate source coordinate")
        occupied[row,col] = True
        start=int(rec["first_landid"]); end=start+n
        row_all[start:end]=row; col_all[start:end]=col
        source_mlai[:,:,start:end] = mlai[:,:43,3:15].transpose(1,2,0)
        source_npp[:, row, col] = anpp[:, :43, -1].T
        if int(rec["chunk"]) % 30 == 0:
            print("loaded", int(rec["chunk"]), "/183", flush=True)
    if occupied.sum() != 91255 or not np.all(np.isfinite(source_mlai)):
        raise ValueError("Source cells incomplete")
    with rasterio.open(src_ref) as sr:
        check = sr.read(1).astype("float32")
        valid = check != sr.nodata
        if not np.array_equal(valid,occupied) or np.max(np.abs(check[valid]-source_npp[0][valid])) > 1e-6:
            raise ValueError("Original source grid does not match previous audit")

    # Map-oriented output: use every satellite yearly value with at least one
    # overlapping LPJ source cell. The count/fraction layers expose partial coverage.
    full = np.zeros((th,tw),dtype="uint8")
    for dr in (0,1):
        for dc in (0,1):
            full[dr:dr+sh, dc:dc+sw] += occupied
    support_ok = full >= 1
    fraction = read(PREP / "china_fraction.tif")
    npp_common = np.ones((th, tw), dtype=bool)
    for year in range(1982, 2025):
        npp_common &= np.isfinite(read(SAT / "final_annual" / "NPP" / f"{year}.tif"))

    with rasterio.open(sat_ref) as ref:
        save(OUT / "LPJ_four_neighbor_count.tif", full.astype("float32"), ref)
        save(OUT/"LPJ_source_coverage_fraction.tif",full.astype("float32")/4,ref)
        for var in ("LAI","NPP"):
            for k,year in enumerate(YEARS):
                if var=="LAI":
                    monthly_sum=np.zeros((th,tw),dtype="float32")
                    standalone_sum=np.zeros((th,tw),dtype="float32")
                    monthly_count=np.zeros((th,tw),dtype="uint8")
                    month_source=range(12)
                else:
                    month_source=(None,)
                for m in month_source:
                    if var=="LAI":
                        x=np.full((sh,sw),np.nan,dtype="float32")
                        x[row_all,col_all]=source_mlai[k,m]
                    else:
                        x=source_npp[k]
                    q0=np.full((th,tw),np.nan,dtype="float32")
                    q1=q0.copy();q2=q0.copy();q3=q0.copy()
                    q0[:sh,:sw]=x
                    q1[:sh,1:sw+1]=x
                    q2[1:sh+1,:sw]=x
                    q3[1:sh+1,1:sw+1]=x
                    regridded=np.divide(np.nan_to_num(q0)+np.nan_to_num(q1)+np.nan_to_num(q2)+np.nan_to_num(q3),
                                        full,out=np.full((th,tw),np.nan,dtype="float32"),where=support_ok)
                    if var=="LAI":
                        standalone_sum += np.where(support_ok,regridded,0)
                        sat_month=read(SAT/"final_monthly_LAI"/f"{year}_{m+1:02d}.tif")
                        keep=np.isfinite(sat_month)&support_ok
                        monthly_sum += np.where(keep,regridded,0)
                        monthly_count += keep
                    else:
                        aligned=regridded
                if var=="LAI":
                    standalone=standalone_sum/12
                    aligned=np.divide(monthly_sum,monthly_count,
                                      out=np.full((th,tw),np.nan,dtype="float32"),
                                      where=monthly_count>=10)
                else:
                    standalone=aligned.copy()
                standalone[~support_ok]=np.nan
                standalone[~(fraction>0)]=np.nan
                save(OUT/"standalone_LPJ"/var/f"{year}.tif",standalone,ref)
                aligned[~support_ok]=np.nan
                spath=SAT/"final_annual"/var/f"{year}.tif"
                satellite=read(spath)
                save(OUT/"standalone_satellite"/var/f"{year}.tif",satellite,ref)
                gap_support = np.isfinite(satellite) if var == "LAI" else npp_common
                pair = np.isfinite(aligned) & np.isfinite(satellite)
                aligned[~pair] = np.nan
                satellite[~pair] = np.nan
                root = OUT / "yearly_map_pairs_2001_switch" / var
                save(root / "LPJ" / f"{year}.tif", aligned, ref)
                save(root / "satellite" / f"{year}.tif", satellite, ref)
                analysis_p = fill_to_support(aligned, standalone, gap_support)
                save(OUT / "analysis_LPJ" / var / f"{year}.tif", analysis_p, ref)
                if year%10==0 or year==1982 or year==2024:
                    print(var,year,"paired",int(pair.sum()),flush=True)


if __name__=="__main__":
    main()


