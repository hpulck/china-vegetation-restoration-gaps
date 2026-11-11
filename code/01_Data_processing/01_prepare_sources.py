"""Prepare native source rasters on a common 0.1-degree China grid.

Only source product files are read here. No previously cropped or spliced raster
enters this pipeline.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from pyhdf.SD import SD, SDC
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, from_bounds

ROOT = Path(os.environ.get("LAI_NPP_WORK", Path(__file__).resolve().parent / "work")).resolve()
BASE = Path(os.environ.get("LAI_NPP_MODIS_ROOT", r"G:\科研数据\04_论文数据\01_LPJ GUESS\Data\merged_files"))
GIMMS_DIR = Path(os.environ.get("LAI_NPP_SOURCES", r"H:\China_LAI_NPP_rebuild_2026\sources"))
GLASS_DIR = GIMMS_DIR / "GLASS_HDF"
MOD_LAI_DIR = BASE / "LAI/Modis/Moth/01TIF"
MOD_NPP_DIR = BASE / "NPP/MODIS/MODIS/01TIF"
CHINA_VECTOR = Path(os.environ.get("LAI_NPP_CHINA_VECTOR", r"G:\科研数据\04_论文数据\01_LPJ GUESS\Data\中国矢量\国界\China.shp"))
GRID = from_origin(73.5, 53.6, 0.1, 0.1)
SHAPE = (355, 616)
CRS = "EPSG:4326"
ND = -9999.0
OUT = ROOT / "prepared"


def warp(src: np.ndarray, transform: Affine, crs: str, method: Resampling = Resampling.average) -> np.ndarray:
    dst = np.full(SHAPE, np.nan, dtype="float32")
    reproject(src, dst, src_transform=transform, src_crs=crs,
              dst_transform=GRID, dst_crs=CRS, src_nodata=np.nan,
              dst_nodata=np.nan, resampling=method, num_threads=2)
    return dst


def save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=SHAPE[0], width=SHAPE[1],
                       count=1, dtype="float32", crs=CRS, transform=GRID,
                       nodata=ND, compress="deflate", predictor=3, tiled=True) as dst:
        dst.write(np.where(np.isfinite(arr), arr, ND).astype("float32"), 1)


def read_gimms_half(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        w = from_bounds(73.4, 18.0, 135.2, 53.7, src.transform).round_offsets().round_lengths()
        w = w.intersection(Window(0, 0, src.width, src.height))
        values = src.read(1, window=w).astype("float32")
        qc = src.read(2, window=w)
        # Product V1.2: 0.001 LAI scale, 65535 fill; QC 0 good, 1 spline, 2 snow/cloud.
        valid = (values < 10000) & (qc <= 1)
        values[~valid] = np.nan
        values *= .001
        return warp(values, src.window_transform(w), src.crs)


def read_tif(path: Path, maximum: float, minimum: float = 0) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        valid = np.isfinite(arr) & (arr >= minimum) & (arr <= maximum)
        if src.nodata is not None:
            valid &= arr != src.nodata
        arr[~valid] = np.nan
        return warp(arr, src.transform, src.crs)


def read_glass(path: Path) -> np.ndarray:
    h = SD(str(path), SDC.READ)
    ds = h.select("NPP")
    # GLASS17B12 V40 annual global 0.05-degree grid, units gC m-2 yr-1.
    r0, r1 = 725, 1441
    c0, c1 = 5067, 6305
    a = np.array(ds[r0:r1, c0:c1], dtype="float32")
    h.end()
    a[(a < 0) | (a > 8000)] = np.nan  # includes 65535 fill
    a /= 1000  # kg C m-2 yr-1
    transform = from_origin(-180 + c0 * .05, 90 - r0 * .05, .05, .05)
    return warp(a, transform, CRS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    boundary = gpd.read_file(CHINA_VECTOR).to_crs(CRS)
    shapes = [(geom, 1) for geom in boundary.geometry]
    # Supersampling captures fractional coastal cells for area-weighted statistics.
    s = 4
    fine = rasterize(shapes, out_shape=(SHAPE[0]*s, SHAPE[1]*s),
                     transform=GRID * Affine.scale(1/s, 1/s), dtype="uint8")
    fraction = fine.reshape(SHAPE[0], s, SHAPE[1], s).mean(axis=(1, 3)).astype("float32")
    save(OUT / "china_fraction.tif", fraction)
    land = fraction > 0
    print("China intersected grid cells", int(land.sum()), flush=True)

    rows = []
    def record(product: str, date: str, arr: np.ndarray, paths: str) -> None:
        arr[~land] = np.nan
        path = OUT / product / f"{date}.tif"
        save(path, arr)
        finite = np.isfinite(arr) & land
        rows.append({"product": product, "date": date, "source": paths,
                     "valid_china_cells": int(finite.sum()),
                     "coverage_china": float(finite.sum()/land.sum()),
                     "min": float(np.nanmin(arr)), "mean": float(np.nanmean(arr)),
                     "max": float(np.nanmax(arr))})

    gimms_paths = {}
    for p in GIMMS_DIR.glob("GIMMS_LAI4g_AVHRR_solely_*/*.tif"):
        hit = re.search(r"_(\d{4})(\d{2})(0[12])\.tif$", p.name)
        if hit:
            gimms_paths[tuple(map(int, hit.groups()))] = p
    print("GIMMS sole-AVHRR half-month files", len(gimms_paths), flush=True)
    if len(gimms_paths) != 34*12*2:
        raise RuntimeError("GIMMS AVHRR source inventory incomplete")

    for year in range(1982, 2016):
        for month in range(1, 13):
            paths = [gimms_paths[(year, month, half)] for half in (1, 2)]
            parts = [read_gimms_half(p) for p in paths]
            count = np.isfinite(parts[0]).astype("uint8") + np.isfinite(parts[1]).astype("uint8")
            total = np.nan_to_num(parts[0], nan=0) + np.nan_to_num(parts[1], nan=0)
            arr = np.divide(total, count, out=np.full(SHAPE, np.nan, dtype="float32"), where=count > 0)
            record("LAI_GIMMS_AVHRR", f"{year}_{month:02d}", arr, " | ".join(str(p) for p in paths))
        print("GIMMS", year, flush=True)

    for year in range(2001, 2025):
        for month in range(1, 13):
            p = MOD_LAI_DIR / f"{year}_{month}.tif"
            record("LAI_MODIS", f"{year}_{month:02d}", read_tif(p, 10, .000001), str(p))
        print("MODIS LAI", year, flush=True)

    for year in range(1982, 2019):
        p = next(GLASS_DIR.glob(f"GLASS17B12.V40.A{year}001.*.hdf"))
        record("NPP_GLASS", str(year), read_glass(p), str(p))
        print("GLASS", year, flush=True)

    for year in range(2001, 2025):
        p = MOD_NPP_DIR / f"MOD17A3HGF_NPP_AnnualMean_{year}.tif"
        record("NPP_MODIS", str(year), read_tif(p, 5, .000001), str(p))
        print("MODIS NPP", year, flush=True)

    with (ROOT / "source_inventory.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
