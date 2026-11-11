"""Calculate signed restoration gaps, select G > 0 and test minimum-P masks."""

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


NODATA = -9999.0
THRESHOLDS = {
    "LAI": (0.0, 0.05, 0.10, 0.20),
    "NPP": (0.0, 0.02, 0.05, 0.10),
}


def read_raster(path):
    with rasterio.open(path) as source:
        data = source.read(1, masked=True).astype("float64").filled(np.nan)
        return data, source.profile.copy()


def require_same_grid(reference, other, path):
    keys = ("width", "height", "transform", "crs")
    if any(reference[key] != other[key] for key in keys):
        raise ValueError(f"Raster grid does not match the reference: {path}")


def write_raster(path, data, profile, **tags):
    profile = dict(profile, driver="GTiff", count=1, dtype="float64", nodata=NODATA,
                   compress="deflate", predictor=3)
    finite = np.isfinite(data)
    if np.any(data[finite] == NODATA):
        raise ValueError(f"Valid values collide with the NoData code: {path}")
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(np.where(finite, data, NODATA), 1)
        destination.update_tags(**tags)


def relative_gap(potential, actual):
    valid = np.isfinite(potential) & np.isfinite(actual) & (potential > 0) & (actual >= 0)
    gap = np.full(potential.shape, np.nan)
    gap[valid] = (potential[valid] - actual[valid]) / potential[valid]
    return gap


def positive_gap(gap):
    return np.where(np.isfinite(gap) & (gap > 0), gap, np.nan)


def calculate(args):
    years = range(args.start, args.end + 1)
    min_years = len(years) if args.min_valid_years is None else args.min_valid_years
    if not 1 <= min_years <= len(years):
        raise ValueError("The period and minimum valid-year count must be positive and consistent.")
    for folder in (args.p_dir, args.a_dir):
        for year in years:
            path = folder / f"{year}.tif"
            if not path.is_file():
                raise FileNotFoundError(path)

    args.output.mkdir(parents=True, exist_ok=False)
    annual = args.output / "annual_G"
    annual.mkdir()
    profile = None
    for year in years:
        p_path = args.p_dir / f"{year}.tif"
        a_path = args.a_dir / f"{year}.tif"
        potential, p_profile = read_raster(p_path)
        actual, a_profile = read_raster(a_path)
        if profile is None:
            profile = p_profile
            count = np.zeros(potential.shape, dtype=np.int32)
            sum_p = np.zeros(potential.shape)
            sum_d = np.zeros(potential.shape)
            sum_g = np.zeros(potential.shape)
        require_same_grid(profile, p_profile, p_path)
        require_same_grid(profile, a_profile, a_path)
        gap = relative_gap(potential, actual)
        valid = np.isfinite(gap)
        count += valid
        sum_p[valid] += potential[valid]
        sum_d[valid] += potential[valid] - actual[valid]
        sum_g[valid] += gap[valid]
        write_raster(annual / f"{year}.tif", gap, profile, formula="(P-A)/P",
                     P_source=str(p_path), A_source=str(a_path))

    support = count >= min_years
    p_mean = np.full(count.shape, np.nan)
    g_period = np.full(count.shape, np.nan)
    np.divide(sum_p, count, out=p_mean, where=support)
    if args.method == "ratio-of-sums":
        np.divide(sum_d, sum_p, out=g_period, where=support)
        formula = "sum(P-A)/sum(P) over valid paired years"
    else:
        np.divide(sum_g, count, out=g_period, where=support)
        formula = "mean of annual (P-A)/P over valid paired years"
    tags = {"period": f"{args.start}-{args.end}", "minimum_valid_years": str(min_years)}
    write_raster(args.output / "P_mean.tif", p_mean, profile, **tags)
    write_raster(args.output / "G_period.tif", g_period, profile, formula=formula, **tags)
    write_raster(args.output / "G_period_positive.tif", positive_gap(g_period), profile,
                 formula=formula, selection="G > 0", **tags)
    write_raster(args.output / "valid_year_count.tif", np.where(count > 0, count, np.nan),
                 profile, **tags)


def select_positive(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    gap, profile = read_raster(args.g)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_raster(args.output, positive_gap(gap), profile, source=str(args.g), selection="G > 0")


def area_weights(fraction, profile):
    transform = profile["transform"]
    crs = profile["crs"]
    if crs is None or crs.to_epsg() != 4326 or transform.b != 0 or transform.d != 0:
        raise ValueError("Area-share calculation requires a north-up EPSG:4326 grid.")
    if np.any(np.isfinite(fraction) & ((fraction < 0) | (fraction > 1))):
        raise ValueError("China coverage fractions must be between 0 and 1.")
    latitudes = transform.f + (np.arange(profile["height"]) + 0.5) * transform.e
    return fraction * np.cos(np.deg2rad(latitudes[:, None]))


def climate_zones(path, scale, profile, shape):
    zones = {"All China": np.ones(shape, dtype=bool)}
    if path is None:
        return zones
    with rasterio.open(path) as source, WarpedVRT(
        source, crs=profile["crs"], transform=profile["transform"],
        width=profile["width"], height=profile["height"],
        dtype="float64", nodata=np.nan, resampling=Resampling.average,
    ) as aligned:
        ai = aligned.read(1, masked=True).filled(np.nan) * scale
    finite = np.isfinite(ai)
    zones.update({
        "Arid": finite & (ai < 0.20),
        "Semiarid": finite & (ai >= 0.20) & (ai < 0.50),
        "Dry subhumid": finite & (ai >= 0.50) & (ai < 0.65),
        "Humid": finite & (ai >= 0.65),
        "Arid + semiarid": finite & (ai < 0.50),
    })
    return zones


def percentage(numerator, denominator):
    return 100.0 * numerator / denominator if denominator > 0 else ""


def sensitivity(args):
    potential, profile = read_raster(args.p)
    gap, g_profile = read_raster(args.g)
    fraction, fraction_profile = read_raster(args.china_fraction)
    require_same_grid(profile, g_profile, args.g)
    require_same_grid(profile, fraction_profile, args.china_fraction)
    weights = area_weights(fraction, profile)
    valid = (fraction > 0) & np.isfinite(potential) & (potential > 0) & np.isfinite(gap)
    high = valid & (gap > 0.70)
    thresholds = THRESHOLDS[args.metric]
    retention = np.full(gap.shape, np.nan)
    retention[high] = sum((potential[high] >= threshold) for threshold in thresholds)

    args.output.mkdir(parents=True, exist_ok=False)
    write_raster(args.output / "high_G_retention_count.tif", retention, profile,
                 metric=args.metric, high_gap_rule="G > 0.70",
                 minimum_P=",".join(map(str, thresholds)),
                 meaning="number of thresholds passed by baseline high-gap pixels")
    zones = climate_zones(args.aridity, args.aridity_scale, profile, gap.shape)
    with (args.output / "Pmin_sensitivity.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "metric", "zone", "minimum_P", "baseline_valid_pixels", "retained_valid_pixels",
            "baseline_high_pixels", "retained_high_pixels", "valid_area_retained_pct",
            "high_area_retained_pct", "high_area_excluded_pct",
        ])
        for name, zone in zones.items():
            valid_zone = valid & zone
            high_zone = high & zone
            valid_area = weights[valid_zone].sum()
            high_area = weights[high_zone].sum()
            for threshold in thresholds:
                included = potential >= threshold
                valid_kept = valid_zone & included
                high_kept = high_zone & included
                writer.writerow([
                    args.metric, name, threshold, valid_zone.sum(), valid_kept.sum(),
                    high_zone.sum(), high_kept.sum(),
                    percentage(weights[valid_kept].sum(), valid_area),
                    percentage(weights[high_kept].sum(), high_area),
                    percentage(weights[high_zone & ~included].sum(), high_area),
                ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    annual = commands.add_parser("calculate", help="Calculate annual and period gaps from aligned P and A.")
    annual.add_argument("--p-dir", type=Path, required=True)
    annual.add_argument("--a-dir", type=Path, required=True)
    annual.add_argument("--start", type=int, default=1982)
    annual.add_argument("--end", type=int, default=2024)
    annual.add_argument("--min-valid-years", type=int)
    annual.add_argument("--method", choices=("ratio-of-sums", "mean-annual"), default="ratio-of-sums")
    annual.add_argument("--output", type=Path, required=True)
    annual.set_defaults(run=calculate)

    positive = commands.add_parser("positive", help="Keep strictly positive values from an existing G raster.")
    positive.add_argument("--g", type=Path, required=True)
    positive.add_argument("--output", type=Path, required=True)
    positive.set_defaults(run=select_positive)

    pmin = commands.add_parser("sensitivity", help="Assess retention under the study's minimum-P thresholds.")
    pmin.add_argument("--metric", choices=tuple(THRESHOLDS), required=True)
    pmin.add_argument("--p", type=Path, required=True)
    pmin.add_argument("--g", type=Path, required=True)
    pmin.add_argument("--china-fraction", type=Path, required=True)
    pmin.add_argument("--aridity", type=Path)
    pmin.add_argument("--aridity-scale", type=float, default=1.0)
    pmin.add_argument("--output", type=Path, required=True)
    pmin.set_defaults(run=sensitivity)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
