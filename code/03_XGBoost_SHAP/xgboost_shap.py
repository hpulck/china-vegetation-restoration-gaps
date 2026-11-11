"""Fit the fixed 12-predictor gap models and export spatial out-of-fold TreeSHAP."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import DMatrix, XGBRegressor


TARGETS = {"GLAI": "GLAI_mean", "GNPP": "GNPP_mean"}
PREDICTORS = {
    "PET__mean": "PET",
    "RH__mean": "RH",
    "PRE_TC__mean": "PRE",
    "DEM": "DEM",
    "TEM_SEASON_SD__mean": "TEMs",
    "PRE_TC__interannual_variability": "PREi",
    "VPD__mean": "VPD",
    "CLD__mean": "CLD",  # Long-term mean cloud cover.
    "NDEP_TOTAL__mean": "NDEP",
    "GRAZING__mean_available": "GRAZ",
    "TEM__mean": "TEM",
    "CLCD_FOREST__twmean_1985_2024": "FC",
}
ADDED = (
    "RH__mean", "NDEP_TOTAL__mean", "GRAZING__mean_available",
    "CLCD_FOREST__twmean_1985_2024",
)
SEED = 20261106


def read_inputs(response_path, feature_path):
    columns = ["pixel_index", "longitude", "latitude", "area_weight", "block", *TARGETS.values()]
    columns.extend(name for name in PREDICTORS if name not in ADDED)
    responses = pd.read_parquet(response_path, columns=columns)
    features = pd.read_parquet(feature_path, columns=["pixel_index", *ADDED])
    data = responses.merge(features, on="pixel_index", how="left", validate="one_to_one")
    valid = np.isfinite(data[list(TARGETS.values())]).all(axis=1)
    return data.loc[valid].reset_index(drop=True)


def spatial_folds(data):
    transformer = Transformer.from_crs(
        4326, "+proj=aea +lat_1=25 +lat_2=47 +lat_0=35 +lon_0=105 +datum=WGS84 +units=m",
        always_xy=True,
    )
    x, y = transformer.transform(data.longitude.to_numpy(float), data.latitude.to_numpy(float))
    xy = np.column_stack((x, y))
    folds = []
    for train, test in GroupKFold(n_splits=5).split(data, groups=data.block):
        distance = cKDTree(xy[test]).query(xy[train], k=1)[0]
        folds.append((train[distance >= 75_000], test))
    return folds


def fit_model(train_x, train_y, train_weights, fold):
    model = XGBRegressor(
        n_estimators=320, max_depth=4, learning_rate=0.04, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=15, reg_lambda=10,
        objective="reg:squarederror", tree_method="hist", n_jobs=4, random_state=SEED + fold,
    )
    return model.fit(train_x, train_y, sample_weight=train_weights)


def metrics(y, prediction, weight):
    return {
        "R2": r2_score(y, prediction, sample_weight=weight),
        "RMSE": np.sqrt(mean_squared_error(y, prediction, sample_weight=weight)),
        "MAE": mean_absolute_error(y, prediction, sample_weight=weight),
        "Bias": np.average(prediction - y, weights=weight),
    }


def shap_interactions(booster, test_x, prediction, shap, rows, destination):
    """Store native interaction entries and check closure against TreeSHAP."""
    for start in range(0, len(rows), 2000):
        end = start + 2000
        batch = DMatrix(test_x[start:end], nthread=4)
        values = booster.predict(batch, pred_interactions=True).astype(float)
        terms = values[:, :-1, :-1]
        np.testing.assert_allclose(terms, terms.swapaxes(1, 2), atol=2e-5, rtol=0)
        np.testing.assert_allclose(terms.sum(axis=2), shap[start:end], atol=2e-5, rtol=0)
        np.testing.assert_allclose(values.sum(axis=(1, 2)), prediction[start:end], atol=2e-5, rtol=0)
        destination[rows[start:end]] = terms


def run_target(data, x, weights, folds, product, output, *, positive_only, interactions):
    y = data[TARGETS[product]].to_numpy(float)
    keep = y > 0 if positive_only else np.ones(len(y), dtype=bool)
    prediction = np.full(len(y), np.nan)
    base = np.full(len(y), np.nan, dtype=np.float32)
    shap = np.full(x.shape, np.nan, dtype=np.float32)
    fold_ids = np.full(len(y), -1)
    pair_values = np.full((len(y), x.shape[1], x.shape[1]), np.nan, dtype=np.float32) if interactions else None
    rows = []
    for fold, (train, test) in enumerate(folds):
        train, test = train[keep[train]], test[keep[test]]
        if len(train) == 0 or len(test) < 2:
            raise ValueError(f"{product} fold {fold + 1} has too few retained samples.")
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        train_x = imputer.fit_transform(x[train])
        test_x = imputer.transform(x[test])
        model = fit_model(train_x, y[train], weights[train], fold)
        predicted = model.predict(test_x)
        booster = model.get_booster()
        contribution = booster.predict(DMatrix(test_x, nthread=4), pred_contribs=True)
        np.testing.assert_allclose(contribution.sum(axis=1), predicted, rtol=1e-5, atol=1e-5)
        prediction[test] = predicted
        shap[test] = contribution[:, :-1]
        base[test] = contribution[:, -1]
        fold_ids[test] = fold
        if pair_values is not None:
            shap_interactions(booster, test_x, predicted, contribution[:, :-1], test, pair_values)
        rows.append({"product": product, "fold": fold, "train_n": len(train), "test_n": len(test),
                     **metrics(y[test], predicted, weights[test])})
        print(f"{product} fold {fold + 1}/5  R²={rows[-1]['R2']:.4f}", flush=True)

    if (fold_ids[keep] < 0).any() or not np.isfinite(shap[keep]).all():
        raise RuntimeError(f"Incomplete held-out predictions or SHAP values for {product}.")
    rows.append({"product": product, "fold": "pooled", "test_n": int(keep.sum()),
                 **metrics(y[keep], prediction[keep], weights[keep])})
    metadata = dict(names=list(PREDICTORS), x=x[keep].astype(np.float32), area_weight=weights[keep],
                    pixel_index=data.pixel_index.to_numpy()[keep], fold=fold_ids[keep])
    np.savez_compressed(output / f"{product}_OOF_SHAP.npz", **metadata,
                        values=shap[keep], base_value=base[keep], observed=y[keep],
                        prediction=prediction[keep])
    if pair_values is not None:
        np.savez_compressed(output / f"{product}_OOF_interactions.npz", **metadata,
                            interactions=pair_values[keep])
    importance = np.average(np.abs(shap[keep].astype(float)), axis=0, weights=weights[keep])
    total = importance.sum()
    ranking = pd.DataFrame({
        "product": product, "factor": list(PREDICTORS), "label": list(PREDICTORS.values()),
        "mean_absolute_oof_shap": importance,
        "share_percent": 100.0 * importance / total if total else np.nan,
    })
    ranking["rank"] = ranking["mean_absolute_oof_shap"].rank(ascending=False, method="min").astype(int)
    return rows, ranking.sort_values("rank", kind="stable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="A new output directory.")
    parser.add_argument("--positive-only", action="store_true", help="For each target keep G > 0.")
    parser.add_argument("--interactions", action="store_true", help="Also calculate exact SHAP interactions.")
    args = parser.parse_args()

    data = read_inputs(args.responses, args.features)
    x = data[list(PREDICTORS)].to_numpy(float)
    weight = data.area_weight.to_numpy(float, copy=True)
    if not len(data) or not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("Area weights must be finite and strictly positive on the model domain.")
    if np.isinf(x).any() or (np.isfinite(x).mean(axis=0) < 0.05).any() or (np.nanstd(x, axis=0) <= 1e-12).any():
        raise ValueError("At least one predictor has infinite values, insufficient coverage or no variation.")
    weight /= weight.mean()
    folds = spatial_folds(data)
    args.output.mkdir(parents=True, exist_ok=False)
    all_metrics, all_rankings = [], []
    for product in TARGETS:
        scores, rankings = run_target(data, x, weight, folds, product, args.output,
                                      positive_only=args.positive_only, interactions=args.interactions)
        all_metrics.extend(scores)
        all_rankings.append(rankings)
    pd.DataFrame(all_metrics).to_csv(args.output / "spatial_oof_metrics.csv", index=False)
    pd.concat(all_rankings, ignore_index=True).to_csv(
        args.output / "SHAP_12_factors.csv", index=False, encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
