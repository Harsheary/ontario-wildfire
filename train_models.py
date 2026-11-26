"""Fit and score the first wildfire models on 2010–2019 leave-one-year-out folds."""

import argparse
import json
import os
from statistics import mean

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from build_baselines import (
    DATA, SEED, SOURCE, TARGETS, YEARS, climatology_loyo, metrics, summarize,
    training_batches, training_plan,
)

CHECKS = DATA / "models_v1.checks.json"
SOURCE_V2 = DATA / "model_table_v2.parquet"
CHECKS_V2 = DATA / "models_v2.checks.json"
CHECKS_V3 = DATA / "models_v3.checks.json"
CAUSE_TARGETS = ("target_human", "target_lightning")
CATCH_METRICS = tuple(f"top_1pct_catch_{name}" for name in ("any", "human", "lightning"))
EXCLUDED = {"cell_id", "feature_date", "year", "doy", "split", "noon_utc_hour"}
LIGHTGBM = {"objective": "binary", "metric": "binary_logloss", "num_leaves": 31,
            "learning_rate": 0.05, "min_data_in_leaf": 200, "feature_fraction": 0.8,
            "is_unbalance": False, "scale_pos_weight": 1.0, "seed": SEED, "verbosity": -1}
MAX_ROUNDS = 1000
STOPPING_ROUNDS = 50
LOGISTIC = {"solver": "newton-cholesky", "l1_ratio": 0.0, "C": 1.0,
            "class_weight": None, "max_iter": 100, "tol": 1e-8}


def read_inputs(source):
    with pq.ParquetFile(source, pre_buffer=False) as parquet:
        dates, cells, groups = training_plan(parquet)
        features = [field.name for field in parquet.schema_arrow
                    if field.name not in EXCLUDED and not field.name.startswith(("target_", "check_"))]
        for name in features:
            dtype = parquet.schema_arrow.field(name).type
            if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype)):
                raise ValueError(f"Expected numeric or boolean feature: {name}")
        design = np.empty((*dates.shape, cells, len(features)), dtype=np.float32)
        flat = design.reshape(dates.size, cells, len(features))
        labels = {name: np.empty((*dates.shape, cells), dtype=np.bool_) for name in TARGETS}
        targets = {name: label.reshape(-1, cells) for name, label in labels.items()}
        for span, batch in training_batches(parquet, dates, cells, groups, (*features, *TARGETS)):
            for i, name in enumerate(features):
                values = batch.column(name).to_numpy(zero_copy_only=False)
                column = flat[span, :, i]
                column[:] = values.reshape(-1, cells)
                if values.dtype.kind == "f" and not np.isfinite(column).all():
                    raise ValueError(f"Nonfinite training feature: {name}")
            for name, target in targets.items():
                target[span] = batch.column(name).to_numpy(zero_copy_only=False).reshape(-1, cells)
    return design, labels, features, groups


def choose_lightgbm_rounds(train, y_train, year_rows, features, params):
    # The fold buffer is in year order; its final year is the inner validation set.
    train_set = lgb.Dataset(train[:-year_rows], label=y_train[:-year_rows], feature_name=features)
    stopping_set = train_set.create_valid(train[-year_rows:], label=y_train[-year_rows:])
    booster = lgb.train(
        params, train_set, num_boost_round=MAX_ROUNDS,
        valid_sets=[stopping_set], valid_names=["early_stopping"],
        callbacks=[lgb.early_stopping(STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(50)],
    )
    return booster.best_iteration, booster.best_score["early_stopping"]["binary_logloss"]


def refit_lightgbm(train, y_train, scored, features, params, best, stopping_loss):
    full_train_set = lgb.Dataset(train, label=y_train, feature_name=features)
    booster = lgb.train(params, full_train_set, num_boost_round=best)
    scores = booster.predict(scored, num_iteration=best)
    gains = booster.feature_importance(importance_type="gain", iteration=best)
    return scores, gains, {"best_iteration": best, "refit_iterations": booster.current_iteration(),
                           "early_stopping_log_loss": stopping_loss}


def fit_lightgbm(train, y_train, year_rows, scored, features, params):
    choice = choose_lightgbm_rounds(train, y_train, year_rows, features, params)
    return refit_lightgbm(train, y_train, scored, features, params, *choice)


def fit_logistic(train, y_train, valid):
    # Float64 keeps the regularized Hessian stable with collinear features.
    # Bound the scaler's work buffers; it sees only training rows.
    scaler = StandardScaler(copy=False)
    for start in range(0, len(train), 131_072):
        scaler.partial_fit(train[start:start + 131_072])
    scaler.transform(train)
    model = LogisticRegression(**LOGISTIC).fit(train, y_train)
    iterations = int(model.n_iter_[0])
    if iterations >= LOGISTIC["max_iter"] - 1:
        raise RuntimeError("Logistic regression did not converge within the iteration limit")
    scored = scaler.transform(valid.astype(np.float64), copy=False)
    return model.predict_proba(scored)[:, 1], iterations


def score(probabilities, labels, totals, k):
    scores = probabilities.reshape(labels["target_any"].shape)
    if not (np.isfinite(scores).all() and scores.min() >= 0 and scores.max() <= 1):
        raise ValueError("Expected finite probabilities in [0, 1]")
    return {**metrics(scores, labels, totals, k),
            "brier_score": float(np.mean(np.square(scores - labels["target_any"]), dtype=np.float64))}


def compare_catch(candidate, reference):
    comparison = {}
    for metric in CATCH_METRICS:
        deltas = {year: candidate[year][metric] - reference[year][metric] for year in YEARS}
        comparison[metric] = {
            "candidate_mean": mean(candidate[year][metric] for year in YEARS),
            "reference_mean": mean(reference[year][metric] for year in YEARS),
            "mean_delta": mean(deltas.values()),
            "win_count": sum(delta > 0 for delta in deltas.values()),
            "win_years": [year for year, delta in deltas.items() if delta > 0],
            "loss_years": [year for year, delta in deltas.items() if delta < 0],
            "tie_years": [year for year, delta in deltas.items() if delta == 0],
        }
    return comparison


def build(threads, version="v1"):
    v2 = version == "v2"
    source, checks = (SOURCE_V2, CHECKS_V2) if v2 else (SOURCE, CHECKS)
    prior = json.loads(CHECKS.read_text(encoding="utf-8")) if v2 else None
    design, targets, features, groups = read_inputs(source)
    variants = {"lightgbm": features}
    if v2:
        from fold_weather import FoldWeather

        if prior["years"] != list(YEARS) or prior["features"] != features[:len(prior["features"])]:
            raise ValueError("V1 report does not match the v2 model years or base features")
        dropped = [name for name in features if name.startswith("fuel_")]
        if not dropped:
            raise ValueError("Fuel-drop ablation requires fuel features")
        keep = [i for i, name in enumerate(features) if name not in dropped]
        projection = {column: i for i, column in enumerate(keep)}
        variants["lightgbm_no_fuel"] = [features[i] for i in keep]
        normals = FoldWeather(YEARS, features, pq.read_schema(source).metadata)
    _, days, cells, n_features = design.shape
    year_rows = days * cells
    k = (cells + 99) // 100
    rates = climatology_loyo(targets["target_any"])
    by_year = {name: {} for name in (*variants, "logistic_regression", "climatology_loyo")}
    folds = {}
    fold_gains = {}
    params = {**LIGHTGBM, "num_threads": threads, "force_col_wise": True}
    design = design.reshape(len(YEARS), year_rows, n_features)
    train_buffer = np.empty((len(YEARS) - 1) * year_rows * n_features, dtype=np.float64)
    train = train_buffer.reshape(-1, n_features)
    y_train = np.empty(len(train), dtype=np.bool_)
    train_years = train.reshape(len(YEARS) - 1, year_rows, n_features)
    train_targets = y_train.reshape(len(YEARS) - 1, days, cells)
    with threadpool_limits(limits=threads):
        for i, year in enumerate(YEARS):
            training_indices = [j for j in range(len(YEARS)) if j != i]
            training_years = [YEARS[j] for j in training_indices]
            train_targets[:i], train_targets[i:] = targets["target_any"][:i], targets["target_any"][i + 1:]
            labels = {name: target[i] for name, target in targets.items()}
            totals = {name: int(target.sum()) for name, target in labels.items()}
            if not all(totals.values()):
                raise ValueError(f"Undefined catch rate for a target without positives in {year}")
            stopping_year = training_years[-1]
            folds[year] = {"training_years": training_years, "training_rows": len(y_train),
                           "validation_rows": year_rows, "positives": totals, "base_rate": totals["target_any"] / year_rows}
            for name, inputs in variants.items():
                print(f"{year}: {name} selects rounds on {stopping_year}", flush=True)
                if name == "lightgbm":
                    train_years[:i], train_years[i:] = design[:i], design[i + 1:]
                    fit_rows, fit_years, valid = train, train_years, design[i].copy() if v2 else design[i]
                    mapping = None
                else:
                    # Reuse the full fit's buffer, which logistic no longer needs.
                    fit_rows = train_buffer[:len(y_train) * len(inputs)].reshape(-1, len(inputs))
                    fit_years = fit_rows.reshape(len(training_indices), year_rows, len(inputs))
                    for j, out in zip(training_indices, fit_years, strict=True):
                        for start in range(0, year_rows, 131_072):
                            span = slice(start, start + 131_072)
                            out[span] = design[j, span][:, keep]
                    valid = design[i][:, keep]
                    mapping = projection
                if v2:
                    normals.apply(design, fit_years, training_indices, training_indices[:-1], mapping)
                    choice = choose_lightgbm_rounds(fit_rows, y_train, year_rows, inputs, params)
                    normals.apply(design, fit_years, training_indices, training_indices, mapping)
                    normals.apply(design, (valid,), (i,), training_indices, mapping)
                    probabilities, gains, tree_fit = refit_lightgbm(fit_rows, y_train, valid, inputs, params, *choice)
                else:
                    probabilities, gains, tree_fit = fit_lightgbm(fit_rows, y_train, year_rows, valid, inputs, params)
                tree_fit["early_stopping_year"] = stopping_year
                folds[year][name] = tree_fit
                by_year[name][year] = score(probabilities, labels, totals, k)
                print(f"{year}: {name} refit with {tree_fit['refit_iterations']} rounds", flush=True)
                if name == "lightgbm":
                    fold_gains[year] = dict(zip(inputs, gains.tolist(), strict=True))
                    probabilities, iterations = fit_logistic(train, y_train, valid)
                    folds[year]["logistic_iterations"] = iterations
                    by_year["logistic_regression"][year] = score(probabilities, labels, totals, k)
            by_year["climatology_loyo"][year] = score(np.broadcast_to(rates[i], (days, cells)), labels, totals, k)
            print(f"{year}: fold complete (logistic iterations: {iterations})", flush=True)

    gains = {feature: mean(fold_gains[year][feature] for year in YEARS) for feature in features}
    top_gains = sorted(gains, key=lambda feature: (-gains[feature], feature))[:15]
    catch = "top_1pct_catch_any"
    wins = [year for year in YEARS if by_year["lightgbm"][year][catch] > by_year["climatology_loyo"][year][catch]]
    report = {
        "source": str(source.relative_to(DATA.parent)), "year_filter": f"year <= {YEARS[-1]}", "years": YEARS,
        "rows_read": sum(rows for _, rows, _ in groups), "row_groups_read": len(groups),
        "partial_row_groups": [{"row_group": index, "training_prefix_rows": rows, "total_rows": total}
                               for index, rows, total in groups if rows != total],
        "features": features, "top_cells_per_day": k, "top_k_ties": "cell_id ascending",
        "pr_auc_method": "Non-interpolated average precision for target_any over all cell-days in each year",
        "brier_method": "Mean squared probability error for target_any over all cell-days in each year",
        "lightgbm_parameters": {**params, "num_boost_round": MAX_ROUNDS, "early_stopping_rounds": STOPPING_ROUNDS},
        "logistic_parameters": {**LOGISTIC, "penalty": "L2", "standardization": "training fold only"},
        "folds": folds,
        "models": {name: {"annual": results, "summary": summarize(results)} for name, results in by_year.items()},
        "lightgbm_climatology_top_1pct_any_wins": {"count": len(wins), "years": wins},
        "lightgbm_gain_by_fold": fold_gains,
        "lightgbm_top_15_mean_gain": [{"feature": name, "mean_gain": gains[name]} for name in top_gains],
    }
    if v2:
        if json.loads(json.dumps(report["models"]["climatology_loyo"])) != prior["models"]["climatology_loyo"]:
            raise ValueError("V1 comparison climatology does not match the v2 folds")
        prior_lightgbm = prior["models"]["lightgbm"]["annual"]
        report["lightgbm_vs_v1"] = compare_catch(by_year["lightgbm"], {year: prior_lightgbm[str(year)] for year in YEARS})
        report["fuel_ablation"] = {
            "dropped_features": dropped,
            "full_vs_no_fuel": compare_catch(by_year["lightgbm"], by_year["lightgbm_no_fuel"]),
        }
        report["anomaly_normals"] = (
            "Refit within each fitting split; exclude each row's year and all years outside that fitting split"
        )
    checks.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def score_combined(human, lightning, labels, totals, k):
    for probabilities in (human, lightning):
        if not (np.isfinite(probabilities).all() and probabilities.min() >= 0 and probabilities.max() <= 1):
            raise ValueError("Expected finite cause probabilities in [0, 1]")
    scores = (human + lightning).reshape(labels["target_any"].shape)
    ranking = metrics(scores, labels, totals, k)
    clipped = int(np.count_nonzero(scores > 1))
    # The cause labels can overlap. The raw sum is a ranking score, not a union probability.
    np.minimum(scores, 1, out=scores)
    brier = float(np.mean(np.square(scores - labels["target_any"]), dtype=np.float64))
    return {**ranking, "brier_score": brier}, clipped


def build_cause_models(threads):
    from fold_weather import FoldWeather

    prior = json.loads(CHECKS_V2.read_text(encoding="utf-8"))
    design, targets, features, groups = read_inputs(SOURCE_V2)
    _, days, cells, n_features = design.shape
    year_rows = days * cells
    k = (cells + 99) // 100
    params = {**LIGHTGBM, "num_threads": threads, "force_col_wise": True}
    report_params = {**params, "num_boost_round": MAX_ROUNDS, "early_stopping_rounds": STOPPING_ROUNDS}
    rows_read = sum(rows for _, rows, _ in groups)
    partial_groups = [{"row_group": index, "training_prefix_rows": rows, "total_rows": total}
                      for index, rows, total in groups if rows != total]
    source = str(SOURCE_V2.relative_to(DATA.parent))
    year_filter = f"year <= {YEARS[-1]}"
    if (prior["source"] != source or prior["year_filter"] != year_filter or prior["years"] != list(YEARS)
            or prior["rows_read"] != rows_read or prior["partial_row_groups"] != partial_groups
            or prior["features"] != features or prior["top_cells_per_day"] != k
            or {key: value for key, value in prior["lightgbm_parameters"].items() if key != "num_threads"}
            != {key: value for key, value in report_params.items() if key != "num_threads"}):
        raise ValueError("V2 reference does not match the input features, folds or LightGBM settings")

    normals = FoldWeather(YEARS, features, pq.read_schema(SOURCE_V2).metadata)
    design = design.reshape(len(YEARS), year_rows, n_features)
    train_years = np.empty((len(YEARS) - 1, year_rows, n_features), dtype=design.dtype)
    train = train_years.reshape(-1, n_features)
    y_train = np.empty((len(CAUSE_TARGETS), len(train)), dtype=np.bool_)
    train_labels = y_train.reshape(len(CAUSE_TARGETS), len(YEARS) - 1, days, cells)
    by_year, folds = {}, {}
    single = prior["models"]["lightgbm"]
    climatology = prior["models"]["climatology_loyo"]
    with threadpool_limits(limits=threads):
        for i, year in enumerate(YEARS):
            training = [j for j in range(len(YEARS)) if j != i]
            training_years = [YEARS[j] for j in training]
            stop = training_years[-1]
            labels = {name: values[i] for name, values in targets.items()}
            totals = {name: int(values.sum()) for name, values in labels.items()}
            fold = {"training_years": training_years, "training_rows": len(train), "validation_rows": year_rows,
                    "positives": totals, "base_rate": totals["target_any"] / year_rows}
            reference = prior["folds"][str(year)]
            if not all(totals.values()) or any(reference[key] != value for key, value in fold.items()):
                raise ValueError(f"{year}: v2 reference fold or target totals do not match")

            train_years[:i], train_years[i:] = design[:i], design[i + 1:]
            for out, name in zip(train_labels, CAUSE_TARGETS, strict=True):
                target = targets[name]
                out[:i], out[i:] = target[:i], target[i + 1:]
            normals.apply(design, train_years, training, training[:-1])
            choices = []
            for name, target in zip(CAUSE_TARGETS, y_train, strict=True):
                print(f"{year}: {name} selects rounds on {stop}", flush=True)
                choices.append(choose_lightgbm_rounds(train, target, year_rows, features, params))

            valid = design[i].copy()
            normals.apply(design, train_years, training, training)
            normals.apply(design, (valid,), (i,), training)
            predictions = []
            for name, target, choice in zip(CAUSE_TARGETS, y_train, choices, strict=True):
                print(f"{year}: {name} refits with {choice[0]} rounds", flush=True)
                prediction, _, fit = refit_lightgbm(train, target, valid, features, params, *choice)
                if fit["refit_iterations"] != choice[0]:
                    raise ValueError(f"{year}: {name} refit did not reach the selected round count")
                predictions.append(prediction)
                fold[name] = {**fit, "early_stopping_year": stop}
            by_year[year], fold["brier_clipped_rows"] = score_combined(*predictions, labels, totals, k)
            folds[year] = fold
            print(f"{year}: cause-model fold complete", flush=True)

    single_annual = {year: single["annual"][str(year)] for year in YEARS}
    climate_annual = {year: climatology["annual"][str(year)] for year in YEARS}
    report = {
        "source": source, "reference": str(CHECKS_V2.relative_to(DATA.parent)), "year_filter": year_filter,
        "years": YEARS, "rows_read": rows_read, "row_groups_read": len(groups), "partial_row_groups": partial_groups,
        "features": features, "targets": CAUSE_TARGETS, "top_cells_per_day": k, "top_k_ties": prior["top_k_ties"],
        "lightgbm_parameters": report_params, "anomaly_normals": prior["anomaly_normals"],
        "combined_score": "P(target_human) + P(target_lightning), unmodified for ranking and PR-AUC",
        "brier_method": "Mean squared error against target_any; cap combined scores at 1 for Brier only",
        "pr_auc_method": prior["pr_auc_method"], "folds": folds,
        "models": {"lightgbm_combined": {"annual": by_year, "summary": summarize(by_year)},
                   "lightgbm_v2_single": single, "climatology_loyo": climatology},
        "combined_vs_v2_single": compare_catch(by_year, single_annual),
        "combined_vs_climatology": compare_catch(by_year, climate_annual),
    }
    CHECKS_V3.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.version == "v3":
        report, checks = build_cause_models(args.threads), CHECKS_V3
    else:
        report = build(args.threads, args.version)
        checks = CHECKS_V2 if args.version == "v2" else CHECKS
    print(f"Scored {report['rows_read']:,} training-era rows across {len(YEARS)} folds; see {checks.relative_to(DATA.parent)}")
