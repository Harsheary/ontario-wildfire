"""Repeat the v1 LightGBM static-feature ablations on the saved v1 folds."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import train_models as tm
from build_baselines import summarize

CHECKS = tm.DATA / "models_v1.ablation.checks.json"
ACCESS = set(("road_length_km road_rail_1km_share nearest_community_km population_per_km2 "
              "pop_missing census_da_coverage_share provincial_park_point_count campground_point_count "
              "recreation_point_count provincial_park_share").split())


def build(threads):
    baseline = json.loads(tm.CHECKS.read_text(encoding="utf-8"))
    design, targets, features, groups = tm.read_inputs(tm.SOURCE)
    _, days, cells, width = design.shape
    year_rows = days * cells
    top = (cells + 99) // 100
    if (baseline["years"] != list(tm.YEARS) or baseline["features"] != features
            or baseline["top_cells_per_day"] != top or baseline["rows_read"] != sum(n for _, n, _ in groups)):
        raise ValueError("V1 report and input table do not match")
    design = design.reshape(len(tm.YEARS), year_rows, width)
    fuel = {f for f in features if f.startswith("fuel_")}
    if len(fuel) != 7 or not ACCESS <= set(features):
        raise ValueError("V1 feature set does not contain the original ablation groups")
    params = {**tm.LIGHTGBM, "num_threads": threads, "force_col_wise": True}
    report_params = {**params, "num_boost_round": tm.MAX_ROUNDS, "early_stopping_rounds": tm.STOPPING_ROUNDS}
    if {**baseline["lightgbm_parameters"], "num_threads": threads} != report_params:
        raise ValueError("V1 report uses different LightGBM parameters")
    full = baseline["models"]["lightgbm"]
    ablations = {}
    with threadpool_limits(limits=threads):
        for name, drop in (("no_access", ACCESS), ("no_fuel", fuel), ("neither", ACCESS | fuel)):
            keep = [j for j, column in enumerate(features) if column not in drop]
            kept = [features[j] for j in keep]
            train = np.empty((len(tm.YEARS) - 1, year_rows, len(keep)), dtype=np.float32)
            scored = np.empty((year_rows, len(keep)), dtype=np.float32)
            y_train = np.empty((len(tm.YEARS) - 1, days, cells), dtype=np.bool_)
            annual = {}
            for i, year in enumerate(tm.YEARS):
                fold = baseline["folds"][str(year)]
                years = [y for y in tm.YEARS if y != year]
                if fold["training_years"] != years or fold["lightgbm"]["early_stopping_year"] != years[-1]:
                    raise ValueError(f"V1 report has a different fold protocol for {year}")
                for j, source in enumerate(keep):
                    train[:i, :, j], train[i:, :, j] = design[:i, :, source], design[i + 1:, :, source]
                    scored[:, j] = design[i, :, source]
                y_train[:i], y_train[i:] = targets["target_any"][:i], targets["target_any"][i + 1:]
                labels = {key: values[i] for key, values in targets.items()}
                totals = {key: int(values.sum()) for key, values in labels.items()}
                if totals != fold["positives"]:
                    raise ValueError(f"V1 target counts disagree for {year}")
                scores, _, fit = tm.fit_lightgbm(train.reshape(-1, len(keep)), y_train.ravel(), year_rows,
                                                  scored, kept, params)
                annual[year] = tm.score(scores, labels, totals, top)
                print(f"{name} {year}: {fit['refit_iterations']} refit rounds", flush=True)
            reference = full["annual"]
            ablations[name] = {
                "dropped_features": sorted(drop), "annual": annual, "summary": summarize(annual),
                "full_wins": {m: [y for y in tm.YEARS if reference[str(y)][m] > annual[y][m]] for m in annual[tm.YEARS[0]]},
                "ties": {m: [y for y in tm.YEARS if reference[str(y)][m] == annual[y][m]] for m in annual[tm.YEARS[0]]},
            }
            del train, scored, y_train
    return {"years": baseline["years"], "top_cells_per_day": top, "baseline": full,
            "ablation_parameters": report_params, "ablation": ablations}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--output", type=Path, default=CHECKS)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.output.write_text(json.dumps(build(args.threads), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
